//! Main thread — owns the DOM (built with Dominator) and the input, and
//! orchestrates the two workers.
//!
//! Flow:
//! 1. Build the canvas + HUD with Dominator and mount them.
//! 2. Ask the GPU what the Settings need (a software adapter, the largest
//!    texture), size the canvas from the Settings' resolution, transfer it to
//!    an `OffscreenCanvas`, and spawn the **render** worker with it and the
//!    anti-aliasing mode.
//! 3. When the render worker reports [`RenderMsg::PhysicsInit`], spawn the
//!    **physics** worker with that payload (the shared-memory binding).
//! 4. Translate keyboard input into the shared [`InputState`] block the physics
//!    worker polls (no per-keystroke `postMessage`).
//! 5. Own the orbit camera: apply every drag and wheel event to it, and once
//!    per main-thread frame write it into the shared [`CameraRing`] the
//!    render worker reads (see that type for why it is a timestamped ring).
//!
//! The Settings modal and the Stats panel are awsm-renderer-kit's
//! (`awsm-renderer-settings`, `awsm-renderer-stats`): this thread shows them
//! and passes their messages to the render worker, which applies the
//! settings and measures the stats.
//!
//! The DOM here stays deliberately thin (a canvas + a static controls HUD whose
//! one live line is a `futures-signals` `Mutable`) — the point of the template
//! is the threading, not the UI.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use awsm_renderer_settings::{
    probe_gpu, GpuInfo, SettingsConfig, SettingsHooks, SettingsOptions, SettingsPanel,
    SettingsStore, SettingsToWorker, SettingsUi,
};
use awsm_renderer_stats::{StatsHooks, StatsPanel, StatsToWorker};
use dominator::{clone, html};
use futures_signals::signal::Mutable;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, HtmlElement, KeyboardEvent, MessageEvent, Worker};

use crate::about::{About, Tab};
use crate::audio::AudioController;
use crate::bootstrap::{spawn_shared_worker, spawn_shared_worker_transfer};
use crate::protocol::{
    abs_time_us, AudioMsg, CameraRing, DropMsg, InputState, OrbitCamera, PhysicsMsg, RenderMsg,
    HELD_BACK, HELD_FORWARD, HELD_LEFT, HELD_RIGHT,
};

/// Shared, lazily-loaded audio. `None` until the export finishes loading; stays
/// silent until a user gesture starts it (browser autoplay policy).
type Audio = Rc<RefCell<Option<AudioController>>>;

/// The render worker, once spawned. The Settings and Stats panels exist
/// before it (their messages before then are dropped; both repeat what
/// matters).
type RenderRef = Rc<RefCell<Option<Worker>>>;

/// The prefix of every `localStorage` key this app writes: the resolution
/// (`res_scale`), the anti-aliasing mode (`aa`), the Stats panel (`stats`)
/// and its groups (`stats_open_<id>`). Every experiment is served from the
/// same origin (awsm.fun), so each app's keys carry its own prefix; change it
/// when starting a new project from this template.
const STORAGE_PREFIX: &str = "awsm_box3d_mt_";

/// The element id of the reduced-quality notice (awsm-renderer-settings
/// builds it): the About modal leaves Escape to it while it is up.
pub(crate) const QUALITY_NOTICE_ID: &str = "quality-notice";

/// The graphics Settings this app offers (awsm-renderer-settings' defaults):
///
/// - **Anti-aliasing**: Off, SMAA, MSAA 4×, TAA, and the TAA upscaling
///   presets Quality (67%), Balanced (58%) and Performance (50%). TAA is the
///   default on a desktop; a touch device and a software GPU start with Off
///   (they optimize for frame rate, and a one-time notice on a touch device
///   points at Settings). A stored choice always wins, and `?aa=<id>`
///   overrides it for one session.
/// - **Resolution** (the fill-rate lever): the canvas renders at
///   `devicePixelRatio × scale` device pixels, 50–100% in 5% steps. Fill
///   rate — not geometry — bounds this scene, so on a weak GPU (or a
///   high-DPR phone) this is what buys the framerate back. The default is
///   native resolution capped at a 2.8 MP pixel budget (never below an
///   effective DPR of 1), and lower on a touch device (60%) or a software
///   GPU (50%). (A Retina MacBook at native 2400×1794 ran this scene's GPU
///   frame over a 120 Hz display's budget; the renderer's per-pixel cost
///   scales with pixel count, so the budget is in pixels, not DPR.) TAA's
///   upscaling presets scale the renderer's internal size on top of it.
fn settings_config() -> SettingsConfig {
    SettingsConfig::default()
}

/// The pre-0.42 per-toggle anti-aliasing keys (no app prefix), migrated to
/// `<prefix>aa` once and then removed.
const LEGACY_AA_KEYS: (&str, &str) = ("awsm_msaa", "awsm_smaa");

/// Build + mount the DOM, then start the worker pipeline.
pub fn start() -> Result<(), JsValue> {
    // `?reset` in the URL forgets every stored setting (resolution,
    // anti-aliasing, the stats panel and its groups) so the load boots with
    // fresh defaults — the escape hatch when a device carries choices from old
    // sessions (which override the device-class defaults and suppress the
    // reduced-quality notice). It runs before anything reads a stored value.
    let store =
        SettingsStore::new(STORAGE_PREFIX).with_legacy_aa_keys(LEGACY_AA_KEYS.0, LEGACY_AA_KEYS.1);
    if store.reset_if_asked() {
        loading_log("?reset — stored settings cleared, booting with defaults");
    }
    store.migrate_legacy_aa();

    let status = Mutable::new("booting…".to_string());
    let about = About::default();
    let render_ref: RenderRef = Rc::new(RefCell::new(None));

    // The Stats panel (top right, hidden by default) and the bottom-left
    // Stats button. awsm-renderer-stats fills the panel from the render
    // worker's reports; it sends the worker which groups are open (only those
    // are measured in detail) and what it measured here (its frame gaps for
    // the refresh rate, and this thread's own work).
    let stats_el = element("div", "stats");
    let stats_button = element("button", "stats-btn");
    let _ = stats_button.set_attribute("type", "button");
    stats_button.set_text_content(Some("Stats"));
    let stats = StatsPanel::mount(
        STORAGE_PREFIX,
        &stats_el,
        Some(&stats_button),
        StatsHooks {
            post: Box::new(clone!(render_ref => move |message: &StatsToWorker| {
                if let Some(render) = render_ref.borrow().as_ref() {
                    post_kit(render, message.to_js());
                }
            })),
            on_help: Some(Box::new(clone!(about => move || about.show(Tab::Stats)))),
            page_part: PAGE_FRAME_PART.to_string(),
            app_sections: None,
            section_rows: None,
            shown: None,
        },
    );
    loading_log("wasm compiled — main thread booting");

    let app = html!("div", {
        .child(html!("canvas" => HtmlCanvasElement, {
            .class("canvas")
            .after_inserted(clone!(status, stats, render_ref => move |canvas| {
                // The GPU probe first: the Settings' defaults depend on it (a
                // software adapter starts lower), and the canvas is sized
                // from them before it goes to the render worker.
                spawn_local(async move {
                    let gpu = probe_gpu().await;
                    tracing::info!(
                        "main thread: gpu — fallback adapter {}, largest 2D texture {}",
                        gpu.is_fallback,
                        gpu.max_texture_dim
                    );
                    if let Err(e) = setup(canvas, status.clone(), stats, store, gpu, render_ref) {
                        status.set(format!("setup error: {e:?}"));
                        tracing::error!("main thread setup: {e:?}");
                    }
                });
            }))
        }))
        .child(html!("div", {
            .class("hud")
            .text("single-player-game-physics\nW/A/S/D or arrows: roll · Space: jump · click: drop a ball\nright-drag: orbit · wheel: zoom\ntouch: swipe to fling the ball · tap to drop\nsound starts on your first key or click\n")
            .child(html!("span", {
                .text_signal(status.signal_cloned())
            }))
        }))
        .child(dominator::DomBuilder::new(stats_el).into_dom())
        .child(dominator::DomBuilder::new(stats_button).into_dom())
        .child(about.button())
        .child(about.modal())
    });

    dominator::append_dom(&dominator::body(), app);
    Ok(())
}

/// The name of the camera frame loop in the Stats panel's page group.
const PAGE_FRAME_PART: &str = "camera frame loop";
/// The name of the sound cues in the Stats panel's page group.
const PAGE_CUES_PART: &str = "sound cues";

/// A new `<tag class="class">`.
fn element(tag: &str, class: &str) -> HtmlElement {
    let el: HtmlElement = dominator::body()
        .owner_document()
        .and_then(|d| d.create_element(tag).ok())
        .expect("create_element")
        .unchecked_into();
    el.set_class_name(class);
    el
}

/// Posts one of the kit's messages (already a JS value) to the render worker.
fn post_kit(render: &Worker, message: Result<JsValue, String>) {
    match message {
        Ok(value) => {
            let _ = render.post_message(&value);
        }
        Err(err) => tracing::error!("main thread: kit message: {err}"),
    }
}

/// Append a line to the loading screen's log (`#loading-log` in index.html —
/// present from the first byte so "loading code…" shows during the wasm
/// fetch). No-op once the overlay is gone.
pub fn loading_log(message: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };
    // Stamp each line with elapsed seconds since navigation — the gaps between
    // stamps are what tell us (and users filing reports) where a load stalls.
    let elapsed = window
        .performance()
        .map(|p| p.now() / 1000.0)
        .unwrap_or_default();
    let message = format!("[{elapsed:6.2}s] {message}");
    web_sys::console::log_1(&JsValue::from_str(&format!("loading: {message}")));
    let Some(log) = document.get_element_by_id("loading-log") else {
        return;
    };
    if let Ok(line) = document.create_element("div") {
        line.set_text_content(Some(&message));
        let _ = log.append_child(&line);
        // Keep the tail visible (the box masks/fades older lines).
        while log.child_element_count() > 14 {
            if let Some(first) = log.first_element_child() {
                first.remove();
            }
        }
    }
}

/// Fade out + drop the loading overlay (first frames are on screen).
fn loading_done() {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(overlay) = document.get_element_by_id("loading") else {
        return;
    };
    let _ = overlay.set_attribute("class", "done");
    // Remove after the CSS fade so it can't swallow clicks.
    let remove = Closure::once_into_js(move || overlay.remove());
    if let Some(window) = web_sys::window() {
        let _ = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(remove.unchecked_ref(), 600);
    }
}

/// Render a single full-screen message and nothing else. Used when the page
/// isn't cross-origin isolated, so the threaded app can't start at all — we
/// show *why* rather than booting into a confusing mid-spawn crash.
pub fn fatal(message: &str) {
    loading_log(&format!("FATAL: {message}"));
    loading_done();
    let app = html!("div", {
        .class("hud")
        .style("pointer-events", "auto")
        .style("max-width", "46em")
        .style("white-space", "pre-wrap")
        .text("single-player-game-physics — cannot start\n\n")
        .child(html!("span", { .text(message) }))
    });
    dominator::append_dom(&dominator::body(), app);
}

/// Runs once the canvas is in the DOM and the GPU has been probed: size it,
/// transfer it, spawn render, and arm the render→physics handoff, the
/// Settings and the input.
fn setup(
    canvas: HtmlCanvasElement,
    status: Mutable<String>,
    stats: StatsPanel,
    store: SettingsStore,
    gpu: GpuInfo,
    render_ref: RenderRef,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;

    // ── Settings: resolution + anti-aliasing (awsm-renderer-settings) ───────
    // The stored choices (or `?aa=`, or the device defaults: touch and the
    // software GPU probed above start lower) give the first canvas size and
    // mode; the bottom-right Settings modal changes them live and stores
    // them. The modal, the reduced-quality notice and the "compiling
    // pipelines" overlay go into the body, after the Settings button.
    let body = dominator::body();
    let settings_button = element("button", "settings-btn");
    let _ = settings_button.set_attribute("type", "button");
    settings_button.set_text_content(Some("Settings"));
    body.append_child(&settings_button)?;
    let settings = SettingsPanel::new(
        &body,
        &canvas,
        store,
        &SettingsOptions::new(&settings_config(), Some(gpu)),
        SettingsUi::default(),
        loading_log,
    );
    // Must be sized BEFORE transfer.
    let (w, h) = settings.backing();
    canvas.set_width(w);
    canvas.set_height(h);

    let offscreen = canvas.transfer_control_to_offscreen()?;
    // Base URL for our same-origin asset fetches (scene.toml, audio export).
    // This is the *directory* the page is served from — NOT just the origin:
    // on GitHub Pages project sites the page lives under `/<repo>/`, and the
    // assets are copied there too, so the origin alone (`https://host`) would
    // 404 (you'd fetch the org root and get its 404 HTML, breaking the TOML
    // parse). On `task dev` the page is at `/`, so this collapses to the origin.
    let base = page_base(&window);

    // The orbit camera. Main owns it (it has the pointer events): drags and
    // the wheel change it in `install_pointer`, which fans the yaw out into
    // the shared `InputState` (physics makes W/A/S/D camera-relative) and onto
    // the audio listener (the stereo image tracks the view). Once per main
    // frame it goes to the render worker through the `CameraRing`.
    let camera = Rc::new(RefCell::new(OrbitCamera::start()));
    // Leaked: it lives in the shared memory for the session, and the render
    // worker gets its address in the spawn payload.
    let camera_ring: &'static CameraRing = Box::leak(Box::new(CameraRing::new()));
    // Set by a drag or wheel; the frame loop writes the ring only then.
    let camera_dirty = Rc::new(Cell::new(false));
    // Time spent on sound cues since the last frame (the Stats panel's page
    // group).
    let cues_ms = Rc::new(Cell::new(0.0));
    install_camera_frames(
        &window,
        camera.clone(),
        camera_ring,
        camera_dirty.clone(),
        stats.clone(),
        cues_ms.clone(),
    )?;

    // Kick off the (async) audio load now; it stays silent until the first
    // gesture. `None` until ready, so input/handlers no-op against it meanwhile.
    let audio: Audio = Rc::new(RefCell::new(None));
    spawn_local(clone!(audio, status, base, camera => async move {
        match AudioController::load(&base).await {
            Ok(mut controller) => {
                // Catch the listener up in case the player orbited during load.
                controller.set_camera_yaw(camera.borrow().yaw);
                *audio.borrow_mut() = Some(controller);
                tracing::info!("main thread: audio loaded");
                loading_log("audio project loaded (roll + hit/land voices)");
            }
            Err(e) => {
                tracing::error!("main thread: audio load failed: {e:?}");
                status.set(format!("audio load error: {e:?}"));
            }
        }
    }));

    // Payload for the render worker: the OffscreenCanvas (transferred) + the
    // asset base URL (the worker's own `blob:` base can't resolve our fetches).
    let payload = js_sys::Object::new();
    set(&payload, "canvas", &offscreen);
    set(&payload, "origin", &JsValue::from_str(&base));
    // The APP base — the directory this page is served from (where index.html +
    // its copy-file'd `workers/` + `vendor/basis/` live) — distinct from
    // `origin`/`base` above, which in dev is the side media server (`MEDIA_BASE`,
    // :9001) that serves the bundle but NOT the Basis codec files. The render
    // worker builds the absolute worker/transcoder URLs from this. It keeps the
    // deploy PATH, not just the origin, so a subpath deploy (GitHub Pages
    // `/<repo>/`, or a CDN `/experiments/<slug>/`) resolves correctly — a bare
    // origin would 404.
    set(&payload, "app_base", &JsValue::from_str(&app_base(&window)));
    // The startup anti-aliasing mode (its id) rides the spawn payload so the
    // render worker BUILDS at it (no post-`Ready` reconcile recompile).
    set(&payload, "aa", &JsValue::from_str(settings.aa().id()));
    set(
        &payload,
        "camera_ptr",
        &JsValue::from_f64(camera_ring as *const CameraRing as usize as f64),
    );
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    // For timing this thread's own work (the Stats panel's page group).
    let perf = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;

    // Shared handle to the physics worker (spawned later, once render hands us
    // the arena binding). We keep it only to hold the worker alive — input no
    // longer flows through it.
    let physics: Rc<RefCell<Option<Worker>>> = Rc::new(RefCell::new(None));

    // The main→physics input channel: a shared-memory block main writes and the
    // physics worker polls (it replaces per-keystroke `postMessage`). Leaked so it
    // lives for the whole session at a stable address we hand the worker, which
    // attaches to the same `WebAssembly.Memory`.
    let input: &'static InputState = Box::leak(Box::new(InputState::new()));

    // Keyboard input is wired up immediately; it writes into `input`, which the
    // physics worker starts polling once spawned. A keypress also starts the audio
    // (the required user gesture).
    install_keyboard(&window, input, audio.clone())?;
    // Ask for keyboard focus up front, so W/A/S/D work without a click first.
    // At the top level this is a no-op; in a cross-origin iframe the browser
    // only allows it once the parent has focused the frame (the awsm.fun
    // shell does), and otherwise the first press on the canvas does it.
    let _ = window.focus();

    // Handle messages coming back from the render worker: the kit's (the
    // "compiling pipelines" overlay, the stats reports) first, then ours.
    let on_render_msg = Closure::<dyn FnMut(MessageEvent)>::new(
        clone!(physics, status, audio, settings, stats, cues_ms, perf => move |e: MessageEvent| {
            let data = e.data();
            if settings.handle_js(&data) || stats.handle_js(&data) {
                return;
            }
            match serde_wasm_bindgen::from_value::<RenderMsg>(data) {
                Ok(RenderMsg::Progress { message }) => {
                    loading_log(&message);
                }
                Ok(RenderMsg::PhysicsInit(mut init)) => {
                    tracing::info!("main thread: render ready, spawning physics worker");
                    loading_log("spawning physics worker…");
                    // Hand physics the shared input block's address (same memory).
                    init.input_ptr = input as *const InputState as usize as f64;
                    match serde_wasm_bindgen::to_value(&init) {
                        Ok(payload) => {
                            // The physics worker streams gameplay AudioMsgs back here
                            // (dispatched into the audio controller once loaded), plus
                            // the occasional PhysicsMsg control request — notably
                            // SpawnTaskWorkers: physics blocks on the frame-tick futex
                            // for its whole life, so MAIN spawns its task-pool workers
                            // (a live event loop is needed to start nested workers and
                            // to see any startup error they post back).
                            let on_phys = Closure::<dyn FnMut(MessageEvent)>::new(clone!(audio, cues_ms, perf => move |e: MessageEvent| {
                                let t0 = perf.now();
                                if let Ok(msg) = serde_wasm_bindgen::from_value::<AudioMsg>(e.data()) {
                                    if let Some(c) = audio.borrow_mut().as_mut() {
                                        c.on_audio(msg);
                                    }
                                    cues_ms.set(cues_ms.get() + perf.now() - t0);
                                } else if let Ok(PhysicsMsg::SpawnTaskWorkers { pool, count }) =
                                    serde_wasm_bindgen::from_value::<PhysicsMsg>(e.data())
                                {
                                    tracing::info!("main thread: spawning {count} physics-task workers");
                                    loading_log(&format!("spawning {count} physics task-pool workers…"));
                                    for i in 0..count {
                                        let payload = js_sys::Object::new();
                                        let _ = js_sys::Reflect::set(&payload, &JsValue::from_str("pool"), &JsValue::from_f64(pool));
                                        let _ = js_sys::Reflect::set(&payload, &JsValue::from_str("index"), &JsValue::from_f64((i + 1) as f64));
                                        let on_task_msg = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
                                            tracing::error!("physics-task worker posted: {:?}", e.data());
                                        });
                                        match spawn_shared_worker("physics-task", &payload, on_task_msg.as_ref().unchecked_ref()) {
                                            // Leak the handle: the pool runs for the session.
                                            Ok(worker) => std::mem::forget(worker),
                                            Err(err) => tracing::error!("spawn physics-task worker: {err:?}"),
                                        }
                                        on_task_msg.forget();
                                    }
                                }
                            }));
                            match spawn_shared_worker("physics", &payload, on_phys.as_ref().unchecked_ref()) {
                                Ok(worker) => *physics.borrow_mut() = Some(worker),
                                Err(err) => {
                                    tracing::error!("spawn physics: {err:?}");
                                    status.set(format!("spawn physics error: {err:?}"));
                                }
                            }
                            on_phys.forget();
                        }
                        Err(err) => tracing::error!("serialize PhysicsInit: {err}"),
                    }
                }
                Ok(RenderMsg::Ready) => {
                    loading_log("first frames rendered — ready");
                    loading_done();
                    // The game is visible now — if this load applied the
                    // reduced touch-device defaults, say so.
                    settings.ready();
                    status.set("playing — roll · Space jump · click drops a ball · right-drag orbit".into());
                    // No AA reconcile here: the render worker BUILT at the
                    // desired mode (it rode the spawn payload).
                }
                Ok(RenderMsg::Error { message }) => {
                    loading_log(&format!("ERROR: {message}"));
                    status.set(format!("render error: {message}"));
                }
                Err(_) => { /* not a RenderMsg (e.g. an init-error blob) — ignore */ }
            }
        }),
    );

    loading_log("spawning render worker…");
    let render = spawn_shared_worker_transfer(
        "render",
        &payload,
        &transfer,
        on_render_msg.as_ref().unchecked_ref(),
    )?;
    on_render_msg.forget();
    // Publish the handle for the Settings and Stats panels' messages.
    *render_ref.borrow_mut() = Some(render.clone());

    // The Settings start following the canvas's layout size (the backing
    // store is CSS size × dpr × the resolution scale; without this it keeps
    // its initial size and the browser stretches it — circles become ovals
    // after any window resize) and their controls. Every change goes to the
    // render worker, which owns the transferred OffscreenCanvas and the
    // renderer.
    settings.start(
        &canvas,
        SettingsHooks {
            on_resize: Box::new(clone!(render => move |width, height| {
                post_kit(&render, SettingsToWorker::Resize { width, height }.to_js());
            })),
            on_anti_alias: Box::new(clone!(render => move |mode| {
                post_kit(&render, SettingsToWorker::AntiAlias { mode }.to_js());
            })),
            open_button: Some(settings_button.into()),
            extra_rows: Vec::new(),
        },
    );

    // Left-clicks drop balls (relayed to render for unprojection); right-drags
    // orbit and the wheel zooms; a pointerdown is also the gesture that starts
    // audio. Orbit drags and the wheel change `camera` (see above).
    install_pointer(&window, &canvas, render, audio, input, camera, camera_dirty)?;

    status.set("loading scene…".into());
    Ok(())
}

// ── Touch fling tuning ───────────────────────────────────────────────────────
/// CSS px a touch must travel before it counts as a swipe — under this it's a
/// tap (→ the browser's click drops a ball).
const TAP_SLOP_PX: f32 = 12.0;
/// The release-velocity window (ms): the fling speed is measured from the
/// oldest sample still inside this window to the lift point, so it captures
/// the terminal flick of the gesture, not the whole drag's average.
const FLING_WINDOW_MS: f64 = 120.0;
/// Swipe speed (CSS px/s) → ball speed (m/s). A brisk phone flick lands around
/// 1000–2500 px/s, i.e. 4–10 m/s before the physics-side cap.
const FLING_GAIN: f32 = 0.004;
/// Swipes slower than this (m/s) are ignored — a slow deliberate drag isn't a
/// throw, and firing it would feel like phantom input.
const FLING_MIN_SPEED: f32 = 0.25;

/// Live tracking of the (single) touch pointer that may become a fling: where
/// it started (tap-vs-swipe slop) and its recent move samples
/// (`event.timeStamp` ms, x, y) for the release velocity.
struct Swipe {
    id: i32,
    origin: (f32, f32),
    samples: Vec<(f64, f32, f32)>,
}

/// Wire pointer input: a **left click drops a ball** — the click point goes to
/// the render worker as a [`DropMsg`] in NDC (render does the unprojection
/// onto the table, through the camera it last drew). A **right-button drag
/// orbits** and the **wheel zooms** main's [`OrbitCamera`] (marking it dirty
/// for the next [`CameraRing`] write); an orbit also fans the new yaw out to
/// the shared [`InputState`] (camera-relative W/A/S/D) and the audio listener
/// (camera-relative stereo).
/// A **touch swipe flings the ball**: the swipe's release velocity becomes a
/// camera-frame fling in the shared [`InputState`] (physics turns it into the
/// ball's horizontal velocity); a short still touch stays a tap → drop.
fn install_pointer(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    render: Worker,
    audio: Audio,
    input: &'static InputState,
    camera: Rc<RefCell<OrbitCamera>>,
    camera_dirty: Rc<Cell<bool>>,
) -> Result<(), JsValue> {
    // Set when a touch swipe just flung the ball: the browser still
    // synthesizes a `click` on the canvas after the pointer sequence, and a
    // fling must not ALSO drop a ball. Consumed (reset) by the click handler.
    let suppress_click = Rc::new(Cell::new(false));

    // Click → drop a ball at the clicked table spot. NDC: x right, y up.
    let click_canvas = canvas.clone();
    let click = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(
        clone!(render, audio, suppress_click => move |e: web_sys::MouseEvent| {
            // A swipe just ended — a fling is not a request for a new ball.
            if suppress_click.take() {
                return;
            }
            // The click is also a user gesture — the first one starts audio,
            // so this very drop is audible.
            if let Some(c) = audio.borrow_mut().as_mut() {
                c.ensure_started();
            }
            let w = click_canvas.client_width().max(1) as f32;
            let h = click_canvas.client_height().max(1) as f32;
            let ndc_x = (e.offset_x() as f32 / w) * 2.0 - 1.0;
            let ndc_y = 1.0 - (e.offset_y() as f32 / h) * 2.0;
            if let Ok(v) = serde_wasm_bindgen::to_value(&DropMsg::Ball { ndc_x, ndc_y }) {
                let _ = render.post_message(&v);
            }
        }),
    );
    canvas.add_event_listener_with_callback("click", click.as_ref().unchecked_ref())?;
    click.forget();

    let dragging = Rc::new(Cell::new(false));

    // ── Touch fling (mobile) ────────────────────────────────────────────────
    // A swipe throws the ball: we track the active touch pointer's recent
    // samples and, on lift, turn the end-of-swipe velocity into a camera-frame
    // fling published through the shared `InputState` (physics rotates it by
    // the camera yaw, same as W/A/S/D — screen-up = away = forward). Only the
    // first finger is tracked, so a second finger can't hijack the gesture.
    let swipe: Rc<RefCell<Option<Swipe>>> = Rc::new(RefCell::new(None));

    // pointerdown on the canvas: the RIGHT button begins an orbit drag (left
    // stays the drop-a-ball click); a touch begins a possible swipe. Any
    // button is a user gesture → start audio.
    let focus_window = window.clone();
    let down = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, audio, swipe => move |e: web_sys::PointerEvent| {
            // Keyboard focus to this page. On awsm.fun it runs inside an
            // iframe, and keys go to whichever frame has focus: a press on the
            // canvas must bring W/A/S/D here, not leave them with the parent.
            let _ = focus_window.focus();
            if e.button() == 2 {
                dragging.set(true);
            }
            // First finger down starts the swipe track; ignore extra fingers.
            if e.pointer_type() == "touch" && swipe.borrow().is_none() {
                let (x, y) = (e.client_x() as f32, e.client_y() as f32);
                *swipe.borrow_mut() = Some(Swipe {
                    id: e.pointer_id(),
                    origin: (x, y),
                    samples: vec![(e.time_stamp(), x, y)],
                });
            }
            if let Some(c) = audio.borrow_mut().as_mut() {
                c.ensure_started();
            }
        }),
    );
    canvas.add_event_listener_with_callback("pointerdown", down.as_ref().unchecked_ref())?;
    down.forget();

    // Suppress the OS context menu on the canvas so the right-drag is a clean
    // orbit gesture (macOS decides the menu at pointerdown, Windows at -up —
    // both arrive here as `contextmenu`).
    let menu = Closure::<dyn FnMut(web_sys::Event)>::new(|e: web_sys::Event| {
        e.prevent_default();
    });
    canvas.add_event_listener_with_callback("contextmenu", menu.as_ref().unchecked_ref())?;
    menu.forget();

    // pointermove on the window so a drag keeps orbiting even off-canvas. The
    // orbit turns main's camera; the new yaw goes straight to physics (shared
    // input block) and the audio listener, and the whole camera to render
    // with the next frame's `CameraRing` write.
    let move_ = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, audio, camera, camera_dirty, swipe => move |e: web_sys::PointerEvent| {
            // Touch swipe: record the sample; velocity is computed at lift.
            if let Some(s) = swipe.borrow_mut().as_mut() {
                if e.pointer_id() == s.id {
                    let t = e.time_stamp();
                    s.samples.push((t, e.client_x() as f32, e.client_y() as f32));
                    // Prune to the release-velocity window, always keeping one
                    // sample older than it as the measurement anchor.
                    while s.samples.len() > 1 && t - s.samples[1].0 > FLING_WINDOW_MS {
                        s.samples.remove(0);
                    }
                    e.prevent_default();
                }
                return;
            }
            if dragging.get() {
                let y = {
                    let mut cam = camera.borrow_mut();
                    cam.orbit(e.movement_x() as f32, e.movement_y() as f32);
                    cam.yaw
                };
                camera_dirty.set(true);
                input.set_camera_yaw(y);
                if let Some(c) = audio.borrow_mut().as_mut() {
                    c.set_camera_yaw(y);
                }
            }
        }),
    );
    window.add_event_listener_with_callback("pointermove", move_.as_ref().unchecked_ref())?;
    move_.forget();

    // pointerup ends the drag and resolves the swipe: past the tap slop it's a
    // gesture (suppress the trailing click); fast enough, it's a fling — the
    // end-of-swipe velocity goes to physics in the camera frame (screen-right
    // = +x, screen-down = +z toward the camera, matching the held-key basis).
    let up = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, swipe, suppress_click => move |e: web_sys::PointerEvent| {
            dragging.set(false);
            let ours = matches!(&*swipe.borrow(), Some(s) if s.id == e.pointer_id());
            if !ours {
                return;
            }
            let Some(s) = swipe.borrow_mut().take() else {
                return;
            };
            let (lx, ly) = (e.client_x() as f32, e.client_y() as f32);
            let travel =
                ((lx - s.origin.0).powi(2) + (ly - s.origin.1).powi(2)).sqrt();
            if travel < TAP_SLOP_PX {
                return; // a tap — the browser's click follows and drops a ball
            }
            suppress_click.set(true);
            // Velocity from the oldest in-window sample to the lift point.
            let t = e.time_stamp();
            let &(t0, x0, y0) = s
                .samples
                .iter()
                .find(|(ts, _, _)| t - ts <= FLING_WINDOW_MS)
                .unwrap_or(&s.samples[0]);
            let dt = ((t - t0) / 1000.0) as f32;
            if dt <= 0.0 {
                return;
            }
            let vx = (lx - x0) / dt * FLING_GAIN;
            let vz = (ly - y0) / dt * FLING_GAIN;
            if (vx * vx + vz * vz).sqrt() >= FLING_MIN_SPEED {
                input.bump_fling(vx, vz);
            }
        }),
    );
    window.add_event_listener_with_callback("pointerup", up.as_ref().unchecked_ref())?;
    up.forget();

    // A cancelled pointer (system gesture, finger slid off) abandons the
    // swipe — no fling from a gesture the user didn't finish.
    let cancel = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, swipe => move |e: web_sys::PointerEvent| {
            dragging.set(false);
            let ours = matches!(&*swipe.borrow(), Some(s) if s.id == e.pointer_id());
            if ours {
                *swipe.borrow_mut() = None;
            }
        }),
    );
    window.add_event_listener_with_callback("pointercancel", cancel.as_ref().unchecked_ref())?;
    cancel.forget();

    // wheel on the canvas zooms (preventDefault so the page doesn't scroll).
    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
        e.prevent_default();
        camera.borrow_mut().zoom(e.delta_y() as f32);
        camera_dirty.set(true);
    });
    canvas.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();

    Ok(())
}

/// A self-rescheduling `requestAnimationFrame` callback.
type FrameLoop = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// The main thread's own `requestAnimationFrame` loop for the camera: in
/// every frame the camera changed in, write it into the [`CameraRing`]
/// stamped with this frame's time. Pointer events are delivered before the
/// frame's rAF callbacks, so each write holds exactly that frame's input; the
/// render worker picks the previous frame's write (see [`CameraRing`]).
fn install_camera_frames(
    window: &web_sys::Window,
    camera: Rc<RefCell<OrbitCamera>>,
    ring: &'static CameraRing,
    dirty: Rc<Cell<bool>>,
    stats: StatsPanel,
    cues_ms: Rc<Cell<f64>>,
) -> Result<(), JsValue> {
    let perf = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;
    let time_origin = perf.time_origin();
    let cb: FrameLoop = Rc::new(RefCell::new(None));
    let next = cb.clone();
    let win = window.clone();
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        let t0 = perf.now();
        if dirty.replace(false) {
            ring.write(abs_time_us(time_origin, t), &camera.borrow());
        }
        // This thread's own work, for the Stats panel's page group: the
        // camera write, and the sound cues played since the last frame.
        stats.page_frame(perf.now() - t0, Some(PAGE_FRAME_PART), true);
        stats.page_frame(cues_ms.take(), Some(PAGE_CUES_PART), false);
        if let Some(f) = next.borrow().as_ref() {
            let _ = win.request_animation_frame(f.as_ref().unchecked_ref());
        }
    }));
    if let Some(f) = cb.borrow().as_ref() {
        window.request_animation_frame(f.as_ref().unchecked_ref())?;
    }
    // Runs for the session; the closure keeps itself alive.
    std::mem::forget(cb);
    Ok(())
}

/// Attach `keydown`/`keyup` listeners that translate WASD/arrows into the shared
/// [`InputState`]'s held-key bits and Space into a jump bump. The physics worker
/// polls that block each step — no `postMessage`.
fn install_keyboard(
    window: &web_sys::Window,
    input: &'static InputState,
    audio: Audio,
) -> Result<(), JsValue> {
    // keydown.
    let down = Closure::<dyn FnMut(KeyboardEvent)>::new(clone!(audio => move |e: KeyboardEvent| {
        // Any keypress is a user gesture — start audio if it's ready.
        if let Some(c) = audio.borrow_mut().as_mut() {
            c.ensure_started();
        }
        let key = e.key();
        if key == " " || key == "Spacebar" {
            // A button reached with the keyboard keeps Space: it presses the
            // button. (One that was clicked does not: `:focus-visible` is the
            // browser's own test for "focused from the keyboard".)
            if on_keyboard_focused_control(&e) {
                return;
            }
            // Edge-triggered: ignore auto-repeat so one press = one hop.
            if !e.repeat() {
                input.bump_jump();
            }
            e.prevent_default();
            return;
        }
        if let Some(mask) = key_mask(&key) {
            input.set_held(mask, true);
        }
    }));
    window.add_event_listener_with_callback("keydown", down.as_ref().unchecked_ref())?;
    down.forget();

    // keyup.
    let up = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
        if let Some(mask) = key_mask(&e.key()) {
            input.set_held(mask, false);
        }
    });
    window.add_event_listener_with_callback("keyup", up.as_ref().unchecked_ref())?;
    up.forget();

    Ok(())
}

/// Keeps a key pressed inside a modal from the game's `window` listeners, so
/// the arrows move a slider, a select or a tab and Space presses a button.
/// Escape goes through: it closes the modal.
///
/// Attach it with `EventOptions::bubbles()`: dominator's plain `.event` listens
/// in the capture phase, where stopping the key would also keep it from the
/// controls inside the modal.
pub(crate) fn keep_keys_in_modal(e: dominator::events::KeyDown) {
    if e.key() != "Escape" {
        e.stop_propagation();
    }
}

/// Whether a key event is aimed at a control that has the keyboard's focus.
fn on_keyboard_focused_control(e: &KeyboardEvent) -> bool {
    e.target()
        .and_then(|target| target.dyn_into::<web_sys::Element>().ok())
        .is_some_and(|el| {
            el.matches("button:focus-visible, select:focus-visible, a[href]:focus-visible")
                .unwrap_or(false)
        })
}

/// The `HELD_*` bit a key drives (WASD or arrows). Unknown keys → `None`.
fn key_mask(key: &str) -> Option<u32> {
    match key {
        "w" | "W" | "ArrowUp" => Some(HELD_FORWARD),
        "s" | "S" | "ArrowDown" => Some(HELD_BACK),
        "a" | "A" | "ArrowLeft" => Some(HELD_LEFT),
        "d" | "D" | "ArrowRight" => Some(HELD_RIGHT),
        _ => None,
    }
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

/// The base URL for the media fetches (`bundle/` and `audio/`), with no
/// trailing slash. Resolution order:
///
/// 1. `?media=<url>` query param — explicit per-tab override (e.g. point one
///    tab at another machine's export server).
/// 2. `MEDIA_BASE` compile-time env — `task dev` sets it to the side media
///    server (`http://127.0.0.1:9001`, see the Taskfile), so a dev build
///    ALWAYS fetches live media: editor exports are picked up on plain
///    reload, with no Trunk rebuild/reload storm and no stale dist copy.
///    `task build` doesn't set it, so production builds fall through.
/// 3. The directory the page is served from ([`app_base`]) — the production
///    path; correct under a subpath deploy (`/<repo>/`, `/experiments/<slug>/`)
///    because it keeps the path, so `{base}/bundle/scene.toml` resolves
///    alongside `index.html`.
///
/// Cross-origin media is fine: everything goes through `fetch` (CORS), which
/// COEP permits when the media server sends `Access-Control-Allow-Origin`.
fn page_base(window: &web_sys::Window) -> String {
    let location = window.location();
    if let Ok(search) = location.search() {
        if let Ok(params) = web_sys::UrlSearchParams::new_with_str(&search) {
            if let Some(media) = params.get("media") {
                if !media.is_empty() {
                    return media.trim_end_matches('/').to_string();
                }
            }
        }
    }
    if let Some(base) = option_env!("MEDIA_BASE") {
        if !base.is_empty() {
            return base.trim_end_matches('/').to_string();
        }
    }
    app_base(window)
}

/// The directory THIS PAGE is served from — origin + the path up to (not
/// including) the trailing `/` — with no trailing slash. This is where Trunk
/// lays the app down: `index.html`, the wasm/JS glue, and the copy-file'd
/// `workers/` + `vendor/basis/`. Keeping the path makes it correct at any
/// deployment depth: the root (`/`), a GitHub Pages project site (`/<repo>/`),
/// or a CDN subpath (`/experiments/<slug>/`). Unlike [`page_base`], it never
/// redirects to the media server, so it always points at the app's own files.
///   `https://host/experiments/foo/`           → `https://host/experiments/foo`
///   `https://host/experiments/foo/index.html` → `https://host/experiments/foo`
///   `http://127.0.0.1:9000/`                   → `http://127.0.0.1:9000`
fn app_base(window: &web_sys::Window) -> String {
    let location = window.location();
    let origin = location.origin().unwrap_or_default();
    let mut path = location.pathname().unwrap_or_default();
    // Drop the trailing filename (or trailing slash), keeping the directory.
    if let Some(idx) = path.rfind('/') {
        path.truncate(idx);
    }
    format!("{}{}", origin.trim_end_matches('/'), path)
}
