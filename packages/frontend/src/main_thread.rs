//! Main thread — owns the DOM (built with Dominator) and the input, and
//! orchestrates the two workers.
//!
//! Flow:
//! 1. Build the canvas + HUD with Dominator and mount them.
//! 2. Size the canvas to native resolution, transfer it to an `OffscreenCanvas`,
//!    and spawn the **render** worker with it.
//! 3. When the render worker reports [`RenderMsg::PhysicsInit`], spawn the
//!    **physics** worker with that payload (the shared-memory binding).
//! 4. Translate keyboard input into the shared [`InputState`] block the physics
//!    worker polls (no per-keystroke `postMessage`).
//!
//! The DOM here stays deliberately thin (a canvas + a static controls HUD whose
//! one live line is a `futures-signals` `Mutable`) — the point of the template
//! is the threading, not the UI.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html};
use futures_signals::signal::Mutable;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MessageEvent, Worker};

use crate::audio::AudioController;
use crate::bootstrap::{spawn_shared_worker, spawn_shared_worker_transfer};
use crate::physics_tasks::TaskPool;
use crate::protocol::{
    AudioMsg, BallMotions, BodyMotion, CameraMsg, DropMsg, InputState, PhysicsMsg, RenderMsg,
    ResizeMsg, CAMERA_ORBIT_SENSITIVITY, HELD_BACK, HELD_FORWARD, HELD_LEFT, HELD_RIGHT,
};

/// Shared, lazily-loaded audio. `None` until the export finishes loading; stays
/// silent until a user gesture starts it (browser autoplay policy).
type Audio = Rc<RefCell<Option<AudioController>>>;

/// Shared-memory addresses the stats panel samples (filled in as the workers
/// come up): the player's [`BodyMotion`] (frame tick + step count), the
/// [`BallMotions`] block (dropped-ball count), and the task pool + its worker
/// count. All are leaked-for-the-session blocks, so raw addresses are stable.
#[derive(Default, Clone, Copy)]
struct StatsRefs {
    motion: Option<usize>,
    balls: Option<usize>,
    pool: Option<(usize, u32)>,
}

/// Previous sample for rate computation (per-second deltas).
#[derive(Default, Clone)]
struct StatsSample {
    frame: u32,
    step: u32,
    claims: Vec<u32>,
}

/// Build + mount the DOM, then start the worker pipeline.
pub fn start() -> Result<(), JsValue> {
    let status = Mutable::new("booting…".to_string());
    let stats = Mutable::new(String::new());
    let about_open = Mutable::new(false);
    loading_log("wasm compiled — main thread booting");

    let app = html!("div", {
        .child(html!("canvas" => HtmlCanvasElement, {
            .class("canvas")
            .after_inserted(clone!(status, stats => move |canvas| {
                if let Err(e) = setup(canvas, status.clone(), stats.clone()) {
                    status.set(format!("setup error: {e:?}"));
                    tracing::error!("main thread setup: {e:?}");
                }
            }))
        }))
        .child(html!("div", {
            .class("hud")
            .text("single-player-game-physics\nW/A/S/D or arrows: roll · Space: jump · click: drop a ball\nright-drag: orbit · wheel: zoom\nsound starts on your first key or click\n")
            .child(html!("span", {
                .text_signal(status.signal_cloned())
            }))
        }))
        .child(html!("div", {
            .class("stats")
            .text_signal(stats.signal_cloned())
        }))
        .child(about_button(&about_open))
        .child(about_modal(&about_open))
    });

    dominator::append_dom(&dominator::body(), app);
    Ok(())
}

/// The bottom-center "About" chip. Lives outside the canvas, so clicking it
/// never drops a ball (the drop listener is on the canvas element itself).
fn about_button(open: &Mutable<bool>) -> dominator::Dom {
    html!("button", {
        .class("about-btn")
        .text("About")
        .event(clone!(open => move |_: dominator::events::Click| {
            open.set_neq(true);
        }))
    })
}

/// The About overlay: what this template is + where it lives. Backdrop click
/// or Escape closes; clicks inside the card don't propagate to the backdrop.
fn about_modal(open: &Mutable<bool>) -> dominator::Dom {
    const REPO: &str = "https://github.com/awsm-fun/awsm-template-physics-multithreaded";
    let link = |href: &str, label: &str| {
        html!("a", {
            .attr("href", href)
            .attr("target", "_blank")
            .attr("rel", "noopener")
            .text(label)
        })
    };
    html!("div", {
        .class("about-overlay")
        .visible_signal(open.signal())
        // Close only when the click lands on the BACKDROP itself — checking
        // the event target (not relying on stop_propagation from the card)
        // so clicks inside the modal never close it.
        .event(clone!(open => move |e: dominator::events::Click| {
            let on_backdrop = e
                .dyn_target::<web_sys::Element>()
                .map(|el| el.class_list().contains("about-overlay"))
                .unwrap_or(false);
            if on_backdrop {
                open.set_neq(false);
            }
        }))
        .global_event(clone!(open => move |e: dominator::events::KeyDown| {
            if e.key() == "Escape" {
                open.set_neq(false);
            }
        }))
        .child(html!("div", {
            .class("about-modal")
            .child(html!("button", {
                .class("about-close")
                .attr("aria-label", "Close")
                .text("×")
                .event(clone!(open => move |_: dominator::events::Click| {
                    open.set_neq(false);
                }))
            }))
            .child(html!("h2", { .text("Multithreaded Physics Demo") }))
            .child(html!("p", {
                .text("A copyable template from the ")
                .child(link("https://awsm.fun", "Awsm"))
                .text(" project — a WebGPU game skeleton running across three \
                       wasm threads plus a physics task pool, all over one \
                       shared WebAssembly.Memory: no postMessage on any hot \
                       path, real atomics, wasm SIMD in the solver.")
            }))
            .child(html!("ul", {
                .child(html!("li", {
                    .text("Rendering: ")
                    .child(link("https://scene.awsm.fun", "AwsmRenderer"))
                }))
                .child(html!("li", {
                    .text("Audio: ")
                    .child(link("https://audio.awsm.fun", "AwsmAudio"))
                }))
                .child(html!("li", {
                    .text("Physics: ")
                    .child(link("https://github.com/erincatto/box3d", "Box3D"))
                }))
            }))
            .child(html!("p", {
                .text("Roll the red ball with WASD, jump with Space, click the table to \
                       drop more balls — every impact is synthesized and spatialized in \
                       real time from the physics contacts. Drag with the right mouse \
                       button to orbit the camera and scroll to zoom; the controls and \
                       the stereo image stay relative to your view from any angle.")
            }))
            .child(html!("p", {
                .text("The stats panel (top right) shows each thread live. A \"task\" is \
                       one slice of Box3D's internal parallel-for: every physics step, \
                       the solver splits its work into small tasks that the physics \
                       thread and the pool workers race to claim from shared memory — \
                       the physics thread help-executes while it waits, so it wins most \
                       races when the scene is small. Drop more balls and watch the \
                       pool's share grow.")
            }))
            .child(html!("p", {
                .class("about-repo")
                .child(link(REPO, "Source on GitHub"))
            }))
            .child(html!("p", {
                .class("about-footer")
                .text("Built with ❤ by David Komer")
            }))
        }))
    })
}

/// Append a line to the loading screen's log (`#loading-log` in index.html —
/// present from the first byte so "loading code…" shows during the wasm
/// fetch). No-op once the overlay is gone.
pub fn loading_log(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(log) = document.get_element_by_id("loading-log") else {
        return;
    };
    if let Ok(line) = document.create_element("div") {
        line.set_text_content(Some(message));
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

/// Runs once the canvas is in the DOM: size it, transfer it, spawn render, and
/// arm the render→physics handoff + keyboard input.
fn setup(
    canvas: HtmlCanvasElement,
    status: Mutable<String>,
    stats: Mutable<String>,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;

    // Size the backing store to CSS size × devicePixelRatio so the worker
    // renders at native resolution. Must happen BEFORE transfer.
    let dpr = window.device_pixel_ratio().max(1.0);
    let w = ((canvas.client_width().max(1) as f64) * dpr).round() as u32;
    let h = ((canvas.client_height().max(1) as f64) * dpr).round() as u32;
    canvas.set_width(w.max(1));
    canvas.set_height(h.max(1));

    let offscreen = canvas.transfer_control_to_offscreen()?;
    // Base URL for our same-origin asset fetches (scene.toml, audio export).
    // This is the *directory* the page is served from — NOT just the origin:
    // on GitHub Pages project sites the page lives under `/<repo>/`, and the
    // assets are copied there too, so the origin alone (`https://host`) would
    // 404 (you'd fetch the org root and get its 404 HTML, breaking the TOML
    // parse). On `task dev` the page is at `/`, so this collapses to the origin.
    let base = page_base(&window);

    // The camera yaw, accumulated from orbit drags in `install_pointer`. Main
    // owns it and fans it out: into the shared `InputState` (physics makes
    // W/A/S/D camera-relative) and onto the audio listener (the stereo image
    // tracks the view). The render thread integrates the same deltas with the
    // same sensitivity into its `OrbitCamera`, so all three agree.
    let camera_yaw = Rc::new(Cell::new(0.0_f32));

    // Kick off the (async) audio load now; it stays silent until the first
    // gesture. `None` until ready, so input/handlers no-op against it meanwhile.
    let audio: Audio = Rc::new(RefCell::new(None));
    spawn_local(clone!(audio, status, base, camera_yaw => async move {
        match AudioController::load(&base).await {
            Ok(mut controller) => {
                // Catch the listener up in case the player orbited during load.
                controller.set_camera_yaw(camera_yaw.get());
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
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    // Shared-memory addresses the stats panel samples, filled in as workers
    // report them (PhysicsInit → motion/balls, SpawnTaskWorkers → pool).
    let stats_refs: Rc<RefCell<StatsRefs>> = Rc::new(RefCell::new(StatsRefs::default()));
    install_stats(&window, stats, stats_refs.clone())?;

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

    // Handle messages coming back from the render worker.
    let on_render_msg = Closure::<dyn FnMut(MessageEvent)>::new(
        clone!(physics, status, audio, stats_refs => move |e: MessageEvent| {
            match serde_wasm_bindgen::from_value::<RenderMsg>(e.data()) {
                Ok(RenderMsg::Progress { message }) => {
                    loading_log(&message);
                }
                Ok(RenderMsg::PhysicsInit(mut init)) => {
                    tracing::info!("main thread: render ready, spawning physics worker");
                    loading_log("spawning physics worker…");
                    // Hand physics the shared input block's address (same memory).
                    init.input_ptr = input as *const InputState as usize as f64;
                    // Remember the shared blocks for the stats panel.
                    {
                        let mut refs = stats_refs.borrow_mut();
                        refs.motion = Some(init.motion_ptr as usize);
                        refs.balls = Some(init.balls_ptr as usize);
                    }
                    match serde_wasm_bindgen::to_value(&init) {
                        Ok(payload) => {
                            // The physics worker streams gameplay AudioMsgs back here
                            // (dispatched into the audio controller once loaded), plus
                            // the occasional PhysicsMsg control request — notably
                            // SpawnTaskWorkers: physics blocks on the frame-tick futex
                            // for its whole life, so MAIN spawns its task-pool workers
                            // (a live event loop is needed to start nested workers and
                            // to see any startup error they post back).
                            let on_phys = Closure::<dyn FnMut(MessageEvent)>::new(clone!(audio, stats_refs => move |e: MessageEvent| {
                                if let Ok(msg) = serde_wasm_bindgen::from_value::<AudioMsg>(e.data()) {
                                    if let Some(c) = audio.borrow_mut().as_mut() {
                                        c.on_audio(msg);
                                    }
                                } else if let Ok(PhysicsMsg::SpawnTaskWorkers { pool, count }) =
                                    serde_wasm_bindgen::from_value::<PhysicsMsg>(e.data())
                                {
                                    tracing::info!("main thread: spawning {count} physics-task workers");
                                    loading_log(&format!("spawning {count} physics task-pool workers…"));
                                    stats_refs.borrow_mut().pool = Some((pool as usize, count));
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
                    status.set("playing — roll · Space jump · click drops a ball · right-drag orbit".into());
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

    // Track canvas layout size → render worker (which owns the transferred
    // OffscreenCanvas and applies the new backing size). Without this the
    // backing store keeps its initial size and the browser stretches it —
    // circles become ovals after any window resize.
    install_resize(&window, &canvas, &render)?;

    // Left-clicks drop balls (relayed to render for unprojection); right-drags
    // orbit and the wheel zooms; a pointerdown is also the gesture that starts
    // audio. Orbit drags also advance `camera_yaw` and fan it out (see above).
    install_pointer(&window, &canvas, render, audio, input, camera_yaw)?;

    status.set("loading scene…".into());
    Ok(())
}

/// Observe the canvas element's layout size (only main sees layout) and relay
/// changes to the render worker as [`ResizeMsg`]s in DEVICE pixels (CSS size ×
/// `devicePixelRatio`, same convention as the initial sizing in [`setup`]).
/// The observer fires once on `observe` per spec — the worker dedups repeats.
/// Mirrors the ResizeObserver pattern in awsm-renderer's editor/model-viewer.
fn install_resize(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    render: &Worker,
) -> Result<(), JsValue> {
    let win = window.clone();
    let render = render.clone();
    let cb = Closure::<dyn FnMut(js_sys::Array)>::new(move |entries: js_sys::Array| {
        let Ok(entry) = entries.get(0).dyn_into::<web_sys::ResizeObserverEntry>() else {
            return;
        };
        let rect = entry.content_rect();
        let dpr = win.device_pixel_ratio().max(1.0);
        let width = ((rect.width() * dpr).round() as u32).max(1);
        let height = ((rect.height() * dpr).round() as u32).max(1);
        if let Ok(v) = serde_wasm_bindgen::to_value(&ResizeMsg::Canvas { width, height }) {
            let _ = render.post_message(&v);
        }
    });
    let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())?;
    observer.observe(canvas);
    // Both live for the page's lifetime.
    cb.forget();
    std::mem::forget(observer);
    Ok(())
}

/// Wire pointer input: a **left click drops a ball** — the click point goes to
/// the render worker as a [`DropMsg`] in NDC (render owns the camera, so it
/// does the unprojection onto the table). A **right-button drag orbits** and
/// the **wheel zooms**, relayed as [`CameraMsg`]s; the orbit drag ALSO
/// accumulates `yaw` here (same deltas × same sensitivity as the render
/// thread's `OrbitCamera`) and fans it out to the shared [`InputState`]
/// (camera-relative W/A/S/D) and the audio listener (camera-relative stereo).
fn install_pointer(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    render: Worker,
    audio: Audio,
    input: &'static InputState,
    yaw: Rc<Cell<f32>>,
) -> Result<(), JsValue> {
    // Click → drop a ball at the clicked table spot. NDC: x right, y up.
    let click_canvas = canvas.clone();
    let click = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(
        clone!(render, audio => move |e: web_sys::MouseEvent| {
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

    // pointerdown on the canvas: the RIGHT button begins an orbit drag (left
    // stays the drop-a-ball click). Any button is a user gesture → start audio.
    let down = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, audio => move |e: web_sys::PointerEvent| {
            if e.button() == 2 {
                dragging.set(true);
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
    // orbit delta goes to render (the visual), and its dx also advances OUR
    // yaw with the same sensitivity — fanned out to physics (shared input
    // block) and the audio listener, keeping all three frames in lockstep.
    let move_ = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, render, audio, yaw => move |e: web_sys::PointerEvent| {
            if dragging.get() {
                let dx = e.movement_x() as f32;
                let dy = e.movement_y() as f32;
                let y = yaw.get() - dx * CAMERA_ORBIT_SENSITIVITY;
                yaw.set(y);
                input.set_camera_yaw(y);
                if let Some(c) = audio.borrow_mut().as_mut() {
                    c.set_camera_yaw(y);
                }
                post_camera(&render, &CameraMsg::Orbit { dx, dy });
            }
        }),
    );
    window.add_event_listener_with_callback("pointermove", move_.as_ref().unchecked_ref())?;
    move_.forget();

    // pointerup anywhere ends the drag.
    let up = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging => move |_e: web_sys::PointerEvent| {
            dragging.set(false);
        }),
    );
    window.add_event_listener_with_callback("pointerup", up.as_ref().unchecked_ref())?;
    up.forget();

    // wheel on the canvas zooms (preventDefault so the page doesn't scroll).
    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(
        clone!(render => move |e: web_sys::WheelEvent| {
            e.prevent_default();
            post_camera(&render, &CameraMsg::Zoom { dy: e.delta_y() as f32 });
        }),
    );
    canvas.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();

    Ok(())
}

fn post_camera(render: &Worker, msg: &CameraMsg) {
    match serde_wasm_bindgen::to_value(msg) {
        Ok(v) => {
            let _ = render.post_message(&v);
        }
        Err(e) => tracing::error!("serialize CameraMsg: {e}"),
    }
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

/// The top-right worker-stats panel: a 1 Hz `setInterval` that samples the
/// shared-memory counters the workers already maintain — no messages, no new
/// instrumentation. Rates are per-second deltas between samples:
/// - render: presented frames (the `BodyMotion` frame-tick),
/// - physics: fixed steps (the `BodyMotion` step counter),
/// - each task-pool worker: solver tasks claimed (`TaskPool` claim counters;
///   index 0 is the physics thread helping inside `finishTask`),
/// - balls: `BallMotions` count.
fn install_stats(
    window: &web_sys::Window,
    stats: Mutable<String>,
    refs: Rc<RefCell<StatsRefs>>,
) -> Result<(), JsValue> {
    let mut last = StatsSample::default();
    let tick = Closure::<dyn FnMut()>::new(move || {
        let refs = *refs.borrow();
        let mut lines = String::from("workers\n");
        lines.push_str("  main      ui · audio · input\n");

        // SAFETY (all three blocks): leaked-for-the-session allocations in the
        // shared `WebAssembly.Memory`; we only read their atomics.
        if let Some(addr) = refs.motion {
            let motion = unsafe { &*(addr as *const BodyMotion) };
            let frame = motion.frame_tick();
            let step = motion.latest_step();
            lines.push_str(&format!(
                "  render    {} fps\n  physics   {} steps/s",
                frame.wrapping_sub(last.frame),
                step.wrapping_sub(last.step),
            ));
            last.frame = frame;
            last.step = step;
        } else {
            lines.push_str("  render    starting…\n  physics   starting…");
        }

        if let Some((addr, count)) = refs.pool {
            let pool = unsafe { &*(addr as *const TaskPool) };
            let claims = pool.claim_counts(count as usize + 1);
            last.claims.resize(claims.len(), 0);
            for (i, (&now, then)) in claims.iter().zip(&last.claims).enumerate() {
                let rate = now.wrapping_sub(*then);
                let name = if i == 0 {
                    "  physics⤷ ".to_string()
                } else {
                    format!("  task {i}    ")
                };
                if rate == 0 {
                    lines.push_str(&format!("\n{name}idle"));
                } else {
                    lines.push_str(&format!("\n{name}{rate} tasks/s"));
                }
            }
            last.claims.copy_from_slice(&claims);
        }

        if let Some(addr) = refs.balls {
            let balls = unsafe { &*(addr as *const BallMotions) };
            lines.push_str(&format!("\n\nballs dropped  {}", balls.count()));
        }

        stats.set(lines);
    });
    window.set_interval_with_callback_and_timeout_and_arguments_0(
        tick.as_ref().unchecked_ref(),
        1000,
    )?;
    tick.forget();
    Ok(())
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
/// 3. The directory the page is served from — we keep the path (not just the
///    origin) so it's correct under a GitHub Pages project base like
///    `/<repo>/`; `{base}/bundle/scene.toml` then resolves alongside
///    `index.html`.
///    `/foo/bar/` → `https://host/foo/bar`, `/foo/index.html` →
///    `https://host/foo`, `/` → `https://host`.
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
    let origin = location.origin().unwrap_or_default();
    let mut path = location.pathname().unwrap_or_default();
    // Drop the trailing filename (or trailing slash), keeping the directory.
    if let Some(idx) = path.rfind('/') {
        path.truncate(idx);
    }
    format!("{}{}", origin.trim_end_matches('/'), path)
}
