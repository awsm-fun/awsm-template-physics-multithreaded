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
//! 5. Own the orbit camera: apply every drag and wheel event to it, and once
//!    per main-thread frame write it into the shared [`CameraRing`] the
//!    render worker reads (see that type for why it is a timestamped ring).
//!
//! The DOM here stays deliberately thin (a canvas + a static controls HUD whose
//! one live line is a `futures-signals` `Mutable`) — the point of the template
//! is the threading, not the UI.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html, with_node};
use futures_signals::signal::{Mutable, SignalExt};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::js_sys;
use web_sys::{
    HtmlCanvasElement, HtmlInputElement, HtmlSelectElement, KeyboardEvent, MessageEvent, Worker,
};

use crate::about::{About, Tab};
use crate::audio::AudioController;
use crate::bootstrap::{spawn_shared_worker, spawn_shared_worker_transfer};
use crate::protocol::{
    abs_time_us, AudioMsg, CameraRing, DropMsg, InputState, OrbitCamera, PhysicsMsg, QualityMsg,
    RenderMsg, ResizeMsg, TimingTotal, HELD_BACK, HELD_FORWARD, HELD_LEFT, HELD_RIGHT,
};
use crate::stats::{install_stats, PageWork, Panel, StatsRefs};

/// Shared, lazily-loaded audio. `None` until the export finishes loading; stays
/// silent until a user gesture starts it (browser autoplay policy).
type Audio = Rc<RefCell<Option<AudioController>>>;

// ── Resolution scale: the one fill-rate lever ────────────────────────────────
// The canvas renders at `devicePixelRatio × scale` device pixels. Fill rate —
// not geometry — bounds this scene, so on a weak GPU (or a high-DPR phone)
// dropping this fraction is what buys the framerate back. It's the user's to
// set (the bottom-right slider); we only pick the DEFAULT, and only while they
// haven't chosen one.
//
// The default is native resolution capped at a PIXEL BUDGET
// ([`DEFAULT_PIXEL_BUDGET`]), further capped by device class (touch / software
// GPU). An earlier version kept native on any desktop ("a big display on a
// fast GPU keeps native"), but measurement disproved that premise: a Retina
// MacBook (M1 Max) at native 2400×1794 ran this scene's GPU frame at ~10 ms —
// over a 120 Hz display's 8.3 ms — so it presented ~90 fps while the CPU stat
// read 0.4 ms. The renderer's per-pixel cost (a deferred, compute-driven
// pipeline) scales with pixel count, so the budget is in pixels, not DPR, and a
// small window keeps native. It never drops below an effective DPR of 1 (a
// default softer than CSS resolution reads as blurry), so a DPR-1 display —
// even a large 4K one — keeps native by default; its user has the slider.
/// Floor of the slider (50% of native). Below this the image turns to mush for
/// little further gain; the slider's `min` in `index.html` mirrors it.
const MIN_SCALE: f64 = 0.5;
/// Ceiling: native resolution. We never supersample here.
const MAX_SCALE: f64 = 1.0;
/// Start on a touch / `pointer: coarse` device (phones, tablets): high DPR × a
/// mobile GPU is the case that tanks to ~20fps at native, so seed it down.
const COARSE_START_SCALE: f64 = 0.6;
/// Start on a software / fallback adapter — it genuinely can't push pixels, so
/// seed lower still. Applied when the render worker reports the adapter.
const FALLBACK_START_SCALE: f64 = 0.5;
/// Backing-store pixels the DEFAULT resolution allows (see the section note).
/// A 1200×897 CSS window at DPR 2 (4.3 MP native) defaults to ~80%. The user's
/// slider still goes to 100%.
const DEFAULT_PIXEL_BUDGET: f64 = 2_800_000.0;
/// Max 2D texture dimension assumed until the render worker reports the real
/// one (WebGPU guarantees at least this). Caps the backing store so a huge
/// display can't ask for a canvas larger than the GPU allows.
const DEFAULT_MAX_TEX: u32 = 8192;
/// `localStorage` key for the persisted user choice. Every experiment is served
/// from the same origin (awsm.fun), so each app's keys carry its own prefix;
/// change it when starting a new project from this template.
const RES_STORAGE_KEY: &str = "awsm_box3d_mt_res_scale";

/// Everything needed to size the canvas backing store: the user's `scale`
/// fraction, the GPU's texture-dimension cap, and the current CSS size + DPR
/// (updated by the `ResizeObserver`). `Copy` so it lives in a `Cell`.
#[derive(Clone, Copy)]
struct ResState {
    scale: f64,
    max_tex: u32,
    css_w: f64,
    css_h: f64,
    dpr: f64,
    /// Device-class ceiling for the DEFAULT scale: [`MAX_SCALE`], or lower on
    /// a touch device / software GPU. Never applied over a user choice.
    auto_cap: f64,
}

impl ResState {
    /// Backing-store size in device pixels: `css × dpr × scale`, each axis
    /// clamped to `[1, max_tex]`.
    fn backing(&self) -> (u32, u32) {
        let s = (self.dpr * self.scale).max(0.01);
        let w = ((self.css_w * s).round() as u32).clamp(1, self.max_tex);
        let h = ((self.css_h * s).round() as u32).clamp(1, self.max_tex);
        (w, h)
    }

    /// The default scale for the current size: native, capped by the pixel
    /// budget and the device-class ceiling, floored at an effective DPR of 1
    /// (and the slider's range).
    fn auto_scale(&self) -> f64 {
        let native_px = self.css_w * self.css_h * self.dpr * self.dpr;
        let budget = if native_px > DEFAULT_PIXEL_BUDGET {
            (DEFAULT_PIXEL_BUDGET / native_px)
                .sqrt()
                .max(1.0 / self.dpr)
        } else {
            MAX_SCALE
        };
        // Round DOWN to the slider's 5% step so the thumb, its label and the
        // real scale agree (the browser would snap an 81% value to 80).
        let stepped = (budget.min(self.auto_cap) * 20.0).floor() / 20.0;
        stepped.clamp(MIN_SCALE, MAX_SCALE)
    }
}

/// Percent form of a scale fraction (0.6 → 60) — the slider's unit.
fn pct_of(scale: f64) -> u32 {
    (scale * 100.0).round() as u32
}

/// The device-class ceiling for the default scale: lower on a touch device.
/// (A fallback GPU lowers it further once the render worker reports in — see
/// the `GpuInfo` handler.)
fn initial_auto_cap(window: &web_sys::Window) -> f64 {
    if coarse_pointer(window) {
        COARSE_START_SCALE
    } else {
        MAX_SCALE
    }
}

/// A touch / `pointer: coarse` device (phones, tablets) — the device class
/// that seeds lower graphics defaults: the resolution scale starts at
/// [`COARSE_START_SCALE`] and MSAA starts off. Only DEFAULTS — a stored user
/// choice always wins.
pub(crate) fn coarse_pointer(window: &web_sys::Window) -> bool {
    window
        .match_media("(pointer: coarse)")
        .ok()
        .flatten()
        .map(|m| m.matches())
        .unwrap_or(false)
}

/// The persisted resolution scale, if the user has set one (clamped to range).
fn stored_scale(window: &web_sys::Window) -> Option<f64> {
    let ls = window.local_storage().ok().flatten()?;
    let raw = ls.get_item(RES_STORAGE_KEY).ok().flatten()?;
    raw.parse::<f64>()
        .ok()
        .map(|v| v.clamp(MIN_SCALE, MAX_SCALE))
}

/// Persist the user's resolution scale (only user drags call this — auto seeds
/// are recomputed each load, never written).
fn store_scale(window: &web_sys::Window, scale: f64) {
    if let Ok(Some(ls)) = window.local_storage() {
        let _ = ls.set_item(RES_STORAGE_KEY, &format!("{scale:.2}"));
    }
}

/// Post the backing-store size for `st` to the render worker (it owns the
/// transferred `OffscreenCanvas`), reusing the existing `ResizeMsg` path.
fn post_resize(render: &Worker, st: &ResState) {
    let (width, height) = st.backing();
    if let Ok(v) = serde_wasm_bindgen::to_value(&ResizeMsg::Canvas { width, height }) {
        let _ = render.post_message(&v);
    }
}

// ── Anti-aliasing setting ────────────────────────────────────────────────────
// ONE three-way choice in the Settings modal, persisted: Off / SMAA / MSAA 4×.
// MSAA+SMAA together isn't offered: on top of a 4× MSAA resolve SMAA buys
// almost nothing (the renderer itself defaults it off for that reason) and
// costs an extra full-screen pre-pass. SMAA earns its place as the cheap AA
// when MSAA is off.
const AA_STORAGE_KEY: &str = "awsm_box3d_mt_aa";
/// The pre-0.42 per-toggle keys — migrated to [`AA_STORAGE_KEY`] once, then
/// cleared (see [`clear_legacy_aa`]).
const LEGACY_AA_STORAGE_KEYS: [&str; 2] = ["awsm_msaa", "awsm_smaa"];
/// `localStorage` key for the stats-panel toggle (default OFF — the panel eats
/// a lot of a phone screen, so it's opt-in via the bottom-left Stats chip).
/// Each panel group's open state is stored under it too (see `stats`).
pub(crate) const STATS_STORAGE_KEY: &str = "awsm_box3d_mt_stats";
/// The element id of the reduced-quality notice (see `quality_notice_modal`).
pub(crate) const QUALITY_NOTICE_ID: &str = "quality-notice";
/// The anti-aliasing modes the Settings modal offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AaMode {
    Off,
    Smaa,
    Msaa,
}

impl AaMode {
    const ALL: [AaMode; 3] = [AaMode::Off, AaMode::Smaa, AaMode::Msaa];

    /// The renderer's `(msaa, smaa)` pair for this mode.
    fn flags(self) -> (bool, bool) {
        match self {
            AaMode::Off => (false, false),
            AaMode::Smaa => (false, true),
            AaMode::Msaa => (true, false),
        }
    }

    /// Storage / `<option>` value.
    fn key(self) -> &'static str {
        match self {
            AaMode::Off => "off",
            AaMode::Smaa => "smaa",
            AaMode::Msaa => "msaa",
        }
    }

    fn label(self) -> &'static str {
        match self {
            AaMode::Off => "Off",
            AaMode::Smaa => "SMAA",
            AaMode::Msaa => "MSAA 4×",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key() == s)
    }
}

/// Default anti-aliasing on a fine-pointer (desktop) device. On
/// `pointer: coarse` devices it defaults to [`AaMode::Off`] instead: like the
/// [`COARSE_START_SCALE`] resolution seed, the mobile defaults optimize
/// purely for frame rate — and because that visibly costs quality, a notice
/// modal points the user at Settings (see `quality_notice_modal`). A stored
/// user choice always wins over either default.
const AA_DEFAULT: AaMode = AaMode::Msaa;

/// The persisted anti-aliasing choice, if the user has made one.
fn stored_aa(window: &web_sys::Window) -> Option<AaMode> {
    let ls = window.local_storage().ok().flatten()?;
    AaMode::parse(&ls.get_item(AA_STORAGE_KEY).ok().flatten()?)
}

/// Persist the user's anti-aliasing choice.
fn store_aa(window: &web_sys::Window, mode: AaMode) {
    if let Ok(Some(ls)) = window.local_storage() {
        let _ = ls.set_item(AA_STORAGE_KEY, mode.key());
    }
}

/// Carry an explicit pre-0.42 choice over to [`AA_STORAGE_KEY`], then drop the
/// legacy per-toggle keys (see [`LEGACY_AA_STORAGE_KEYS`]). The old keys were
/// only ever written by a user toggle, so they are real choices: MSAA off maps
/// to SMAA if SMAA was on, else Off; MSAA on maps to MSAA (SMAA alongside it
/// isn't offered any more). Unset legacy keys leave the new default in place.
fn clear_legacy_aa(window: &web_sys::Window) {
    let Ok(Some(ls)) = window.local_storage() else {
        return;
    };
    let get = |k: &str| ls.get_item(k).ok().flatten();
    let (msaa, smaa) = (
        get(LEGACY_AA_STORAGE_KEYS[0]),
        get(LEGACY_AA_STORAGE_KEYS[1]),
    );
    if get(AA_STORAGE_KEY).is_none() {
        let migrated = match (msaa.as_deref(), smaa.as_deref()) {
            (Some("1"), _) => Some(AaMode::Msaa),
            (Some("0"), Some("1")) => Some(AaMode::Smaa),
            (Some("0"), _) => Some(AaMode::Off),
            (None, Some("1")) => Some(AaMode::Msaa),
            _ => None,
        };
        if let Some(mode) = migrated {
            let _ = ls.set_item(AA_STORAGE_KEY, mode.key());
        }
    }
    for key in LEGACY_AA_STORAGE_KEYS {
        let _ = ls.remove_item(key);
    }
}

/// Read a persisted boolean setting (`"1"`/`"0"`), `None` if unset.
pub(crate) fn stored_bool(window: &web_sys::Window, key: &str) -> Option<bool> {
    let ls = window.local_storage().ok().flatten()?;
    match ls.get_item(key).ok().flatten()?.as_str() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

/// Persist a boolean setting.
pub(crate) fn store_bool(window: &web_sys::Window, key: &str, val: bool) {
    if let Ok(Some(ls)) = window.local_storage() {
        let _ = ls.set_item(key, if val { "1" } else { "0" });
    }
}

/// Post an anti-aliasing change to the render worker (it recompiles the
/// affected pipelines — see [`QualityMsg`]).
fn post_quality(render: &Worker, msaa: bool, smaa: bool) {
    if let Ok(v) = serde_wasm_bindgen::to_value(&QualityMsg::AntiAlias { msaa, smaa }) {
        let _ = render.post_message(&v);
    }
}

/// `?reset` in the URL wipes every persisted app setting (resolution, MSAA,
/// SMAA, the stats panel and its groups) so the load boots with fresh defaults — the escape
/// hatch when a device carries choices from old sessions (which override the
/// device-class seeds and suppress the reduced-quality notice) and the
/// browser's site-data UI won't surface them. Must run before ANY stored
/// value is read, i.e. first thing in [`start`].
fn maybe_reset_settings(window: &web_sys::Window) {
    let has_reset = window
        .location()
        .search()
        .ok()
        .and_then(|s| web_sys::UrlSearchParams::new_with_str(&s).ok())
        .map(|p| p.get("reset").is_some())
        .unwrap_or(false);
    if !has_reset {
        return;
    }
    if let Ok(Some(ls)) = window.local_storage() {
        for key in [
            RES_STORAGE_KEY,
            AA_STORAGE_KEY,
            LEGACY_AA_STORAGE_KEYS[0],
            LEGACY_AA_STORAGE_KEYS[1],
            STATS_STORAGE_KEY,
        ] {
            let _ = ls.remove_item(key);
        }
        for key in crate::stats::open_keys() {
            let _ = ls.remove_item(&key);
        }
    }
    loading_log("?reset — stored settings cleared, booting with defaults");
}

/// Build + mount the DOM, then start the worker pipeline.
pub fn start() -> Result<(), JsValue> {
    if let Some(window) = web_sys::window() {
        maybe_reset_settings(&window);
    }
    let status = Mutable::new("booting…".to_string());
    // The Stats panel (hidden by default; see `stats`).
    let stats = Panel::new();
    let about = About::default();
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
            .text("single-player-game-physics\nW/A/S/D or arrows: roll · Space: jump · click: drop a ball\nright-drag: orbit · wheel: zoom\ntouch: swipe to fling the ball · tap to drop\nsound starts on your first key or click\n")
            .child(html!("span", {
                .text_signal(status.signal_cloned())
            }))
        }))
        .child(stats.dom(clone!(about => move || about.show(Tab::Stats))))
        .child(stats.button())
        .child(about.button())
        .child(about.modal())
    });

    dominator::append_dom(&dominator::body(), app);
    Ok(())
}

/// The one-time reduced-quality notice for touch devices: the coarse-pointer
/// seeds (60% resolution, both anti-aliasing toggles off) trade visuals for
/// frame rate, so the first time they apply we say so and point at Settings.
/// Opened by the `Ready` handler (once the game is actually on screen);
/// closes via OK, ×, backdrop, or Escape. Shows on every load where the
/// seeds applied; it stops once the user makes their own Settings choices
/// (those persist and replace the seeds).
fn quality_notice_modal(open: &Mutable<bool>) -> dominator::Dom {
    html!("div", {
        .class("settings-overlay")
        .attr("id", QUALITY_NOTICE_ID)
        // Above the About modal, which may already be open when this shows.
        .style("z-index", "25")
        .visible_signal(open.signal())
        .event_with_options(&dominator::EventOptions::bubbles(), keep_keys_in_modal)
        .event(clone!(open => move |e: dominator::events::Click| {
            let on_backdrop = e
                .dyn_target::<web_sys::Element>()
                .map(|el| el.class_list().contains("settings-overlay"))
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
            .class("settings-modal")
            .child(html!("button", {
                .class("about-close")
                .attr("aria-label", "Close")
                .text("×")
                .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
            }))
            .child(html!("h2", { .text("Display Quality Reduced") }))
            .child(html!("p", {
                .class("notice-text")
                .text("Display quality was reduced to keep the frame rate up on \
                       touch devices. You can adjust the resolution and \
                       anti-aliasing any time in Settings (bottom right).")
            }))
            .child(html!("button", {
                .class("notice-ok")
                .text("OK")
                .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
            }))
        }))
    })
}

/// The blocking "compiling pipelines" overlay, raised while the render worker
/// recompiles for a Settings anti-aliasing change
/// ([`RenderMsg::AaCompileStart`] → [`RenderMsg::AaCompileDone`]). Not
/// user-dismissable — it lowers itself when the worker reports done. `None` =
/// hidden; `Some(line)` = shown, with the worker's live progress line.
fn aa_compile_modal(state: &Mutable<Option<String>>) -> dominator::Dom {
    html!("div", {
        .class("settings-overlay")
        // Above the Settings modal that triggered it (both are z-index 20).
        .style("z-index", "30")
        .visible_signal(state.signal_cloned().map(|s| s.is_some()))
        .child(html!("div", {
            .class("settings-modal")
            .child(html!("h2", { .text("Compiling Pipelines") }))
            .child(html!("p", {
                .class("notice-text")
                .text("Rebuilding the render pipelines for the new \
                       anti-aliasing settings. The first switch in each \
                       direction compiles shaders; after that it's cached and \
                       instant.")
            }))
            .child(html!("p", {
                .class("settings-hint")
                .text_signal(state.signal_cloned().map(|s| s.unwrap_or_default()))
            }))
        }))
    })
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

/// Runs once the canvas is in the DOM: size it, transfer it, spawn render, and
/// arm the render→physics handoff + keyboard input.
fn setup(
    canvas: HtmlCanvasElement,
    status: Mutable<String>,
    stats: Rc<Panel>,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;

    // ── Resolution scale (the fill-rate lever) ──────────────────────────────
    // Size the backing store to CSS size × devicePixelRatio × the chosen scale.
    // The scale starts from a stored preference / device-class seed and is then
    // driven live by the bottom-right slider (see `install_resolution_control`).
    // Must be sized BEFORE transfer. `max_tex` is refined once the render worker
    // reports the GPU's real limit (the `GpuInfo` handler below).
    let mut res0 = ResState {
        scale: MAX_SCALE,
        max_tex: DEFAULT_MAX_TEX,
        css_w: canvas.client_width().max(1) as f64,
        css_h: canvas.client_height().max(1) as f64,
        dpr: window.device_pixel_ratio().max(1.0),
        auto_cap: initial_auto_cap(&window),
    };
    // A stored user choice wins; otherwise the budgeted default.
    res0.scale = stored_scale(&window).unwrap_or_else(|| res0.auto_scale());
    let res = Rc::new(Cell::new(res0));
    // Whether the user has an explicit choice — if so, auto-seeding (touch /
    // fallback GPU) must never override it.
    let user_pref = Rc::new(Cell::new(stored_scale(&window).is_some()));
    // Slider position + label (percent). Auto-seeds set this too, moving the
    // thumb; the slider's own input handler sets it on drag.
    let res_pct = Mutable::new(pct_of(res.get().scale));
    // Anti-aliasing toggles (persisted). Applied to the render worker once
    // it's ready (see the `Ready` handler) and on toggle. Like the resolution
    // seed above, a coarse-pointer device defaults BOTH off — the mobile
    // seeds optimize purely for frame rate (a one-time notice modal below
    // tells the user and points at Settings).
    let coarse = coarse_pointer(&window);
    clear_legacy_aa(&window);
    let aa =
        Mutable::new(stored_aa(&window).unwrap_or(if coarse { AaMode::Off } else { AA_DEFAULT }));
    // ── Reduced-quality notice (touch devices) ──────────────────────────────
    // If any of the coarse-device seeds actually applied (no stored user
    // choice overrode it), quality was visibly traded for frame rate — say so
    // when the game first becomes visible (the `Ready` handler opens it), and
    // point at Settings. Shown on EVERY such load — informing the user beats
    // sparing them a dismissal; it goes quiet once they've made their own
    // Settings choices (those persist and replace the seeds).
    let show_quality_notice =
        coarse && (stored_scale(&window).is_none() || stored_aa(&window).is_none());
    let notice_open = Mutable::new(false);
    dominator::append_dom(&dominator::body(), quality_notice_modal(&notice_open));
    // The "compiling pipelines" overlay for runtime AA changes (render-worker-
    // driven via `RenderMsg::AaCompile*`).
    let aa_compile: Mutable<Option<String>> = Mutable::new(None);
    dominator::append_dom(&dominator::body(), aa_compile_modal(&aa_compile));
    let (w, h) = res.get().backing();
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
    // This thread's own per-frame work, for the Stats panel's page group.
    let page_work = Rc::new(PageWork::default());
    install_camera_frames(
        &window,
        camera.clone(),
        camera_ring,
        camera_dirty.clone(),
        page_work.clone(),
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
    // The desired startup anti-aliasing rides the spawn payload so the render
    // worker BUILDS at it (no post-`Ready` reconcile recompile).
    let (msaa0, smaa0) = aa.get().flags();
    set(&payload, "msaa", &JsValue::from_bool(msaa0));
    set(&payload, "smaa", &JsValue::from_bool(smaa0));
    set(
        &payload,
        "camera_ptr",
        &JsValue::from_f64(camera_ring as *const CameraRing as usize as f64),
    );
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    // Shared-memory addresses the stats panel samples, filled in as workers
    // report them (PhysicsInit → motion/balls, SpawnTaskWorkers → pool).
    let stats_refs: Rc<RefCell<StatsRefs>> = Rc::new(RefCell::new(StatsRefs::default()));
    // Flipped by the `Ready` handler below: the stats warm-up doesn't even
    // start until the renderer has committed and presented real frames.
    let render_ready = Rc::new(Cell::new(false));
    // For timing this thread's own work (the Stats panel's page group).
    let perf = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;
    // The renderer's latest `RenderMsg::Timings` (only sent while asked).
    let render_timings: Rc<RefCell<Vec<TimingTotal>>> = Rc::new(RefCell::new(Vec::new()));
    install_stats(
        &window,
        stats,
        stats_refs.clone(),
        render_ready.clone(),
        render_timings.clone(),
        page_work.clone(),
    )?;

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

    // Holds the render worker once spawned, so the message handler (installed
    // as that worker's `onmessage`, hence built before it exists) can post the
    // resolution resize back to it when `GpuInfo` arrives. Set synchronously
    // right after spawn, before any message is delivered.
    let render_ref: Rc<RefCell<Option<Worker>>> = Rc::new(RefCell::new(None));

    // Handle messages coming back from the render worker.
    let on_render_msg = Closure::<dyn FnMut(MessageEvent)>::new(
        clone!(physics, status, audio, stats_refs, res, user_pref, res_pct, render_ref, render_ready, render_timings, page_work, perf, notice_open, aa_compile => move |e: MessageEvent| {
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
                            let on_phys = Closure::<dyn FnMut(MessageEvent)>::new(clone!(audio, stats_refs, page_work, perf => move |e: MessageEvent| {
                                let t0 = perf.now();
                                if let Ok(msg) = serde_wasm_bindgen::from_value::<AudioMsg>(e.data()) {
                                    if let Some(c) = audio.borrow_mut().as_mut() {
                                        c.on_audio(msg);
                                    }
                                    page_work.cues(perf.now() - t0);
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
                    // The stats warm-up clock starts here (real frames exist).
                    render_ready.set(true);
                    // The game is visible now — if this load applied the
                    // reduced touch-device defaults, say so.
                    if show_quality_notice {
                        notice_open.set_neq(true);
                    }
                    status.set("playing — roll · Space jump · click drops a ball · right-drag orbit".into());
                    // No AA reconcile here: the render worker BUILT at the
                    // desired config (it rode the spawn payload), so the old
                    // post-`Ready` full-pipeline recompile is gone.
                }
                Ok(RenderMsg::GpuInfo { is_fallback, max_texture_dim }) => {
                    let mut st = res.get();
                    st.max_tex = max_texture_dim.max(1);
                    // A software adapter can't push pixels — seed lower, but
                    // never override a choice the user has already made.
                    if is_fallback {
                        st.auto_cap = st.auto_cap.min(FALLBACK_START_SCALE);
                        if !user_pref.get() && st.scale > st.auto_scale() {
                            st.scale = st.auto_scale();
                            res_pct.set(pct_of(st.scale));
                            loading_log(&format!(
                                "software GPU detected — starting at {}% resolution",
                                pct_of(st.scale)
                            ));
                        }
                    }
                    res.set(st);
                    // Re-apply with the real texture cap (and any lowered scale).
                    if let Some(r) = render_ref.borrow().as_ref() {
                        post_resize(r, &st);
                    }
                }
                Ok(RenderMsg::AaCompileStart { msaa, smaa }) => {
                    aa_compile.set(Some(format!(
                        "MSAA 4× {} · SMAA {}",
                        if msaa { "on" } else { "off" },
                        if smaa { "on" } else { "off" }
                    )));
                }
                Ok(RenderMsg::AaCompileProgress { message }) => {
                    aa_compile.set(Some(message));
                }
                Ok(RenderMsg::AaCompileDone) => aa_compile.set(None),
                Ok(RenderMsg::Timings { entries }) => *render_timings.borrow_mut() = entries,
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
    // Publish the handle so the message handler can resize on `GpuInfo` (set
    // now, before the event loop can deliver any message).
    *render_ref.borrow_mut() = Some(render.clone());

    // Track canvas layout size → render worker (which owns the transferred
    // OffscreenCanvas and applies the new backing size). Now scale-aware: the
    // backing store is CSS size × dpr × the chosen resolution scale. Without
    // this the backing store keeps its initial size and the browser stretches
    // it — circles become ovals after any window resize.
    install_resize(
        &window,
        &canvas,
        &render,
        res.clone(),
        user_pref.clone(),
        res_pct.clone(),
    )?;

    // The bottom-right Settings button + modal: the resolution slider (the
    // fill-rate lever) plus the anti-aliasing mode. Applies live and persists.
    install_settings(
        window.clone(),
        render.clone(),
        res.clone(),
        user_pref,
        res_pct,
        aa,
    )?;

    // Left-clicks drop balls (relayed to render for unprojection); right-drags
    // orbit and the wheel zooms; a pointerdown is also the gesture that starts
    // audio. Orbit drags and the wheel change `camera` (see above).
    install_pointer(&window, &canvas, render, audio, input, camera, camera_dirty)?;

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
    res: Rc<Cell<ResState>>,
    user_pref: Rc<Cell<bool>>,
    res_pct: Mutable<u32>,
) -> Result<(), JsValue> {
    let win = window.clone();
    let render = render.clone();
    let cb = Closure::<dyn FnMut(js_sys::Array)>::new(move |entries: js_sys::Array| {
        let Ok(entry) = entries.get(0).dyn_into::<web_sys::ResizeObserverEntry>() else {
            return;
        };
        let rect = entry.content_rect();
        // Record the new CSS size + DPR (it can change when a window moves
        // between monitors) and re-derive the backing store at the current
        // resolution scale.
        let mut st = res.get();
        st.dpr = win.device_pixel_ratio().max(1.0);
        st.css_w = rect.width().max(1.0);
        st.css_h = rect.height().max(1.0);
        // The default resolution is size-dependent (pixel budget): re-derive
        // it — but never over a user's slider choice.
        if !user_pref.get() {
            st.scale = st.auto_scale();
            res_pct.set_neq(pct_of(st.scale));
        }
        res.set(st);
        post_resize(&render, &st);
    });
    let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())?;
    observer.observe(canvas);
    // Both live for the page's lifetime.
    cb.forget();
    std::mem::forget(observer);
    Ok(())
}

/// The bottom-right **Settings** button + modal: the graphics controls that are
/// the user's to set. Everything here trades GPU cost for image quality —
/// exactly the levers that rescue a weak-GPU / high-DPR framerate:
/// - **Resolution** (the fill-rate lever): renders at `devicePixelRatio × scale`
///   device pixels. Default from [`ResState::auto_scale`] (+ the
///   fallback-GPU nudge in the `GpuInfo` handler); a drag applies live via the
///   shared [`ResState`] + [`post_resize`], persists, and marks `user_pref` so no
///   auto-seed overrides the human.
/// - **Anti-aliasing** (Off / SMAA / MSAA 4×, see [`AaMode`]): applied by the
///   render worker ([`QualityMsg`] → `set_anti_aliasing`); persisted.
///
/// The modal mirrors the About one (backdrop / Escape close). The controls live
/// here rather than in `start`'s static DOM because they need the render
/// [`Worker`] + the shared resolution state, which only exist after setup.
#[allow(clippy::too_many_arguments)]
fn install_settings(
    window: web_sys::Window,
    render: Worker,
    res: Rc<Cell<ResState>>,
    user_pref: Rc<Cell<bool>>,
    res_pct: Mutable<u32>,
    aa: Mutable<AaMode>,
) -> Result<(), JsValue> {
    let open = Mutable::new(false);

    let button = html!("button", {
        .class("settings-btn")
        .text("Settings")
        .event(clone!(open => move |_: dominator::events::Click| open.set_neq(true)))
    });

    // Resolution slider row.
    let resolution_row = html!("div", {
        .class("settings-row")
        .child(html!("label", {
            .class("settings-label")
            .text_signal(res_pct.signal().map(|p| format!("Resolution — {p}%")))
        }))
        .child(html!("input" => HtmlInputElement, {
            .class("res-slider")
            .attr("type", "range")
            // Mirror MIN_SCALE / MAX_SCALE (as percent).
            .attr("min", "50")
            .attr("max", "100")
            .attr("step", "5")
            .attr("aria-label", "render resolution")
            // Drive the thumb from `res_pct` so auto-seeds (touch / fallback GPU)
            // move it too; a user drag sets `res_pct` right back to the same
            // value, so there's no feedback wobble.
            .prop_signal("value", res_pct.signal().map(|p| p.to_string()))
            .with_node!(el => {
                .event(clone!(res, user_pref, res_pct, window, render => move |_: dominator::events::Input| {
                    let pct = el.value().parse::<f64>().unwrap_or(100.0);
                    let scale = (pct / 100.0).clamp(MIN_SCALE, MAX_SCALE);
                    let mut st = res.get();
                    st.scale = scale;
                    res.set(st);
                    user_pref.set(true);
                    res_pct.set(pct_of(scale));
                    store_scale(&window, scale);
                    post_resize(&render, &st);
                }))
            })
        }))
    });

    // Anti-aliasing: one three-way choice (see [`AaMode`]). A change persists
    // and posts the renderer's `(msaa, smaa)` pair; the render worker applies
    // it (recompiling behind the "compiling pipelines" modal).
    let aa_row = html!("div", {
        .class("settings-row")
        .child(html!("label", { .class("settings-label").text("Anti-aliasing") }))
        .child(html!("select" => HtmlSelectElement, {
            .class("settings-select")
            .attr("aria-label", "anti-aliasing")
            .children(AaMode::ALL.into_iter().map(|m| html!("option", {
                .attr("value", m.key())
                .text(m.label())
            })))
            .prop_signal("value", aa.signal().map(|m| m.key().to_string()))
            .with_node!(el => {
                .event(clone!(aa, window, render => move |_: dominator::events::Change| {
                    let Some(mode) = AaMode::parse(&el.value()) else { return };
                    if aa.replace(mode) != mode {
                        store_aa(&window, mode);
                        let (msaa, smaa) = mode.flags();
                        post_quality(&render, msaa, smaa);
                    }
                }))
            })
        }))
    });

    let modal = html!("div", {
        .class("settings-overlay")
        .visible_signal(open.signal())
        .event_with_options(&dominator::EventOptions::bubbles(), keep_keys_in_modal)
        .event(clone!(open => move |e: dominator::events::Click| {
            let on_backdrop = e
                .dyn_target::<web_sys::Element>()
                .map(|el| el.class_list().contains("settings-overlay"))
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
            .class("settings-modal")
            .child(html!("button", {
                .class("about-close")
                .attr("aria-label", "Close")
                .text("×")
                .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
            }))
            .child(html!("h2", { .text("Settings") }))
            .child(resolution_row)
            .child(aa_row)
            .child(html!("p", {
                .class("settings-hint")
                .text("Adjust these settings if your framerate is low.")
            }))
        }))
    });

    dominator::append_dom(&dominator::body(), button);
    dominator::append_dom(&dominator::body(), modal);
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
    page_work: Rc<PageWork>,
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
        page_work.frame(perf.now() - t0);
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
