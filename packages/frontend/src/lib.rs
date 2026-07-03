// `core::arch::wasm32::memory_atomic_wait32` (the physics thread's precise sleep)
// is still unstable; this crate is nightly-only anyway (see `rust-toolchain.toml`).
#![feature(stdarch_wasm_atomic_wait)]

//! **single-player-game-physics** — a copyable template for *playing* a scene
//! exported from `awsm-scene`, rendered with `awsm-renderer`, across three real
//! wasm threads over one shared `WebAssembly.Memory`:
//!
//! - **Main thread** ([`main_thread`]): owns the DOM (built with Dominator) and
//!   captures input. Spawns the render worker, then the physics worker.
//! - **Render thread** ([`render_thread`]): hosts `awsm-renderer`, loads the
//!   exported `scene.toml` via the player loader, and renders every frame —
//!   reading the sphere's transform straight out of shared memory.
//! - **Physics thread** ([`physics_thread`]): runs a Box3D world (a dynamic
//!   sphere on a static ground — C code compiled into this same wasm module),
//!   applies input, and writes the sphere's transform into the shared arena
//!   every step.
//!
//! A single wasm bundle serves all three; the active role is chosen at runtime
//! (the `wasm-bindgen-rayon` pattern). The threaded build profile (nightly +
//! `+atomics` + `build-std`) and COOP/COEP headers are what make the shared
//! memory possible — see `Taskfile.yml`, `Trunk.toml`, `rust-toolchain.toml`.

pub mod audio;
pub mod bootstrap;
pub mod main_thread;
pub mod physics_tasks;
pub mod physics_thread;
pub mod protocol;
pub mod render_thread;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

/// `true` when running inside a `DedicatedWorkerGlobalScope`.
pub fn is_worker_scope() -> bool {
    js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
}

/// Single entry point. `wasm-bindgen` runs this automatically on every `init()`
/// (main thread *and* every worker). On the main thread it boots the app; in a
/// worker it does nothing — the worker's real work is triggered explicitly by
/// the bootstrap JS calling [`mt_worker_start`] after init returns.
#[wasm_bindgen(start)]
pub fn boot() -> Result<(), JsValue> {
    install_tracing();
    if is_worker_scope() {
        Ok(())
    } else {
        main_thread_boot()
    }
}

fn main_thread_boot() -> Result<(), JsValue> {
    tracing::info!("single-player-game-physics: main-thread boot");
    let isolated = crossorigin_isolated();
    let has_sab = shared_array_buffer_available();
    tracing::info!("crossOriginIsolated = {isolated}, SharedArrayBuffer = {has_sab}");
    if !isolated || !has_sab {
        // Hard fail — this template is threads-only. Without cross-origin
        // isolation there's no shared `WebAssembly.Memory`, so there's nothing
        // to degrade to: stop here with a clear message instead of limping on
        // and throwing an opaque `SharedArrayBuffer` DataCloneError mid-spawn.
        //
        // On GitHub Pages this state is transient on the very FIRST visit —
        // `coi-serviceworker.js` reloads the page the moment its worker takes
        // control, after which we're isolated. If you're seeing this stick,
        // the host isn't sending COOP: same-origin + COEP: require-corp and the
        // service-worker shim didn't register (see index.html / Trunk.toml).
        let msg = "Cross-origin isolation is OFF — multithreading is unavailable, \
                   so this build cannot run. It needs COOP: same-origin + COEP: \
                   require-corp (sent by `task dev`, or re-imposed by \
                   coi-serviceworker.js on static hosts like GitHub Pages).";
        tracing::error!("{msg}");
        main_thread::fatal(msg);
        return Err(JsValue::from_str(msg));
    }
    main_thread::start()
}

/// The worker-side entry point the bootstrap JS calls after init. Dispatches on
/// `role`; `payload` is the per-role data posted with the init message.
#[wasm_bindgen]
pub fn mt_worker_start(role: String, payload: JsValue) -> Result<(), JsValue> {
    install_tracing();
    match role.as_str() {
        "render" => render_thread::start(payload),
        "physics" => physics_thread::start(payload),
        "physics-task" => physics_tasks::start(payload),
        other => {
            tracing::warn!("unknown worker role {other:?}");
            Ok(())
        }
    }
}

/// Install the browser-console tracing subscriber (idempotent — safe to call on
/// the main thread and in every worker).
pub fn install_tracing() {
    use tracing_subscriber::prelude::*;
    // Surface panic messages in the console — with `panic = abort` on wasm a
    // panic otherwise dies as an opaque `RuntimeError: unreachable`.
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&format!("PANIC: {info}")));
    }));
    // The default `fmt` time formatter calls `SystemTime::now()`, which panics
    // on wasm32; `without_time` strips it (the console prepends its own time).
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .without_time()
        .with_writer(tracing_web::MakeWebConsoleWriter::new())
        .with_target(false);
    // Unfiltered, every dependency's debug!/trace! (the renderer is chatty)
    // funnels through the console writer ON THE MAIN/RENDER THREAD — console
    // logging is expensive enough to show up as frame hiccups. Keep our own
    // crate at debug (the audio-cue lines are how headless checks observe
    // impacts — they only fire on contacts, which is cheap).
    let filter = tracing_subscriber::EnvFilter::new("info,single_player_game_physics=debug");
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init();
}

/// `globalThis.crossOriginIsolated` from whichever scope is active.
pub fn crossorigin_isolated() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("crossOriginIsolated"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// `typeof SharedArrayBuffer !== "undefined"` in the active scope.
pub fn shared_array_buffer_available() -> bool {
    js_sys::Reflect::has(&js_sys::global(), &JsValue::from_str("SharedArrayBuffer"))
        .unwrap_or(false)
}
