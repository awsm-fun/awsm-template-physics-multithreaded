//! Render thread — hosts `awsm-renderer` on an `OffscreenCanvas`, loads the
//! editor-exported `scene.toml` through the **player** path, and renders every
//! frame.
//!
//! This is the shipped game's render loop: the editor's `export_player_bundle`
//! emits `scene.toml` (a serialized runtime [`awsm_renderer_scene::Scene`]); we
//! fetch it same-origin, deserialize with `scene_from_toml`, and materialize it
//! with [`load_scene_for_player`]. The scene's own colliders become the physics
//! world (table + walls static, ball dynamic — see [`derive_physics`]). Physics
//! publishes every fixed step's pose, stamped with the time it is due, into a
//! shared [`BodyMotion`](crate::protocol::BodyMotion) ring. Each frame starts by
//! publishing the frame's time (which paces physics), then draws the world at
//! the previous frame's time minus one step, **interpolated** between the two
//! published steps around it — Fix Your Timestep's `lerp(previous, current,
//! alpha)`, one frame behind because physics runs on its own thread (see the
//! `protocol` module docs). The camera comes from main through shared memory
//! too ([`CameraRing`]). No per-frame messaging. Lighting is entirely the
//! scene's (the loader instantiates its light nodes); its clips, texture flows
//! and particles advance each frame, before the bodies are written.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use awsm_renderer::buffer::shared_arena::foreign_write;
use awsm_renderer::transforms::TransformKey;
use awsm_renderer_settings::{
    apply_anti_aliasing, post_to_page, scene_anti_aliasing, AaMode, SettingsToPage,
    SettingsToWorker,
};
use awsm_renderer_stats::{
    post_report, stats_logging, FrameInput, RendererTimings, StatsCollector, StatsToWorker,
};
use glam::{Mat4, Quat, Vec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

use crate::protocol::{
    abs_time_us, time_diff_us, BallMotions, BodyMotion, Bracket, CameraRing, ColliderInit,
    ColliderShapeMsg, DropMsg, OrbitCamera, PhysicsInit, PoseRing, RenderMsg, DETAIL_PHYSICS,
    ROLE_FLOOR, STEP_US,
};
use crate::stats::{self, Phase, PhysicsStats};

thread_local! {
    /// The Stats panel's numbers (awsm-renderer-stats), from the worker's
    /// start: the page's `stats-view` and `stats-page` can arrive before the
    /// renderer exists.
    static STATS: RefCell<StatsCollector> = RefCell::new(StatsCollector::new(stats::config()));
}

/// The awsm-renderer release the vendored Basis transcoder files came with,
/// appended to their URLs as `?v=…`: those files keep their names from one
/// release to the next, so a browser that cached an older transcoder (wasm
/// included: the worker carries the query over to it) would otherwise pair
/// it with the new JS. Bump it together with awsm-renderer in Cargo.toml.
const BASIS_VERSION: &str = "0.46.0";

/// A drawn ball that moves farther than this in one frame (m) teleported
/// (physics' fall-through respawn): under TAA its motion history is reset,
/// so the respawned ball does not reproject the history of where it fell
/// from. A ball never moves this far in a frame on its own (its speed is
/// capped well below it).
const TELEPORT_JUMP_M: f32 = 1.5;

/// What a click needs to become a dropped ball: the shared drop block and the
/// tabletop (both known once the scene is loaded).
struct DropTarget {
    balls: &'static BallMotions,
    floor: Option<([f32; 3], [f32; 3])>,
    ball_radius: f32,
}

/// The boxed RAF callback, self-referenced so the render loop can reschedule
/// itself each frame (and stays alive for the worker's lifetime). The `f64`
/// is requestAnimationFrame's DOMHighResTimeStamp — the VSYNC-aligned frame
/// time. Frame timing MUST come from it, not from `performance.now()` inside
/// the callback: frames present on the vsync grid, but the callback itself
/// runs with scheduling delay (±1–3 ms under load), and timing "now" bakes
/// that delay straight into displayed positions as judder.
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// A body's pose at display time `display_us` (see [`PoseRing::bracket`]):
/// lerp/slerp between the two published steps around it. Also says whether
/// physics was late — its newest step older than the display time — in which
/// case the newest pose is held (never extrapolated). With no display time
/// yet (the first frame) the newest pose is used.
fn sample_at(ring: &PoseRing, display_us: Option<u32>) -> (Vec3, Quat, bool) {
    let pose = |p: crate::protocol::StepPose| (Vec3::from_array(p.pos), Quat::from_array(p.quat));
    let Some(display_us) = display_us else {
        let (p, q) = pose(ring.read_latest());
        return (p, q, false);
    };
    match ring.bracket(display_us) {
        Bracket::Between { older, newer } => {
            let span = time_diff_us(newer.time_us, older.time_us).max(1) as f32;
            let alpha = (time_diff_us(display_us, older.time_us) as f32 / span).clamp(0.0, 1.0);
            let (p0, q0) = pose(older);
            let (p1, q1) = pose(newer);
            (p0.lerp(p1, alpha), q0.slerp(q1, alpha), false)
        }
        Bracket::Late(newest) => {
            let (p, q) = pose(newest);
            (p, q, true)
        }
        Bracket::Early(oldest) => {
            let (p, q) = pose(oldest);
            (p, q, false)
        }
    }
}

/// Base-color factor for the PLAYER ball (multiplies the albedo texture) — a
/// red tint so yours reads instantly against the silver click-dropped balls.
const PLAYER_TINT: [f32; 4] = [1.0, 0.25, 0.2, 1.0];

/// Lens + clip planes for the one camera this app has. Shared by the render
/// path (`set_camera`) and the click-unprojection so the two can't drift apart
/// — a mismatch here silently drops balls in the wrong place.
const CAMERA_FOV_Y_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 400.0;

/// Worker entry: unpack the transferred `OffscreenCanvas` + page origin, build
/// the WebGPU device, and kick off the async load+render.
pub fn start(payload: JsValue) -> Result<(), JsValue> {
    use awsm_renderer::core::renderer::{AwsmRendererWebGpuBuilder, DeviceRequestLimits};
    use awsm_renderer::web_global::navigator_gpu;
    use awsm_renderer_scene_loader::basis::{configure, BasisWorkerConfig};

    let canvas: web_sys::OffscreenCanvas =
        js_sys::Reflect::get(&payload, &JsValue::from_str("canvas"))?.unchecked_into();
    // The worker has a `blob:` base URL, so relative fetches can't resolve —
    // main passes the page origin so we can build the absolute scene.toml URL.
    let origin = js_sys::Reflect::get(&payload, &JsValue::from_str("origin"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    // Provide the Basis (KTX2/BasisU) codec URLs — the crate hardcodes none.
    // We run this worker's load path against a `blob:` base, so the URLs must be
    // ABSOLUTE. `app_base` is the directory the app is served from (Trunk
    // copy-file serves workers/basis-worker.js + vendor/basis/… there). It keeps
    // the deploy PATH, not just the origin, so a subpath deploy (e.g.
    // /experiments/<slug>/) resolves correctly — a bare origin would 404. It is
    // also NOT `origin` above, which in dev is the separate live-media server
    // (no codec files).
    {
        let app_base = js_sys::Reflect::get(&payload, &JsValue::from_str("app_base"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let base = app_base.trim_end_matches('/');
        configure(BasisWorkerConfig::player(
            format!("{base}/workers/basis-worker.js?v={BASIS_VERSION}"),
            format!("{base}/vendor/basis/basis_transcoder.js?v={BASIS_VERSION}"),
        ));
    }
    // The STARTUP anti-aliasing mode (main's Settings: the stored choice,
    // `?aa=`, or the device default) rides the spawn payload as its id, so
    // the renderer BUILDS with it — compiling exactly the variants this
    // session needs, with no reconcile recompile after `Ready`.
    let aa_id = js_sys::Reflect::get(&payload, &JsValue::from_str("aa"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    let desired_aa = AaMode::from_id(&aa_id)
        .ok_or_else(|| JsValue::from_str(&format!("render worker: unknown aa mode {aa_id:?}")))?;
    // The camera ring main writes (see `CameraRing`): leaked by main in the
    // same `WebAssembly.Memory`, so its address is valid here.
    let camera_ptr = js_sys::Reflect::get(&payload, &JsValue::from_str("camera_ptr"))
        .ok()
        .and_then(|v| v.as_f64())
        .filter(|p| *p > 0.0)
        .ok_or_else(|| JsValue::from_str("render worker: no camera_ptr"))?;
    // SAFETY: see above — a live, never-freed `CameraRing`.
    let camera_ring: &'static CameraRing = unsafe { &*(camera_ptr as usize as *const CameraRing) };

    // This worker's messages from main, from now on (the bootstrap's init
    // handler has done its job): the stats panel talks to the collector even
    // while the scene loads, and Settings changes made meanwhile wait in
    // `pending_aa` for the render loop.
    let camera = Rc::new(RefCell::new(OrbitCamera::start()));
    let pending_aa: Rc<Cell<Option<AaMode>>> = Rc::new(Cell::new(None));
    let drop_target: Rc<RefCell<Option<DropTarget>>> = Rc::new(RefCell::new(None));
    install_render_input(
        camera.clone(),
        canvas.clone(),
        drop_target.clone(),
        pending_aa.clone(),
    )?;
    // The panel's "starting…" until the scene is on screen.
    post_report(STATS.with_borrow(StatsCollector::placeholder));

    post_progress("render worker: requesting WebGPU device…");
    let gpu =
        navigator_gpu().ok_or_else(|| JsValue::from_str("render worker: no navigator.gpu"))?;
    let gpu_builder = AwsmRendererWebGpuBuilder::new_with_offscreen_canvas(gpu, canvas)
        .with_device_request_limits(DeviceRequestLimits::max_all());

    wasm_bindgen_futures::spawn_local(async move {
        let started = run(
            gpu_builder,
            origin,
            desired_aa,
            camera_ring,
            camera,
            pending_aa,
            drop_target,
        )
        .await;
        if let Err(err) = started {
            tracing::error!("render thread: {err:?}");
            post_to_main(&RenderMsg::Error {
                message: format!("{err:?}"),
            });
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run(
    gpu_builder: awsm_renderer::core::renderer::AwsmRendererWebGpuBuilder,
    origin: String,
    desired_aa: AaMode,
    camera_ring: &'static CameraRing,
    camera: Rc<RefCell<OrbitCamera>>,
    pending_aa: Rc<Cell<Option<AaMode>>>,
    drop_target: Rc<RefCell<Option<DropTarget>>>,
) -> Result<(), JsValue> {
    use awsm_renderer::camera::CameraParams;
    use awsm_renderer::AwsmRendererBuilder;

    // The builder reports its coarse boot phases through this handler — wire
    // them to the loading screen. Since the deferred-boot renderer, `build()`
    // compiles no pipelines (they're reserved and compiled by the labeled
    // `ensure_config_pipelines` step below), so only `Init` — device +
    // core GPU resources — takes visible time here.
    let mut renderer = AwsmRendererBuilder::new(gpu_builder)
        .with_anti_aliasing(desired_aa.to_anti_aliasing())
        // GPU timing for the Stats panel's GPU group: `Frame` tier =
        // timestamp queries on every pass, folded into one whole-frame span
        // (cheap; silently off without `timestamp-query`). The CPU tier is
        // `Off` except while the renderer group is open (the stats
        // collector's `begin_renderer_frame` switches it).
        .with_logging(stats_logging())
        .with_phase_handler(|phase| {
            use awsm_renderer::RendererLoadingPhase as P;
            let message = match phase {
                P::Init => "renderer init: GPU device + core resources…",
                P::CompilingShaders | P::BuildingPipelines | P::Ready => return,
            };
            post_progress(message);
        })
        .build()
        .await
        .map_err(|e| JsValue::from_str(&format!("build renderer: {e}")))?;
    post_progress(&format!(
        "render worker: WebGPU device + renderer ready (anti-aliasing: {})",
        AaMode::from_anti_aliasing(&renderer.anti_aliasing).label()
    ));

    // Shared mode must be enabled BEFORE the load so every scene node gets an
    // arena slot — the sphere's slot is then foreign-writable by physics.
    renderer.transforms.enable_shared_arena();

    // ── Warm-up + scene fetch, CONCURRENTLY ─────────────────────────────────
    // `build()` compiled NO pipeline — it only reserved them. We know our config
    // here (AA rode the spawn payload), so we warm exactly that set now via
    // `ensure_config_pipelines` (only what's needed — not an eager compile of
    // everything). That warm-up is GPU/driver work; the `scene.toml` fetch is
    // network work; the two share no data, so we run them under one `join!` and
    // let the single-threaded executor interleave them — the compile hides the
    // fetch latency instead of following it. First visit really compiles (the
    // browser caches from then on); warm visits are a no-op.
    let bundle_base = format!("{}/bundle", origin.trim_end_matches('/'));
    let scene_url = format!("{bundle_base}/scene.toml");
    post_progress("compiling core render pipelines + fetching scene… (first visit can take a while — cached after)");
    let (compiled, scene) =
        futures::join!(renderer.ensure_config_pipelines(), fetch_scene(&scene_url));
    let compiled =
        compiled.map_err(|e| JsValue::from_str(&format!("ensure_config_pipelines: {e}")))?;
    let mut scene =
        scene.map_err(|e| JsValue::from_str(&format!("load scene {scene_url}: {e}")))?;
    // The loader applies the scene's authored anti-aliasing. This app chooses
    // AA itself (the Settings prefs the renderer was built with), so write that
    // choice into the scene: the loader then keeps it and compiles nothing
    // twice.
    scene.post_process.anti_aliasing = scene_anti_aliasing(&renderer.anti_aliasing);
    tracing::info!(
        "render thread: loaded scene {scene_url} ({} nodes)",
        scene.nodes.len()
    );
    post_progress(&format!(
        "core pipelines ready ({compiled}) · scene parsed ({} nodes)",
        scene.nodes.len()
    ));

    // Find the ball's *mesh* node. The renderer's shared-arena mode is FLAT —
    // every transform slot holds an absolute world matrix with no parent→child
    // propagation (see `Transforms::descend_pack_arena`) — so physics must drive
    // the slot of the node that actually carries the mesh, not the parent group.
    //
    // The exported "Ball" is a GROUP whose geometry lives on a child mesh node
    // ("Ball_Mesh"). Its physics counterparts (this scene's `Collider` nodes)
    // are derived below, not hard-coded — the table + walls become static Box3D
    // colliders and the ball a dynamic one.
    let ball_group = scene
        .nodes
        .iter()
        .find(|n| n.name == "Ball")
        .ok_or_else(|| JsValue::from_str("scene has no node named 'Ball'"))?;
    let ball_node_id = find_mesh_node(ball_group)
        .ok_or_else(|| JsValue::from_str("'Ball' has no renderable mesh node to drive"))?;

    // Pull the physics world straight out of the scene's collider nodes.
    let (colliders, spawn, ball_visual_scale) = derive_physics(&scene, ball_node_id);
    tracing::info!(
        "render thread: derived {} colliders from scene (ball scale {ball_visual_scale}, spawn {spawn:?})",
        colliders.len()
    );

    // Fetch the bundle's `assets/` (env cubemaps, textures, imported-mesh glbs)
    // from our own origin as the loader requests them — the exported files sit
    // next to `scene.toml` under `bundle/` (see `index.html`'s copy-dir of
    // `media/bundle`). `HttpAssets` is the loader's ready-made web asset source
    // (feature = "http"), so the template carries no bespoke fetching glue.
    let assets = awsm_renderer_scene_loader::assets::HttpAssets::new(bundle_base.clone());
    // `LoadPhase::label()` is the loader's human-readable progress line
    // ("Fetching textures 3/9…"), deduped — several phases re-report the same
    // counts back-to-back, which reads as log spam rather than progress.
    let mut last_phase_line = String::new();
    let loaded = awsm_renderer_scene_loader::load_scene_for_player(
        &mut renderer,
        &scene,
        &assets,
        |phase| {
            let line = phase.label();
            // "0/0" phases are commit bookkeeping over an empty registry —
            // noise, not progress.
            if line != last_phase_line && !line.contains("0/0") {
                post_progress(&line);
                last_phase_line = line;
            }
        },
    )
    .await
    .map_err(|e| JsValue::from_str(&format!("load_scene_for_player: {e}")))?;
    // The loader builds the scene's particle emitters; the host ticks them
    // each frame (as it advances the clips), in the render loop below.
    let mut particles = awsm_renderer_scene_loader::ParticleDriver::new(&loaded);

    // Lighting is whatever the scene authored — the loader instantiates every
    // light node (the `Sun`, plus any you add in the editor). No code-side lights.

    // Relay GPU-commit progress, deduped (the callback fires per resolution).
    // This commit is normally a cheap no-op — the loader already ran the real
    // one — so most loads print nothing here.
    let mut last_commit_line = String::new();
    renderer
        .commit_load(|stats| {
            let Some(line) = stats.phase_label() else {
                return;
            };
            if line != last_commit_line && !line.contains("0/0") {
                post_progress(&line);
                last_commit_line = line;
            }
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("commit_load: {e}")))?;
    renderer.update_transforms();
    post_progress("gpu commit complete");

    // ── The PLAYER ball is always distinct ───────────────────────────────────
    // Clicking mints duplicates of this mesh (below), and a duplicate copies
    // the source mesh's material key — so the player's look must be swapped in
    // up front, leaving the original material as the duplicates' source of
    // truth via `ball_material`.
    //
    // Preferred: the scene ships the player look as a MATERIAL VARIANT on the
    // ball mesh itself (`material_variants`, the editor's "Material variants"
    // section — e.g. the red-stripe billiard skin) — the loader pre-builds
    // every variant into a ready key (`LoadedScene::node_material_variants`).
    // Fallback for scenes without one: tint the base material's color factor
    // red (multiplies the full texture).
    let ball_mesh = loaded
        .nodes
        .get(&ball_node_id)
        .and_then(|h| h.meshes.first().copied())
        .ok_or_else(|| JsValue::from_str("ball node has no mesh"))?;
    let ball_material = renderer
        .meshes
        .get(ball_mesh)
        .map_err(|e| JsValue::from_str(&format!("ball mesh lookup: {e}")))?
        .material_key;
    let variant_material = loaded
        .node_material_variants
        .get(&ball_node_id)
        .and_then(|vs| vs.iter().find(|v| v.name == "Ball_Player_Red"))
        .map(|v| v.key);
    let player_key = match variant_material {
        Some(key) => {
            tracing::info!("player ball: using the authored 'Ball_Player_Red' variant");
            key
        }
        None => {
            tracing::info!("player ball: no 'Ball_Player_Red' variant — tinting the base material");
            let mut tinted = renderer
                .materials
                .get(ball_material)
                .map_err(|e| JsValue::from_str(&format!("ball material lookup: {e}")))?
                .clone();
            if let awsm_renderer::materials::Material::Pbr(pbr) = &mut tinted {
                pbr.base_color_factor = PLAYER_TINT;
            }
            renderer.materials.insert(
                tinted,
                &renderer.textures,
                &renderer.dynamic_materials,
                &renderer.extras_pool,
            )
        }
    };
    renderer
        .set_mesh_material(ball_mesh, player_key)
        .map_err(|e| JsValue::from_str(&format!("set player material: {e}")))?;

    // ── Click-to-drop shared block ───────────────────────────────────────────
    // Leaked for the session. Clicks arrive from main as [`DropMsg`]s, get
    // unprojected onto the tabletop here (only this thread knows the camera),
    // and become drop requests physics polls. The RAF loop below mints a
    // silver visual duplicate whenever physics reports a new ball.
    let balls: &'static BallMotions = Box::leak(Box::new(BallMotions::new()));
    let ball_radius = colliders
        .iter()
        .find(|c| c.dynamic)
        .and_then(|c| match c.shape {
            ColliderShapeMsg::Ball { radius } => Some(radius),
            _ => None,
        })
        .unwrap_or(0.5);
    let floor = floor_box(&colliders);

    // ── Allocate the shared motion buffer + hand physics what it needs ──────
    // The render thread owns the renderer's transform arena AND this `BodyMotion`
    // buffer: physics publishes prev/curr poses into it, and THIS thread reads
    // them, interpolates, and writes the resulting matrix into the arena slot. So
    // we keep the arena binding here and send physics only the buffer address.
    let ball_tk = loaded
        .nodes
        .get(&ball_node_id)
        .map(|h| h.transform)
        .ok_or_else(|| JsValue::from_str("ball node produced no transform"))?;
    let dirty_words_addr = renderer
        .transforms
        .arena_dirty_words_addr()
        .ok_or_else(|| JsValue::from_str("shared arena not enabled"))?;
    let binding = renderer
        .transforms
        .arena_slot_binding(ball_tk)
        .ok_or_else(|| JsValue::from_str("ball slot has no arena binding"))?;

    // Leaked so it lives in the shared heap for the whole session; physics
    // reconstructs it from this address (same `WebAssembly.Memory`).
    let motion: &'static BodyMotion =
        Box::leak(Box::new(BodyMotion::new(spawn, [0.0, 0.0, 0.0, 1.0])));
    let motion_ptr = motion as *const BodyMotion as usize;

    post_to_main(&RenderMsg::PhysicsInit(PhysicsInit {
        colliders,
        spawn,
        motion_ptr: motion_ptr as f64,
        // Main owns the keyboard, so it allocates the shared InputState and fills
        // this in before spawning physics; we just leave it null.
        input_ptr: 0.0,
        balls_ptr: balls as *const BallMotions as usize as f64,
    }));
    tracing::info!("render thread: posted PhysicsInit, starting render loop");
    post_progress("starting render loop; spawning physics…");

    // ── Render loop: orbit camera + interpolated ball ───────────────────────
    // Main owns the camera (it has the pointer events) and writes it into the
    // shared `CameraRing` once per main-thread frame; each frame we copy one
    // slot of it into `camera`, the camera we draw and unproject clicks with.
    // Clicks can become balls from now on (see `install_render_input`).
    *drop_target.borrow_mut() = Some(DropTarget {
        balls,
        floor,
        ball_radius,
    });
    // Runtime anti-aliasing: main posts the kit's `anti-alias` on a Settings
    // change; the input handler leaves the mode in `pending_aa`, and the
    // render loop applies it off the render path (see the RAF closure).
    // `reconfiguring` gates frames out while the async recompile holds the
    // renderer.
    let reconfiguring: Rc<Cell<bool>> = Rc::new(Cell::new(false));

    // `Option` so an async anti-aliasing reconfig can take the renderer OUT of
    // the cell for the duration of its awaits (rather than holding the RefCell
    // borrow across them); the `reconfiguring` flag keeps the render loop off
    // the cell while it's `None`. See the RAF closure.
    #[allow(clippy::arc_with_non_send_sync)]
    let cell = Rc::new(RefCell::new(Some(renderer)));
    #[allow(clippy::arc_with_non_send_sync)]
    let raf: RafCell = Rc::new(RefCell::new(None));
    let raf_init = raf.clone();
    let raf_run = raf.clone();
    let cell_loop = cell.clone();
    let camera_loop = camera.clone();

    // The previous frame's time (`abs_time_us` of its rAF timestamp), which
    // this frame draws at, minus one step — `None` on the first frame. See
    // the `protocol` module docs.
    let mut last_frame_us: Option<u32> = None;
    // The previous PRESENTED frame's vsync timestamp: the clips and
    // particles advance by the time since it.
    let mut last_presented: Option<f64> = None;
    let mut frame_count: u32 = 0;
    // The Stats panel's physics group (the collector itself is `STATS`).
    let mut physics_stats = PhysicsStats::default();
    // One arena binding per minted (dropped-ball) visual duplicate, with its
    // transform (for a teleport's motion-history reset).
    let mut ball_bindings: Vec<(
        awsm_renderer::buffer::shared_arena::SlotBinding,
        TransformKey,
    )> = Vec::new();
    // Where each ball was drawn last frame (the player first, then the
    // dropped balls), to spot a teleport.
    let mut drawn_at: Vec<Vec3> = Vec::new();

    // Worker-scope `performance` for the frame-work clock. This is NOT frame
    // pacing (the vsync timestamp does that) — it measures how long this
    // thread's work takes, for the Stats panel's renderer group.
    let perf = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("performance"))
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok())
        .ok_or_else(|| JsValue::from_str("render: no performance.now"))?;
    // Frame times cross threads, so they are absolute (each worker has its
    // own `timeOrigin`).
    let time_origin = perf.time_origin();

    *raf_init.borrow_mut() = Some(Closure::new(move |vsync_ms: f64| {
        // An async anti-aliasing reconfig has taken the renderer out of the cell
        // — skip the whole frame (never touch the renderer) and just reschedule.
        if reconfiguring.get() {
            if let Some(cb) = raf_run.borrow().as_ref() {
                let _ =
                    awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
            }
            return;
        }
        // A pending Settings anti-aliasing change: apply it off the render path.
        // `set_anti_aliasing` + `commit_load` are async and recompile the new
        // config's pipeline variants (cached, so a re-toggle is cheap). We take
        // the renderer OUT of the cell for the awaits (so no RefCell borrow is
        // held across them); the `reconfiguring` guard keeps every frame off the
        // renderer until it's put back. `take()` coalesces rapid toggles.
        if let Some(mode) = pending_aa.take() {
            reconfiguring.set(true);
            let cell = cell_loop.clone();
            let done = reconfiguring.clone();
            wasm_bindgen_futures::spawn_local(async move {
                // Own the renderer across the awaits (borrow released here).
                let Some(mut renderer) = cell.borrow_mut().take() else {
                    done.set(false);
                    return;
                };
                // awsm-renderer-settings posts main the `aa-compile-*`
                // messages that raise and lower the "compiling pipelines"
                // overlay (the first switch in each direction really
                // compiles; later ones hit the variant cache, and switching
                // between TAA presets compiles nothing and shows no overlay).
                // `aa-compile-done` is always posted, even on failure.
                let applied = apply_anti_aliasing(&mut renderer, mode, |event| {
                    post_to_page(&SettingsToPage::from(event));
                })
                .await;
                match applied {
                    Ok(()) => tracing::info!(
                        "render thread: anti-aliasing applied: {}",
                        AaMode::from_anti_aliasing(&renderer.anti_aliasing).label()
                    ),
                    Err(message) => {
                        tracing::error!("render thread: {message}");
                        post_to_main(&RenderMsg::Error { message });
                    }
                }
                *cell.borrow_mut() = Some(renderer);
                done.set(false);
            });
            if let Some(cb) = raf_run.borrow().as_ref() {
                let _ =
                    awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
            }
            return;
        }
        // Frame-work clock starts here — everything above is early-out paths.
        let work_t0 = perf.now();
        let mut cell_ref = cell_loop.borrow_mut();
        let Some(r) = cell_ref.as_mut() else {
            // Renderer momentarily taken out for a reconfig — skip this frame.
            if let Some(cb) = raf_run.borrow().as_ref() {
                let _ =
                    awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
            }
            return;
        };

        STATS.with_borrow_mut(|stats| {
            // Phase timing for the Stats panel, only while it shows the
            // renderer's breakdown (awsm-renderer-stats).
            stats.begin_renderer_frame(r, work_t0);
            let timing = stats.renderer_detail();
            let lap = |stats: &mut StatsCollector, phase: Phase| {
                if timing {
                    stats.lap(phase as usize, perf.now());
                }
            };

            // ── Frame start: publish this frame's time, waking physics ──────────
            // The rAF argument is the frame's VSYNC-aligned time (see `RafCell`).
            let frame_us = abs_time_us(time_origin, vsync_ms);
            // What this frame shows: the world at the PREVIOUS frame's time minus
            // one step — inside the pair of steps that frame's burst left (Fix
            // Your Timestep's lerp by alpha), decided before the new burst starts.
            let display_us = last_frame_us.map(|t| t.wrapping_sub(STEP_US));
            // The camera main wrote at least half a frame ago: its previous
            // frame's (see `CameraRing`).
            let camera_cutoff_us = last_frame_us
                .map(|t| t.wrapping_add((time_diff_us(frame_us, t).max(0) / 2) as u32));
            last_frame_us = Some(frame_us);
            // The span since the previous presented frame (ms) advances the
            // clips and particles below; 0 on the first frame.
            let frame_ms = last_presented
                .replace(vsync_ms)
                .map_or(0.0, |prev| (vsync_ms - prev).max(0.0));
            // Physics now has the whole frame period to simulate up to
            // `frame_us`; the next frame draws the result.
            motion.begin_frame(frame_us);

            if let Some(cutoff) = camera_cutoff_us {
                if let Some((yaw, pitch, radius)) = camera_ring.read_at(cutoff) {
                    let mut cam = camera_loop.borrow_mut();
                    cam.yaw = yaw;
                    cam.pitch = pitch;
                    cam.radius = radius;
                }
            }
            lap(stats, Phase::ReadApp);

            // The scene's authored clips and texture UV flows, then its particles.
            // Before the ball writes below, so physics has the last word on the
            // slots it drives. (Don't animate a ball or its parents, though:
            // `update_transforms` rewrites the slots of a subtree a clip moved,
            // over the physics pose.)
            let _ = r.update_animations(frame_ms);
            particles.tick(r, (frame_ms / 1000.0) as f32);
            lap(stats, Phase::Animate);

            let (pos, rot, late) = sample_at(motion.ring(), display_us);
            // A teleport (see `TELEPORT_JUMP_M`): under TAA the ball's motion
            // history goes, in the frame the jump is drawn.
            let teleported = |drawn_at: &mut Vec<Vec3>, index: usize, pos: Vec3| {
                if drawn_at.len() <= index {
                    drawn_at.push(pos);
                    return false;
                }
                let jumped = drawn_at[index].distance(pos) > TELEPORT_JUMP_M;
                drawn_at[index] = pos;
                jumped
            };
            if teleported(&mut drawn_at, 0, pos) {
                r.reset_motion_history(ball_tk);
            }
            // Bake the ball mesh's authored scale in (the arena slot is the absolute
            // world matrix). This thread is the sole writer of this slot.
            let mat =
                Mat4::from_scale_rotation_translation(Vec3::splat(ball_visual_scale), rot, pos);
            let cols = mat.to_cols_array();
            // SAFETY: `binding` + `dirty_words_addr` address the ball's 64-byte arena
            // slot in the shared `WebAssembly.Memory`, alive for the session; `cols`
            // is exactly 16 f32 = 64 bytes.
            let bytes = unsafe { std::slice::from_raw_parts(cols.as_ptr() as *const u8, 64) };
            unsafe {
                foreign_write(binding, dirty_words_addr, bytes);
            }

            // Dropped balls: physics bumps `count` after initializing each new
            // slot, so mint the silver visual duplicate for any ball we haven't
            // seen yet (transform + shared-arena binding), then interpolate every
            // slot with the same alpha (all bodies advance in the same steps).
            let ball_count = balls.count();
            while ball_bindings.len() < ball_count {
                let newest = balls.slot(ball_bindings.len()).read_latest();
                let (cp, cq) = (newest.pos, newest.quat);
                let tk = r.transforms.insert(
                    awsm_renderer::transforms::Transform {
                        translation: Vec3::from_array(cp),
                        rotation: Quat::from_array(cq),
                        scale: Vec3::splat(ball_visual_scale),
                    },
                    None,
                );
                match r.duplicate_mesh_with_transform(ball_mesh, tk) {
                    Ok(dup) => {
                        // The duplicate copies the source's CURRENT material —
                        // which is the player's red tint. Silver it back.
                        if let Err(err) = r.set_mesh_material(dup, ball_material) {
                            tracing::warn!("dropped ball material reset failed: {err}");
                        }
                        match r.transforms.arena_slot_binding(tk) {
                            Some(b) => ball_bindings.push((b, tk)),
                            None => {
                                tracing::error!("dropped ball has no arena slot");
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        tracing::error!("dropped ball duplicate failed: {err}");
                        break;
                    }
                }
            }
            for (slot_index, (b, tk)) in ball_bindings.iter().enumerate() {
                // Same display time as the player — every body advances in the
                // same fixed steps.
                let (pos, rot, _) = sample_at(balls.slot(slot_index), display_us);
                if teleported(&mut drawn_at, slot_index + 1, pos) {
                    r.reset_motion_history(*tk);
                }
                let mat =
                    Mat4::from_scale_rotation_translation(Vec3::splat(ball_visual_scale), rot, pos);
                let cols = mat.to_cols_array();
                // SAFETY: same contract as the player-ball write above — each
                // binding addresses that duplicate's 64-byte arena slot.
                let bytes = unsafe { std::slice::from_raw_parts(cols.as_ptr() as *const u8, 64) };
                unsafe {
                    foreign_write(*b, dirty_words_addr, bytes);
                }
            }
            lap(stats, Phase::ReadApp);

            let cam = camera_loop.borrow();
            // ONE camera API: a view matrix plus projection params. The renderer
            // supplies the depth convention (reverse-Z by default) and the live
            // surface aspect itself, so neither can drift from what it actually
            // renders with — hand-rolling a projection here would invert every
            // depth test. Focus on the orbit pivot so DoF tracks the dolly;
            // aperture stays at the f/5.6 default.
            let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
            let mut params =
                CameraParams::perspective(CAMERA_FOV_Y_DEG.to_radians(), CAMERA_NEAR, CAMERA_FAR);
            params.focus_distance = cam.radius;
            let _ = r.set_camera(view, params);
            drop(cam);
            lap(stats, Phase::SetCamera);
            // Folds the ball slot we just wrote (plus any other dirty slots) into
            // world matrices for this frame.
            r.update_transforms();
            lap(stats, Phase::UpdateTransforms);
            if let Err(err) = r.render(None) {
                tracing::warn!("render thread: render error: {err}");
            }
            lap(stats, Phase::Render);

            // The frame's stats: about once a second the collector closes a
            // window, the physics group is built from the shared counters, and
            // the report goes to main's Stats panel.
            let input = FrameInput {
                late,
                ..FrameInput::new(vsync_ms, perf.now() - work_t0)
            };
            if let Some(window) = stats.end_frame(input, &RendererTimings) {
                let section = physics_stats.section(stats, &window, motion, balls.count());
                post_report(stats.report(&window, vec![section]));
            }
            // Physics times its step phases only while its group is open.
            motion.set_detail(if stats.is_open(crate::stats::PHYSICS) {
                DETAIL_PHYSICS
            } else {
                0
            });

            frame_count = frame_count.wrapping_add(1);
            if frame_count == 3 {
                post_to_main(&RenderMsg::Ready);
                // The scene is on screen: the stats warm up from here.
                post_report(stats.mark_ready());
            }
        });
        if let Some(cb) = raf_run.borrow().as_ref() {
            let _ = awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }));
    if let Some(cb) = raf_init.borrow().as_ref() {
        awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref())?;
    }
    // Keep the renderer + RAF closure alive for the session.
    std::mem::forget(raf);
    std::mem::forget(cell);
    Ok(())
}

/// Fetch + deserialize a same-origin player-bundle `scene.toml`.
async fn fetch_scene(url: &str) -> Result<awsm_renderer_scene::Scene, String> {
    // `no-cache` = revalidate: a runtime fetch isn't busted by a page
    // refresh, so a re-exported scene.toml would otherwise keep serving
    // stale from the browser cache.
    let text = gloo_net::http::Request::get(url)
        .cache(web_sys::RequestCache::NoCache)
        .send()
        .await
        .map_err(|e| format!("fetch: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read: {e}"))?;
    awsm_renderer_scene::project_dir::scene_from_toml(&text).map_err(|e| format!("parse: {e}"))
}

/// Depth-first search for the first node carrying renderable mesh geometry in
/// `node`'s subtree (including `node` itself), returning its [`NodeId`]. The
/// exported "Sphere" is a group; its geometry sits on a child mesh node, and
/// the renderer's flat shared-arena needs the mesh node's own slot driven (see
/// the call site).
fn find_mesh_node(node: &awsm_renderer_scene::EditorNode) -> Option<awsm_renderer_scene::NodeId> {
    use awsm_renderer_scene::NodeKind;
    if matches!(
        node.kind,
        NodeKind::Mesh { .. } | NodeKind::SkinnedMesh { .. } | NodeKind::ClusterMesh { .. }
    ) {
        return Some(node.id);
    }
    node.children.iter().find_map(find_mesh_node)
}

/// Walk the scene's node tree and pull out everything physics needs straight
/// from the authored `Collider` nodes: each collider in **world** space (table +
/// walls static, the ball dynamic), the ball's spawn point, and the ball mesh's
/// world scale (which must be baked into the flat arena slot physics writes).
///
/// World transforms are composed by hand (parent × local, scale→rotate→translate
/// per node) because the scene crate exposes no world-transform helper and the
/// renderer hasn't folded ancestors into its slots at this point.
fn derive_physics(
    scene: &awsm_renderer_scene::Scene,
    ball_mesh_id: awsm_renderer_scene::NodeId,
) -> (Vec<ColliderInit>, [f32; 3], f32) {
    let mut colliders = Vec::new();
    let mut spawn = [0.0, 0.6, 0.0];
    let mut ball_scale = 1.0_f32;
    for node in &scene.nodes {
        // A collider counts as "the ball" when it lives under the `Ball` group.
        let in_ball = node.name == "Ball";
        walk_node(
            node,
            Mat4::IDENTITY,
            in_ball,
            ball_mesh_id,
            &mut colliders,
            &mut spawn,
            &mut ball_scale,
        );
    }
    (colliders, spawn, ball_scale)
}

#[allow(clippy::too_many_arguments)]
fn walk_node(
    node: &awsm_renderer_scene::EditorNode,
    parent_world: Mat4,
    in_ball: bool,
    ball_mesh_id: awsm_renderer_scene::NodeId,
    colliders: &mut Vec<ColliderInit>,
    spawn: &mut [f32; 3],
    ball_scale: &mut f32,
) {
    use awsm_renderer_scene::NodeKind;
    let t = &node.transform;
    let local = Mat4::from_scale_rotation_translation(
        Vec3::from_array(t.scale),
        glam::Quat::from_array(t.rotation), // [x, y, z, w] — matches glam
        Vec3::from_array(t.translation),
    );
    let world = parent_world * local;

    if node.id == ball_mesh_id {
        // Uniform in this scene; take X as the representative scale.
        *ball_scale = world.to_scale_rotation_translation().0.x;
    }

    if let NodeKind::Collider(shape) = &node.kind {
        let (scale, rot, tr) = world.to_scale_rotation_translation();
        let role = if in_ball {
            crate::protocol::ROLE_BALL
        } else if node.name.contains("Wall") {
            crate::protocol::ROLE_WALL
        } else {
            crate::protocol::ROLE_FLOOR
        };
        if in_ball {
            *spawn = tr.to_array();
        }
        if let Some(shape) = collider_shape_msg(shape, scale) {
            colliders.push(ColliderInit {
                shape,
                translation: tr.to_array(),
                rotation: [rot.x, rot.y, rot.z, rot.w],
                dynamic: in_ball,
                role,
            });
        }
    }

    for child in &node.children {
        walk_node(
            child,
            world,
            in_ball,
            ball_mesh_id,
            colliders,
            spawn,
            ball_scale,
        );
    }
}

/// Map a scene `ColliderShape` to the physics-thread wire shape, folding the
/// node's accumulated per-axis world **scale** into the shape extents (a
/// physics collider has no scale of its own — the placement is a rotation +
/// translation). Per-axis folding is exact for the axis-aligned boxes this
/// scene uses; rotated non-uniformly-scaled colliders would shear (not
/// representable — the max-axis fallbacks below are conservative).
///
/// Ellipsoid is dropped (unused in this scene).
fn collider_shape_msg(
    shape: &awsm_renderer_scene::ColliderShape,
    scale: Vec3,
) -> Option<ColliderShapeMsg> {
    use awsm_renderer_scene::ColliderShape as S;
    let s = scale.abs();
    let radial = s.x.max(s.z);
    Some(match *shape {
        S::Box { half_extents } => ColliderShapeMsg::Cuboid {
            half_extents: [
                half_extents[0] * s.x,
                half_extents[1] * s.y,
                half_extents[2] * s.z,
            ],
        },
        S::Sphere { radius } => ColliderShapeMsg::Ball {
            radius: radius * s.x.max(s.y).max(s.z),
        },
        S::Capsule {
            half_height,
            radius,
        } => ColliderShapeMsg::Capsule {
            half_height: half_height * s.y,
            radius: radius * radial,
        },
        S::Cylinder {
            half_height,
            radius,
        } => ColliderShapeMsg::Cylinder {
            half_height: half_height * s.y,
            radius: radius * radial,
        },
        S::Cone {
            half_height,
            radius,
        } => ColliderShapeMsg::Cone {
            half_height: half_height * s.y,
            radius: radius * radial,
        },
        // Unused in this scene (and the Box3D wire has no hull/ellipsoid
        // shape) — dropped, same policy as Ellipsoid always had.
        S::Ellipsoid { .. } | S::ConvexHull { .. } => return None,
    })
}

/// The tabletop's box collider: `(half_extents, translation)` — the click drop
/// zone. `None` if the scene has no recognizable floor box.
fn floor_box(colliders: &[ColliderInit]) -> Option<([f32; 3], [f32; 3])> {
    colliders.iter().find_map(|c| {
        if c.dynamic || c.role != ROLE_FLOOR {
            return None;
        }
        match c.shape {
            ColliderShapeMsg::Cuboid { half_extents } => Some((half_extents, c.translation)),
            _ => None,
        }
    })
}

/// Worker thread: current backing-store aspect ratio (width / height).
fn aspect(canvas: &web_sys::OffscreenCanvas) -> f32 {
    (canvas.width().max(1) as f32) / (canvas.height().max(1) as f32)
}

/// Install this worker's `onmessage`, for the rest of its life (it replaces
/// the bootstrap's init handler, which has done its job): the kit's messages
/// — Settings' `anti-alias` and `resize`, the stats panel's `stats-view` and
/// `stats-page` — and [`DropMsg`] clicks. (The camera does not come through
/// here: see [`CameraRing`].)
///
/// A click becomes a ball once the scene is loaded (`drop_target`): unproject
/// the NDC point through the current camera onto the tabletop plane, clamp
/// it to the table interior (inside the rails), and file a drop request in
/// the shared [`BallMotions`] block for physics to pick up on its next step.
fn install_render_input(
    camera: Rc<RefCell<OrbitCamera>>,
    canvas: web_sys::OffscreenCanvas,
    drop_target: Rc<RefCell<Option<DropTarget>>>,
    pending_aa: Rc<Cell<Option<AaMode>>>,
) -> Result<(), JsValue> {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    let cb = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        match SettingsToWorker::from_js(&data) {
            // A Settings anti-aliasing change — stash it for the render loop
            // to apply (it owns the renderer; we can't touch it from here).
            // A newer change replaces one not applied yet.
            Some(Ok(SettingsToWorker::AntiAlias { mode })) => {
                pending_aa.set(Some(mode));
                return;
            }
            // Canvas resize (the Settings panel follows the canvas's layout
            // size and the resolution slider): apply the new device-pixel
            // backing size to the transferred OffscreenCanvas. The render
            // loop recomputes the camera aspect from the canvas every frame,
            // and the surface reconfigures to the new size on the next
            // present.
            Some(Ok(SettingsToWorker::Resize { width, height })) => {
                if width > 0 && height > 0 {
                    canvas.set_width(width);
                    canvas.set_height(height);
                }
                return;
            }
            Some(Err(err)) => {
                post_to_main(&RenderMsg::Error {
                    message: format!("a settings message did not parse: {err}"),
                });
                return;
            }
            None => {}
        }
        match StatsToWorker::from_js(&data) {
            Some(Ok(message)) => {
                STATS.with_borrow_mut(|stats| stats.handle(message));
                return;
            }
            Some(Err(err)) => {
                post_to_main(&RenderMsg::Error {
                    message: format!("a stats message did not parse: {err}"),
                });
                return;
            }
            None => {}
        }
        if let Ok(DropMsg::Ball { ndc_x, ndc_y }) = serde_wasm_bindgen::from_value::<DropMsg>(data)
        {
            let target = drop_target.borrow();
            let Some(DropTarget {
                balls,
                floor: Some((floor_he, floor_tr)),
                ball_radius,
            }) = target.as_ref()
            else {
                return;
            };
            let (balls, floor_he, floor_tr, ball_radius) =
                (*balls, *floor_he, *floor_tr, *ball_radius);
            let cam = camera.borrow();
            let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
            let projection = Mat4::perspective_rh(
                CAMERA_FOV_Y_DEG.to_radians(),
                aspect(&canvas),
                CAMERA_NEAR,
                CAMERA_FAR,
            );
            drop(cam);
            // Unproject the click ray and intersect the tabletop plane.
            let inv = (projection * view).inverse();
            let p0 = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
            let p1 = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
            let dir = p1 - p0;
            let table_top = floor_tr[1] + floor_he[1];
            if dir.y.abs() < 1e-6 {
                return; // grazing ray — no usable intersection
            }
            let t = (table_top - p0.y) / dir.y;
            if t <= 0.0 {
                return; // clicked the sky
            }
            // Clamp inside the rails so the ball always lands on the felt.
            let margin = ball_radius + 0.25;
            let x = (p0.x + t * dir.x).clamp(
                floor_tr[0] - floor_he[0] + margin,
                floor_tr[0] + floor_he[0] - margin,
            );
            let z = (p0.z + t * dir.z).clamp(
                floor_tr[2] - floor_he[2] + margin,
                floor_tr[2] + floor_he[2] - margin,
            );
            balls.request_drop(x, z);
        }
    });
    scope.set_onmessage(Some(cb.as_ref().unchecked_ref()));
    cb.forget();
    Ok(())
}

/// Post a human-readable load-progress line to main (the loading screen).
fn post_progress(message: &str) {
    post_to_main(&RenderMsg::Progress {
        message: message.to_string(),
    });
}

/// Serialize a [`RenderMsg`] and post it to the main thread.
fn post_to_main(msg: &RenderMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    match serde_wasm_bindgen::to_value(msg) {
        Ok(v) => {
            let _ = scope.post_message(&v);
        }
        Err(e) => tracing::error!("render thread: serialize RenderMsg: {e}"),
    }
}
