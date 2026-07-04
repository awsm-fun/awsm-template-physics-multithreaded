//! Physics thread — a **Box3D** world built from the scene's collider nodes.
//!
//! The exported scene's `Collider` nodes become the world: the table + walls are
//! static bodies (box hulls), the ball is a dynamic sphere (see [`PhysicsInit`]).
//! The ball drops in and bounces; WASD/arrow keys roll it, Space makes it hop.
//!
//! Box3D is C (vendored at `vendor/box3d`), compiled into this same wasm module
//! by `box3d-sys` — same shared memory, no bridge, no copies. This thread is the
//! world's only owner and the only caller of `b3World_Step`; the step itself
//! fans Box3D's internal parallel-for out across the wasm **task pool**
//! ([`crate::physics_tasks`]) — extra web workers that claim solver tasks from
//! shared memory (this thread helps too, inside `finishTask`). On machines with
//! no core headroom the world falls back to `workerCount = 1` (Box3D's inline
//! serial path — no scheduler at all).
//!
//! The sim runs at a **fixed timestep** (`protocol::SIM_HZ`) decoupled from the wall clock: an
//! accumulator tracks elapsed real time and we step Box3D exactly as many fixed
//! steps as that time bought (capped, to avoid a spiral of death) — so the
//! simulation stays correct and real-time regardless of how heavy a step gets.
//! Each step we publish the ball's prev/curr pose into the shared [`BodyMotion`]
//! buffer; the render thread interpolates and draws it.
//!
//! **The render thread paces the sim, not a timer.** Each loop iteration blocks
//! on the [`BodyMotion`] frame-tick (`memory.atomic.wait`) until render presents a
//! frame, then advances and publishes the pose that frame's successor will read.
//! This locks the step cadence to vsync: a physics timer running on its *own*
//! clock beats against the display's refresh, and every phase-slip becomes a
//! one-frame stutter. See [`BodyMotion`].
//!
//! Blocking the worker like this means it has no event loop to receive messages
//! on, so **input is shared memory, not messages**: main writes held keys / jumps
//! into the [`InputState`] block and we poll it each step. Audio cues still go
//! back to main as [`AudioMsg`] (the ball's roll + its wall/floor impacts) —
//! classified from Box3D's begin-touch contact events plus the pre-step
//! velocity (the speed the ball *arrives* with, which the step then reflects).

use std::collections::HashMap;
use std::ffi::c_void;

use box3d_sys as b3;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

use crate::protocol::{
    AudioMsg, BallMotions, BodyMotion, ColliderShapeMsg, InputState, PhysicsInit, HELD_BACK,
    HELD_FORWARD, HELD_LEFT, HELD_RIGHT, ROLE_FLOOR, ROLE_WALL, SIM_HZ,
};

/// Fixed simulation step, derived from the shared [`SIM_HZ`] (the one rate knob).
/// `FIXED_DT_SECS` is what Box3D integrates; the `_MS` form drives the
/// accumulator. Render derives the *same* value for its interpolation `alpha`.
const FIXED_DT_SECS: f32 = (1.0 / SIM_HZ) as f32;
const FIXED_DT_MS: f64 = 1000.0 / SIM_HZ;
/// Box3D's internal solver sub-steps per `b3World_Step` (its *Soft Step* solver
/// iterates inside one step). Upstream advises 4 at a 60 Hz step; our outer rate
/// is already 240 Hz, so 1 sub-step lands on the same 240 Hz solver cadence —
/// raise this only if contact quality demands it (cost is linear).
const SUB_STEPS: i32 = 1;
/// Clamp a single elapsed gap (e.g. a backgrounded tab) so we never try to
/// simulate minutes of backlog at once — the most real time one iteration absorbs.
const MAX_FRAME_MS: f64 = 100.0;
/// Cap on fixed steps per loop iteration — bounds catch-up so a hitch can't
/// trigger a spiral of death. Derived to always cover a full [`MAX_FRAME_MS`] gap
/// at the current rate (≈6 at 60 Hz, ≈24 at 240 Hz), so it scales with [`SIM_HZ`].
const MAX_SUBSTEPS: u32 = (MAX_FRAME_MS / 1000.0 * SIM_HZ + 0.999) as u32;
const GRAVITY: f32 = -9.81;
/// How far above its authored position the ball starts, so it drops in.
const DROP_HEIGHT: f32 = 2.0;
/// Horizontal rolling **acceleration** (m/s²) applied while a movement key is
/// held. Expressed per *second*, not per tick, so the feel is identical at any
/// [`SIM_HZ`]: each step adds `MOVE_ACCEL * FIXED_DT_SECS` to the velocity. The
/// impulse is mass-scaled at apply time so the feel is independent of ball size.
const MOVE_ACCEL: f32 = 27.0;
/// Cap on horizontal (xz) rolling speed. The per-tick kick is applied
/// continuously while a key is held; with the ball's light damping that would
/// otherwise accumulate into a runaway velocity, so we clamp the horizontal
/// speed each tick (vertical/jump motion is left untouched).
const MAX_SPEED: f32 = 6.0;
/// Launch velocity for a jump (also mass-scaled into an impulse).
const JUMP_DV: f32 = 3.2;
/// Cap on the horizontal speed a touch fling can set (m/s). Deliberately above
/// [`MAX_SPEED`] so a hard flick reads as a real throw, but bounded so the
/// ball can't be launched arbitrarily fast over the rails.
const FLING_MAX: f32 = 9.0;
/// Restitution/friction for the static table + walls.
const STATIC_RESTITUTION: f32 = 0.5;
const STATIC_FRICTION: f32 = 0.9;
/// Restitution/friction/damping for the dynamic ball.
const BALL_RESTITUTION: f32 = 0.6;
const BALL_FRICTION: f32 = 0.9;
const BALL_LINEAR_DAMPING: f32 = 0.1;
const BALL_ANGULAR_DAMPING: f32 = 0.4;
/// Target rate (Hz) for the continuous roll cue. The audio side glides between
/// values, so ~20 Hz stays smooth; kept as a rate so the message traffic doesn't
/// balloon with [`SIM_HZ`].
const ROLL_CUE_HZ: f64 = 20.0;
/// Send the roll cue every Nth step — derived so it fires ~[`ROLL_CUE_HZ`] at any
/// sim rate (3 at 60 Hz, 12 at 240 Hz). Impacts are sent immediately.
const ROLL_EVERY: u32 = (SIM_HZ / ROLL_CUE_HZ + 0.5) as u32;
/// Below this normalized roll speed the ball is treated as effectively still.
const ROLL_FLOOR: f32 = 0.04;
/// Vertical speed (m/s) that maps a landing to full intensity.
const LAND_FULL_SPEED: f32 = 6.5;
/// Minimum approach speed (m/s) for a contact to count as a real impact — below
/// this it's resting/grinding contact, not a hit (avoids machine-gun retriggers
/// when a ball is shoved steadily against a wall or settling on the floor).
/// Enforced by Box3D itself: this is the world's `hitEventThreshold`.
const HIT_MIN_SPEED: f32 = 0.8;
/// Minimum *time* between successive impacts of the same kind — a debounce so
/// shoving the ball against a rail doesn't machine-gun the knock.
const IMPACT_COOLDOWN_SECS: f64 = 0.13;
/// The same debounce in steps, derived from [`SIM_HZ`] (8 at 60 Hz, 31 at 240 Hz).
const IMPACT_COOLDOWN: u32 = (IMPACT_COOLDOWN_SECS * SIM_HZ + 0.5) as u32;
/// Below this height the ball has escaped the table; we drop it back to spawn
/// (the walls are low rails, so a determined shove can pop the ball over them).
const FALL_LIMIT: f32 = -2.0;
/// Step at which the SIMD-parity probe logs the ball's exact pose bits (20 s
/// in — long settled by then). Deterministic absent input/clicks, so a scalar
/// (`BOX3D_WASM_SCALAR=1`) and a SIMD build must print identical bits.
const PARITY_TICK: u32 = (SIM_HZ * 20.0) as u32;

const fn v3(x: f32, y: f32, z: f32) -> b3::b3Vec3 {
    b3::b3Vec3 { x, y, z }
}

const QUAT_IDENTITY: b3::b3Quat = b3::b3Quat {
    v: v3(0.0, 0.0, 0.0),
    s: 1.0,
};

/// Box3D assert hook: surface the condition in the console before the trap
/// (without this an assert is an opaque `RuntimeError: unreachable`).
/// Returning nonzero keeps the debugger break (loud failure).
unsafe extern "C" fn box3d_assert(
    condition: *const core::ffi::c_char,
    file: *const core::ffi::c_char,
    line: i32,
) -> i32 {
    let cstr = |p: *const core::ffi::c_char| {
        if p.is_null() {
            "?".into()
        } else {
            core::ffi::CStr::from_ptr(p).to_string_lossy()
        }
    };
    tracing::error!("BOX3D ASSERT: {} ({}:{line})", cstr(condition), cstr(file));
    1
}

/// Route Box3D's printf output (its default assert/warning formatting) and
/// asserts to the console. Idempotent; called once at physics startup.
fn install_box3d_hooks() {
    b3::wasm_shim::set_shim_log(|msg| tracing::warn!("box3d: {msg}"));
    unsafe { b3::b3SetAssertFcn(box3d_assert) };
}

/// Worker entry: deserialize the [`PhysicsInit`] payload, build the world, wire
/// up input, and start the fixed-step loop.
pub fn start(payload: JsValue) -> Result<(), JsValue> {
    let init: PhysicsInit = serde_wasm_bindgen::from_value(payload)
        .map_err(|e| JsValue::from_str(&format!("physics: bad PhysicsInit: {e}")))?;
    install_box3d_hooks();
    tracing::info!(
        "physics thread: starting Box3D world — {SIM_HZ} Hz (dt {FIXED_DT_MS:.2}ms, \
         {SUB_STEPS} solver sub-steps, max_substeps {MAX_SUBSTEPS}, roll every {ROLL_EVERY}, \
         impact cooldown {IMPACT_COOLDOWN})"
    );

    // The render thread owns this buffer (in shared memory); we publish poses into
    // it. SAFETY: `motion_ptr` is a live address in the same `WebAssembly.Memory`,
    // kept alive (leaked) by the render thread for the session.
    let motion: &'static BodyMotion = unsafe { &*(init.motion_ptr as usize as *const BodyMotion) };

    // `performance.now()` (this worker's clock) drives the accumulator.
    let perf = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("performance"))
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok())
        .ok_or_else(|| JsValue::from_str("physics: no performance.now"))?;

    // ── The task pool: Box3D's parallelism on real wasm threads ─────────────
    // Pool size: leave headroom for the 3 existing threads (main/render/
    // physics), cap at 4 — the demo's point is the skeleton, not saturation.
    // With 0 pool workers (small machines) the world runs Box3D's serial
    // fallback and the pool is never allocated.
    let hardware = js_sys::global()
        .unchecked_into::<web_sys::DedicatedWorkerGlobalScope>()
        .navigator()
        .hardware_concurrency() as i32;
    let pool_workers = (hardware - 3).clamp(0, 4) as u32;
    let pool: Option<&'static crate::physics_tasks::TaskPool> = if pool_workers > 0 {
        let pool = crate::physics_tasks::TaskPool::leak_new();
        // MAIN spawns the workers: this thread blocks on the frame-tick futex
        // for its whole life and never services its event loop again, so it
        // can neither reliably start nested workers nor observe their errors.
        // The gap until they come online is safe — finishTask help-executes
        // everything itself meanwhile (the pool just adds parallelism).
        post_physics(&crate::protocol::PhysicsMsg::SpawnTaskWorkers {
            pool: pool.addr() as f64,
            count: pool_workers,
        });
        tracing::info!(
            "physics thread: task pool requested — {pool_workers} workers \
             (hardware_concurrency {hardware}), Box3D workerCount {}",
            pool_workers + 1
        );
        Some(pool)
    } else {
        tracing::info!(
            "physics thread: no task pool (hardware_concurrency {hardware}) — Box3D serial"
        );
        None
    };

    // ── Build the world from the scene's collider list ──────────────────────
    // With a pool: our enqueue/finish callbacks + workerCount = pool + 1 (the
    // stepping thread participates via finishTask's help-execution). Without:
    // workerCount == 1 → Box3D's inline serial task path — its internal
    // pthread scheduler (which doesn't exist on wasm) is never touched either
    // way.
    let world = unsafe {
        let mut def = b3::b3DefaultWorldDef();
        def.gravity = v3(0.0, GRAVITY, 0.0);
        // Impacts are voiced from hit events; this is the speed gate.
        def.hitEventThreshold = HIT_MIN_SPEED;
        match pool {
            Some(pool) => {
                def.workerCount = pool_workers + 1;
                def.enqueueTask = Some(crate::physics_tasks::enqueue_task);
                def.finishTask = Some(crate::physics_tasks::finish_task);
                def.userTaskContext = pool.addr() as *mut c_void;
            }
            None => def.workerCount = 1,
        }
        b3::b3CreateWorld(&def)
    };
    if !unsafe { b3::b3World_IsValid(world) } {
        return Err(JsValue::from_str("physics: b3CreateWorld failed"));
    }

    // Static geometry: the table + walls, one static body per collider placed at
    // its world pose. Each shape carries its gameplay role in userData so
    // contact events can be classified.
    for c in init.colliders.iter().filter(|c| !c.dynamic) {
        unsafe {
            let mut body_def = b3::b3DefaultBodyDef();
            body_def.position = v3(c.translation[0], c.translation[1], c.translation[2]);
            body_def.rotation = b3::b3Quat {
                v: v3(c.rotation[0], c.rotation[1], c.rotation[2]),
                s: c.rotation[3],
            };
            let body = b3::b3CreateBody(world, &body_def);

            let mut shape_def = b3::b3DefaultShapeDef();
            shape_def.baseMaterial.friction = STATIC_FRICTION;
            shape_def.baseMaterial.restitution = STATIC_RESTITUTION;
            shape_def.userData = c.role as usize as *mut c_void;
            // Contact events are OR'd across a pair's shapes (contact.c) — the
            // player ball's own enable already covers ball-vs-static. Enabling
            // here too would flood the arrays with dropped-ball contacts.
            shape_def.enableContactEvents = false;
            create_shape(body, &shape_def, &c.shape);
        }
    }

    // The dynamic ball, dropped from above its authored spawn.
    let ball_def = init
        .colliders
        .iter()
        .find(|c| c.dynamic)
        .ok_or_else(|| JsValue::from_str("physics: no dynamic collider (ball) in scene"))?;
    let spawn = v3(init.spawn[0], init.spawn[1] + DROP_HEIGHT, init.spawn[2]);
    let (ball_body, ball_shape) = unsafe {
        let mut body_def = b3::b3DefaultBodyDef();
        body_def.r#type = b3::b3_dynamicBody;
        body_def.position = spawn;
        body_def.linearDamping = BALL_LINEAR_DAMPING;
        body_def.angularDamping = BALL_ANGULAR_DAMPING;
        body_def.enableSleep = false;
        // Continuous collision so the ball can't tunnel through the thin rails
        // at speed (Box3D bullets do CCD against static + non-bullet bodies).
        body_def.isBullet = true;
        let body = b3::b3CreateBody(world, &body_def);

        let mut shape_def = b3::b3DefaultShapeDef();
        shape_def.density = 1.0;
        shape_def.baseMaterial.friction = BALL_FRICTION;
        shape_def.baseMaterial.restitution = BALL_RESTITUTION;
        shape_def.userData = ball_def.role as usize as *mut c_void;
        // Contact events maintain the grounded/contacts set (roll cue); hit
        // events voice the impacts (they carry position + approach speed).
        shape_def.enableContactEvents = true;
        shape_def.enableHitEvents = true;
        let shape = create_shape(body, &shape_def, &ball_def.shape);
        (body, shape)
    };
    // Mass (from the shape's density) — used to scale input impulses so the
    // ball handles the same regardless of its authored radius.
    let ball_mass = unsafe { b3::b3Body_GetMass(ball_body) };
    if ball_mass <= 0.0 || !ball_mass.is_finite() {
        return Err(JsValue::from_str("physics: ball has no mass"));
    }

    // ── Click-dropped balls ──────────────────────────────────────────────────
    // The shared block render allocated: it files drop requests (clicks,
    // unprojected onto the table) and reads back the pose slots we publish.
    // SAFETY: render leaked the block in the same `WebAssembly.Memory`.
    let balls: &'static BallMotions = unsafe { &*(init.balls_ptr as usize as *const BallMotions) };
    let mut last_drop_seq = 0u32;
    let ball_radius = match ball_def.shape {
        ColliderShapeMsg::Ball { radius } => radius,
        _ => 0.5,
    };
    // Per dropped ball: body id + its drop point (fall-through respawn target).
    let mut drop_bodies: Vec<b3::b3BodyId> = Vec::new();
    let mut drop_spawns: Vec<b3::b3Vec3> = Vec::new();

    // ── Input: the shared block main writes; we poll it each step ────────────
    // SAFETY: `input_ptr` is a live address in the same `WebAssembly.Memory`,
    // leaked by the main thread for the session.
    let input: &'static InputState = unsafe { &*(init.input_ptr as usize as *const InputState) };
    let mut last_jump_seq = input.jump_seq();
    let mut last_fling_seq = input.fling_seq();

    // Shapes the ball is currently touching (`other shape index1` → role),
    // maintained from begin/end contact events. Membership answers "grounded";
    // a begin event is by definition a *fresh* contact → impact classification.
    let mut contacts: HashMap<i32, u8> = HashMap::new();
    let mut tick: u32 = 0;
    // Last tick each impact kind fired, for debouncing.
    let mut last_wall_tick: u32 = 0;
    let mut last_land_tick: u32 = 0;
    let mut last_clack_tick: u32 = 0;

    // Step-cost telemetry window (avg b3World_Step wall time, logged 1/min).
    let mut step_ms_acc: f64 = 0.0;
    let mut step_ms_count: u32 = 0;

    // Fixed-timestep accumulator; every step's pose is published into the
    // shared pose rings (render samples them at its own display time).
    let mut total_steps: u32 = 0;
    let mut accumulator: f64 = 0.0;
    let mut last_time = perf.now();

    // ── Driver: render-paced loop. Block until the render thread presents a
    // frame, then catch the fixed-step sim up to elapsed real time and publish
    // the pose the next frame will read — so the sim steps in phase with vsync,
    // not against a second free-running clock (see `BodyMotion`) ─────────────
    let mut last_frame = motion.frame_tick();
    loop {
        last_frame = motion.wait_frame(last_frame);
        let now = perf.now();
        let mut frame = now - last_time;
        last_time = now;
        if frame > MAX_FRAME_MS {
            frame = MAX_FRAME_MS; // a long stall (backgrounded tab) — don't binge-simulate
        }
        accumulator += frame;

        let mut substeps: u32 = 0;

        while accumulator >= FIXED_DT_MS && substeps < MAX_SUBSTEPS {
            // Whole-step CPU clock for the stats panel: input poll + solver
            // step + event classification + pose publish (everything this
            // iteration does), accumulated in the shared block as µs.
            let body_t0 = perf.now();
            let mut teleported = false;
            tick = tick.wrapping_add(1);
            total_steps = total_steps.wrapping_add(1);

            // ── Pending click-drop? Create the ball before stepping ────────
            // (Render unprojected the click and filed the request; if several
            // clicks land inside one ~4 ms poll window the latest wins.)
            if let Some((x, z)) = balls.poll_drop(&mut last_drop_seq) {
                if drop_bodies.len() < crate::protocol::MAX_BALLS {
                    let spawn_at = v3(x, DROP_HEIGHT + init.spawn[1], z);
                    unsafe {
                        let mut body_def = b3::b3DefaultBodyDef();
                        body_def.r#type = b3::b3_dynamicBody;
                        body_def.position = spawn_at;
                        body_def.linearDamping = BALL_LINEAR_DAMPING;
                        body_def.angularDamping = BALL_ANGULAR_DAMPING;
                        // Cheaper than the player: sleep allowed (a settled
                        // pile costs ~nothing), no bullet CCD.
                        let body = b3::b3CreateBody(world, &body_def);
                        let mut shape_def = b3::b3DefaultShapeDef();
                        shape_def.density = 1.0;
                        shape_def.baseMaterial.friction = BALL_FRICTION;
                        shape_def.baseMaterial.restitution = BALL_RESTITUTION;
                        shape_def.userData = crate::protocol::ROLE_BALL as usize as *mut c_void;
                        // Hit events so its drop + collisions make sound too.
                        shape_def.enableHitEvents = true;
                        b3::b3CreateSphereShape(
                            body,
                            &shape_def,
                            &b3::b3Sphere {
                                center: v3(0.0, 0.0, 0.0),
                                radius: ball_radius,
                            },
                        );
                        drop_bodies.push(body);
                    }
                    drop_spawns.push(spawn_at);
                    let index = drop_bodies.len() - 1;
                    // Seed the ring with a valid pose at the current step (a
                    // snap, so interpolation near the boundary is inert), THEN
                    // publish the count (Release) — render mints the visual
                    // when it sees it.
                    balls.slot(index).publish_snap(
                        total_steps,
                        [spawn_at.x, spawn_at.y, spawn_at.z],
                        [0.0, 0.0, 0.0, 1.0],
                    );
                    balls.set_count(drop_bodies.len());
                } else {
                    tracing::warn!(
                        "ball drop ignored — MAX_BALLS ({}) reached",
                        crate::protocol::MAX_BALLS
                    );
                }
            }

            // Apply input before stepping (polled from the shared block).
            unsafe {
                let held = input.held();
                let mut dir = v3(0.0, 0.0, 0.0);
                if held & HELD_FORWARD != 0 {
                    dir.z -= 1.0;
                }
                if held & HELD_BACK != 0 {
                    dir.z += 1.0;
                }
                if held & HELD_LEFT != 0 {
                    dir.x -= 1.0;
                }
                if held & HELD_RIGHT != 0 {
                    dir.x += 1.0;
                }
                let dir_sq = dir.x * dir.x + dir.z * dir.z;
                if dir_sq > 0.0 {
                    // `dir` is in the CAMERA frame (W = away from the camera).
                    // Rotate it into world space by the camera yaw main mirrors
                    // into the input block, so W/A/S/D stay view-relative at any
                    // orbit angle. Basis from `OrbitCamera::eye()` — the eye
                    // sits at (sinθ, cosθ)·r, so camera-forward is (−sinθ,
                    // −cosθ) and camera-right is (cosθ, −sinθ); at θ = 0 this
                    // is the identity (forward = −Z, right = +X).
                    let yaw = input.camera_yaw();
                    let (s, c) = yaw.sin_cos();
                    let wx = dir.x * c + dir.z * s;
                    let wz = -dir.x * s + dir.z * c;
                    // Acceleration × step dt × mass → a velocity kick that's the
                    // same per *second* at any SIM_HZ (see `MOVE_ACCEL`).
                    // (Rotation preserves length, so `dir_sq` still normalizes.)
                    let k = MOVE_ACCEL * FIXED_DT_SECS * ball_mass / dir_sq.sqrt();
                    b3::b3Body_ApplyLinearImpulseToCenter(ball_body, v3(wx * k, 0.0, wz * k), true);
                    // Clamp horizontal speed so the held impulse can't run away,
                    // leaving the vertical component (gravity / jumps) intact.
                    let v = b3::b3Body_GetLinearVelocity(ball_body);
                    let horiz_sq = v.x * v.x + v.z * v.z;
                    if horiz_sq > MAX_SPEED * MAX_SPEED {
                        let scale = MAX_SPEED / horiz_sq.sqrt();
                        b3::b3Body_SetLinearVelocity(ball_body, v3(v.x * scale, v.y, v.z * scale));
                    }
                }
                // Jump is edge-triggered: act when main's counter has advanced.
                let seq = input.jump_seq();
                if seq != last_jump_seq {
                    last_jump_seq = seq;
                    b3::b3Body_ApplyLinearImpulseToCenter(
                        ball_body,
                        v3(0.0, JUMP_DV * ball_mass, 0.0),
                        true,
                    );
                }
                // Touch fling — edge-triggered like the jump. The swipe
                // velocity arrives in the camera frame; rotate by the same yaw
                // as the held keys, then SET the horizontal velocity: a throw
                // should read as "the ball goes where I flicked, at the speed
                // I flicked", not as a nudge on top of whatever motion it had.
                // Vertical velocity is left alone (gravity / jumps).
                if let Some((fx, fz)) = input.poll_fling(&mut last_fling_seq) {
                    let yaw = input.camera_yaw();
                    let (s, c) = yaw.sin_cos();
                    let mut wx = fx * c + fz * s;
                    let mut wz = -fx * s + fz * c;
                    let speed_sq = wx * wx + wz * wz;
                    if speed_sq > FLING_MAX * FLING_MAX {
                        let k = FLING_MAX / speed_sq.sqrt();
                        wx *= k;
                        wz *= k;
                    }
                    let v = b3::b3Body_GetLinearVelocity(ball_body);
                    b3::b3Body_SetLinearVelocity(ball_body, v3(wx, v.y, wz));
                }
            }

            // Recycle the pool's task slots — everything from the previous
            // step is COMPLETE (each parallel-for finishes before returning).
            if let Some(pool) = pool {
                pool.reset();
            }
            let step_t0 = perf.now();
            unsafe { b3::b3World_Step(world, FIXED_DT_SECS, SUB_STEPS) };
            step_ms_acc += perf.now() - step_t0;
            step_ms_count += 1;

            // Telemetry, ~once/minute at debug: avg b3World_Step wall time
            // over the window (the M6 scalar-vs-SIMD benchmark reads this)
            // and, with a pool, the per-executor claim counts (index 0 = this
            // thread helping in finishTask, 1.. = pool workers).
            if tick.is_multiple_of(SIM_HZ as u32 * 60) && step_ms_count > 0 {
                tracing::debug!(
                    "avg b3World_Step: {:.3} ms over {} steps{}",
                    step_ms_acc / step_ms_count as f64,
                    step_ms_count,
                    match pool {
                        Some(pool) => format!(
                            "; task pool claims (self, workers..): {:?}",
                            pool.claim_counts(pool_workers as usize + 1)
                        ),
                        None => String::new(),
                    }
                );
                step_ms_acc = 0.0;
                step_ms_count = 0u32;
            }
            // M6 parity probe: the exact pose bits after a fixed number of
            // steps. Spawns are seeded and there's no input this early, so a
            // scalar and a SIMD build must print identical bits.
            if tick == PARITY_TICK {
                let p = unsafe { b3::b3Body_GetPosition(ball_body) };
                let q = unsafe { b3::b3Body_GetRotation(ball_body) };
                tracing::info!(
                    "parity @ step {PARITY_TICK}: pos [{:08x} {:08x} {:08x}] quat [{:08x} {:08x} {:08x} {:08x}]",
                    p.x.to_bits(), p.y.to_bits(), p.z.to_bits(),
                    q.v.x.to_bits(), q.v.y.to_bits(), q.v.z.to_bits(), q.s.to_bits()
                );
            }

            // Safety net: if the ball escaped (shoved over a low rail), drop it
            // back to spawn rather than losing it off-world.
            if unsafe { b3::b3Body_GetPosition(ball_body) }.y < FALL_LIMIT {
                unsafe {
                    b3::b3Body_SetTransform(ball_body, spawn, QUAT_IDENTITY);
                    b3::b3Body_SetLinearVelocity(ball_body, v3(0.0, 0.0, 0.0));
                    b3::b3Body_SetAngularVelocity(ball_body, v3(0.0, 0.0, 0.0));
                }
                contacts.clear();
                teleported = true;
            }

            let (pos, quat) = unsafe {
                let p = b3::b3Body_GetPosition(ball_body);
                let q = b3::b3Body_GetRotation(ball_body);
                ([p.x, p.y, p.z], [q.v.x, q.v.y, q.v.z, q.s])
            };

            // ── Contact + hit events → grounded set + impact sounds ────────
            // Begin/end touch events (player ball only) maintain the contacts
            // map — grounded ⇒ roll cue. HIT events voice the impacts for
            // EVERY ball (player and click-dropped alike): each carries the
            // contact point and the approach speed, so the drop thud, rail
            // knocks, and ball-on-ball clacks all cue with a real intensity.
            unsafe {
                let events = b3::b3World_GetContactEvents(world);
                for i in 0..events.beginCount as usize {
                    let ev = &*events.beginEvents.add(i);
                    let other = if ev.shapeIdA == ball_shape {
                        ev.shapeIdB
                    } else if ev.shapeIdB == ball_shape {
                        ev.shapeIdA
                    } else {
                        continue;
                    };
                    contacts.insert(other.index1, b3::b3Shape_GetUserData(other) as usize as u8);
                }
                for i in 0..events.endCount as usize {
                    let ev = &*events.endEvents.add(i);
                    let other = if ev.shapeIdA == ball_shape {
                        ev.shapeIdB
                    } else if ev.shapeIdB == ball_shape {
                        ev.shapeIdA
                    } else {
                        continue;
                    };
                    contacts.remove(&other.index1);
                }

                for i in 0..events.hitCount as usize {
                    let ev = &*events.hitEvents.add(i);
                    let role_a = b3::b3Shape_GetUserData(ev.shapeIdA) as usize as u8;
                    let role_b = b3::b3Shape_GetUserData(ev.shapeIdB) as usize as u8;
                    let (x, y, z) = (ev.point.x, ev.point.y, ev.point.z);
                    // Voice by the "surface" hit: a rail knock, a floor thud,
                    // or (ball-on-ball) the same knock timbre.
                    if role_a == ROLE_WALL || role_b == ROLE_WALL {
                        if tick.wrapping_sub(last_wall_tick) >= IMPACT_COOLDOWN {
                            last_wall_tick = tick;
                            let intensity = (ev.approachSpeed / MAX_SPEED).clamp(0.12, 1.0);
                            post_audio(&AudioMsg::WallHit { x, y, z, intensity });
                        }
                    } else if role_a == ROLE_FLOOR || role_b == ROLE_FLOOR {
                        if tick.wrapping_sub(last_land_tick) >= IMPACT_COOLDOWN {
                            last_land_tick = tick;
                            let intensity = (ev.approachSpeed / LAND_FULL_SPEED).clamp(0.12, 1.0);
                            post_audio(&AudioMsg::Land { x, y, z, intensity });
                        }
                    } else if tick.wrapping_sub(last_clack_tick) >= IMPACT_COOLDOWN {
                        // ball-on-ball → the steel-sphere clack (its own
                        // debounce window, so a rail knock and a clack in the
                        // same instant both sound — they're distinct events).
                        last_clack_tick = tick;
                        let intensity = (ev.approachSpeed / MAX_SPEED).clamp(0.12, 1.0);
                        post_audio(&AudioMsg::BallClack { x, y, z, intensity });
                    }
                }
            }
            let grounded = contacts.values().any(|&role| role != ROLE_WALL);

            // ── Continuous roll cue (throttled) ────────────────────────────
            if tick.is_multiple_of(ROLL_EVERY) {
                let v = unsafe { b3::b3Body_GetLinearVelocity(ball_body) };
                let horiz = (v.x * v.x + v.z * v.z).sqrt();
                // Only "rolling" while in contact with the table.
                let speed = if grounded {
                    (horiz / MAX_SPEED).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let speed = if speed < ROLL_FLOOR { 0.0 } else { speed };
                post_audio(&AudioMsg::Roll {
                    speed,
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                });
            }

            // ── Dropped balls: publish this step's pose into their rings ───
            // Same fall-through safety net as the player, snapping back to
            // the ball's own drop point (publish_snap: no teleport sweep).
            for (i, body) in drop_bodies.iter().enumerate() {
                unsafe {
                    let mut p = b3::b3Body_GetPosition(*body);
                    let ball_teleported = p.y < FALL_LIMIT;
                    if ball_teleported {
                        b3::b3Body_SetTransform(*body, drop_spawns[i], QUAT_IDENTITY);
                        b3::b3Body_SetLinearVelocity(*body, v3(0.0, 0.0, 0.0));
                        b3::b3Body_SetAngularVelocity(*body, v3(0.0, 0.0, 0.0));
                        p = b3::b3Body_GetPosition(*body);
                    }
                    let q = b3::b3Body_GetRotation(*body);
                    let curr_p = [p.x, p.y, p.z];
                    let curr_q = [q.v.x, q.v.y, q.v.z, q.s];
                    if ball_teleported {
                        balls.slot(i).publish_snap(total_steps, curr_p, curr_q);
                    } else {
                        balls.slot(i).publish(total_steps, curr_p, curr_q);
                    }
                }
            }

            // Publish the player's pose for THIS step (snap on teleport so the
            // render thread never interpolates across the respawn).
            if teleported {
                motion.publish_snap(total_steps, pos, quat);
            } else {
                motion.publish(total_steps, pos, quat);
            }
            motion.add_step_work_us(((perf.now() - body_t0) * 1000.0).max(0.0) as u32);

            accumulator -= FIXED_DT_MS;
            substeps += 1;
        }

        if substeps == MAX_SUBSTEPS {
            accumulator = 0.0; // hit the cap — drop the backlog (avoid a spiral)
        }
    }
}

/// Create the Box3D shape for a scene collider on `body` (which carries the
/// world pose; the shape is in the body's local frame). Capsule/cylinder/cone
/// are Y-axis-aligned, matching the scene's local frames.
///
/// SAFETY: `body` must be a live body id and `def` a cookie-valid shape def.
unsafe fn create_shape(
    body: b3::b3BodyId,
    def: &b3::b3ShapeDef,
    shape: &ColliderShapeMsg,
) -> b3::b3ShapeId {
    match *shape {
        ColliderShapeMsg::Cuboid {
            half_extents: [x, y, z],
        } => {
            let hull = b3::b3MakeBoxHull(x, y, z);
            b3::b3CreateHullShape(body, def, &hull.base)
        }
        ColliderShapeMsg::Ball { radius } => b3::b3CreateSphereShape(
            body,
            def,
            &b3::b3Sphere {
                center: v3(0.0, 0.0, 0.0),
                radius,
            },
        ),
        ColliderShapeMsg::Capsule {
            half_height,
            radius,
        } => b3::b3CreateCapsuleShape(
            body,
            def,
            &b3::b3Capsule {
                center1: v3(0.0, -half_height, 0.0),
                center2: v3(0.0, half_height, 0.0),
                radius,
            },
        ),
        // The hull builders return heap hulls; the world interns a copy into its
        // hull database on shape creation, so destroying ours right after is safe
        // (upstream samples do exactly this).
        ColliderShapeMsg::Cylinder {
            half_height,
            radius,
        } => {
            // Spans y ∈ [yOffset, yOffset + height] → center with -half_height.
            let hull = b3::b3CreateCylinder(2.0 * half_height, radius, -half_height, 16);
            let shape = b3::b3CreateHullShape(body, def, hull);
            b3::b3DestroyHull(hull);
            shape
        }
        ColliderShapeMsg::Cone {
            half_height,
            radius,
        } => {
            // b3CreateCone builds a truncated cone spanning y ∈ [0, height]
            // (no offset — base sits at the node origin rather than centered)
            // and asserts both radii > 0, so the apex is 5% of the base:
            // acceptable approximations; the shipped scene has no cones.
            let hull = b3::b3CreateCone(2.0 * half_height, radius, radius * 0.05, 16);
            let shape = b3::b3CreateHullShape(body, def, hull);
            b3::b3DestroyHull(hull);
            shape
        }
    }
}

/// Serialize an [`AudioMsg`] and post it to the main thread (which owns the
/// WebAudio players).
fn post_audio(msg: &AudioMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    if let Ok(v) = serde_wasm_bindgen::to_value(msg) {
        let _ = scope.post_message(&v);
    }
}

/// Serialize a [`PhysicsMsg`](crate::protocol::PhysicsMsg) control request and
/// post it to the main thread.
fn post_physics(msg: &crate::protocol::PhysicsMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    if let Ok(v) = serde_wasm_bindgen::to_value(msg) {
        let _ = scope.post_message(&v);
    }
}
