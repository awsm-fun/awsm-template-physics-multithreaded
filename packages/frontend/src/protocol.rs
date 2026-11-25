//! The typed messages — and shared-memory blocks — that cross the three threads.
//!
//! Only the cold paths are `postMessage` (serialized with `serde` +
//! `serde_wasm_bindgen`). The hot paths are shared memory instead: per-step
//! ball poses + the render→physics frame tick and frame time live in
//! [`BodyMotion`] (physics publishes every step's pose with its simulation
//! time, render interpolates and ticks back), keyboard input lives in
//! [`InputState`] (main writes, physics polls), and the camera lives in
//! [`CameraRing`] (main writes once per main-thread frame, render reads once
//! per render frame). See [`crate::physics_thread`].
//!
//! ```text
//!   main ──(spawn, OffscreenCanvas)──▶ render        [RenderMsg back to main]
//!   render ──(PhysicsInit, via main)─▶ physics
//!   main ──(held keys / jumps)──▶ physics            (InputState, shared memory)
//!   main ──(yaw/pitch/radius per frame)──▶ render    (CameraRing, shared memory)
//!   physics ◀──(frame tick + frame time)── render    (BodyMotion, shared memory)
//!   physics ──(pose + sim time, every step)──▶ render (BodyMotion, NOT a message)
//!   physics ──(AudioMsg)──▶ main                     (gameplay sound cues)
//! ```
//!
//! ## One clock: Fix Your Timestep across two threads
//!
//! The render thread's `requestAnimationFrame` timestamp is the only clock
//! that paces the simulation. At the START of each frame, render publishes
//! that timestamp `T_k` (absolute: `performance.timeOrigin + rAF time`, as
//! wrapping microseconds) and bumps the frame tick. Physics wakes, adds
//! `T_k − T_(k−1)` to its fixed-step accumulator, runs as many whole steps as
//! that pays for, and stamps every step's pose with the time it is due:
//! `T_k − accumulator` (computed from the f64 accumulator, never by adding a
//! rounded step length). Render, in frame `k+1`, draws at display time
//! `D = T_k − STEP` and interpolates between the two published steps around
//! it, which is exactly Fix Your Timestep's `lerp(previous, current,
//! accumulator / STEP)` — one frame later, because the burst runs on its own
//! thread while frame `k` draws. If the burst has not finished in time, the
//! newest step is older than `D`: render holds it (no extrapolation) and
//! counts the frame as a late frame (the Stats panel's physics group).

use core::arch::wasm32::{memory_atomic_notify, memory_atomic_wait32};
use core::sync::atomic::{fence, AtomicU32, Ordering};

use serde::{Deserialize, Serialize};

/// **The simulation's fixed step rate, in hertz — the single knob that sets the
/// physics/render timestep.** The fixed step is `1 / SIM_HZ` seconds; the sim
/// advances this many physics steps per second of real time. Both worker threads
/// read *this* constant (physics integrates at `1/SIM_HZ`; render derives the same
/// value for its interpolation `alpha`), so the two can never disagree.
///
/// ## Why a *fixed* step — and why retuning it is safe
///
/// A rigid-body solver is only stable and deterministic at a **constant** `dt`;
/// variable-`dt` stepping makes restitution / friction / penetration drift frame
/// to frame. So the world always advances in `1/SIM_HZ` chunks and the physics
/// loop simply runs *as many* of them as real time has bought since the last frame
/// (the accumulator in [`crate::physics_thread`]). Because the render thread
/// *paces* that loop — one wake per presented frame (see [`BodyMotion`]) — raising
/// `SIM_HZ` means **more sub-steps per frame**, not a faster or slower game.
///
/// ## Why raising it helps — and what it costs
///
/// * **Latency** — rendering interpolates between the last two published poses, so
///   it shows the world up to one step in the past. Smaller step ⇒ less of it: the
///   interpolation floor is `~1000/SIM_HZ` ms (≈16.7 at 60, ≈4.2 at 240).
/// * **Collision accuracy** — the ball moves a shorter distance per step, so less
///   penetration and less reliance on CCD.
/// * **Cost** — CPU scales linearly (2× the rate ⇒ 2× the `step()` calls). Each
///   step here is ~0.2 ms, so even 240 Hz is a few % of one core; a heavy sim is
///   where the trade-off would start to bite.
///
/// ## What makes this a *one-number* change
///
/// Nothing else is written in "ticks". Every force / threshold in
/// [`crate::physics_thread`] is either a per-*second* physical quantity
/// (velocities, accelerations, gravity, damping — all `dt`-invariant) or is
/// *derived* from `SIM_HZ` (the roll-cue divider, the impact cooldown, the
/// catch-up cap). Change this number and the feel is identical — only the step
/// granularity (hence latency + cost) moves. **If you add a new per-step force,
/// express it per second and multiply by the step `dt`** — don't bake the rate
/// into a constant, or you'll reintroduce the very coupling this removes.
pub const SIM_HZ: f64 = 240.0;

/// One fixed step in microseconds, rounded up: render's display time is
/// `T − STEP_US`. (The exact length is `1e6 / SIM_HZ`; rounding up keeps the
/// display time at or before the newest step a finished burst publishes.)
pub const STEP_US: u32 = (1_000_000.0 / SIM_HZ + 0.999) as u32;

/// An absolute time (`performance.timeOrigin + t_ms`, in milliseconds since
/// the epoch) as wrapping microseconds. Every worker has its own
/// `timeOrigin`, so times that cross threads are absolute; u32 µs wraps every
/// ~71 minutes, so compare them only with [`time_diff_us`].
pub fn abs_time_us(time_origin_ms: f64, t_ms: f64) -> u32 {
    ((time_origin_ms + t_ms) * 1000.0) as u64 as u32
}

/// `a − b` in microseconds for two wrapping [`abs_time_us`] times that are
/// within ±35 minutes of each other.
pub fn time_diff_us(a: u32, b: u32) -> i32 {
    a.wrapping_sub(b) as i32
}

/// The Stats panel's physics breakdown costs something to measure, so the
/// physics thread times its step phases only while this bit is set in
/// [`BodyMotion::detail`]. The render worker sets it while the panel shows the
/// physics group expanded (its stats collector knows the open groups).
pub const DETAIL_PHYSICS: u32 = 1 << 2;

/// The phases of one physics step the Stats panel lists (while its physics
/// group is expanded), in the order they run; see `physics_thread`.
pub const STEP_PHASES: [&str; 4] = [
    "input + ball drops",
    "Box3D step",
    "contacts + sound cues",
    "publish poses",
];

/// Everything the physics thread needs to start driving the scene's ball. The
/// render thread owns the renderer (hence the shared transform arena) *and*
/// allocates the [`BodyMotion`] buffer, so it produces this once the scene is
/// loaded and posts it back to main, which relays it as the physics worker's
/// startup payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsInit {
    /// Every collider derived from the scene's `Collider` nodes, in world space.
    /// Exactly one is `dynamic` (the ball); the rest are static (table + walls).
    pub colliders: Vec<ColliderInit>,
    /// Where the dynamic ball starts (its collider's authored world translation).
    pub spawn: [f32; 3],
    /// Address (in the shared `WebAssembly.Memory`) of the render-owned
    /// [`BodyMotion`] — valid in the physics worker because it attaches to the
    /// same memory. Physics publishes the ball's prev/curr pose here each step.
    pub motion_ptr: f64,
    /// Address of the main-thread-owned [`InputState`] in the same shared memory.
    /// The render thread leaves this 0; main fills it in before spawning physics
    /// (it owns the keyboard, so it allocates the block). The physics worker reads
    /// held keys / jumps straight from it — no per-input `postMessage`.
    pub input_ptr: f64,
    /// Address of the render-owned [`BallMotions`] block in the same shared
    /// memory: the click-to-drop request queue plus one pose slot per dropped
    /// ball. Physics polls the drop requests and publishes each ball's
    /// prev/curr pose; the render thread mints a visual duplicate per new ball
    /// and interpolates each slot into its arena binding.
    pub balls_ptr: f64,
}

/// The most balls a session can drop (clicks past this are ignored). Bounded
/// by the renderer's spatial index more than by Box3D — see BOX3D.md's rstar
/// note: ≥~300 concurrently-moving bodies can trip an upstream rstar panic.
pub const MAX_BALLS: usize = 200;

/// Click-to-drop shared block: a tiny request "queue" (latest click wins
/// within one 240 Hz poll window) plus a pose slot per dropped ball.
///
/// Roles: **main** captures the click and sends it to **render** (which owns
/// the camera) to unproject onto the table; render writes the drop request
/// here. **Physics** (which has no event loop — it blocks on the frame-tick
/// futex) polls the request each step, creates the body, publishes the slot,
/// then bumps `count` (Release) — the render thread mints the visual duplicate
/// when it sees `count` grow, reading a valid pose from the slot.
#[repr(C)]
pub struct BallMotions {
    /// Bumped once per drop request (render-side write).
    drop_seq: AtomicU32,
    /// Requested drop position on the table (f32 bits), valid at `drop_seq`.
    drop_x: AtomicU32,
    drop_z: AtomicU32,
    /// Balls created so far (physics-side write, Release after slot init).
    count: AtomicU32,
    /// On the heap: `MAX_BALLS` pose rings are about half a megabyte, too big
    /// to build on a thread's stack and then move. (It is still in the one
    /// shared wasm memory, so every thread reaches it through `&self`.)
    slots: Box<[PoseRing]>,
}

impl BallMotions {
    pub fn new() -> Self {
        BallMotions {
            drop_seq: AtomicU32::new(0),
            drop_x: AtomicU32::new(0),
            drop_z: AtomicU32::new(0),
            count: AtomicU32::new(0),
            slots: (0..MAX_BALLS)
                .map(|_| PoseRing::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]))
                .collect(),
        }
    }

    /// Render side: request a ball drop at table position `(x, z)`. If two
    /// clicks land inside one physics poll window (~4 ms) the later position
    /// wins and one ball spawns — acceptable for a click cadence.
    pub fn request_drop(&self, x: f32, z: f32) {
        self.drop_x.store(x.to_bits(), Ordering::Relaxed);
        self.drop_z.store(z.to_bits(), Ordering::Relaxed);
        self.drop_seq.fetch_add(1, Ordering::Release);
    }

    /// Physics side: consume any pending drop request. Returns the requested
    /// `(x, z)` when `last_seq` is behind (and advances it to current).
    pub fn poll_drop(&self, last_seq: &mut u32) -> Option<(f32, f32)> {
        let seq = self.drop_seq.load(Ordering::Acquire);
        if seq == *last_seq {
            return None;
        }
        *last_seq = seq;
        Some((
            f32::from_bits(self.drop_x.load(Ordering::Relaxed)),
            f32::from_bits(self.drop_z.load(Ordering::Relaxed)),
        ))
    }

    /// Balls created so far (Acquire — pairs with physics' Release so a newly
    /// visible slot always holds a valid pose).
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }

    /// Physics side: publish that slot `count-1` is initialized.
    pub fn set_count(&self, count: usize) {
        self.count.store(count as u32, Ordering::Release);
    }

    pub fn slot(&self, index: usize) -> &PoseRing {
        &self.slots[index]
    }
}

impl Default for BallMotions {
    fn default() -> Self {
        Self::new()
    }
}

/// Pose-ring depth, in steps (~133 ms of history at 240 Hz). Render only
/// ever reads the two steps around its display time, at most one burst behind
/// the newest; the ring is deep enough that a whole burst (up to
/// `MAX_SUBSTEPS`, see `physics_thread`) can be written while render reads
/// the pair the previous burst left — the seqlock and step tag catch
/// anything else.
pub const POSE_RING: usize = 32;

/// One step's published pose, as read back: where the body was, and the
/// absolute time ([`abs_time_us`]) that step is due.
#[derive(Debug, Clone, Copy)]
pub struct StepPose {
    pub step: u32,
    pub time_us: u32,
    pub pos: [f32; 3],
    pub quat: [f32; 4],
}

/// Where a display time falls in a body's published steps (see
/// [`PoseRing::bracket`]).
#[derive(Debug, Clone, Copy)]
pub enum Bracket {
    /// `older.time_us <= D < newer.time_us`: interpolate.
    Between { older: StepPose, newer: StepPose },
    /// The newest published step is before `D`: physics has not finished
    /// the burst this frame shows. Hold it (no extrapolation).
    Late(StepPose),
    /// Nothing in the ring is old enough (a body younger than `D`, or the
    /// startup seed): hold the oldest available step.
    Early(StepPose),
}

/// One fixed step's published pose, seqlock'd, tagged with its absolute step
/// index (so a reader can tell whether the slot still holds the step it
/// wants — the ring overwrites in place) and the time the step is due.
#[repr(C)]
pub struct PoseSlot {
    version: AtomicU32,
    step: AtomicU32,
    /// When this step is due, as [`abs_time_us`]: `T − accumulator` for the
    /// frame time `T` whose burst produced it.
    time_us: AtomicU32,
    /// `pos[3], quat[4]` as f32 bit patterns.
    fields: [AtomicU32; 7],
}

impl PoseSlot {
    fn new() -> Self {
        PoseSlot {
            version: AtomicU32::new(0),
            step: AtomicU32::new(u32::MAX),
            time_us: AtomicU32::new(0),
            fields: core::array::from_fn(|_| AtomicU32::new(0)),
        }
    }

    fn write(&self, step: u32, time_us: u32, pos: [f32; 3], quat: [f32; 4]) {
        let v = self.version.load(Ordering::Relaxed);
        self.version.store(v.wrapping_add(1), Ordering::Relaxed); // odd: in flight
        fence(Ordering::Release);
        self.step.store(step, Ordering::Relaxed);
        self.time_us.store(time_us, Ordering::Relaxed);
        let vals = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]];
        for (slot, val) in self.fields.iter().zip(vals) {
            slot.store(val.to_bits(), Ordering::Relaxed);
        }
        fence(Ordering::Release);
        self.version.store(v.wrapping_add(2), Ordering::Relaxed); // even: done
    }

    /// Untorn read, `None` if the slot holds a different step (overwritten) or
    /// a write is racing.
    fn read(&self, expect_step: u32) -> Option<StepPose> {
        for _ in 0..64 {
            let v1 = self.version.load(Ordering::Relaxed);
            if v1 & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            fence(Ordering::Acquire);
            let step = self.step.load(Ordering::Relaxed);
            let time_us = self.time_us.load(Ordering::Relaxed);
            let mut f = [0.0_f32; 7];
            for (i, slot) in self.fields.iter().enumerate() {
                f[i] = f32::from_bits(slot.load(Ordering::Relaxed));
            }
            fence(Ordering::Acquire);
            if self.version.load(Ordering::Relaxed) == v1 {
                if step != expect_step {
                    return None;
                }
                return Some(StepPose {
                    step,
                    time_us,
                    pos: [f[0], f[1], f[2]],
                    quat: [f[3], f[4], f[5], f[6]],
                });
            }
        }
        None
    }
}

/// A ring of the last [`POSE_RING`] published step poses for one body, plus
/// the latest published step index. Physics (sole writer) publishes every
/// fixed step with its due time; render finds the two steps around its
/// display time ([`Self::bracket`]) and interpolates between them.
#[repr(C)]
pub struct PoseRing {
    latest: AtomicU32,
    slots: [PoseSlot; POSE_RING],
}

impl PoseRing {
    /// Ring seeded with the given resting pose at step 0.
    pub fn new(pos: [f32; 3], quat: [f32; 4]) -> Self {
        let ring = PoseRing {
            latest: AtomicU32::new(0),
            slots: core::array::from_fn(|_| PoseSlot::new()),
        };
        // Step 0 is the startup seed: it has no real time, and `bracket`
        // treats it as such.
        ring.slots[0].write(0, 0, pos, quat);
        ring
    }

    /// Writer side (physics): publish the pose at absolute step `step`, due
    /// at `time_us`.
    pub fn publish(&self, step: u32, time_us: u32, pos: [f32; 3], quat: [f32; 4]) {
        self.slots[step as usize % POSE_RING].write(step, time_us, pos, quat);
        self.latest.store(step, Ordering::Release);
    }

    /// Writer side: publish a **teleport** — the pose is written for `step`
    /// AND `step - 1` (due at `prev_time_us`), so a reader interpolating
    /// across the boundary snaps instead of sweeping the body through the
    /// world.
    pub fn publish_snap(
        &self,
        step: u32,
        time_us: u32,
        prev_time_us: u32,
        pos: [f32; 3],
        quat: [f32; 4],
    ) {
        self.slots[step.wrapping_sub(1) as usize % POSE_RING].write(
            step.wrapping_sub(1),
            prev_time_us,
            pos,
            quat,
        );
        self.publish(step, time_us, pos, quat);
    }

    /// Latest published absolute step.
    pub fn latest_step(&self) -> u32 {
        self.latest.load(Ordering::Acquire)
    }

    /// The pose published for `step`, if the ring still holds it.
    pub fn read_step(&self, step: u32) -> Option<StepPose> {
        self.slots[step as usize % POSE_RING].read(step)
    }

    /// The latest pose (always available once seeded).
    pub fn read_latest(&self) -> StepPose {
        loop {
            let step = self.latest_step();
            if let Some(pose) = self.read_step(step) {
                return pose;
            }
            // Racing the writer onto the next step — retry with the new latest.
            core::hint::spin_loop();
        }
    }

    /// The published steps around display time `display_us`: walks back from
    /// the newest step to the first one due at or before it. Usually one or
    /// two steps back — the display time trails the newest step by less than
    /// one step plus one burst.
    pub fn bracket(&self, display_us: u32) -> Bracket {
        let newest = self.read_latest();
        if newest.step == 0 {
            return Bracket::Early(newest); // only the startup seed so far
        }
        if time_diff_us(newest.time_us, display_us) < 0 {
            return Bracket::Late(newest);
        }
        let mut newer = newest;
        for _ in 1..POSE_RING {
            let step = newer.step.wrapping_sub(1);
            if step == 0 {
                break; // the seed has no real time
            }
            match self.read_step(step) {
                Some(older) if time_diff_us(older.time_us, display_us) <= 0 => {
                    return Bracket::Between { older, newer };
                }
                Some(older) => newer = older,
                // Overwritten, or before this body existed.
                None => break,
            }
        }
        Bracket::Early(newer)
    }
}

/// The player ball's motion channel in the shared `WebAssembly.Memory`: a
/// [`PoseRing`] of recent step poses (physics publishes every fixed step with
/// its due time; the render thread interpolates between the two steps around
/// its display time), plus the frame tick and frame time that pace the sim.
///
/// Pose is stored as position + quaternion (not a baked matrix) precisely so the
/// reader can lerp position and *slerp* rotation correctly.
///
/// ## Frame tick + frame time: render's clock paces the sim
///
/// At the START of each frame the render thread stores the frame's rAF
/// timestamp and bumps `frame_tick` ([`begin_frame`]); the physics thread
/// blocks on the tick ([`wait_frame`]) and feeds its accumulator from the
/// difference between successive frame times — not from its own clock read
/// at wake-up, which carries the render thread's CPU time and the futex wake
/// latency and would turn a steady 4 steps per 60 Hz frame into 3/4/5. With
/// one clock the step count per frame only changes when the frame time does
/// (see the module docs). Bumping at the start of the frame gives physics the
/// whole frame period for its burst. The wait/notify is a futex on the tick
/// word (`memory.atomic.wait32` / `notify`).
///
/// [`begin_frame`]: BodyMotion::begin_frame
/// [`wait_frame`]: BodyMotion::wait_frame
#[repr(C)]
pub struct BodyMotion {
    /// Pose history the render thread samples (see [`PoseRing`]).
    ring: PoseRing,
    /// Bumped by render at the start of every frame; physics blocks on it
    /// (see the type docs).
    frame_tick: AtomicU32,
    /// The latest frame's rAF timestamp as [`abs_time_us`], stored before the
    /// tick bump. Physics steps to it.
    frame_time_us: AtomicU32,
    /// 0 until render has stored its first `frame_time_us`, then 1 forever.
    /// Until then physics runs on its own clock.
    frame_time_valid: AtomicU32,
    /// Accumulated physics-thread CPU time, in MICROSECONDS (wrapping): the
    /// full cost of each fixed step — input poll, `b3World_Step`, contact/hit
    /// classification, pose publish — summed across steps. The render
    /// worker's stats divide its delta by the step-count delta for avg
    /// ms/step.
    step_work_us: AtomicU32,
    /// Written by render: [`DETAIL_PHYSICS`] while the Stats panel's physics
    /// group is expanded. Physics reads it once per frame.
    detail: AtomicU32,
    /// Written by physics once, if it runs a task pool: the
    /// [`TaskPool`](crate::physics_tasks::TaskPool)'s address and its worker
    /// count, for the render worker's stats (the solver's work split).
    pool_addr: AtomicU32,
    pool_workers: AtomicU32,
    /// Physics side, only while [`DETAIL_PHYSICS`] is set: running totals of
    /// each [`STEP_PHASES`] phase in NANOSECONDS (wrapping), and the steps
    /// they cover. Republished a few times a second, not per step, under a
    /// seqlock (odd `step_phase_version` = being written) so a reader never
    /// pairs one publish's totals with another's step count.
    step_phase_version: AtomicU32,
    step_phase_ns: [AtomicU32; STEP_PHASES.len()],
    step_phase_steps: AtomicU32,
}

impl BodyMotion {
    /// Ring seeded with the given resting pose.
    pub fn new(pos: [f32; 3], quat: [f32; 4]) -> Self {
        BodyMotion {
            ring: PoseRing::new(pos, quat),
            frame_tick: AtomicU32::new(0),
            frame_time_us: AtomicU32::new(0),
            frame_time_valid: AtomicU32::new(0),
            step_work_us: AtomicU32::new(0),
            detail: AtomicU32::new(0),
            pool_addr: AtomicU32::new(0),
            pool_workers: AtomicU32::new(0),
            step_phase_version: AtomicU32::new(0),
            step_phase_ns: [const { AtomicU32::new(0) }; STEP_PHASES.len()],
            step_phase_steps: AtomicU32::new(0),
        }
    }

    /// Render side: which Stats breakdowns to measure (`DETAIL_*` bits).
    pub fn set_detail(&self, bits: u32) {
        self.detail.store(bits, Ordering::Relaxed);
    }

    /// Physics side: the `DETAIL_*` bits render last set.
    pub fn detail(&self) -> u32 {
        self.detail.load(Ordering::Relaxed)
    }

    /// Physics side: the task pool it runs (its address in the shared memory
    /// and its worker count), published once for the render worker's stats.
    pub fn set_pool(&self, addr: usize, workers: u32) {
        self.pool_workers.store(workers, Ordering::Relaxed);
        self.pool_addr.store(addr as u32, Ordering::Release);
    }

    /// Render side: the task pool's address and worker count, once physics
    /// has one.
    pub fn pool(&self) -> Option<(usize, u32)> {
        let addr = self.pool_addr.load(Ordering::Acquire);
        (addr != 0).then(|| (addr as usize, self.pool_workers.load(Ordering::Relaxed)))
    }

    /// Physics side: publish the running per-phase totals (ms) and the steps
    /// they cover. Stored as wrapping ns; readers diff two reads.
    pub fn set_step_phases(&self, totals_ms: &[f64; STEP_PHASES.len()], steps: u64) {
        let v = self.step_phase_version.load(Ordering::Relaxed);
        self.step_phase_version
            .store(v.wrapping_add(1), Ordering::Relaxed);
        fence(Ordering::Release);
        for (slot, ms) in self.step_phase_ns.iter().zip(totals_ms) {
            slot.store((ms * 1e6) as u64 as u32, Ordering::Relaxed);
        }
        self.step_phase_steps.store(steps as u32, Ordering::Relaxed);
        fence(Ordering::Release);
        self.step_phase_version
            .store(v.wrapping_add(2), Ordering::Relaxed);
    }

    /// Render (stats) side: `(per-phase ns, steps)`, all wrapping; `None` if a publish
    /// kept racing the read (the next sample gets it).
    pub fn step_phases(&self) -> Option<([u32; STEP_PHASES.len()], u32)> {
        for _ in 0..64 {
            let v1 = self.step_phase_version.load(Ordering::Relaxed);
            if v1 & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            fence(Ordering::Acquire);
            let ns = core::array::from_fn(|i| self.step_phase_ns[i].load(Ordering::Relaxed));
            let steps = self.step_phase_steps.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            if self.step_phase_version.load(Ordering::Relaxed) == v1 {
                return Some((ns, steps));
            }
        }
        None
    }

    /// Physics side: add one fixed step's CPU work time (µs) to the total.
    pub fn add_step_work_us(&self, us: u32) {
        self.step_work_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Accumulated physics-thread work time (µs, wrapping) — diff two reads
    /// and divide by the step-count delta for avg time per step.
    pub fn step_work_us(&self) -> u32 {
        self.step_work_us.load(Ordering::Relaxed)
    }

    /// Render side, at the start of a frame: publish the frame's time
    /// ([`abs_time_us`] of its rAF timestamp), then bump the tick, waking
    /// physics to simulate up to it. Cheap when physics is already running
    /// (the `notify` is a no-op with no waiter; physics compares tick
    /// *values*, so it can't miss a bump).
    pub fn begin_frame(&self, time_us: u32) {
        self.frame_time_us.store(time_us, Ordering::Relaxed);
        self.frame_time_valid.store(1, Ordering::Relaxed);
        self.frame_tick.fetch_add(1, Ordering::Release);
        // SAFETY: `frame_tick` lives in the shared memory both threads attach to.
        unsafe {
            memory_atomic_notify(self.frame_tick.as_ptr() as *mut i32, 1);
        }
    }

    /// Current frame-tick value (physics seeds its last-seen with this).
    pub fn frame_tick(&self) -> u32 {
        self.frame_tick.load(Ordering::Relaxed)
    }

    /// Physics side: the latest frame time render published, or `None`
    /// before its first frame. Read after [`Self::wait_frame`] (whose Acquire
    /// pairs with the tick's Release), so it is at least that frame's time.
    pub fn frame_time_us(&self) -> Option<u32> {
        (self.frame_time_valid.load(Ordering::Relaxed) != 0)
            .then(|| self.frame_time_us.load(Ordering::Relaxed))
    }

    /// Physics side: block until the frame-tick advances past `last`, returning the
    /// new value. Worker scope only — `memory.atomic.wait` traps on the main
    /// thread. Spurious wakeups + already-advanced ticks are both handled by the
    /// value re-check, so a presented frame is never missed.
    pub fn wait_frame(&self, last: u32) -> u32 {
        loop {
            let cur = self.frame_tick.load(Ordering::Acquire);
            if cur != last {
                return cur;
            }
            // Block until `bump_frame` notifies (or a spurious wake): waits only
            // while the word still equals `last`. -1 = no timeout.
            // SAFETY: shared-memory address, called on the physics worker.
            unsafe {
                memory_atomic_wait32(self.frame_tick.as_ptr() as *mut i32, last as i32, -1);
            }
        }
    }

    /// Latest published absolute step.
    pub fn latest_step(&self) -> u32 {
        self.ring.latest_step()
    }

    /// The underlying pose ring: physics publishes into it, render brackets
    /// its display time in it (see [`PoseRing::bracket`]).
    pub fn ring(&self) -> &PoseRing {
        &self.ring
    }
}

/// Held-movement bits packed into [`InputState::held`].
pub const HELD_FORWARD: u32 = 1 << 0;
pub const HELD_BACK: u32 = 1 << 1;
pub const HELD_LEFT: u32 = 1 << 2;
pub const HELD_RIGHT: u32 = 1 << 3;

/// Lock-free input shared from main → physics, living in the shared
/// `WebAssembly.Memory` (like [`BodyMotion`], but the other direction). Main owns
/// the keyboard and pointer and is the sole writer; the physics worker is the
/// sole reader and samples it once per fixed step. This replaces per-keystroke
/// `postMessage`, which matters because the physics worker now runs a *blocking*
/// paced loop (`memory.atomic.wait`) with no event loop to deliver messages to —
/// see [`crate::physics_thread`]. Plain relaxed atomics: the fields are
/// independent and nothing else is ordered against them.
#[repr(C)]
pub struct InputState {
    /// Bitset of currently-held movement keys (`HELD_*`).
    held: AtomicU32,
    /// Movement keys pressed since physics last looked (`HELD_*`): set with
    /// `held` on keydown, taken (cleared) by physics once per step and OR'd
    /// into `held`. A tap whose keydown AND keyup both land between two steps
    /// clears `held` before physics ever sees it; this latch still gives it
    /// one step, so every press is simulated.
    pressed: AtomicU32,
    /// Bumped once per discrete jump press. Physics edge-detects a jump by
    /// diffing this against the value it last saw (so a press = exactly one hop,
    /// even though physics polls rather than receives an event).
    jump_seq: AtomicU32,
    /// Camera yaw in radians (`f32::to_bits`) of main's [`OrbitCamera`], set on
    /// every orbit drag. Physics rotates the held-key roll direction by this
    /// so W/A/S/D stay camera-relative at any orbit angle. (It is the newest
    /// yaw, about a frame ahead of the one on screen — a few degrees at most
    /// mid-drag, and the same as before the camera moved to [`CameraRing`].)
    camera_yaw: AtomicU32,
    /// Touch fling: bumped once per swipe gesture (edge-detected like
    /// `jump_seq`). The velocity is valid at the seq bump — values are stored
    /// first, then the seq (Release), pairing with the Acquire in
    /// [`poll_fling`](Self::poll_fling).
    fling_seq: AtomicU32,
    /// Swipe release velocity (m/s, `f32::to_bits`) in the CAMERA frame —
    /// x right, −z away from the camera, the same convention as the held-key
    /// direction. Physics rotates it by `camera_yaw` exactly like W/A/S/D.
    fling_x: AtomicU32,
    fling_z: AtomicU32,
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

impl InputState {
    pub const fn new() -> Self {
        InputState {
            held: AtomicU32::new(0),
            pressed: AtomicU32::new(0),
            jump_seq: AtomicU32::new(0),
            camera_yaw: AtomicU32::new(0), // 0u32 == 0.0f32.to_bits()
            fling_seq: AtomicU32::new(0),
            fling_x: AtomicU32::new(0),
            fling_z: AtomicU32::new(0),
        }
    }

    /// Main side: set/clear one held-key bit.
    pub fn set_held(&self, mask: u32, down: bool) {
        if down {
            self.held.fetch_or(mask, Ordering::Relaxed);
            self.pressed.fetch_or(mask, Ordering::Relaxed);
        } else {
            self.held.fetch_and(!mask, Ordering::Relaxed);
        }
    }

    /// Main side: register a discrete jump press.
    pub fn bump_jump(&self) {
        self.jump_seq.fetch_add(1, Ordering::Relaxed);
    }

    /// Physics side, once per step: the keys to act on this step — those held
    /// now plus any pressed since the previous step (see `pressed`).
    pub fn take_held(&self) -> u32 {
        self.held.load(Ordering::Relaxed) | self.pressed.swap(0, Ordering::Relaxed)
    }

    /// Physics side: the jump counter (diff against your last-seen to edge-detect).
    pub fn jump_seq(&self) -> u32 {
        self.jump_seq.load(Ordering::Relaxed)
    }

    /// Main side: publish a touch-swipe fling with the given CAMERA-frame
    /// velocity (m/s — x right, −z away from the camera). Values first, then
    /// the seq bump (Release), so [`poll_fling`](Self::poll_fling) never reads
    /// a stale pair.
    pub fn bump_fling(&self, vx: f32, vz: f32) {
        self.fling_x.store(vx.to_bits(), Ordering::Relaxed);
        self.fling_z.store(vz.to_bits(), Ordering::Relaxed);
        self.fling_seq.fetch_add(1, Ordering::Release);
    }

    /// Physics side: consume a pending fling. Returns the camera-frame swipe
    /// velocity when `last_seq` is behind (and advances it to current). Like
    /// [`BallMotions::poll_drop`], if two swipes land inside one poll window
    /// the later one wins — fine at gesture cadence.
    pub fn poll_fling(&self, last_seq: &mut u32) -> Option<(f32, f32)> {
        let seq = self.fling_seq.load(Ordering::Acquire);
        if seq == *last_seq {
            return None;
        }
        *last_seq = seq;
        Some((
            f32::from_bits(self.fling_x.load(Ordering::Relaxed)),
            f32::from_bits(self.fling_z.load(Ordering::Relaxed)),
        ))
    }

    /// Physics side: the fling counter's current value (seed your last-seen).
    pub fn fling_seq(&self) -> u32 {
        self.fling_seq.load(Ordering::Relaxed)
    }

    /// Main side: publish the camera yaw (radians) after an orbit drag.
    pub fn set_camera_yaw(&self, yaw: f32) {
        self.camera_yaw.store(yaw.to_bits(), Ordering::Relaxed);
    }

    /// Physics side: the camera yaw (radians) to rotate roll input by.
    pub fn camera_yaw(&self) -> f32 {
        f32::from_bits(self.camera_yaw.load(Ordering::Relaxed))
    }
}

/// One scene-derived collider, ready to hand to the physics engine (Box3D). The shape is in local
/// space; `translation`/`rotation` place it in the world. Scale is intentionally
/// NOT carried — the collider has no scale (its placement is a rotation +
/// translation isometry); the fit lives entirely in the shape extents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColliderInit {
    pub shape: ColliderShapeMsg,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    /// `true` for the one dynamic body (the ball); `false` for static geometry.
    pub dynamic: bool,
    /// Gameplay role, used to classify collision sounds — see `ROLE_*`.
    pub role: u8,
}

/// Collider shapes the physics thread knows how to build (mirrors
/// `awsm_renderer_scene::ColliderShape`; ellipsoid is dropped — unused here).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "kebab-case")]
pub enum ColliderShapeMsg {
    Cuboid { half_extents: [f32; 3] },
    Ball { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
    Cylinder { half_height: f32, radius: f32 },
    Cone { half_height: f32, radius: f32 },
}

/// Collider role tags, shared between the render-thread builder and the
/// physics-thread collision classifier.
pub const ROLE_FLOOR: u8 = 0; // the tabletop — ball landings thud here
pub const ROLE_WALL: u8 = 1; // the rails — ball knocks against these
pub const ROLE_BALL: u8 = 2; // the dynamic ball itself

/// Physics thread → main thread: control requests. Separate from [`AudioMsg`]
/// so the audio dispatch stays a plain cue stream.
///
/// `SpawnTaskWorkers` exists because the physics worker itself cannot reliably
/// spawn sub-workers: it blocks on the frame-tick futex for its whole life, so
/// it never services its event loop again — and a nested worker's startup (and
/// any error it posts back) needs a live parent loop. Main has one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phys", rename_all = "kebab-case")]
pub enum PhysicsMsg {
    /// Spawn `count` task-pool workers (role `physics-task`) attached to the
    /// shared [`TaskPool`](crate::physics_tasks::TaskPool) at address `pool`.
    SpawnTaskWorkers { pool: f64, count: u32 },
}

/// Physics thread → main thread: gameplay-driven audio cues. The physics thread
/// owns the ball's motion + contacts, so it decides when sounds fire and how
/// loud; main relays each to the (main-thread-owned) WebAudio players. Cold-ish:
/// `Roll` is throttled, the impacts are event-driven.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "audio", rename_all = "kebab-case")]
pub enum AudioMsg {
    /// Continuous rolling state: normalized speed (0..1) + ball world position.
    Roll { speed: f32, x: f32, y: f32, z: f32 },
    /// The ball struck a wall — `intensity` (0..1) from the impact speed.
    WallHit {
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    },
    /// The ball landed on the table — `intensity` (0..1) from the drop speed.
    Land {
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    },
    /// Two balls collided — `intensity` (0..1) from the approach speed. Voiced
    /// by the steel-sphere clack (`sfx_ball_clack`, a modal-synthesis worklet
    /// whose `intensity` param this drives); falls back to the wall knock when
    /// the loaded audio export predates the clack.
    BallClack {
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    },
}

/// Render thread → main thread. (The Settings' and the Stats panel's
/// messages — `aa-compile-*`, `stats` — are awsm-renderer-kit's own, posted
/// alongside these; main hands them to the panels first.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "msg", rename_all = "kebab-case")]
pub enum RenderMsg {
    /// A human-readable load-progress line for the loading screen (device
    /// building, scene fetch, loader phases, GPU commit stats, …).
    Progress { message: String },
    /// Scene loaded; here's what physics needs. Main relays this to a freshly
    /// spawned physics worker.
    PhysicsInit(PhysicsInit),
    /// The first few frames have rendered — the sphere is on screen.
    Ready,
    /// Something failed in the render worker.
    Error { message: String },
}

/// Main thread → render thread: the user clicked the canvas at the given NDC
/// (x right, y up, both −1..1). Render (which owns the camera) unprojects it
/// onto the tabletop and files a drop request in [`BallMotions`] — main can't
/// tell physics directly (physics has no event loop), and only render knows
/// the camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "drop", rename_all = "kebab-case")]
pub enum DropMsg {
    Ball { ndc_x: f32, ndc_y: f32 },
}

/// Radians of camera yaw (and pitch) per CSS pixel of drag.
pub const CAMERA_ORBIT_SENSITIVITY: f32 = 0.005;

/// A minimal mouse-driven orbit camera (after the renderer's `model-tests`
/// `OrbitCamera`): it circles a fixed `look_at` point at spherical
/// `(yaw, pitch, radius)`. Right-drag to orbit, wheel to dolly (no pan — the
/// table can never leave the frame).
///
/// The MAIN thread owns the live one: it applies every drag and wheel event
/// (with the clamps below), fans the yaw out to physics and audio, and once
/// per main-thread frame writes `(yaw, pitch, radius)` into the
/// [`CameraRing`]. The render thread keeps a copy it overwrites from the ring
/// each frame, and only uses it for drawing and click unprojection.
#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    pub look_at: glam::Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl OrbitCamera {
    /// Just under 90° so the camera never flips over the pole.
    pub const PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    /// Floor keeps the eye above the table rim — never under the felt, and the
    /// playfield stays readable even at the shallowest angle.
    pub const PITCH_MIN: f32 = 0.15;
    pub const MIN_RADIUS: f32 = 2.0;
    pub const MAX_RADIUS: f32 = 25.0;

    /// The opening view, the same on both threads: head-on from +Z, tilted
    /// ~26° down, looking just above the table.
    pub fn start() -> Self {
        Self {
            look_at: glam::Vec3::new(0.0, 0.4, 0.0),
            radius: 9.0,
            yaw: 0.0,
            pitch: 0.46,
        }
    }

    /// Apply a drag delta (CSS pixels): horizontal → yaw, vertical → pitch.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * CAMERA_ORBIT_SENSITIVITY;
        self.pitch =
            (self.pitch - dy * CAMERA_ORBIT_SENSITIVITY).clamp(Self::PITCH_MIN, Self::PITCH_MAX);
    }

    /// Apply a wheel delta — positive `dy` (scroll down) dollies out.
    pub fn zoom(&mut self, dy: f32) {
        self.radius = (self.radius * (1.0 + dy * 0.001)).clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
    }

    /// Camera world position from spherical coords (yaw 0 = looking from +Z).
    pub fn eye(&self) -> glam::Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.look_at
            + glam::Vec3::new(
                self.radius * cp * sy,
                self.radius * sp,
                self.radius * cp * cy,
            )
    }
}

/// Slots in the [`CameraRing`]: render reads a slot about one main-thread
/// frame old, so a few frames of history is plenty.
pub const CAMERA_RING: usize = 4;

/// One main-thread frame's camera, seqlock'd (odd `seq` = being written).
#[repr(C)]
struct CameraSlot {
    seq: AtomicU32,
    /// The main-thread rAF timestamp it was written in, as [`abs_time_us`].
    time_us: AtomicU32,
    yaw: AtomicU32,
    pitch: AtomicU32,
    radius: AtomicU32,
}

/// The camera, main → render, once per main-thread frame (shared memory, so
/// no message queue sits between pointer input and the render loop).
///
/// Why a ring of timestamped frames instead of "the latest value": pointer
/// events reach the main thread once per frame, aligned to ITS
/// `requestAnimationFrame`, while the render worker runs its own
/// `requestAnimationFrame`. Reading "the latest" races the two: a render frame
/// that starts just before main writes sees nothing new, and the next sees two
/// frames' worth — the orbit judders 0/2. Instead render takes the newest
/// slot written at least half a frame before its own frame time, which is
/// main's PREVIOUS frame: exactly one frame of input per drawn frame, one
/// frame of latency, no race.
#[repr(C)]
pub struct CameraRing {
    /// Slots written so far (wrapping); the newest is `(writes − 1) % CAMERA_RING`.
    writes: AtomicU32,
    slots: [CameraSlot; CAMERA_RING],
}

impl Default for CameraRing {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraRing {
    pub fn new() -> Self {
        CameraRing {
            writes: AtomicU32::new(0),
            slots: core::array::from_fn(|_| CameraSlot {
                seq: AtomicU32::new(0),
                time_us: AtomicU32::new(0),
                yaw: AtomicU32::new(0),
                pitch: AtomicU32::new(0),
                radius: AtomicU32::new(0),
            }),
        }
    }

    /// Main side (the sole writer), once per main-thread frame in which the
    /// camera changed: overwrite the oldest slot.
    pub fn write(&self, time_us: u32, camera: &OrbitCamera) {
        let n = self.writes.load(Ordering::Relaxed);
        let slot = &self.slots[n as usize % CAMERA_RING];
        let v = slot.seq.load(Ordering::Relaxed);
        slot.seq.store(v.wrapping_add(1), Ordering::Relaxed); // odd: in flight
        fence(Ordering::Release);
        slot.time_us.store(time_us, Ordering::Relaxed);
        slot.yaw.store(camera.yaw.to_bits(), Ordering::Relaxed);
        slot.pitch.store(camera.pitch.to_bits(), Ordering::Relaxed);
        slot.radius
            .store(camera.radius.to_bits(), Ordering::Relaxed);
        fence(Ordering::Release);
        slot.seq.store(v.wrapping_add(2), Ordering::Relaxed); // even: done
        self.writes.store(n.wrapping_add(1), Ordering::Release);
    }

    /// Render side: `(yaw, pitch, radius)` from the newest slot written at or
    /// before `cutoff_us`, skipping a slot being rewritten. `None` before the
    /// first write, or when no slot qualifies (keep the camera you have).
    pub fn read_at(&self, cutoff_us: u32) -> Option<(f32, f32, f32)> {
        let n = self.writes.load(Ordering::Acquire);
        for back in 1..=(n.min(CAMERA_RING as u32)) {
            let slot = &self.slots[n.wrapping_sub(back) as usize % CAMERA_RING];
            let v1 = slot.seq.load(Ordering::Relaxed);
            if v1 & 1 != 0 {
                continue; // being rewritten (main has lapped us) — try older
            }
            fence(Ordering::Acquire);
            let time_us = slot.time_us.load(Ordering::Relaxed);
            let yaw = f32::from_bits(slot.yaw.load(Ordering::Relaxed));
            let pitch = f32::from_bits(slot.pitch.load(Ordering::Relaxed));
            let radius = f32::from_bits(slot.radius.load(Ordering::Relaxed));
            fence(Ordering::Acquire);
            if slot.seq.load(Ordering::Relaxed) != v1 {
                continue; // torn — try older
            }
            if time_diff_us(time_us, cutoff_us) <= 0 {
                return Some((yaw, pitch, radius));
            }
        }
        None
    }
}

// Main → physics input is no longer a message: it's the shared-memory
// [`InputState`] above (main writes, physics polls). See [`crate::physics_thread`].
