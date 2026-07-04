//! The typed messages — and shared-memory blocks — that cross the three threads.
//!
//! Only the cold paths are `postMessage` (serialized with `serde` +
//! `serde_wasm_bindgen`). The two *hot* paths are shared memory instead: per-frame
//! ball motion + the render→physics frame-tick live in [`BodyMotion`] (physics
//! publishes prev/curr poses, render reads/interpolates and ticks back), and
//! keyboard input lives in [`InputState`] (main writes, physics polls). See
//! [`crate::physics_thread`].
//!
//! ```text
//!   main ──(spawn, OffscreenCanvas)──▶ render        [RenderMsg back to main]
//!   render ──(PhysicsInit, via main)─▶ physics
//!   main ──(held keys / jumps)──▶ physics            (InputState, shared memory)
//!   physics ◀──(frame-tick)── render                 (BodyMotion, shared memory)
//!   physics ──(prev/curr pose, every step)──▶ render (BodyMotion, NOT a message)
//!   physics ──(AudioMsg)──▶ main                     (gameplay sound cues)
//! ```

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
    slots: [PoseRing; MAX_BALLS],
}

impl BallMotions {
    pub fn new() -> Self {
        BallMotions {
            drop_seq: AtomicU32::new(0),
            drop_x: AtomicU32::new(0),
            drop_z: AtomicU32::new(0),
            count: AtomicU32::new(0),
            slots: core::array::from_fn(|_| PoseRing::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])),
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

/// Pose-ring depth, in steps (~133 ms of history at 240 Hz). The ring is a
/// **jitter buffer**: the render thread samples it at its own display time, so
/// any wobble or burstiness in when physics wakes and publishes — up to this
/// many steps — is absorbed by *data*, not clamped by an interpolation window.
/// Measured wobble is ±1–2 steps on a loaded desktop but 6–12 steps on
/// phones (thread contention delays the physics wake, then the accumulator
/// catches up in a burst), and the render thread's display lag adapts to
/// that envelope — so the ring must hold the worst adaptive lag plus margin;
/// only genuine hiccups (tab hidden, physics stall) fall off the end.
pub const POSE_RING: usize = 32;

/// One fixed step's published pose, seqlock'd, tagged with its absolute step
/// index so a reader can tell whether the slot still holds the step it wants
/// (the ring overwrites in place).
#[repr(C)]
pub struct PoseSlot {
    version: AtomicU32,
    step: AtomicU32,
    /// `pos[3], quat[4]` as f32 bit patterns.
    fields: [AtomicU32; 7],
}

impl PoseSlot {
    fn new() -> Self {
        PoseSlot {
            version: AtomicU32::new(0),
            step: AtomicU32::new(u32::MAX),
            fields: core::array::from_fn(|_| AtomicU32::new(0)),
        }
    }

    fn write(&self, step: u32, pos: [f32; 3], quat: [f32; 4]) {
        let v = self.version.load(Ordering::Relaxed);
        self.version.store(v.wrapping_add(1), Ordering::Relaxed); // odd: in flight
        fence(Ordering::Release);
        self.step.store(step, Ordering::Relaxed);
        let vals = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]];
        for (slot, val) in self.fields.iter().zip(vals) {
            slot.store(val.to_bits(), Ordering::Relaxed);
        }
        fence(Ordering::Release);
        self.version.store(v.wrapping_add(2), Ordering::Relaxed); // even: done
    }

    /// Untorn read, `None` if the slot holds a different step (overwritten) or
    /// a write is racing.
    fn read(&self, expect_step: u32) -> Option<([f32; 3], [f32; 4])> {
        for _ in 0..64 {
            let v1 = self.version.load(Ordering::Relaxed);
            if v1 & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            fence(Ordering::Acquire);
            let step = self.step.load(Ordering::Relaxed);
            let mut f = [0.0_f32; 7];
            for (i, slot) in self.fields.iter().enumerate() {
                f[i] = f32::from_bits(slot.load(Ordering::Relaxed));
            }
            fence(Ordering::Acquire);
            if self.version.load(Ordering::Relaxed) == v1 {
                if step != expect_step {
                    return None;
                }
                return Some(([f[0], f[1], f[2]], [f[3], f[4], f[5], f[6]]));
            }
        }
        None
    }
}

/// A ring of the last [`POSE_RING`] published step poses for one body, plus
/// the latest published step index. Physics (sole writer) publishes every
/// fixed step; render samples the ring at a step-indexed display time (see
/// `render_thread`'s display cursor). Replaces the earlier prev/curr
/// double-buffer, whose one-step interpolation window could not absorb the
/// real ±1–2 step wobble between the two threads' clocks — the display rammed
/// the window's rails and juddered.
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
        ring.slots[0].write(0, pos, quat);
        ring
    }

    /// Writer side (physics): publish the pose at absolute step `step`.
    pub fn publish(&self, step: u32, pos: [f32; 3], quat: [f32; 4]) {
        self.slots[step as usize % POSE_RING].write(step, pos, quat);
        self.latest.store(step, Ordering::Release);
    }

    /// Writer side: publish a **teleport** — the pose is written for `step`
    /// AND `step - 1`, so a reader interpolating across the boundary snaps
    /// instead of sweeping the body through the world.
    pub fn publish_snap(&self, step: u32, pos: [f32; 3], quat: [f32; 4]) {
        self.slots[step.wrapping_sub(1) as usize % POSE_RING].write(
            step.wrapping_sub(1),
            pos,
            quat,
        );
        self.publish(step, pos, quat);
    }

    /// Latest published absolute step.
    pub fn latest_step(&self) -> u32 {
        self.latest.load(Ordering::Acquire)
    }

    /// The pose published for `step`, if the ring still holds it.
    pub fn read_step(&self, step: u32) -> Option<([f32; 3], [f32; 4])> {
        self.slots[step as usize % POSE_RING].read(step)
    }

    /// The latest pose (always available once seeded).
    pub fn read_latest(&self) -> ([f32; 3], [f32; 4]) {
        loop {
            let step = self.latest_step();
            if let Some(pose) = self.read_step(step) {
                return pose;
            }
            // Racing the writer onto the next step — retry with the new latest.
            core::hint::spin_loop();
        }
    }
}

/// The player ball's motion channel in the shared `WebAssembly.Memory`: a
/// [`PoseRing`] of recent step poses (physics publishes every fixed step; the
/// render thread samples the ring at its display-time cursor and
/// lerps/slerps between the straddling steps), plus the frame-tick futex that
/// paces the sim.
///
/// Pose is stored as position + quaternion (not a baked matrix) precisely so the
/// reader can lerp position and *slerp* rotation correctly.
///
/// ## Frame-tick: render drives the physics cadence
///
/// `frame_tick` flows the *other* way (render → physics). The render thread bumps
/// it once per presented frame ([`bump_frame`]); the physics thread blocks on it
/// ([`wait_frame`]) instead of running its own timer. This deliberately makes the
/// sim step *in phase with vsync*: a free-running sim clock and the display's
/// refresh are two independent clocks that **beat**, and every phase-slip
/// surfaces as a one-frame stutter no matter how steady each clock is on its own.
/// Driving physics from the frame it feeds removes the beat at the source — each
/// presented frame gets a freshly-published pose, with the fixed-timestep
/// accumulator still decoupling sim rate from refresh rate (so 120/144 Hz just
/// means multiple frames between steps, interpolated). The wait/notify is a futex
/// on the tick word (`memory.atomic.wait32` / `notify`).
///
/// [`bump_frame`]: BodyMotion::bump_frame
/// [`wait_frame`]: BodyMotion::wait_frame
#[repr(C)]
pub struct BodyMotion {
    /// Pose history the render thread samples (see [`PoseRing`]).
    ring: PoseRing,
    /// Bumped by render once per presented frame; physics blocks on it to pace
    /// the sim to vsync (see the type docs).
    frame_tick: AtomicU32,
    /// Accumulated render-thread CPU time, in MICROSECONDS (wrapping): how long
    /// each frame's work — interpolation, transform update, encode + submit —
    /// took on the render thread, summed across frames. The stats panel divides
    /// its delta by the `frame_tick` delta for avg ms/frame: a real workload
    /// metric, unlike fps, which is vsync-capped and monitor-dependent. GPU
    /// time is NOT included (the submit returns before the GPU finishes).
    frame_work_us: AtomicU32,
    /// Accumulated physics-thread CPU time, in MICROSECONDS (wrapping): the
    /// full cost of each fixed step — input poll, `b3World_Step`, contact/hit
    /// classification, pose publish — summed across steps. The stats panel
    /// divides its delta by the step-count delta for avg ms/step.
    step_work_us: AtomicU32,
    /// Running max of the display cursor's |re-anchor error| (steps, `f32`
    /// bits — IEEE ordering matches `u32` for non-negatives, so `fetch_max`
    /// on the bits is max on the value). Render records it per frame; the
    /// stats panel reads-and-resets. This is the render/physics sync-health
    /// number: sustained values beyond the cursor's jitter budget
    /// (`TARGET_LAG` steps) mean the interpolation is ramming its rails and
    /// motion visibly pulses.
    sync_err_bits: AtomicU32,
    /// The render thread's CURRENT adaptive display lag (steps, `f32` bits) —
    /// how far the display cursor trails the newest published pose. Grows to
    /// absorb observed publish burstiness (see the render thread's cursor);
    /// published here for the stats panel's sync line.
    display_lag_bits: AtomicU32,
}

impl BodyMotion {
    /// Ring seeded with the given resting pose.
    pub fn new(pos: [f32; 3], quat: [f32; 4]) -> Self {
        BodyMotion {
            ring: PoseRing::new(pos, quat),
            frame_tick: AtomicU32::new(0),
            frame_work_us: AtomicU32::new(0),
            step_work_us: AtomicU32::new(0),
            sync_err_bits: AtomicU32::new(0),
            display_lag_bits: AtomicU32::new(0),
        }
    }

    /// Render side: record this frame's display-cursor |error| (steps).
    pub fn note_sync_err(&self, err_abs: f32) {
        self.sync_err_bits
            .fetch_max(err_abs.to_bits(), Ordering::Relaxed);
    }

    /// Stats side: the max |cursor error| (steps) since the last call, which
    /// resets the running max to zero.
    pub fn take_sync_err(&self) -> f32 {
        f32::from_bits(self.sync_err_bits.swap(0, Ordering::Relaxed))
    }

    /// Render side: publish the current adaptive display lag (steps).
    pub fn set_display_lag(&self, lag: f32) {
        self.display_lag_bits
            .store(lag.to_bits(), Ordering::Relaxed);
    }

    /// Stats side: the render thread's current adaptive display lag (steps).
    pub fn display_lag(&self) -> f32 {
        f32::from_bits(self.display_lag_bits.load(Ordering::Relaxed))
    }

    /// Render side: add one frame's CPU work time (µs) to the running total.
    pub fn add_frame_work_us(&self, us: u32) {
        self.frame_work_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Accumulated render-thread work time (µs, wrapping) — diff two reads and
    /// divide by the `frame_tick` delta for avg time per frame.
    pub fn frame_work_us(&self) -> u32 {
        self.frame_work_us.load(Ordering::Relaxed)
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

    /// Render side: signal that a frame was presented, waking the physics thread
    /// to advance the sim for the next one. Cheap when physics is already running
    /// (the `notify` is a no-op with no waiter; physics compares tick *values*, so
    /// it can't miss a bump).
    pub fn bump_frame(&self) {
        self.frame_tick.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `frame_tick` lives in the shared memory both threads attach to.
        unsafe {
            memory_atomic_notify(self.frame_tick.as_ptr() as *mut i32, 1);
        }
    }

    /// Current frame-tick value (physics seeds its last-seen with this).
    pub fn frame_tick(&self) -> u32 {
        self.frame_tick.load(Ordering::Relaxed)
    }

    /// Physics side: block until the frame-tick advances past `last`, returning the
    /// new value. Worker scope only — `memory.atomic.wait` traps on the main
    /// thread. Spurious wakeups + already-advanced ticks are both handled by the
    /// value re-check, so a presented frame is never missed.
    pub fn wait_frame(&self, last: u32) -> u32 {
        loop {
            let cur = self.frame_tick.load(Ordering::Relaxed);
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

    /// Writer side (physics): publish the pose at absolute step `step`.
    pub fn publish(&self, step: u32, pos: [f32; 3], quat: [f32; 4]) {
        self.ring.publish(step, pos, quat);
    }

    /// Writer side: publish a teleport (snap — see [`PoseRing::publish_snap`]).
    pub fn publish_snap(&self, step: u32, pos: [f32; 3], quat: [f32; 4]) {
        self.ring.publish_snap(step, pos, quat);
    }

    /// Latest published absolute step.
    pub fn latest_step(&self) -> u32 {
        self.ring.latest_step()
    }

    /// The pose published for `step`, if the ring still holds it.
    pub fn read_step(&self, step: u32) -> Option<([f32; 3], [f32; 4])> {
        self.ring.read_step(step)
    }

    /// The latest pose (always available once seeded).
    pub fn read_latest(&self) -> ([f32; 3], [f32; 4]) {
        self.ring.read_latest()
    }

    /// The underlying pose ring (render samples it — see `sample_ring` in the
    /// render thread).
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
    /// Bumped once per discrete jump press. Physics edge-detects a jump by
    /// diffing this against the value it last saw (so a press = exactly one hop,
    /// even though physics polls rather than receives an event).
    jump_seq: AtomicU32,
    /// Camera yaw in radians (`f32::to_bits`), accumulated by main from the
    /// orbit drags with [`CAMERA_ORBIT_SENSITIVITY`] — the same integration the
    /// render thread's `OrbitCamera` does, so the two stay in lockstep. Physics
    /// rotates the held-key roll direction by this so W/A/S/D stay
    /// camera-relative at any orbit angle.
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
        } else {
            self.held.fetch_and(!mask, Ordering::Relaxed);
        }
    }

    /// Main side: register a discrete jump press.
    pub fn bump_jump(&self) {
        self.jump_seq.fetch_add(1, Ordering::Relaxed);
    }

    /// Physics side: the current held-key bitset.
    pub fn held(&self) -> u32 {
        self.held.load(Ordering::Relaxed)
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

/// Render thread → main thread.
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
    /// GPU capability facts the render worker learns at startup, so main can
    /// seed the resolution scale and cap the backing store. `is_fallback` = a
    /// software adapter (can't push pixels → main starts conservative);
    /// `max_texture_dim` = the device's max 2D texture size (a huge display's
    /// backing store can't exceed it). Posted once, before the scene load.
    GpuInfo {
        is_fallback: bool,
        max_texture_dim: u32,
    },
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

/// Main thread → render thread: a runtime quality change. Anti-aliasing is a
/// renderer-*pipeline* setting — the MSAA sample count is baked into every
/// render pipeline + its targets, and SMAA is a post pass compiled into the
/// effects shader — so main can't apply it directly. It sends the desired flags
/// and the render worker calls `set_anti_aliasing` + `commit_load` (which
/// recompiles only the new config's variants; already-seen configs are cached,
/// so toggling back is cheap). Cold path — one message per Settings toggle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "quality", rename_all = "kebab-case")]
pub enum QualityMsg {
    /// Enable/disable MSAA 4× and the SMAA post pass (independent toggles).
    AntiAlias { msaa: bool, smaa: bool },
}

/// Main thread → render thread: the canvas element's size changed (main's
/// `ResizeObserver` — only main sees layout). The render worker owns the
/// transferred `OffscreenCanvas`, so it applies the new **device-pixel**
/// backing size there; the per-frame camera update picks up the new aspect.
/// Without this the backing store stays at its initial size and the browser
/// stretches it — circles render as ovals after any window resize.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "resize", rename_all = "kebab-case")]
pub enum ResizeMsg {
    Canvas { width: u32, height: u32 },
}

/// Radians of camera yaw per CSS pixel of horizontal drag. Lives here because
/// TWO integrators must agree on it: the render thread's `OrbitCamera` (the
/// visual) and the main thread's accumulated yaw (fed to physics for
/// camera-relative W/A/S/D and to audio for the orbiting listener). Both apply
/// `yaw -= dx * CAMERA_ORBIT_SENSITIVITY` to the same deltas from the same
/// start, so they stay in lockstep without ever exchanging the angle itself.
pub const CAMERA_ORBIT_SENSITIVITY: f32 = 0.005;

/// Main thread → render thread. The main thread owns the DOM (hence the
/// pointer events), but the camera lives in the render thread, so orbit/zoom
/// gestures are forwarded here. Cold path — one message per pointer move/wheel
/// tick, never per frame. (No pan: the camera stays aimed at the table so it
/// can never leave the frame.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cam", rename_all = "kebab-case")]
pub enum CameraMsg {
    /// A drag delta (pointer movement in CSS pixels) while the right mouse
    /// button is held — orbits the camera (yaw from `dx`, pitch from `dy`).
    Orbit { dx: f32, dy: f32 },
    /// A wheel delta — dollies the camera in/out.
    Zoom { dy: f32 },
}

// Main → physics input is no longer a message: it's the shared-memory
// [`InputState`] above (main writes, physics polls). See [`crate::physics_thread`].
