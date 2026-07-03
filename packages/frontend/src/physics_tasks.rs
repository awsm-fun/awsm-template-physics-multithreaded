//! The wasm task pool that makes Box3D's internal parallelism real.
//!
//! Box3D fans `b3World_Step`'s work out through `b3ParallelFor`
//! (`vendor/box3d/src/parallel_for.c`): it enqueues up to `workerCount` tasks
//! via the world def's `enqueueTask` callback, then blocks on `finishTask` for
//! each. On native that's backed by its pthread scheduler; here it's THIS pool
//! — plain web workers attached to the same shared `WebAssembly.Memory`,
//! coordinating through atomics + wasm futexes (`memory.atomic.wait32` /
//! `notify`), the same primitives the render→physics frame-tick already uses.
//! No postMessage anywhere: a "task" is two words (C function-table index +
//! context pointer) in shared memory.
//!
//! The design mirrors Box3D's own `scheduler.c` semantics exactly, because
//! `b3FinishTaskCallback`'s contract requires it: **the waiting thread must
//! help execute other pending tasks** while it waits, or nested parallel
//! phases can deadlock when pool threads < enqueued tasks.
//!
//! Safety facts this leans on:
//! - `parallel_for.c` bakes `workerIndex` into each TASK (task i → index i),
//!   so per-worker solver state is really per-task — any anonymous thread may
//!   run any task.
//! - Every `b3ParallelFor` finishes all its tasks before returning, so all
//!   slots are COMPLETE between world steps — [`TaskPool::reset`] (called
//!   before each `b3World_Step`) can safely recycle them.
//! - C function pointers are wasm function-table indices, valid on every
//!   thread of the same module.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::ffi::{c_char, c_void};

use box3d_sys as b3;
use wasm_bindgen::prelude::*;
use web_sys::js_sys;

/// Mirrors Box3D's `B3_MAX_TASKS` (constants.h) — the most tasks a world step
/// can enqueue (`b3ParallelFor` runs overflow inline itself; ours does too).
pub const MAX_TASKS: usize = 256;
/// Mirrors `B3_MAX_WORKERS`; sizes the per-worker claim counters.
pub const MAX_WORKERS: usize = 32;

const FREE: u32 = 0;
const PENDING: u32 = 1;
const CLAIMED: u32 = 2;
const COMPLETE: u32 = 3;

/// One enqueued Box3D task: the C callback (function-table index) + its
/// context, guarded by a status word that doubles as the finish-wait futex.
#[repr(C)]
pub struct TaskSlot {
    status: AtomicU32,
    callback: AtomicUsize,
    context: AtomicUsize,
}

/// The shared pool block. Allocated once (leaked) by the physics thread;
/// workers receive its address at spawn. Writer/reader roles:
/// - `enqueueTask` (physics thread, inside `b3World_Step`) appends slots and
///   signals the semaphore;
/// - pool workers (and the physics thread inside `finishTask`) claim slots by
///   CAS and execute them;
/// - [`reset`](Self::reset) (physics thread, between steps) recycles slots.
#[repr(C)]
pub struct TaskPool {
    /// Next free slot index (may run past MAX_TASKS; overflow runs inline).
    next_slot: AtomicU32,
    /// Counting semaphore of unclaimed tasks; futex word for idle workers.
    sem: AtomicU32,
    /// Per-executor claim counts: index 0 = the physics thread (helping in
    /// `finishTask`), 1.. = pool workers. Proof-of-parallelism telemetry.
    claims: [AtomicU32; MAX_WORKERS],
    tasks: [TaskSlot; MAX_TASKS],
}

fn futex_wait(word: &AtomicU32, expected: u32) {
    unsafe {
        // Returns immediately if the value already changed; -1 = no timeout.
        core::arch::wasm32::memory_atomic_wait32(word.as_ptr() as *mut i32, expected as i32, -1);
    }
}

fn futex_notify(word: &AtomicU32, count: u32) {
    unsafe {
        core::arch::wasm32::memory_atomic_notify(word.as_ptr() as *mut i32, count);
    }
}

impl TaskPool {
    /// Allocate the pool in the shared heap, leaked for the session.
    #[allow(clippy::new_ret_no_self)]
    pub fn leak_new() -> &'static TaskPool {
        #[allow(clippy::declare_interior_mutable_const)]
        const SLOT: TaskSlot = TaskSlot {
            status: AtomicU32::new(FREE),
            callback: AtomicUsize::new(0),
            context: AtomicUsize::new(0),
        };
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: AtomicU32 = AtomicU32::new(0);
        Box::leak(Box::new(TaskPool {
            next_slot: AtomicU32::new(0),
            sem: AtomicU32::new(0),
            claims: [ZERO; MAX_WORKERS],
            tasks: [SLOT; MAX_TASKS],
        }))
    }

    pub fn addr(&'static self) -> usize {
        self as *const TaskPool as usize
    }

    /// Recycle all slots. ONLY safe between world steps — every task is
    /// COMPLETE then (each `b3ParallelFor` finishes its tasks before
    /// returning), so no executor can be touching a slot.
    pub fn reset(&self) {
        let used = self.next_slot.swap(0, Ordering::Relaxed) as usize;
        for slot in &self.tasks[..used.min(MAX_TASKS)] {
            slot.status.store(FREE, Ordering::Relaxed);
        }
    }

    /// Snapshot of the claim counters (first `n` executors).
    pub fn claim_counts(&self, n: usize) -> Vec<u32> {
        self.claims[..n.min(MAX_WORKERS)]
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }

    /// Claim and run one pending task. `who` indexes the claim counters.
    /// Returns false when nothing was pending.
    fn execute_one(&self, who: usize) -> bool {
        let count = (self.next_slot.load(Ordering::Acquire) as usize).min(MAX_TASKS);
        for slot in &self.tasks[..count] {
            if slot.status.load(Ordering::Relaxed) != PENDING {
                continue;
            }
            if slot
                .status
                .compare_exchange(PENDING, CLAIMED, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            unsafe {
                let cb: b3::b3TaskCallback =
                    core::mem::transmute(slot.callback.load(Ordering::Relaxed));
                cb(slot.context.load(Ordering::Relaxed) as *mut c_void);
            }
            slot.status.store(COMPLETE, Ordering::Release);
            // Wake any finishTask waiter parked on this slot.
            futex_notify(&slot.status, u32::MAX);
            self.claims[who.min(MAX_WORKERS - 1)].fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    /// Counting-semaphore signal: one credit per enqueued task (at most one
    /// worker picks up each task, so one wake per enqueue suffices —
    /// `scheduler.c` does the same).
    fn sem_signal(&self) {
        self.sem.fetch_add(1, Ordering::Release);
        futex_notify(&self.sem, 1);
    }

    /// Counting-semaphore wait: take a credit, sleeping on the futex while
    /// the count is zero.
    fn sem_wait(&self) {
        loop {
            let credits = self.sem.load(Ordering::Acquire);
            if credits > 0 {
                if self
                    .sem
                    .compare_exchange(credits, credits - 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
            } else {
                futex_wait(&self.sem, 0);
            }
        }
    }
}

/// `b3WorldDef.enqueueTask` — called by Box3D from inside `b3World_Step` on
/// the stepping thread. Appends a slot and wakes one worker; on slot overflow
/// runs the task inline and returns null ("executed serially" per the Box3D
/// contract, so `finishTask` is skipped for it).
///
/// # Safety
/// `user_context` must be the [`TaskPool`] address handed to `b3CreateWorld`.
pub unsafe extern "C" fn enqueue_task(
    task: b3::b3TaskCallback,
    task_context: *mut c_void,
    user_context: *mut c_void,
    _name: *const c_char,
) -> *mut c_void {
    let pool = &*(user_context as *const TaskPool);
    let index = pool.next_slot.fetch_add(1, Ordering::Relaxed) as usize;
    if index >= MAX_TASKS {
        task(task_context);
        return core::ptr::null_mut();
    }
    let slot = &pool.tasks[index];
    slot.callback.store(task as usize, Ordering::Relaxed);
    slot.context.store(task_context as usize, Ordering::Relaxed);
    slot.status.store(PENDING, Ordering::Release);
    pool.sem_signal();
    slot as *const TaskSlot as *mut c_void
}

/// `b3WorldDef.finishTask` — block until `user_task` completes, HELPING run
/// other pending tasks meanwhile (required: with fewer pool threads than
/// tasks, the stepping thread must drain work itself or nested phases stall).
///
/// # Safety
/// `user_task` is a slot pointer previously returned by [`enqueue_task`] (or
/// null, which is a no-op); `user_context` is the same pool address.
pub unsafe extern "C" fn finish_task(user_task: *mut c_void, user_context: *mut c_void) {
    if user_task.is_null() {
        return;
    }
    let pool = &*(user_context as *const TaskPool);
    let slot = &*(user_task as *const TaskSlot);
    loop {
        let status = slot.status.load(Ordering::Acquire);
        if status == COMPLETE {
            return;
        }
        if !pool.execute_one(0) {
            // Nothing to steal — sleep until this slot's status changes.
            futex_wait(&slot.status, status);
        }
    }
}

/// Worker entry (`mt_worker_start` role `"physics-task"`): park on the
/// semaphore, drain pending tasks, repeat — forever (the pool lives for the
/// session). Pure compute over shared memory; no JS after this point.
pub fn start(payload: JsValue) -> Result<(), JsValue> {
    let addr = js_sys::Reflect::get(&payload, &JsValue::from_str("pool"))
        .ok()
        .and_then(|v| v.as_f64())
        .ok_or_else(|| JsValue::from_str("physics-task: missing pool address"))?
        as usize;
    let index = js_sys::Reflect::get(&payload, &JsValue::from_str("index"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as usize;
    // SAFETY: the physics thread leaked the pool in this same shared memory.
    let pool: &'static TaskPool = unsafe { &*(addr as *const TaskPool) };
    tracing::info!("physics-task worker {index}: online");
    loop {
        pool.sem_wait();
        while pool.execute_one(index) {}
    }
}
