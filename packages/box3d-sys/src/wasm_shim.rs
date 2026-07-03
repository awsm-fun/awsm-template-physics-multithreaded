//! The Rust half of the wasm libc shim (`wasm32-unknown-unknown` only).
//!
//! Provides the symbols the vendored Box3D C objects import but no libc
//! exists to supply:
//!
//! - **Allocator**: `malloc` / `aligned_alloc` / `free` backed by the Rust
//!   global allocator. C's `free` carries no size/alignment, so each
//!   allocation stores its base pointer + layout in a header just below the
//!   returned address.
//! - **Math**: transcendentals with no wasm instruction (`sinf`, `atan2f`, …)
//!   via the `libm` crate. `sqrtf`/`floorf`/`fabsf`/… never reach here — the
//!   C is compiled with `-fno-math-errno` so clang lowers them to wasm ops.
//! - **Timer/profiling** (`b3GetTicks` family): return 0 — only feeds
//!   `b3Profile`, worthless without a clock and never on a hot path.
//! - **Internal scheduler** (`b3CreateScheduler` family, from the excluded
//!   `scheduler.c`): referenced by `physics_world.c` but unreachable when the
//!   world def supplies task callbacks or `workerCount == 1` — stubbed to
//!   trap LOUDLY if ever hit (means the world was misconfigured).
//! - **Mutex** (`b3CreateMutex` family, from the excluded `timer.c`): real
//!   spinlocks — `physics_world.c` and `recording.c` do take them.
//! - **Log sink**: `b3dsys_shim_log`, the target of the shim `printf` (Box3D
//!   assert/warning text). Route it somewhere visible with [`set_shim_log`].

#![allow(clippy::missing_safety_doc)]

use core::ffi::{c_char, c_void, CStr};
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

// ─── log sink ────────────────────────────────────────────────────────────────

static LOG_FN: AtomicUsize = AtomicUsize::new(0);

/// Route Box3D's printf output (asserts, warnings) somewhere visible — call
/// once at startup, e.g. `set_shim_log(|msg| tracing::warn!("{msg}"))`.
pub fn set_shim_log(f: fn(&str)) {
    LOG_FN.store(f as usize, Ordering::Relaxed);
}

#[no_mangle]
unsafe extern "C" fn b3dsys_shim_log(message: *const c_char) {
    let f = LOG_FN.load(Ordering::Relaxed);
    if f == 0 || message.is_null() {
        return;
    }
    let f: fn(&str) = core::mem::transmute(f);
    f(&CStr::from_ptr(message).to_string_lossy());
}

fn shim_log(msg: &str) {
    let f = LOG_FN.load(Ordering::Relaxed);
    if f != 0 {
        let f: fn(&str) = unsafe { core::mem::transmute(f) };
        f(msg);
    }
}

// ─── allocator ───────────────────────────────────────────────────────────────

/// Header stored immediately below every pointer we hand out: the base address
/// and total size of the underlying Rust allocation (wasm32: 2 × 4 bytes).
const HEADER: usize = 2 * core::mem::size_of::<usize>();
/// Base alignment of the underlying allocation; also the header's alignment.
const BASE_ALIGN: usize = 16;

unsafe fn alloc_impl(size: usize, align: usize) -> *mut c_void {
    let size = size.max(1);
    let align = align.max(BASE_ALIGN);
    // Room for the header below the aligned address, wherever it lands.
    let total = size + align + HEADER;
    let layout = core::alloc::Layout::from_size_align(total, BASE_ALIGN).unwrap();
    let base = std::alloc::alloc(layout);
    if base.is_null() {
        return core::ptr::null_mut();
    }
    let user = (base as usize + HEADER + align - 1) & !(align - 1);
    let header = (user - HEADER) as *mut usize;
    header.write(base as usize);
    header.add(1).write(total);
    user as *mut c_void
}

#[no_mangle]
unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    alloc_impl(size, BASE_ALIGN)
}

#[no_mangle]
unsafe extern "C" fn aligned_alloc(alignment: usize, size: usize) -> *mut c_void {
    alloc_impl(size, alignment)
}

#[no_mangle]
unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let header = (ptr as usize - HEADER) as *const usize;
    let base = header.read() as *mut u8;
    let total = header.add(1).read();
    let layout = core::alloc::Layout::from_size_align(total, BASE_ALIGN).unwrap();
    std::alloc::dealloc(base, layout);
}

// ─── math (no wasm instruction → libm) ───────────────────────────────────────

macro_rules! libm_shim {
    ($($name:ident($($arg:ident),+)),+ $(,)?) => {
        $(
            #[no_mangle]
            unsafe extern "C" fn $name($($arg: f32),+) -> f32 {
                libm::$name($($arg),+)
            }
        )+
    };
}

libm_shim! {
    sinf(x), cosf(x), tanf(x), asinf(x), acosf(x), atanf(x),
    expf(x), logf(x),
    atan2f(y, x), powf(base, exp), fmodf(x, y),
    remainderf(x, y), nextafterf(from, to),
}

// ─── b3Hash ──────────────────────────────────────────────────────────────────

/// Port of `timer.c`'s `b3Hash` (djb2 folded 8 bytes/iteration) — the one
/// target-independent function parked in the excluded file. wasm32 is
/// little-endian, so the big-endian byte-swap branch is dropped. Hull/mesh
/// content identity depends on this matching the C exactly — re-check against
/// `vendor/box3d/src/timer.c` on any submodule bump (the sys tests can't run
/// on wasm, but native runs the C original; divergence would only affect
/// recording replay hashes, not simulation).
#[no_mangle]
unsafe extern "C" fn b3Hash(hash: u32, data: *const u8, count: i32) -> u32 {
    let mut result = hash;
    let count = count as usize;
    let mut i = 0;
    while i + 8 <= count {
        let mut bytes = [0u8; 8];
        core::ptr::copy_nonoverlapping(data.add(i), bytes.as_mut_ptr(), 8);
        let word = u64::from_le_bytes(bytes);
        result = (result << 5).wrapping_add(result).wrapping_add(word as u32);
        result = (result << 5)
            .wrapping_add(result)
            .wrapping_add((word >> 32) as u32);
        i += 8;
    }
    while i < count {
        result = (result << 5)
            .wrapping_add(result)
            .wrapping_add(*data.add(i) as u32);
        i += 1;
    }
    result
}

// ─── timer / profiling stubs (feed b3Profile only) ───────────────────────────

#[no_mangle]
extern "C" fn b3GetTicks() -> u64 {
    0
}

#[no_mangle]
extern "C" fn b3GetMilliseconds(_ticks: u64) -> f32 {
    0.0
}

#[no_mangle]
unsafe extern "C" fn b3GetMillisecondsAndReset(ticks: *mut u64) -> f32 {
    if !ticks.is_null() {
        *ticks = 0;
    }
    0.0
}

#[no_mangle]
extern "C" fn b3Yield() {
    core::hint::spin_loop();
}

// ─── internal scheduler stubs (must never run — see module docs) ─────────────

fn scheduler_unreachable(what: &str) -> ! {
    shim_log(what);
    core::arch::wasm32::unreachable()
}

#[no_mangle]
extern "C" fn b3CreateScheduler(_worker_count: i32) -> *mut c_void {
    scheduler_unreachable(
        "box3d-sys: b3CreateScheduler reached on wasm — the world def must set \
         enqueueTask/finishTask (or workerCount = 1); the internal pthread \
         scheduler does not exist in this build",
    );
}

#[no_mangle]
extern "C" fn b3DestroyScheduler(_scheduler: *mut c_void) {
    scheduler_unreachable("box3d-sys: b3DestroyScheduler reached on wasm");
}

#[no_mangle]
extern "C" fn b3ResetScheduler(_scheduler: *mut c_void) {
    scheduler_unreachable("box3d-sys: b3ResetScheduler reached on wasm");
}

#[no_mangle]
extern "C" fn b3SchedulerEnqueueTask(
    _task: *mut c_void,
    _task_context: *mut c_void,
    _user_context: *mut c_void,
    _name: *const c_char,
) -> *mut c_void {
    scheduler_unreachable("box3d-sys: b3SchedulerEnqueueTask reached on wasm");
}

#[no_mangle]
extern "C" fn b3SchedulerFinishTask(_user_task: *mut c_void, _user_context: *mut c_void) {
    scheduler_unreachable("box3d-sys: b3SchedulerFinishTask reached on wasm");
}

// ─── mutex (spinlock — physics_world.c and recording.c take these) ───────────

#[no_mangle]
extern "C" fn b3CreateMutex() -> *mut c_void {
    Box::into_raw(Box::new(AtomicU32::new(0))) as *mut c_void
}

#[no_mangle]
unsafe extern "C" fn b3DestroyMutex(m: *mut c_void) {
    if !m.is_null() {
        drop(Box::from_raw(m as *mut AtomicU32));
    }
}

#[no_mangle]
unsafe extern "C" fn b3LockMutex(m: *mut c_void) {
    let lock = &*(m as *const AtomicU32);
    while lock
        .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}

#[no_mangle]
unsafe extern "C" fn b3UnlockMutex(m: *mut c_void) {
    (*(m as *const AtomicU32)).store(0, Ordering::Release);
}
