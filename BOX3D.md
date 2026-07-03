# Box3D on wasm — reference notes

How [Box3D](https://github.com/erincatto/box3d) (Erin Catto's 3D physics
engine, C17, MIT) is compiled **into this template's shared-memory wasm
bundle** and driven multithreaded. This is a reference for anyone touching
`packages/box3d-sys` or `packages/frontend/src/physics_*`.

## The shape of the integration

One wasm module, one shared `WebAssembly.Memory`. The Box3D C sources
(`vendor/box3d`, a git submodule) are compiled by `packages/box3d-sys/build.rs`
(the `cc` crate + a wasm-capable clang) and linked into the same bundle as the
Rust code — NOT built as a separate Emscripten module. Nothing per-step or
per-frame crosses a `postMessage` boundary: poses, input, sim pacing, and the
solver's own task system all run over shared memory + atomics.

- **Host builds** (for `cargo test -p box3d-sys`) compile ALL of
  `vendor/box3d/src/*.c` against the real libc — including `timer.c` /
  `scheduler.c` (pthreads) — so the FFI is natively testable, internal
  scheduler included.
- **wasm builds** exclude `timer.c` + `scheduler.c` (no libc, no pthreads),
  substitute shim libc headers, and build every object with
  `-matomics -mbulk-memory -mmutable-globals` (wasm-ld refuses objects without
  atomics in a `--shared-memory` link).

### Why not Emscripten

Emscripten produces its own module + JS glue + its own pthread runtime and
memory. Sharing one `WebAssembly.Memory` between emcc output and our
wasm-bindgen bundle would require matching allocators/TLS/stack layout — not
realistic. Bridging two modules means copies or postMessage, violating the
design. Direct clang → same-module link is strictly better here, and Box3D's
pluggable task system removes the only thing Emscripten would have provided
(pthreads).

## Box3D facts (verified against the vendored submodule)

- **API is box2d-v3-style**: `b3DefaultWorldDef()` → `b3CreateWorld` →
  `b3World_Step(world, dt, subStepCount)`; `b3CreateBody` + `b3DefaultBodyDef`;
  shapes via `b3CreateSphereShape` / `b3CreateCapsuleShape` /
  `b3CreateHullShape` (+ `b3MakeBoxHull(hx,hy,hz)` in `collision.h`).
- **Def structs carry a validation cookie** (`internalValue`): ALWAYS
  initialize via `b3Default*Def()` across FFI and mutate fields — never
  hand-construct. A layout mismatch surfaces as an assert on create; wire
  `b3SetAssertFcn` FIRST so it reaches the console instead of trapping
  opaquely.
- **Shapes**: sphere, capsule, convex hull, mesh, heightfield. No
  cylinder/cone primitives, but `collision.h` has heap-allocated hull builders
  `b3CreateCylinder(height, radius, yOffset, sides)` and
  `b3CreateCone(height, r1, r2, slices)` (returns `b3HullData*`, free with
  `b3DestroyHull`).
- **Events** (`b3World_GetContactEvents`): begin/end-touch events + **hit
  events**. Hit events carry the contact point and `approachSpeed` — ideal for
  impact-audio intensity — and are gated by `worldDef.hitEventThreshold` plus
  per-shape `enableHitEvents`. Contact-event enables are **OR'd across a
  pair's shapes** (`contact.c`): enabling on one dynamic shape covers
  dynamic-vs-static pairs, so leave statics' `enableContactEvents = false` or
  the arrays flood.
- **Threading is pluggable** (`physics_world.c`): with
  `def.workerCount > 0 && def.enqueueTask && def.finishTask`, Box3D uses OUR
  task system and never touches its own scheduler. `workerCount == 1` is the
  fully-serial fallback. Only `workerCount > 1` *without* callbacks spawns
  pthreads — never do that on wasm. Contract details:
  - `void* enqueueTask(b3TaskCallback*, void* taskContext, void* userContext,
    const char* name)`; returning NULL means "ran serially, finishTask won't
    be called". `finishTask(userTask, userContext)` must **help-execute**
    pending tasks while waiting (mirror `src/scheduler.c`, or pools smaller
    than the task count deadlock).
  - `B3_MAX_WORKERS = 32`, `B3_MAX_TASKS = 256` (`constants.h`).
- **Platform hooks**: `b3SetAllocator(allocFcn, freeFcn)` — `b3FreeFcn` gets
  NO size, so an allocator shim must self-describe allocations (size/align
  header trick); honor the `alignment` argument. Plus `b3SetAssertFcn` and the
  log callback.
- **Determinism**: upstream builds with `-ffp-contract=off`
  (box2d.org/posts/2024/08/determinism) — build.rs mirrors it; it's what makes
  the SIMD-vs-scalar parity check meaningful.
- **World lifecycle is not thread-safe**: the world table is a global;
  creating/destroying worlds from two OS threads at once trips an internal
  assert. Irrelevant in the app (one world), but host tests serialize world
  creation via a mutex.

## The wasm libc shim (`packages/box3d-sys/shim/`)

- `-nostdlibinc` drops the (nonexistent) system includes; clang's builtin
  headers (stdint/stdbool/stddef/stdarg/float/limits + `wasm_simd128.h`)
  remain. Shim headers provide string/stdio/stdlib/math/assert/inttypes —
  declarations only. `inttypes.h` must fully replace the builtin one (its
  `#include_next` fails under `-nostdlibinc`); `PRIx64 = "llx"`.
- `vsnprintf`/`snprintf` are real (vendored `stb_sprintf.h`, public domain) so
  assert/log messages format; `fopen` stubs to NULL (call sites check),
  `printf`/`puts` route to Rust `tracing`.
- Rust-side (`src/wasm_shim.rs`): the allocator (size-header trick over the
  Rust global allocator), transcendentals via the `libm` crate
  (`sinf/atan2f/remainderf/nextafterf/…` — no wasm instruction for them;
  `sqrtf`/`fabsf`/etc. lower to instructions, helped by `-fno-math-errno`),
  spinlock-backed mutexes, trap-loud scheduler stubs (unreachable given our
  worldDef), and **`b3Hash` ported to Rust** — it lives at the bottom of the
  excluded `timer.c` upstream.
- `memcpy/memset/memmove/memcmp` come from Rust's compiler-builtins already in
  the link.
- Layout safety: `shim/sizes.c` reports C `sizeof` for every mirrored struct;
  host tests assert them against the Rust `size_of` (26 tests,
  `cargo test -p box3d-sys`).

## Task pool (`packages/frontend/src/physics_tasks.rs`)

A `#[repr(C)]` shared `TaskPool` (leaked) with `B3_MAX_TASKS` slots
`{status, callback, ctx}`, a futex counting semaphore
(`memory_atomic_wait32/notify`), and per-executor claim counters. Pool workers
CAS-claim pending slots and call the C callback (a wasm function-table index —
valid on every thread). `finishTask` help-executes per the scheduler.c
contract. `worldDef.workerCount = pool_threads + 1` (the stepping thread
participates). Slots reset before each `b3World_Step` — safe because every
`b3ParallelFor` completes before returning.

- **Pool workers are spawned by MAIN, not by the physics thread**: the physics
  worker blocks on the frame-tick futex for its whole life and never services
  its event loop again, so it can't start nested workers — it posts
  `PhysicsMsg::SpawnTaskWorkers` to main. (If physics tries to spawn them
  itself the children never come online; the symptom is claim counters stuck
  at `[all-self, 0, 0, 0, 0]`.)
- Worker count: `hardware_concurrency − 3` existing threads, clamped `1..=4`.
- Task callbacks are pure compute into shared memory — no wasm-bindgen/JS from
  pool workers after spawn.
- Blocking `memory_atomic_wait32` is forbidden on the browser main thread —
  every pool/finish/pacing wait lives on worker threads.
- Observed healthy claim balance over a multi-minute soak:
  roughly even across the stepping thread + 4 pool workers.

## SIMD (wasm simd128)

Box3D's contact solver is built around 4-wide float SIMD (`b3V32` in
`src/simd.h`; consumers: `contact_solver.c`, `solver.c`, `dynamic_tree.c`,
`mesh.c`, `height_field.c`) — scalar-only forfeits the headline perf.

- `-DB3_CPU_WASM` flips `core.h` onto its `B3_SIMD_SSE2` path **without any
  submodule edit**; our `shim/include/emmintrin.h` maps the complete 33-symbol
  SSE surface onto `wasm_simd128.h` (`__m128 = v128_t`). All ops used are
  IEEE-exact (upstream deliberately avoids `_mm_rsqrt_ps`), so the port
  matches SSE2 **bit-for-bit**.
- Argument-order traps (the reason for hand-mapping):
  - `_mm_min_ps(a,b)` = `wasm_f32x4_pmin(b,a)`, `_mm_max_ps(a,b)` =
    `wasm_f32x4_pmax(b,a)` — plain `wasm_f32x4_min/max` are NaN-propagating
    and −0.0-aware, SSE returns the second operand on NaN.
  - `_mm_andnot_ps(a,b)` = `wasm_v128_andnot(b,a)`.
  - `_mm_shuffle_ps`/`_MM_SHUFFLE` stay macros (`__builtin_shufflevector`
    needs immediate lane indices; all call sites pass immediates).
  - `_mm_load_ss` → `wasm_v128_load32_zero`; the
    `_mm_castpd_ps(_mm_load_sd(p))` idiom → `wasm_v128_load64_zero`;
    `_mm_movemask_ps` → `wasm_i32x4_bitmask`; `_mm_pause` → no-op.
- `BOX3D_WASM_SCALAR=1` (build.rs env) reverts to the scalar path for A/B.
- **Measured results**: bit-exact scalar/SIMD parity at step 4800 (pose bits
  identical across repeated runs), 0.307 ms vs 0.394 ms avg step on the
  200-ball pile — **~22% faster**. The parity probe (`PARITY_TICK` in
  `physics_thread.rs`) logs exact pose bits at 20 s; it's deterministic absent
  input/clicks.
- `+simd128` is also in `THREADED_RUSTFLAGS` (glam/renderer benefit; every
  WebGPU-capable browser has simd128).

## Renderer integration notes (awsm-renderer 0.11)

- **Runtime ball visuals are mesh duplicates, not GPU instancing**:
  `duplicate_mesh_with_transform(src_mesh, tk)` shares the source's GPU
  geometry + material and rides the same PBR/shadow pipeline.
  `transforms.insert()` after load still allocates a shared-arena slot, so
  `arena_slot_binding(tk)` + `foreign_write` work for every duplicate. Real
  instancing (`set_mesh_instances`) exists but takes owner-side `&[Transform]`
  re-uploads per frame, bypassing the foreign-write arena — the escape hatch
  for thousands of movers.
- **A duplicate copies the source mesh's CURRENT material key** — swap the
  player's tinted material in first and duplicates inherit it. The player's
  red is a cloned material (clone → mutate `base_color` factor →
  `materials.insert` → `set_mesh_material`); duplicates get reset to the
  original key at mint time.
- **rstar ceiling**: above ~300 concurrently-moving bodies the renderer's
  rstar 0.12.2 spatial index can panic ("This is a bug in rstar", recursive
  insert under per-frame churn). `protocol::MAX_BALLS = 200` keeps margin.

## Sync & fluidity

The pose-ring jitter buffer + vsync-timestamp display cursor are documented in
the README ("The threads"). Reference points for future debugging:

- Physics honestly publishes 3/4/5 steps per 60 Hz frame (frame wobble
  quantized against the 240 Hz grid) — a 1-step interpolation window can't
  absorb it; the 16-step ring + 2-step trailing cursor does.
- Frame time must come from the rAF `DOMHighResTimeStamp` argument, never
  `performance.now()` inside the callback (scheduling delay becomes judder).
- 1 Hz `sync:` debug telemetry (steps/frame histogram + cursor-err band) in
  `render_thread.rs` is the canonical smoothness diagnostic; cursor err should
  sit well inside ±2 steps.
- **Judder discriminator**: inject a pure rAF-driven DOM square (devtools
  console) next to the canvas — if it hitches at the same moments as the
  ball, the judder is compositor frame skips (the platform floor), not the
  app pipeline.

## Ball↔ball clack — measured reference (CC0 freesound #539854)

The `audio-worklet-clack` DSP is calibrated against an offline analysis of a
real billiard impact recording (44.1 kHz mono, 239 ms). The numbers that
matter (guessing them instead of measuring reliably sounds wrong):

- Band energy: 0–500 Hz **0.5%**, 500–1k **0.2%**, 1–2k **11.4%**,
  **2–4 kHz 87.9%**, 4–8k **0.1%** — there is NO low "body thump", and the
  4–8 kHz content exists only in the first millisecond.
- The spectrum is a dense cluster of comparable peaks packed into
  ~1.8–2.9 kHz (2250/1950/2400/2175/2100/2275…) — too dense to read as a
  pitch. Sparse resonators (octave-spread modes) sound like a struck pan.
- Envelope: attack 1.3 ms, −20 dB in **5.9 ms**, then a quiet 2 kHz tail to
  −40 dB at ~91 ms.
- The balls RATTLE: secondary contact bumps at ~11.8/16.5/22.7/29.8 ms with
  relative amplitudes ~0.33/0.28/0.20/0.17.

The worklet mirrors these: dense 8-mode jittered cluster + 2 mid + 2 quiet
tail modes, per-contact low-passed noise bursts, intensity-gated micro-bounce
re-excitations, per-voice seeded jitter (no two clacks identical). `intensity`
is the runtime control (contact time, tilt, loudness, bounce count);
`brightness`/`ring` are authoring knobs.

## Browser verification recipe

For any physics/rendering change, verified via the chrome-devtools MCP:

1. **Serve**: `task dev` in the background; poll until trunk reports the
   server on `http://127.0.0.1:9000`. First compile is slow (build-std +
   optimized deps); trunk auto-rebuilds on edits — confirm a fresh "✅ success"
   before re-testing.
2. **Open**: `new_page` → screenshot → table + ball rendered.
3. **Console preconditions**: `crossOriginIsolated = true`, physics startup
   line, render Ready, zero errors. A `RuntimeError: unreachable` usually
   means a Box3D assert or FFI layout mismatch — check `b3SetAssertFcn` output.
4. **Drive input**: movement is held-state — dispatch `keydown`, wait ~1 s,
   then `keyup` (`window.dispatchEvent(new KeyboardEvent(...))`). Space is
   edge-triggered. Clicks: dispatch `MouseEvent('click', {clientX, clientY})`
   on the canvas to drop balls.
5. **Verify motion** by before/after screenshots; **verify audio** via the
   `audio cue:` debug console lines (wall-hit/land with intensity — the roll
   cue is deliberately not logged).
6. **Soak**: leave it running minutes, watch for frozen pose logs (deadlock)
   or new errors. Teardown: close the page, kill the dev server.

## Gotchas

- Every C object needs `-matomics -mbulk-memory` (wasm-ld shared-memory link).
- An env `RUSTFLAGS` **replaces** `.cargo/config.toml` rustflags — the
  Taskfile repeats the base cfgs (`web_sys_unstable_apis`, getrandom);
  don't lose them.
- `cc`'s `.opt_level(2)` is pinned in build.rs so dev builds get a fast
  physics core; the Rust shim needs
  `[profile.dev.package.box3d-sys] opt-level = 3` at the workspace root
  because `package."*"` does NOT cover workspace members.
- `b3FreeFcn` gets no size → allocator header trick (see shim notes).
- COOP/COEP (`Trunk.toml` headers + `coi-serviceworker.js` for Pages) is what
  makes `SharedArrayBuffer` exist — any new hosting must send the same two
  headers.
- Worker `onerror` gives useless "Script error." unless the handler extracts
  `ErrorEvent.message`; a `std::panic::set_hook` logging to console is
  installed so wasm panics don't die as opaque `unreachable`s (both live in
  `lib.rs` / `bootstrap.rs`).
