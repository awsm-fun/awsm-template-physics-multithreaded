//! Build the vendored Box3D C library (vendor/box3d) for the current target.
//!
//! Two very different configurations (see BOX3D.md):
//!
//! - **Host** (mac/linux, for `cargo test -p box3d-sys`): compile ALL of
//!   `vendor/box3d/src/*.c` against the real libc — including `timer.c` /
//!   `scheduler.c` (pthreads) — so the FFI surface is testable natively,
//!   internal scheduler included.
//!
//! - **wasm32-unknown-unknown**: there is no libc and no pthreads. `timer.c`
//!   and `scheduler.c` are EXCLUDED (their symbols are stubbed in Rust; the
//!   task system is provided by the game via `b3WorldDef.enqueueTask`), shim
//!   libc headers stand in for the missing sysroot, and every object is built
//!   with `+atomics,+bulk-memory` so wasm-ld accepts them into the
//!   `--shared-memory` module. Apple clang cannot emit wasm — a wasm-capable
//!   clang is discovered below (env override → Homebrew LLVM → PATH).
//!
//! `shim/sizes.c` (struct-layout ground truth for the FFI tests) compiles on
//! both targets; the Rust tests that read it run on host.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Env var the `cc` crate itself honors for this target; we respect it first.
const CC_ENV: &str = "CC_wasm32_unknown_unknown";

/// Locations a wasm-capable clang typically lives, probed in order.
const CLANG_CANDIDATES: &[&str] = &[
    "/opt/homebrew/opt/llvm/bin/clang", // Homebrew LLVM, Apple Silicon
    "/usr/local/opt/llvm/bin/clang",    // Homebrew LLVM, Intel mac
    "clang",                            // distro clang (Linux/CI) has all backends
];

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let is_wasm = target == "wasm32-unknown-unknown";
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("../../vendor/box3d");

    let mut build = cc::Build::new();
    build
        .std("c17")
        .opt_level(2)
        // Upstream builds with contraction off for cross-platform determinism
        // (https://box2d.org/posts/2024/08/determinism/); required for the M6
        // SIMD-vs-scalar parity test to be meaningful.
        .flag("-ffp-contract=off")
        .include(vendor.join("include"))
        .include(vendor.join("src"))
        .warnings(false);

    if is_wasm {
        let clang = wasm_clang();
        // SIMD by default: `-DB3_CPU_WASM` flips core.h onto its SSE2 path and
        // our shim emmintrin.h maps those intrinsics to wasm simd128 (see
        // BOX3D.md). BOX3D_WASM_SCALAR=1 reverts to the scalar path for A/B
        // parity + benchmark runs.
        let scalar = env::var("BOX3D_WASM_SCALAR").is_ok_and(|v| v == "1");
        println!(
            "cargo:warning=box3d-sys: wasm clang = {clang}{}",
            if scalar {
                " (SCALAR build)"
            } else {
                " (simd128)"
            }
        );
        build
            .compiler(&clang)
            // Must match the Rust side's `-C target-feature` (Taskfile.yml):
            // wasm-ld refuses objects without atomics in a --shared-memory link.
            .flag("-matomics")
            .flag("-mbulk-memory")
            .flag("-mmutable-globals")
            // No sysroot: drop the (nonexistent) system include dirs and use
            // our shim libc headers. clang's builtin headers (stdint/stdbool/
            // stddef/stdarg/float/limits) live in its resource dir and remain.
            .flag("-nostdlibinc")
            .include("shim/include")
            // sqrtf/floorf/fabsf/… lower to wasm instructions instead of
            // libcalls (Box3D never reads errno; results are IEEE-exact).
            .flag("-fno-math-errno");
        if !scalar {
            build.flag("-msimd128").define("B3_CPU_WASM", None);
        }
        build
            .files(box3d_sources(
                &vendor, /* exclude_platform_files */ true,
            ))
            .file("shim/wasm_libc.c")
            .file("shim/sizes.c")
            .file("shim/probe.c");
    } else {
        // Full native library: every .c in vendor/box3d/src, real libc/pthreads.
        build
            .files(box3d_sources(&vendor, false))
            .file("shim/sizes.c")
            .file("shim/probe.c");
    }

    build.compile("box3d");
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed=../../vendor/box3d/src");
    println!("cargo:rerun-if-changed=../../vendor/box3d/include");
    println!("cargo:rerun-if-env-changed={CC_ENV}");
    println!("cargo:rerun-if-env-changed=BOX3D_WASM_SCALAR");
}

/// Every Box3D C source. With `exclude_platform_files` (wasm), `timer.c`
/// (pthreads/OS clocks/mutexes) and `scheduler.c` (internal thread pool) are
/// dropped — their referenced symbols come from Rust stubs in
/// `src/wasm_shim.rs`, and the task system is supplied via `b3WorldDef`.
fn box3d_sources(vendor: &Path, exclude_platform_files: bool) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(vendor.join("src"))
        .expect("box3d-sys: vendor/box3d/src missing — run `git submodule update --init`")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "c"))
        .filter(|p| {
            !exclude_platform_files
                || !matches!(
                    p.file_name().and_then(|n| n.to_str()),
                    Some("timer.c") | Some("scheduler.c")
                )
        })
        .collect();
    files.sort();
    files
}

/// Find a clang that can emit wasm32 objects. Order: explicit env override,
/// then known Homebrew locations, then PATH. Fails the build with an
/// actionable message if none qualifies.
fn wasm_clang() -> String {
    if let Ok(cc) = env::var(CC_ENV) {
        if !cc.is_empty() {
            return cc; // explicit override wins, unvalidated on purpose
        }
    }
    for cand in CLANG_CANDIDATES {
        if *cand != "clang" && !Path::new(cand).exists() {
            continue;
        }
        if supports_wasm(cand) {
            return (*cand).to_string();
        }
    }
    panic!(
        "box3d-sys: no wasm-capable clang found (Apple clang has no wasm backend). \
         Install one with `brew install llvm` (macOS) or `apt install clang` (Linux), \
         or point {CC_ENV} at one."
    );
}

/// `clang --print-targets` lists `wasm32` only when the backend is compiled in.
fn supports_wasm(cc: &str) -> bool {
    Command::new(cc)
        .arg("--print-targets")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("wasm32"))
        .unwrap_or(false)
}
