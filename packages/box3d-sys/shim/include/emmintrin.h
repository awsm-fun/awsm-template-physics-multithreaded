// SSE2 → wasm simd128 compatibility header for Box3D on wasm32-unknown-unknown.
//
// Box3D's SIMD math (`src/simd.h` / `simd.c`, consumed by the contact solver
// and friends) is written against SSE2. Compiled with `-msimd128
// -DB3_CPU_WASM`, core.h selects that SSE2 path and this header maps the small
// intrinsic surface it actually uses (33 intrinsics, all 4-wide float ops —
// census in BOX3D.md) onto clang's builtin `wasm_simd128.h`. Every mapped op
// is IEEE-exact, so results are bit-identical to the scalar build (Box3D
// deliberately avoids the one approximate SSE op, `_mm_rsqrt_ps`, for exactly
// this determinism — it is intentionally NOT provided here).
//
// Two argument-order traps, handled below — get these wrong and NaN/-0.0
// edge-case behavior diverges from real SSE:
// - `MINPS(a,b)` is `a < b ? a : b` while wasm `pmin(x,y)` is `y < x ? y : x`,
//   so `_mm_min_ps(a,b)` = `wasm_f32x4_pmin(b,a)` (same swap for max);
// - `ANDNPS(a,b)` is `~a & b` while wasm `andnot(x,y)` is `x & ~y`, so
//   `_mm_andnot_ps(a,b)` = `wasm_v128_andnot(b,a)`.
#pragma once

#include <wasm_simd128.h>

// The real SSE `__m128` is a GCC/clang float vector; using the same shape
// makes brace initializers (`static const b3V32 x = {0,0,0,0}`), the b3128
// union overlay, and `__builtin_shufflevector` all work unchanged.
typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
typedef double __m128d __attribute__((__vector_size__(16), __aligned__(16)));

// Bit-reinterpret between our float vector and wasm_simd128.h's v128_t.
#define B3DSYS_V128(a) ((v128_t)(a))
#define B3DSYS_M128(a) ((__m128)(a))

#define _MM_SHUFFLE(fp3, fp2, fp1, fp0) (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))

// ── arithmetic ────────────────────────────────────────────────────────────────

static inline __m128 _mm_add_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_add(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_sub_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_sub(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_mul_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_mul(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_div_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_div(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_sqrt_ps(__m128 a)
{
	return B3DSYS_M128(wasm_f32x4_sqrt(B3DSYS_V128(a)));
}

// MINPS(a,b) = a < b ? a : b (NaN/tie → b). wasm pmin(x,y) = y < x ? y : x —
// the argument SWAP below makes the semantics exactly equal. Same for max.
static inline __m128 _mm_min_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_pmin(B3DSYS_V128(b), B3DSYS_V128(a)));
}

static inline __m128 _mm_max_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_pmax(B3DSYS_V128(b), B3DSYS_V128(a)));
}

// ── bitwise ───────────────────────────────────────────────────────────────────

static inline __m128 _mm_and_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_v128_and(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_or_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_v128_or(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_xor_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_v128_xor(B3DSYS_V128(a), B3DSYS_V128(b)));
}

// ANDNPS(a,b) = ~a & b; wasm andnot(x,y) = x & ~y — hence the swap.
static inline __m128 _mm_andnot_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_v128_andnot(B3DSYS_V128(b), B3DSYS_V128(a)));
}

// ── comparisons (all-ones / all-zeros lane masks, same as SSE) ───────────────

static inline __m128 _mm_cmpeq_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_eq(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_cmplt_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_lt(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_cmple_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_le(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_cmpgt_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_gt(B3DSYS_V128(a), B3DSYS_V128(b)));
}

static inline __m128 _mm_cmpge_ps(__m128 a, __m128 b)
{
	return B3DSYS_M128(wasm_f32x4_ge(B3DSYS_V128(a), B3DSYS_V128(b)));
}

// ── construction / loads / stores ─────────────────────────────────────────────

static inline __m128 _mm_set1_ps(float v)
{
	return B3DSYS_M128(wasm_f32x4_splat(v));
}

#define _mm_set_ps1 _mm_set1_ps

// setr = "reverse" of set = memory order — identical to wasm_f32x4_make.
static inline __m128 _mm_setr_ps(float e0, float e1, float e2, float e3)
{
	return B3DSYS_M128(wasm_f32x4_make(e0, e1, e2, e3));
}

static inline __m128 _mm_setzero_ps(void)
{
	return B3DSYS_M128(wasm_f32x4_const_splat(0.0f));
}

static inline __m128 _mm_load_ps(const float* p)
{
	return B3DSYS_M128(wasm_v128_load(p));
}

static inline void _mm_store_ps(float* p, __m128 a)
{
	wasm_v128_store(p, B3DSYS_V128(a));
}

// Load one float into lane 0, zero the rest.
static inline __m128 _mm_load_ss(const float* p)
{
	return B3DSYS_M128(wasm_v128_load32_zero(p));
}

// Load one double (i.e. 8 bytes = two packed floats in Box3D's usage) into the
// low half, zero the high half. Box3D only uses this via the
// `_mm_castpd_ps(_mm_load_sd(...))` two-floats idiom.
static inline __m128d _mm_load_sd(const double* p)
{
	return (__m128d)(wasm_v128_load64_zero(p));
}

static inline __m128 _mm_castpd_ps(__m128d a)
{
	return (__m128)(a);
}

// ── lane extraction / movement ────────────────────────────────────────────────

static inline float _mm_cvtss_f32(__m128 a)
{
	return wasm_f32x4_extract_lane(B3DSYS_V128(a), 0);
}

// Sign bit of each lane → 4-bit mask (identical to MOVMSKPS).
static inline int _mm_movemask_ps(__m128 a)
{
	return (int)wasm_i32x4_bitmask(B3DSYS_V128(a));
}

// The shuffle family must stay macros: __builtin_shufflevector requires
// compile-time-constant lane indices (every Box3D call site passes literals).
#define _mm_shuffle_ps(a, b, imm)                                                                                                \
	((__m128)__builtin_shufflevector((__m128)(a), (__m128)(b), ((imm) & 0x3), (((imm) >> 2) & 0x3),                              \
									 ((((imm) >> 4) & 0x3) + 4), ((((imm) >> 6) & 0x3) + 4)))

#define _mm_movelh_ps(a, b) ((__m128)__builtin_shufflevector((__m128)(a), (__m128)(b), 0, 1, 4, 5))

#define _mm_unpacklo_ps(a, b) ((__m128)__builtin_shufflevector((__m128)(a), (__m128)(b), 0, 4, 1, 5))

#define _mm_unpackhi_ps(a, b) ((__m128)__builtin_shufflevector((__m128)(a), (__m128)(b), 2, 6, 3, 7))

// ── misc ──────────────────────────────────────────────────────────────────────

// Spin-wait hint; wasm has no pause instruction.
static inline void _mm_pause(void)
{
}
