// Minimal libc shim for compiling Box3D to wasm32-unknown-unknown (no sysroot).
//
// sqrtf/floorf/ceilf/fabsf/truncf lower to native wasm instructions (we compile
// with -fno-math-errno so clang uses the intrinsics). The transcendentals have
// no wasm instruction and resolve to Rust `libm` wrappers at link time
// (box3d-sys wasm_shim).
#pragma once

float sqrtf( float x );
float fabsf( float x );
float floorf( float x );
float ceilf( float x );
float truncf( float x );
float sinf( float x );
float cosf( float x );
float tanf( float x );
float asinf( float x );
float acosf( float x );
float atanf( float x );
float atan2f( float y, float x );
float expf( float x );
float logf( float x );
float powf( float base, float exponent );
float fmodf( float x, float y );
float remainderf( float x, float y );
float nextafterf( float from, float to );

#define isnan( x ) __builtin_isnan( x )
#define isinf( x ) __builtin_isinf( x )
#define isfinite( x ) __builtin_isfinite( x )

#define INFINITY ( __builtin_inff() )
#define NAN ( __builtin_nanf( "" ) )
