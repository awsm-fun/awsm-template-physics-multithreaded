// Minimal libc shim for compiling Box3D to wasm32-unknown-unknown (no sysroot).
// clang's builtin inttypes.h unconditionally #include_next's a hosted header,
// so this must fully replace it. Box3D only uses PRIx64; the common 32/64-bit
// print macros are provided for good measure (wasm32: int64_t == long long).
#pragma once

#include <stdint.h>

#define PRId32 "d"
#define PRIu32 "u"
#define PRIx32 "x"
#define PRId64 "lld"
#define PRIu64 "llu"
#define PRIx64 "llx"
#define PRIuPTR "u"
#define PRIxPTR "x"
