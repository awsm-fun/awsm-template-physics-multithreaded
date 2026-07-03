// Minimal libc shim for compiling Box3D to wasm32-unknown-unknown (no sysroot).
// Box3D routes its own asserts through B3_ASSERT / b3SetAssertFcn; this is only
// for stray <assert.h> users (e.g. base.h's unknown-compiler fallback).
#pragma once

#ifdef NDEBUG
#define assert( expression ) ( (void)0 )
#else
#define assert( expression ) ( ( expression ) ? (void)0 : __builtin_trap() )
#endif
