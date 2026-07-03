// Minimal libc shim for compiling Box3D to wasm32-unknown-unknown (no sysroot).
// malloc/aligned_alloc/free are implemented in Rust (box3d-sys wasm_shim,
// backed by the Rust global allocator); qsort/exit live in shim/wasm_libc.c.
#pragma once

#include <stddef.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

void* malloc( size_t size );
void* aligned_alloc( size_t alignment, size_t size );
void free( void* ptr );

void qsort( void* base, size_t count, size_t size, int ( *compare )( const void*, const void* ) );

_Noreturn void exit( int status );
_Noreturn void abort( void );
