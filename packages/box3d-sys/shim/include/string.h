// Minimal libc shim for compiling Box3D to wasm32-unknown-unknown (no sysroot).
// Declarations only — mem* are provided by Rust's compiler-builtins at link
// time; str* live in shim/wasm_libc.c.
#pragma once

#include <stddef.h>

void* memset( void* dest, int ch, size_t count );
void* memcpy( void* dest, const void* src, size_t count );
void* memmove( void* dest, const void* src, size_t count );
int memcmp( const void* lhs, const void* rhs, size_t count );

size_t strlen( const char* str );
int strcmp( const char* lhs, const char* rhs );
char* strncpy( char* dest, const char* src, size_t count );
