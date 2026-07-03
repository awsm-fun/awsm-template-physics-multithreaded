// Minimal libc shim for compiling Box3D to wasm32-unknown-unknown (no sysroot).
//
// printf/snprintf/vsnprintf are REAL (stb_sprintf-backed; printf routes to the
// Rust log hook) because Box3D formats assert/warning messages through them.
// The file API is a stub: fopen always returns NULL and Box3D's dump/record
// paths check for that — there is no filesystem in the browser.
#pragma once

#include <stddef.h>

typedef struct b3dsysFILE FILE;

#define EOF ( -1 )
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

int printf( const char* format, ... );
int snprintf( char* buffer, size_t bufsz, const char* format, ... );
int vsnprintf( char* buffer, size_t bufsz, const char* format, __builtin_va_list vlist );

FILE* fopen( const char* filename, const char* mode );
int fclose( FILE* stream );
int fflush( FILE* stream );
size_t fread( void* buffer, size_t size, size_t count, FILE* stream );
size_t fwrite( const void* buffer, size_t size, size_t count, FILE* stream );
int fseek( FILE* stream, long offset, int origin );
long ftell( FILE* stream );
int fprintf( FILE* stream, const char* format, ... );
int vfprintf( FILE* stream, const char* format, __builtin_va_list vlist );
int fscanf( FILE* stream, const char* format, ... );
char* fgets( char* str, int count, FILE* stream );
