// The C half of the wasm libc shim (wasm32-unknown-unknown only).
//
// Real pieces: vsnprintf/snprintf/printf via the vendored stb_sprintf (public
// domain) — Box3D formats its assert + warning messages through these, and
// printf routes the formatted text to the Rust log hook (`b3dsys_shim_log`).
// qsort is real too: sensor.c sorts overlap results with it (functional, not
// diagnostic). Everything file-shaped is a stub — fopen returns NULL and
// Box3D's dump/recording paths check for that.
//
// mem* are NOT defined here: Rust's compiler-builtins provide them at link
// time. malloc/aligned_alloc/free are in Rust (src/wasm_shim.rs), backed by
// the Rust global allocator.

#define STB_SPRINTF_IMPLEMENTATION
#include "stb_sprintf.h"

#include <stddef.h>
#include <stdio.h>

// Implemented in Rust (box3d-sys wasm_shim) — receives NUL-terminated text.
extern void b3dsys_shim_log( const char* message );

// ── printf family ────────────────────────────────────────────────────────────

int vsnprintf( char* buffer, size_t bufsz, const char* format, __builtin_va_list vlist )
{
	return stbsp_vsnprintf( buffer, (int)bufsz, format, vlist );
}

int snprintf( char* buffer, size_t bufsz, const char* format, ... )
{
	__builtin_va_list args;
	__builtin_va_start( args, format );
	int n = stbsp_vsnprintf( buffer, (int)bufsz, format, args );
	__builtin_va_end( args );
	return n;
}

int printf( const char* format, ... )
{
	char buffer[512];
	__builtin_va_list args;
	__builtin_va_start( args, format );
	int n = stbsp_vsnprintf( buffer, (int)sizeof( buffer ), format, args );
	__builtin_va_end( args );
	b3dsys_shim_log( buffer );
	return n;
}

// ── string.h leftovers (mem* come from compiler-builtins) ────────────────────

size_t strlen( const char* str )
{
	const char* s = str;
	while ( *s )
	{
		++s;
	}
	return (size_t)( s - str );
}

int strcmp( const char* lhs, const char* rhs )
{
	while ( *lhs && *lhs == *rhs )
	{
		++lhs;
		++rhs;
	}
	return (int)(unsigned char)*lhs - (int)(unsigned char)*rhs;
}

char* strncpy( char* dest, const char* src, size_t count )
{
	size_t i = 0;
	for ( ; i < count && src[i]; ++i )
	{
		dest[i] = src[i];
	}
	for ( ; i < count; ++i )
	{
		dest[i] = 0;
	}
	return dest;
}

// ── qsort (shell sort — small arrays, no recursion, no allocation) ──────────

static void b3dsys_swap_bytes( char* a, char* b, size_t size )
{
	for ( size_t i = 0; i < size; ++i )
	{
		char t = a[i];
		a[i] = b[i];
		b[i] = t;
	}
}

void qsort( void* base, size_t count, size_t size, int ( *compare )( const void*, const void* ) )
{
	if ( base == NULL || count < 2 || size == 0 || compare == NULL )
	{
		return;
	}
	char* items = (char*)base;
	// Ciura gap sequence, extended by *2.25.
	static const size_t gaps[] = { 990202, 440086, 195594, 86927, 38633, 17171, 7631, 3392, 1750, 701, 301, 132, 57, 23, 10, 4, 1 };
	for ( int g = 0; g < (int)( sizeof( gaps ) / sizeof( gaps[0] ) ); ++g )
	{
		size_t gap = gaps[g];
		if ( gap >= count )
		{
			continue;
		}
		for ( size_t i = gap; i < count; ++i )
		{
			for ( size_t j = i; j >= gap && compare( items + ( j - gap ) * size, items + j * size ) > 0; j -= gap )
			{
				b3dsys_swap_bytes( items + ( j - gap ) * size, items + j * size, size );
			}
		}
	}
}

// ── stdio file API — stubs (no filesystem; fopen == NULL and callers bail) ──

FILE* fopen( const char* filename, const char* mode )
{
	(void)filename;
	(void)mode;
	return NULL;
}

int fclose( FILE* stream )
{
	(void)stream;
	return EOF;
}

int fflush( FILE* stream )
{
	(void)stream;
	return 0;
}

size_t fread( void* buffer, size_t size, size_t count, FILE* stream )
{
	(void)buffer;
	(void)size;
	(void)count;
	(void)stream;
	return 0;
}

size_t fwrite( const void* buffer, size_t size, size_t count, FILE* stream )
{
	(void)buffer;
	(void)size;
	(void)count;
	(void)stream;
	return 0;
}

int fseek( FILE* stream, long offset, int origin )
{
	(void)stream;
	(void)offset;
	(void)origin;
	return -1;
}

long ftell( FILE* stream )
{
	(void)stream;
	return -1;
}

int fprintf( FILE* stream, const char* format, ... )
{
	(void)stream;
	(void)format;
	return 0;
}

int vfprintf( FILE* stream, const char* format, __builtin_va_list vlist )
{
	(void)stream;
	(void)format;
	(void)vlist;
	return 0;
}

int fscanf( FILE* stream, const char* format, ... )
{
	(void)stream;
	(void)format;
	return EOF;
}

char* fgets( char* str, int count, FILE* stream )
{
	(void)str;
	(void)count;
	(void)stream;
	return NULL;
}

// ── stdlib odds and ends ─────────────────────────────────────────────────────

_Noreturn void exit( int status )
{
	(void)status;
	__builtin_trap();
}

_Noreturn void abort( void )
{
	__builtin_trap();
}
