// Struct-size ground truth for the hand-written Rust FFI mirrors.
//
// Each function returns the C compiler's sizeof for a type mirrored in
// src/lib.rs; the Rust tests assert `size_of::<T>()` matches. This catches
// field-order/padding/pointer-width drift on EVERY target we compile for
// (wasm32 has 4-byte pointers, hosts have 8 — the mirrors use Rust pointer
// types precisely so both agree).

#include "box3d/box3d.h"

#include <stdint.h>

#define B3DSYS_SIZEOF( T )                                                                                                       \
	int32_t b3dsys_sizeof_##T( void )                                                                                            \
	{                                                                                                                            \
		return (int32_t)sizeof( T );                                                                                             \
	}

B3DSYS_SIZEOF( b3Vec3 )
B3DSYS_SIZEOF( b3Quat )
B3DSYS_SIZEOF( b3Transform )
B3DSYS_SIZEOF( b3AABB )
B3DSYS_SIZEOF( b3Plane )
B3DSYS_SIZEOF( b3WorldId )
B3DSYS_SIZEOF( b3BodyId )
B3DSYS_SIZEOF( b3ShapeId )
B3DSYS_SIZEOF( b3Capacity )
B3DSYS_SIZEOF( b3WorldDef )
B3DSYS_SIZEOF( b3MotionLocks )
B3DSYS_SIZEOF( b3BodyDef )
B3DSYS_SIZEOF( b3Filter )
B3DSYS_SIZEOF( b3SurfaceMaterial )
B3DSYS_SIZEOF( b3ShapeDef )
B3DSYS_SIZEOF( b3Sphere )
B3DSYS_SIZEOF( b3Capsule )
B3DSYS_SIZEOF( b3HullData )
B3DSYS_SIZEOF( b3BoxHull )
B3DSYS_SIZEOF( b3ContactEvents )
B3DSYS_SIZEOF( b3ContactBeginTouchEvent )
B3DSYS_SIZEOF( b3ContactEndTouchEvent )
B3DSYS_SIZEOF( b3ContactHitEvent )
