//! `box3d-sys` — hand-written FFI for the subset of Box3D this template uses.
//!
//! Every struct here mirrors `vendor/box3d/include/box3d/*.h` field-for-field
//! (`types.h`, `id.h`, `math_functions.h`). Layout drift is the #1 foot-gun of
//! hand-written FFI, so `shim/sizes.c` exports the C compiler's `sizeof` for
//! each mirrored type and the tests below assert equality — run them on the
//! host after any Box3D submodule bump.
//!
//! Conventions:
//! - C names are kept verbatim (`#[allow(non_snake_case)]` etc.) — this is a
//!   `-sys`-style crate; ergonomics live in the game crate.
//! - Def structs MUST be obtained from the `b3Default*Def()` functions (they
//!   carry a validation cookie in `internalValue`), then mutated.
//! - `b3Pos` is `b3Vec3` — this build does not define `BOX3D_DOUBLE_PRECISION`.
//! - Callbacks we never install are typed as opaque `unsafe extern "C" fn()`;
//!   consult `types.h` for their real signatures before ever setting one.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use core::ffi::{c_char, c_void};

#[cfg(target_arch = "wasm32")]
pub mod wasm_shim;

// ─── math_functions.h ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Single-precision build: a world position is just a vector.
pub type b3Pos = b3Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct b3Quat {
    pub v: b3Vec3,
    pub s: f32,
}

impl Default for b3Quat {
    fn default() -> Self {
        b3Quat {
            v: b3Vec3::default(),
            s: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3Transform {
    pub p: b3Vec3,
    pub q: b3Quat,
}

/// Single-precision build: same as [`b3Transform`].
pub type b3WorldTransform = b3Transform;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3Matrix3 {
    pub cx: b3Vec3,
    pub cy: b3Vec3,
    pub cz: b3Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3AABB {
    pub lowerBound: b3Vec3,
    pub upperBound: b3Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3Plane {
    pub normal: b3Vec3,
    pub offset: f32,
}

// ─── id.h ────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3WorldId {
    pub index1: u16,
    pub generation: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3BodyId {
    pub index1: i32,
    pub world0: u16,
    pub generation: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3ShapeId {
    pub index1: i32,
    pub world0: u16,
    pub generation: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3ContactId {
    pub index1: i32,
    pub world0: u16,
    pub padding: i16,
    pub generation: u32,
}

impl b3BodyId {
    pub fn is_null(self) -> bool {
        self.index1 == 0
    }
}

// ─── types.h: callbacks ──────────────────────────────────────────────────────

/// A Box3D task, handed to the user task system by `enqueueTask`.
pub type b3TaskCallback = unsafe extern "C" fn(taskContext: *mut c_void);

/// User task system: enqueue. Return null to signal "executed serially,
/// don't call finishTask"; otherwise the returned pointer is handed back to
/// [`b3FinishTaskCallback`] as `userTask`.
pub type b3EnqueueTaskCallback = unsafe extern "C" fn(
    task: b3TaskCallback,
    taskContext: *mut c_void,
    userContext: *mut c_void,
    taskName: *const c_char,
) -> *mut c_void;

/// User task system: block until the given enqueued task completes. Must
/// help-execute other pending tasks while waiting (see `src/scheduler.c` for
/// the reference behavior) or nested parallel-for phases can deadlock.
pub type b3FinishTaskCallback =
    unsafe extern "C" fn(userTask: *mut c_void, userContext: *mut c_void);

/// Assert override: return 0 to skip the debugger break.
pub type b3AssertFcn =
    unsafe extern "C" fn(condition: *const c_char, fileName: *const c_char, lineNumber: i32) -> i32;

/// Allocator overrides (`b3SetAllocator`). NOTE: the free callback receives no
/// size — a Rust-side allocator must self-describe its allocations.
pub type b3AllocFcn = unsafe extern "C" fn(size: i32, alignment: i32) -> *mut c_void;
pub type b3FreeFcn = unsafe extern "C" fn(mem: *mut c_void);

/// Callbacks this template never installs — opaque on purpose; see `types.h`
/// for the real signatures before setting one.
pub type b3OpaqueCallback = unsafe extern "C" fn();

// ─── types.h: defs ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3Capacity {
    pub staticShapeCount: i32,
    pub dynamicShapeCount: i32,
    pub staticBodyCount: i32,
    pub dynamicBodyCount: i32,
    pub contactCount: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct b3WorldDef {
    pub gravity: b3Vec3,
    pub restitutionThreshold: f32,
    pub hitEventThreshold: f32,
    pub contactHertz: f32,
    pub contactDampingRatio: f32,
    pub contactSpeed: f32,
    pub maximumLinearSpeed: f32,
    pub frictionCallback: Option<b3OpaqueCallback>,
    pub restitutionCallback: Option<b3OpaqueCallback>,
    pub enableSleep: bool,
    pub enableContinuous: bool,
    pub workerCount: u32,
    pub enqueueTask: Option<b3EnqueueTaskCallback>,
    pub finishTask: Option<b3FinishTaskCallback>,
    pub userTaskContext: *mut c_void,
    pub userData: *mut c_void,
    pub createDebugShape: Option<b3OpaqueCallback>,
    pub destroyDebugShape: Option<b3OpaqueCallback>,
    pub userDebugShapeContext: *mut c_void,
    pub capacity: b3Capacity,
    /// DO NOT SET — written by `b3DefaultWorldDef` (validation cookie).
    pub internalValue: i32,
}

/// `b3BodyType` (C enum → int).
pub type b3BodyType = i32;
pub const b3_staticBody: b3BodyType = 0;
pub const b3_kinematicBody: b3BodyType = 1;
pub const b3_dynamicBody: b3BodyType = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3MotionLocks {
    pub linearX: bool,
    pub linearY: bool,
    pub linearZ: bool,
    pub angularX: bool,
    pub angularY: bool,
    pub angularZ: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct b3BodyDef {
    pub r#type: b3BodyType,
    pub position: b3Pos,
    pub rotation: b3Quat,
    pub linearVelocity: b3Vec3,
    pub angularVelocity: b3Vec3,
    pub linearDamping: f32,
    pub angularDamping: f32,
    pub gravityScale: f32,
    pub sleepThreshold: f32,
    pub name: *const c_char,
    pub userData: *mut c_void,
    pub motionLocks: b3MotionLocks,
    pub enableSleep: bool,
    pub isAwake: bool,
    pub isBullet: bool,
    pub isEnabled: bool,
    pub allowFastRotation: bool,
    pub enableContactRecycling: bool,
    /// DO NOT SET — written by `b3DefaultBodyDef` (validation cookie).
    pub internalValue: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3Filter {
    pub categoryBits: u64,
    pub maskBits: u64,
    pub groupIndex: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3SurfaceMaterial {
    pub friction: f32,
    pub restitution: f32,
    pub rollingResistance: f32,
    pub tangentVelocity: b3Vec3,
    pub userMaterialId: u64,
    pub customColor: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct b3ShapeDef {
    pub userData: *mut c_void,
    pub materials: *mut b3SurfaceMaterial,
    pub materialCount: i32,
    pub baseMaterial: b3SurfaceMaterial,
    pub density: f32,
    pub explosionScale: f32,
    pub filter: b3Filter,
    pub enableCustomFiltering: bool,
    pub isSensor: bool,
    pub enableSensorEvents: bool,
    pub enableContactEvents: bool,
    pub enableHitEvents: bool,
    pub enablePreSolveEvents: bool,
    pub invokeContactCreation: bool,
    pub updateBodyMass: bool,
    /// DO NOT SET — written by `b3DefaultShapeDef` (validation cookie).
    pub internalValue: i32,
}

// ─── types.h: shapes ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3Sphere {
    pub center: b3Vec3,
    pub radius: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct b3Capsule {
    pub center1: b3Vec3,
    pub center2: b3Vec3,
    pub radius: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3HullVertex {
    pub edge: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3HullHalfEdge {
    pub next: u8,
    pub twin: u8,
    pub origin: u8,
    pub face: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct b3HullFace {
    pub edge: u8,
}

/// @note Has data hanging off the end (the offsets index past `self`) — never
/// copy a bare `b3HullData`; copy the containing allocation ([`b3BoxHull`] is
/// safe to copy whole, its arrays travel with it).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct b3HullData {
    pub version: u64,
    pub byteCount: i32,
    pub hash: u32,
    pub aabb: b3AABB,
    pub surfaceArea: f32,
    pub volume: f32,
    pub innerRadius: f32,
    pub center: b3Vec3,
    pub centralInertia: b3Matrix3,
    pub vertexCount: i32,
    pub vertexOffset: i32,
    pub pointOffset: i32,
    pub edgeCount: i32,
    pub edgeOffset: i32,
    pub faceCount: i32,
    pub faceOffset: i32,
    pub planeOffset: i32,
    pub padding: i32,
}

/// A box as an embedded convex hull (`b3MakeBoxHull`). Safe to move/copy as a
/// whole; pass `&self.base` to `b3CreateHullShape`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct b3BoxHull {
    pub base: b3HullData,
    pub boxVertices: [b3HullVertex; 8],
    pub boxPoints: [b3Vec3; 8],
    pub boxEdges: [b3HullHalfEdge; 24],
    pub boxFaces: [b3HullFace; 6],
    pub padding: [u8; 2],
    pub boxPlanes: [b3Plane; 6],
}

// ─── types.h: contact events ─────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct b3ContactBeginTouchEvent {
    pub shapeIdA: b3ShapeId,
    pub shapeIdB: b3ShapeId,
    pub contactId: b3ContactId,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct b3ContactEndTouchEvent {
    pub shapeIdA: b3ShapeId,
    pub shapeIdB: b3ShapeId,
    pub contactId: b3ContactId,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct b3ContactHitEvent {
    pub shapeIdA: b3ShapeId,
    pub shapeIdB: b3ShapeId,
    pub contactId: b3ContactId,
    pub point: b3Pos,
    pub normal: b3Vec3,
    /// Approach speed of the two shapes; always positive (m/s).
    pub approachSpeed: f32,
    pub userMaterialIdA: u64,
    pub userMaterialIdB: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct b3ContactEvents {
    pub beginEvents: *mut b3ContactBeginTouchEvent,
    pub endEvents: *mut b3ContactEndTouchEvent,
    pub hitEvents: *mut b3ContactHitEvent,
    pub beginCount: i32,
    pub endCount: i32,
    pub hitCount: i32,
}

// ─── Functions ───────────────────────────────────────────────────────────────

extern "C" {
    // base.h
    pub fn b3SetAllocator(allocFcn: b3AllocFcn, freeFcn: b3FreeFcn);
    pub fn b3GetByteCount() -> i32;
    pub fn b3SetAssertFcn(assertFcn: b3AssertFcn);

    // types.h defaults
    pub fn b3DefaultWorldDef() -> b3WorldDef;
    pub fn b3DefaultBodyDef() -> b3BodyDef;
    pub fn b3DefaultShapeDef() -> b3ShapeDef;
    pub fn b3DefaultFilter() -> b3Filter;
    pub fn b3DefaultSurfaceMaterial() -> b3SurfaceMaterial;

    // box3d.h — world
    pub fn b3CreateWorld(def: *const b3WorldDef) -> b3WorldId;
    pub fn b3DestroyWorld(worldId: b3WorldId);
    pub fn b3World_IsValid(id: b3WorldId) -> bool;
    pub fn b3World_Step(worldId: b3WorldId, timeStep: f32, subStepCount: i32);
    pub fn b3World_GetContactEvents(worldId: b3WorldId) -> b3ContactEvents;

    // box3d.h — body
    pub fn b3CreateBody(worldId: b3WorldId, def: *const b3BodyDef) -> b3BodyId;
    pub fn b3DestroyBody(bodyId: b3BodyId);
    pub fn b3Body_GetPosition(bodyId: b3BodyId) -> b3Pos;
    pub fn b3Body_GetRotation(bodyId: b3BodyId) -> b3Quat;
    pub fn b3Body_GetLinearVelocity(bodyId: b3BodyId) -> b3Vec3;
    pub fn b3Body_SetLinearVelocity(bodyId: b3BodyId, linearVelocity: b3Vec3);
    pub fn b3Body_GetAngularVelocity(bodyId: b3BodyId) -> b3Vec3;
    pub fn b3Body_SetAngularVelocity(bodyId: b3BodyId, angularVelocity: b3Vec3);
    pub fn b3Body_ApplyLinearImpulseToCenter(bodyId: b3BodyId, impulse: b3Vec3, wake: bool);
    pub fn b3Body_GetMass(bodyId: b3BodyId) -> f32;
    pub fn b3Body_SetTransform(bodyId: b3BodyId, position: b3Pos, rotation: b3Quat);

    // box3d.h — shapes
    pub fn b3CreateSphereShape(
        bodyId: b3BodyId,
        def: *const b3ShapeDef,
        sphere: *const b3Sphere,
    ) -> b3ShapeId;
    pub fn b3CreateCapsuleShape(
        bodyId: b3BodyId,
        def: *const b3ShapeDef,
        capsule: *const b3Capsule,
    ) -> b3ShapeId;
    pub fn b3CreateHullShape(
        bodyId: b3BodyId,
        def: *const b3ShapeDef,
        hull: *const b3HullData,
    ) -> b3ShapeId;
    pub fn b3Shape_GetUserData(shapeId: b3ShapeId) -> *mut c_void;

    // collision.h — hull helpers
    pub fn b3MakeBoxHull(hx: f32, hy: f32, hz: f32) -> b3BoxHull;
    pub fn b3MakeCubeHull(halfWidth: f32) -> b3BoxHull;
    /// Heap-allocated (via the installed allocator); destroy with [`b3DestroyHull`].
    pub fn b3CreateCylinder(height: f32, radius: f32, yOffset: f32, sides: i32) -> *mut b3HullData;
    pub fn b3CreateCone(height: f32, radius1: f32, radius2: f32, slices: i32) -> *mut b3HullData;
    pub fn b3DestroyHull(hull: *mut b3HullData);

    // shim/probe.c — toolchain probe, all targets
    pub fn b3dsys_probe(x: i32) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn probe_links_and_runs() {
        assert_eq!(unsafe { b3dsys_probe(1) }, 42);
    }

    // Struct-size ground truth from the C compiler (shim/sizes.c). A mismatch
    // means the Rust mirror drifted from types.h — fix the mirror, never the C.
    macro_rules! size_checks {
        ($($name:ident: $ty:ty),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    extern "C" {
                        fn $name() -> i32;
                    }
                    assert_eq!(
                        unsafe { $name() } as usize,
                        size_of::<$ty>(),
                        concat!("C sizeof vs Rust size_of mismatch for ", stringify!($ty)),
                    );
                }
            )+
        };
    }

    size_checks! {
        b3dsys_sizeof_b3Vec3: b3Vec3,
        b3dsys_sizeof_b3Quat: b3Quat,
        b3dsys_sizeof_b3Transform: b3Transform,
        b3dsys_sizeof_b3AABB: b3AABB,
        b3dsys_sizeof_b3Plane: b3Plane,
        b3dsys_sizeof_b3WorldId: b3WorldId,
        b3dsys_sizeof_b3BodyId: b3BodyId,
        b3dsys_sizeof_b3ShapeId: b3ShapeId,
        b3dsys_sizeof_b3Capacity: b3Capacity,
        b3dsys_sizeof_b3WorldDef: b3WorldDef,
        b3dsys_sizeof_b3MotionLocks: b3MotionLocks,
        b3dsys_sizeof_b3BodyDef: b3BodyDef,
        b3dsys_sizeof_b3Filter: b3Filter,
        b3dsys_sizeof_b3SurfaceMaterial: b3SurfaceMaterial,
        b3dsys_sizeof_b3ShapeDef: b3ShapeDef,
        b3dsys_sizeof_b3Sphere: b3Sphere,
        b3dsys_sizeof_b3Capsule: b3Capsule,
        b3dsys_sizeof_b3HullData: b3HullData,
        b3dsys_sizeof_b3BoxHull: b3BoxHull,
        b3dsys_sizeof_b3ContactEvents: b3ContactEvents,
        b3dsys_sizeof_b3ContactBeginTouchEvent: b3ContactBeginTouchEvent,
        b3dsys_sizeof_b3ContactEndTouchEvent: b3ContactEndTouchEvent,
        b3dsys_sizeof_b3ContactHitEvent: b3ContactHitEvent,
    }

    /// Box3D's world table is a global — creating worlds from two test threads
    /// at once trips an internal assert (SIGTRAP). Worlds themselves are fine;
    /// only cross-thread world *lifecycle* needs serializing.
    static WORLD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Drop a dynamic sphere onto a static box-hull floor and let it settle.
    /// Exercises the def cookies, world/body/shape creation, stepping, and
    /// pose readback — the whole M1 FFI surface — at the given worker count.
    fn falling_sphere(worker_count: u32) {
        let _guard = WORLD_LOCK.lock().unwrap();
        unsafe {
            let mut world_def = b3DefaultWorldDef();
            world_def.gravity = b3Vec3 {
                x: 0.0,
                y: -9.81,
                z: 0.0,
            };
            world_def.workerCount = worker_count;
            let world = b3CreateWorld(&world_def);
            assert!(b3World_IsValid(world), "world creation failed");

            // Static floor: top surface at y = 0.1 (same dims as the scene's table).
            let floor_def = b3DefaultBodyDef();
            let floor = b3CreateBody(world, &floor_def);
            let hull = b3MakeBoxHull(3.0, 0.1, 2.0);
            let floor_shape_def = b3DefaultShapeDef();
            let fs = b3CreateHullShape(floor, &floor_shape_def, &hull.base);
            assert!(fs.index1 != 0, "floor shape creation failed");

            // Dynamic sphere, radius 0.5, dropped from y = 2.
            let mut ball_def = b3DefaultBodyDef();
            ball_def.r#type = b3_dynamicBody;
            ball_def.position = b3Vec3 {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            };
            let ball = b3CreateBody(world, &ball_def);
            let sphere = b3Sphere {
                center: b3Vec3::default(),
                radius: 0.5,
            };
            let ball_shape_def = b3DefaultShapeDef();
            let bs = b3CreateSphereShape(ball, &ball_shape_def, &sphere);
            assert!(bs.index1 != 0, "ball shape creation failed");
            assert!(b3Body_GetMass(ball) > 0.0, "dynamic ball has no mass");

            // 2 simulated seconds: drop (~0.55 s) + settle.
            for _ in 0..120 {
                b3World_Step(world, 1.0 / 60.0, 4);
            }

            let p = b3Body_GetPosition(ball);
            let v = b3Body_GetLinearVelocity(ball);
            // Rest pose: floor top (0.1) + radius (0.5).
            assert!(
                (p.y - 0.6).abs() < 0.05,
                "ball should rest on the floor at y≈0.6, got y={} (v.y={})",
                p.y,
                v.y
            );
            assert!(p.x.abs() < 0.01 && p.z.abs() < 0.01, "ball drifted: {p:?}");

            b3DestroyWorld(world);
        }
    }

    #[test]
    fn falling_sphere_serial() {
        falling_sphere(1);
    }

    /// workerCount > 1 without task callbacks → Box3D's internal pthread
    /// scheduler (host only). Proves the vendored threading code links + runs.
    #[test]
    fn falling_sphere_threaded() {
        falling_sphere(4);
    }
}
