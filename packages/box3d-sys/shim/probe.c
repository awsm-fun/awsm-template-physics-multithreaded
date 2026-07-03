// M0 toolchain probe: proves the cc → clang → wasm32 object chain works with
// the threaded flags before any real Box3D source is in the build. Replaced by
// the real shim (stb_sprintf vsnprintf + stdio stubs) in M2.
int b3dsys_probe( int x )
{
	return x + 41;
}
