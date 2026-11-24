// The one translation unit of rtcd_geometry that compiles the tinyobjloader implementation.
//
// Kept in its own object file on purpose: with the implementation in a separate object of the static library, the
// linker only pulls this object when nothing else defines tinyobj::LoadObj, so an executable that compiles its own
// implementation (as the parity tests with the removed old stack's RTCD/Meshes/mesh.h did) links without duplicate
// symbols.
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
