// The one translation unit of rtcd_geometry that compiles the tinyobjloader implementation.
//
// Kept in its own object file on purpose: the old stack's RTCD/Meshes/mesh.h also defines
// TINYOBJLOADER_IMPLEMENTATION, and the parity tests include it next to rtcd_geometry. With the implementation in a
// separate object of the static library, the linker only pulls this object when nothing else defines tinyobj::LoadObj,
// so a parity test (which defines it through mesh.h) links without duplicate symbols.
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
