# New-stack modules. Dependency direction (a module may only depend on modules above it):
#
#   rtcd_core      context, streams, device buffers, errors, CUDA algorithms
#   rtcd_geometry  mesh loading, OBB, edges                        -> core
#   rtcd_robot     robot model (URDF), kinematics kernels           -> geometry
#   rtcd_scene     obstacles, scene GAS, edge rays                  -> geometry
#   rtcd_programs  OptiX IR device programs (embedded)              (OptiX headers, include/ only)
#   rtcd_detect    backends, pipeline, CollisionChecker             -> robot, scene, programs
#   rtcd_config    JSON -> config structs                           (include/ + nlohmann-json only)
#
# Every module target gets include/ on its PUBLIC include path; private headers stay in src/<module>/ and are
# reached through the module's PRIVATE include path (src/). No global include_directories, no CU_SRC-style
# source lists: a .cu belongs to exactly one module and is compiled once.
# Placeholder sources (placeholder.cpp / placeholder.cu) only exist so the skeleton builds; delete a module's
# placeholder when the first real source is added.

add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(robot)
add_subdirectory(scene)
add_subdirectory(programs)
add_subdirectory(detect)
add_subdirectory(config)

# Convenience target for apps and tests: the whole public library.
add_library(rtcd INTERFACE)
target_link_libraries(rtcd INTERFACE rtcd_detect rtcd_config)
