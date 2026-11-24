# New-stack libraries (docs/design/interfaces.md, section 2). Dependency direction (a library may only depend on
# libraries above it; wave / agent in brackets):
#
#   rtcd_core           errors, Logger, device buffers, Context, internal algorithms / accel   [1a]  CUDA, OptiX
#   rtcd_config         CheckerConfig validate / resolved                                      [1b]  (nothing)
#   rtcd_json           JSON <-> config / scene / robot descriptions                            [1b]  -> config, nlohmann-json
#   rtcd_geometry       OBJ mesh loading, edges, OBB fit, sphere files                          [1c]  -> core
#   rtcd_scene          Scene: obstacle GASes + IAS, edge rays, obstacle OBBs                   [1c]  -> geometry
#   rtcd_robot          RobotModel (URDF), kinematics kernels                                   [1d]  -> geometry
#   rtcd_backend_mesh   createMeshBackend + embedded OptiX IR                                   [2f]  -> robot, scene
#   rtcd_backend_ray    createRayBackend + embedded OptiX IR                                    [2g]  -> robot, scene
#   rtcd_backend_curve  createCurveBackend + embedded OptiX IR, curve kernels                   [2h]  -> robot, scene
#   rtcd_detector       CollisionChecker: pipeline, SBT, batch loop, dispatch                   [2e]  -> config, backends
#   (rtcd_testsupport   tests/support: reference executor, cached inputs / goldens              [1e]  -> robot, scene)
#
# Every library gets include/ on its PUBLIC include path; private headers stay in src/<module>/ and are reached
# through the PRIVATE include path (src/). No global include_directories, no CU_SRC-style source lists: a .cu
# belongs to exactly one library and is compiled once. Each backend embeds its own OptiX programs with
# rtcd_add_optix_ir (cmake/RTCDTargets.cmake); there is no shared programs library.
# Every library above has real sources; the skeleton-stage placeholder sources (placeholder.cpp / placeholder.cu) are gone.
# A new library starts with at least one real source file.

add_subdirectory(core)
add_subdirectory(config)
add_subdirectory(json)
add_subdirectory(geometry)
add_subdirectory(scene)
add_subdirectory(robot)
add_subdirectory(backend_mesh)
add_subdirectory(backend_ray)
add_subdirectory(backend_curve)
add_subdirectory(detector)

# Umbrella target for apps and tests: the whole public library.
add_library(rtcd INTERFACE)
target_link_libraries(rtcd INTERFACE
  rtcd_detector rtcd_backend_mesh rtcd_backend_ray rtcd_backend_curve rtcd_config rtcd_json)
