# rtcd_detector (Wave 2e): include/rtcd/checker.h -- the CollisionChecker facade and the mode-agnostic detector
# core: OptiX module / program groups / pipeline / SBT from a backend's ProgramSet, the batch loop over
# LaunchPlans, host dispatch, the param-upload cache, stats. Switches on the mode once, at construction, to call
# createMeshBackend / createRayBackend / createCurveBackend.
#
# Object-file split (this is a STATIC library; the linker pulls only the objects whose symbols are needed):
#   detector_core.cpp, optix_pipeline.cpp, stream_workers.cpp   the core; takes its backend factory as a parameter
#   collision_checker.cpp                                        the facade; references createBackendForMode
#   backend_factory.cpp                                          createBackendForMode = the ONLY references to the
#                                                                three backend factories
# so tests of the core with a fake backend (tests/detector) link without the backend libraries' factories.
add_library(rtcd_detector STATIC
  detector_core.h
  detector_core.cpp
  detector_kernels.h
  detector_kernels.cu
  phase_trace.h
  optix_pipeline.h
  optix_pipeline.cpp
  stream_workers.h
  stream_workers.cpp
  backend_factory.h
  backend_factory.cpp
  collision_checker.cpp
)
target_include_directories(rtcd_detector
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_detector
  PUBLIC  rtcd_config rtcd_robot rtcd_scene
  PRIVATE rtcd_backend_mesh rtcd_backend_ray rtcd_backend_curve)
# optix_stubs.h pulls in windows.h: keep its min / max macros out of the way.
target_compile_definitions(rtcd_detector PRIVATE NOMINMAX)
set_target_properties(rtcd_detector PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_detector)
