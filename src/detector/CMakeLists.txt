# rtcd_detector (Wave 2e): include/rtcd/checker.h -- the CollisionChecker facade and the mode-agnostic detector
# core: OptiX module / program groups / pipeline / SBT from a backend's ProgramSet, the batch loop over
# LaunchPlans, host dispatch, the param-upload cache, stats. Switches on the mode once, at construction, to call
# createMeshBackend / createRayBackend / createCurveBackend.
add_library(rtcd_detector STATIC
  placeholder.cpp
)
target_include_directories(rtcd_detector
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_detector
  PUBLIC  rtcd_config rtcd_robot rtcd_scene
  PRIVATE rtcd_backend_mesh rtcd_backend_ray rtcd_backend_curve)
set_target_properties(rtcd_detector PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_detector)
