# rtcd_detect: backends (IAS, RAY, LINEAR, QUAD; prepare() -> LaunchPlan), the detector pipeline (one loop over
# plans) and the public CollisionChecker facade.
add_library(rtcd_detect STATIC
  placeholder.cpp
)
target_include_directories(rtcd_detect
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_detect
  PUBLIC  rtcd_robot rtcd_scene
  PRIVATE rtcd_programs)
set_target_properties(rtcd_detect PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_detect)
