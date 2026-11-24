# rtcd_scene (Wave 1 c): include/rtcd/scene.h -- Scene: per-obstacle triangle GAS + IAS, both edge-ray sets,
# obstacle OBBs (built with rtcd_geometry and internal/accel.h of rtcd_core).
add_library(rtcd_scene STATIC
  scene.cpp
)
target_include_directories(rtcd_scene
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_scene
  PUBLIC  rtcd_geometry)
set_target_properties(rtcd_scene PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_scene)
