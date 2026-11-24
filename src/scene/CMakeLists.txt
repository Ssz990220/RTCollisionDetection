# rtcd_scene: obstacles (SceneDesc -> device meshes), scene GAS, edge rays.
add_library(rtcd_scene STATIC
  placeholder.cpp
)
target_include_directories(rtcd_scene
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_scene
  PUBLIC  rtcd_geometry)
set_target_properties(rtcd_scene PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_scene)
