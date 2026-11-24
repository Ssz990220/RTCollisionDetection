# rtcd_geometry: mesh loading (tiny_obj_loader), OBB, edges.
add_library(rtcd_geometry STATIC
  placeholder.cpp
)
target_include_directories(rtcd_geometry
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_geometry
  PUBLIC  rtcd_core
  PRIVATE rtcd_tinyobjloader Eigen3::Eigen)
set_target_properties(rtcd_geometry PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_geometry)
