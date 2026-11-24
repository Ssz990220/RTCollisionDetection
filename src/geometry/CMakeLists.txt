# rtcd_geometry (Wave 1 c): the free functions of include/rtcd/geometry.h -- OBJ mesh loading reproducing the
# old meshModel::loadOBJ (tiny_obj_loader), unique / loop edges, best-fit OBB, sphere-file I/O. Host only.
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
