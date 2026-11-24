# rtcd_geometry (Wave 1 c): the free functions of include/rtcd/geometry.h -- OBJ mesh loading reproducing the
# old meshModel::loadOBJ (tiny_obj_loader), unique / loop edges, best-fit OBB, sphere-file I/O. Host only.
#
# tinyobj_impl.cpp is the only TU with TINYOBJLOADER_IMPLEMENTATION (separate object, see the comment there).
add_library(rtcd_geometry STATIC
  obj_mesh.cpp
  edges.cpp
  obb_fit.cpp
  spheres.cpp
  tinyobj_impl.cpp
)
target_include_directories(rtcd_geometry
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_geometry
  PUBLIC  rtcd_core
  PRIVATE rtcd_tinyobjloader Eigen3::Eigen)
set_target_properties(rtcd_geometry PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_geometry)
