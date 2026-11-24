# Tests of rtcd_geometry (Wave 1 c owns this folder).
#
# test_geometry_unit       no old stack: sphere-file rules, edge construction, loader errors, OBB basics.
# test_geometry_reference  every Panda link mesh / sphere file and every obstacle of the three shelf scenes vs the
#                          stored summaries of the old loader (tests/data/reference/meshes.txt).
# test_geometry_parity     compares with the old loader (RTCD/Meshes/mesh.h, Obb.h) on the same meshes and, with
#                          RTCD_WRITE_REFERENCE=<dir>, writes meshes.txt. Uses OLD-STACK headers (target `meshes` and
#                          the old directory include paths): delete it together with the old stack.
rtcd_add_new_test(test_geometry_unit LABEL new_unit
  SOURCES test_geometry_unit.cpp
  LIBS rtcd_geometry)

rtcd_add_new_test(test_geometry_reference LABEL new_unit
  SOURCES test_geometry_reference.cpp
  LIBS rtcd_geometry rtcd_json rtcd_testsupport urdfdom::urdf_parser urdfdom::urdfdom_model)

rtcd_add_new_test(test_geometry_parity LABEL new_unit
  SOURCES test_geometry_parity.cpp
  LIBS rtcd_geometry rtcd_testsupport meshes)
target_include_directories(test_geometry_parity PRIVATE
  ${PROJECT_SOURCE_DIR}/RTCD ${PROJECT_SOURCE_DIR}/3rdparty ${CMAKE_BINARY_DIR})
