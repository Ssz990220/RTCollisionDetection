# Tests of rtcd_geometry.
#
# test_geometry_unit       sphere-file rules, edge construction, loader errors, OBB basics.
# test_geometry_reference  every Panda link mesh / sphere file and every obstacle of the three shelf scenes vs the
#                          stored summaries of the old loader (tests/data/reference/meshes.txt).
rtcd_add_new_test(test_geometry_unit LABEL new_unit
  SOURCES test_geometry_unit.cpp
  LIBS rtcd_geometry)

rtcd_add_new_test(test_geometry_reference LABEL new_unit
  SOURCES test_geometry_reference.cpp
  LIBS rtcd_geometry rtcd_json rtcd_testsupport urdfdom::urdf_parser urdfdom::urdfdom_model)
