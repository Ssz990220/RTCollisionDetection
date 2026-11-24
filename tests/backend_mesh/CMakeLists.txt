# Tests of rtcd_backend_mesh (Wave 2f owns this folder).

# Backend through the test-support ReferenceExecutor / ReferenceBatchRunner (no detector core): ProgramSet,
# synchronizesHost, golden on every prepare path (masked, compaction, OBB off) with identical hit counts,
# launch == false when nothing survives, launch params, argument errors.
rtcd_add_new_test(test_backend_mesh LABEL new_gpu
  SOURCES test_backend_mesh.cpp mesh_test_fixture.h
  LIBS rtcd_backend_mesh rtcd_testsupport rtcd_json)

# Per-item hit counts equal the stored counts of the old CollisionDetector<MESH> of the IAS regression
# (tests/data/reference/ias_hit_counts.*).
rtcd_add_new_test(test_backend_mesh_reference LABEL new_gpu
  SOURCES test_backend_mesh_reference.cpp mesh_test_fixture.h
  LIBS rtcd_backend_mesh rtcd_testsupport rtcd_json)

# OLD-STACK parity + REFERENCE GENERATOR (deleted with the old stack): per-item hit counts equal the old
# CollisionDetector<MESH> of the IAS regression (side-by-side old half, profile reg_ias); with
# RTCD_WRITE_REFERENCE=<dir> it writes ias_hit_counts.txt / .bin.
if(TARGET rtcd_sbs_old_shelf)
  rtcd_add_new_test(test_backend_mesh_vs_old LABEL new_gpu
    SOURCES test_backend_mesh_vs_old.cpp mesh_test_fixture.h
    LIBS rtcd_backend_mesh rtcd_testsupport rtcd_json rtcd_sbs_old_shelf)
endif()

# Golden regression variants of the IAS setup (the default run is the automatic rtcd_regression_ias of
# tests/apps): every prepare path of the mesh backend. Engine auto: the reference runner until the
# CollisionChecker exists, the checker afterwards.
if(TARGET rtcd_regression)
  set(rtcd_mesh_reg_variants
    "obb_off|obb_culling=false"
    "always_compact|mesh.compaction_min_poses=0"
    "never_compact|mesh.compaction_min_poses=100000"
    "batch37|max_batch_size=37,mesh.compaction_min_poses=20,streams=3")
  foreach(v IN LISTS rtcd_mesh_reg_variants)
    string(REPLACE "|" ";" parts "${v}")
    list(GET parts 0 name)
    list(GET parts 1 ov)
    add_test(NAME rtcd_regression_ias_${name}
      COMMAND rtcd_regression --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_ias.json --override ${ov}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(rtcd_regression_ias_${name} PROPERTIES LABELS new_gpu)
  endforeach()
endif()
