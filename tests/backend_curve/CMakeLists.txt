# Tests of rtcd_backend_curve.

# The curve kernels (fused / unfused point generation, control points, segment OBBs) vs the stored hashes of the
# old robot.cu / sweptVolume.cu kernels' outputs (tests/data/reference/curve_kernels.txt), byte for byte; fused ==
# unfused; segment containment; segment compaction vs a host reference. Includes the backend's private header.
rtcd_add_new_test(test_curve_kernels LABEL new_gpu
  SOURCES test_curve_kernels.cpp
  LIBS rtcd_backend_curve rtcd_json rtcd_testsupport)
target_include_directories(test_curve_kernels PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(test_curve_kernels PRIVATE NOMINMAX)

# createCurveBackend end to end through the ReferenceBatchRunner: ProgramSet, refit policies vs the goldens, OBB
# culling on == off (design doc Q5), cubic determinism, construction errors.
rtcd_add_new_test(test_backend_curve LABEL new_gpu
  SOURCES test_backend_curve.cpp
  LIBS rtcd_backend_curve rtcd_json rtcd_testsupport)
target_compile_definitions(test_backend_curve PRIVATE NOMINMAX)

# Refit policies (auto / always / never / every 3) x item counts, three consecutive checks each: the backend
# deviates from the golden exactly as the old detector did (stored: tests/data/reference/refit_flags.txt).
rtcd_add_new_test(test_refit_reference LABEL new_gpu
  SOURCES test_refit_reference.cpp
  LIBS rtcd_backend_curve rtcd_json rtcd_testsupport)
target_compile_definitions(test_refit_reference PRIVATE NOMINMAX)

# Golden gate of this backend through the regression app with the reference engine (no detector core), in
# addition to the rtcd_regression_* entries (tests/apps): quad twice more per geometry flag (the gate wants >= 3
# runs: the old quad path once had an intermittent race), and both curve modes with OBB culling on (must still match
# the goldens).
set(rtcd_curve_reg rtcd_regression --engine reference)
foreach(run 2 3)
  add_test(NAME backend_curve_regression_quad_run${run}
    COMMAND ${rtcd_curve_reg} --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_quad.json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  add_test(NAME backend_curve_regression_quad_geomflag_none_run${run}
    COMMAND ${rtcd_curve_reg} --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_quad.json
            --override curve.geometry_flags=none
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(backend_curve_regression_quad_run${run} backend_curve_regression_quad_geomflag_none_run${run}
    PROPERTIES LABELS "new_gpu;backend_curve")
endforeach()
foreach(mode linear quad)
  add_test(NAME backend_curve_regression_${mode}_obb
    COMMAND ${rtcd_curve_reg} --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_${mode}.json
            --override obb_culling=true
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(backend_curve_regression_${mode}_obb PROPERTIES LABELS "new_gpu;backend_curve")
endforeach()
set_tests_properties(test_curve_kernels test_backend_curve test_refit_reference PROPERTIES LABELS "new_gpu;backend_curve")
