# Tests of rtcd_backend_curve (Wave 2h owns this folder).

# OLD-STACK PARITY: the curve kernels (fused / unfused point generation, control points, segment OBBs) vs the old
# robot.cu / sweptVolume.cu kernels, byte for byte; segment compaction vs a host reference. Includes the backend's
# private header (src/backend_curve/curve_kernels.h) and old-stack headers. Delete the old-stack half with the old
# stack.
rtcd_add_new_test(test_curve_kernels LABEL new_gpu
  SOURCES test_curve_kernels.cpp
  LIBS rtcd_backend_curve rtcd_json rtcd_testsupport rtcd_legacy_kernels)
target_include_directories(test_curve_kernels PRIVATE
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/RTCD ${PROJECT_SOURCE_DIR}/3rdparty ${CMAKE_BINARY_DIR})
target_compile_definitions(test_curve_kernels PRIVATE NOMINMAX)

# createCurveBackend end to end through the ReferenceBatchRunner: ProgramSet, refit policies vs the goldens, OBB
# culling on == off (design doc Q5), cubic determinism, construction errors.
rtcd_add_new_test(test_backend_curve LABEL new_gpu
  SOURCES test_backend_curve.cpp
  LIBS rtcd_backend_curve rtcd_json rtcd_testsupport)
target_compile_definitions(test_backend_curve PRIVATE NOMINMAX)

# Golden gate of this backend through the regression app with the reference engine (no detector needed), in
# addition to the automatic rtcd_regression_* entries (tests/apps): quad twice more per geometry flag (the gate
# wants >= 3 runs: the old quad path once had an intermittent race), and both curve modes with OBB culling on
# (must still match the goldens).
if(TARGET rtcd_regression)
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
endif()
set_tests_properties(test_curve_kernels test_backend_curve PROPERTIES LABELS "new_gpu;backend_curve")
