# Tests of the apps.

# The golden regression (apps/regression) as ctest entries, one per configs/setups/regression_<mode>.json, plus quad
# with curve.geometry_flags=none (the second quad gate). Labels new_gpu AND regression: `ctest -L regression` runs
# the four goldens (with the ray hit-mode variants of tests/backend_ray), as it did for the old stack's
# ccRegression_<mode> tests.
foreach(mode ias ray linear quad)
  add_test(NAME rtcd_regression_${mode}
    COMMAND rtcd_regression --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_${mode}.json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(rtcd_regression_${mode} PROPERTIES LABELS "new_gpu;regression")
endforeach()
add_test(NAME rtcd_regression_quad_geomflag_none
  COMMAND rtcd_regression --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_quad.json
          --override curve.geometry_flags=none
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(rtcd_regression_quad_geomflag_none PROPERTIES LABELS "new_gpu;regression")

# Smoke tests of the benchmark apps (apps/bench) and the demos (apps/demo): they only check that the apps start,
# run a few cases and exit 0 (label new_gpu). Benchmarks run with --benchmark_dry_run (one iteration per case, no
# warm-up) and write their JSON to the build tree; no timing is checked or meaningful here.
set(rtcd_smoke_out "${CMAKE_CURRENT_BINARY_DIR}/smoke")
file(MAKE_DIRECTORY "${rtcd_smoke_out}")
add_test(NAME rtcd_bench_discrete_smoke
  COMMAND rtcd_bench_discrete --benchmark_dry_run=true "--benchmark_filter=BM_(IAS|RAY)/(1|16|1024)/"
          --benchmark_out=${rtcd_smoke_out}/discrete_shelf.json
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(rtcd_bench_discrete_smoke PROPERTIES LABELS new_gpu)
foreach(scene shelf_simple shelf_dense)
  add_test(NAME rtcd_bench_discrete_smoke_${scene}
    COMMAND rtcd_bench_discrete --scene ${scene} --benchmark_dry_run=true "--benchmark_filter=BM_(IAS|RAY)/16/"
            --benchmark_out=${rtcd_smoke_out}/discrete_${scene}.json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(rtcd_bench_discrete_smoke_${scene} PROPERTIES LABELS new_gpu)
endforeach()
# every family (BM_LINEAR, BM_QUAD, BM_MESH, BM_RAY, all template arguments) at 1 and 16 trajectories
add_test(NAME rtcd_bench_curve_smoke
  COMMAND rtcd_bench_curve --benchmark_dry_run=true "--benchmark_filter=/(1|16)/"
          --benchmark_out=${rtcd_smoke_out}/curve_shelf.json
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(rtcd_bench_curve_smoke PROPERTIES LABELS new_gpu)
add_test(NAME rtcd_bench_curve_smoke_shelf_dense
  COMMAND rtcd_bench_curve --scene ${PROJECT_SOURCE_DIR}/configs/scenes/shelf_dense.json --benchmark_dry_run=true
          "--benchmark_filter=BM_(LINEAR|QUAD).*/16/" --benchmark_out=${rtcd_smoke_out}/curve_shelf_dense.json
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(rtcd_bench_curve_smoke_shelf_dense PROPERTIES LABELS new_gpu)
foreach(mode ias ray)
  add_test(NAME rtcd_demo_discrete_smoke_${mode}
    COMMAND rtcd_demo_discrete --setup demo_${mode} --counts 1,16,1000,4096 --repeats 1
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(rtcd_demo_discrete_smoke_${mode} PROPERTIES LABELS new_gpu)
endforeach()
foreach(mode linear quad)
  add_test(NAME rtcd_demo_curve_smoke_${mode}
    COMMAND rtcd_demo_curve --setup ${PROJECT_SOURCE_DIR}/configs/setups/demo_${mode}.json --counts 1,16,1000,4096
            --repeats 1
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(rtcd_demo_curve_smoke_${mode} PROPERTIES LABELS new_gpu)
endforeach()
# a regression setup works as a demo setup too
add_test(NAME rtcd_demo_curve_smoke_regression_setup
  COMMAND rtcd_demo_curve --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_quad.json --counts 1,4096 --repeats 1
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(rtcd_demo_curve_smoke_regression_setup PROPERTIES LABELS new_gpu)
