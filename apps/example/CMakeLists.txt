# rtcd_example: minimal end-to-end use of the public API (example.cpp; quoted in README.md, so the README snippet
# is always code that builds). Links only the umbrella target `rtcd`.
rtcd_add_app(rtcd_example SOURCES example.cpp LIBS rtcd)

# Smoke tests (label new_gpu, configured with BUILD_TESTS=ON like every new-stack test): the example runs on each
# regression setup and exits 0. Only the exit code is checked (a library error makes the example return 1); the
# flags themselves are covered by the golden regression (tests/apps).
if(BUILD_TESTS)
  foreach(mode ias ray linear quad)
    add_test(NAME rtcd_example_${mode}
      COMMAND rtcd_example ${PROJECT_SOURCE_DIR}/configs/setups/regression_${mode}.json
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(rtcd_example_${mode} PROPERTIES LABELS new_gpu)
  endforeach()
endif()

