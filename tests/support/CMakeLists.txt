# rtcd_testsupport: include/rtcd/internal/reference_executor.h (ReferenceExecutor, ReferenceBatchRunner) plus test /
# benchmark utilities: the committed reference data of tests/data (cached regression inputs, goldens, stored
# references; tests/data/README.md), golden read / compare, flag-diff printing, legacy env knobs, timing statistics.
# Utility headers live here and are included as <support/...> (tests/ is on the PUBLIC include path). Added from the
# top-level CMakeLists.txt (also used by apps/), not by the tests/ folder scan.
#
# Object-file split matters: tests that only use the host utilities must not pull reference_executor.obj (which
# needs OptiX / rtcd_core at link time), so every utility is its own .cpp.
add_library(rtcd_testsupport STATIC
  paths.cpp
  inputs.cpp
  golden.cpp
  options.cpp
  timing.cpp
  reference.cpp
  reference_executor.cpp
)
target_include_directories(rtcd_testsupport
  PUBLIC  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_testsupport
  PUBLIC  rtcd_core rtcd_config rtcd_robot rtcd_scene)
set_target_properties(rtcd_testsupport PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_testsupport)
