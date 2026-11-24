# rtcd_testsupport (Wave 1 e): include/rtcd/internal/reference_executor.h (ReferenceExecutor,
# ReferenceBatchRunner) plus test / benchmark utilities: cached inputs from data/motions exactly as
# RTCD/CollisionDetector/Test/ccRegression.cpp loads them, golden read / compare, flag-diff printing.
# Utility headers live here and are included as <support/...> (tests/ is on the PUBLIC include path).
# Added from the top-level CMakeLists.txt (also used by apps/), not by the tests/ folder scan.
add_library(rtcd_testsupport STATIC
  placeholder.cpp
)
target_include_directories(rtcd_testsupport
  PUBLIC  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_testsupport
  PUBLIC  rtcd_core rtcd_config rtcd_robot rtcd_scene)
set_target_properties(rtcd_testsupport PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_testsupport)
