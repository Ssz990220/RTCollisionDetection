# New-stack benchmarks: rtcd_bench_discrete (old RTCD/Benchmark/Discrete: benchmarkShelf / ...ShelfSimple /
# ...DenseShelf) and rtcd_bench_curve (old RTCD/Benchmark/Curve: BM_CURVE_Shelf / ...ShelfSimple / ...DenseShelf),
# timing CollisionChecker::checkDevice. One binary per benchmark kind; the scene is a runtime option (--scene).
# Same benchmark names and argument ranges as the old executables (bench_common.h). Run by benchmark.sh /
# benchmark.bat; smoke-tested in tests/apps.
#
# The executables need the CollisionChecker (src/detector + the three backends). In a tree where that is not
# implemented (RTCD_HAVE_CHECKER false) only the compile check rtcd_bench_compile is built.
rtcd_new_stack_status()

add_library(rtcd_bench_compile OBJECT bench_common.cpp bench_discrete.cpp bench_curve.cpp)
target_link_libraries(rtcd_bench_compile PRIVATE rtcd rtcd_testsupport benchmark::benchmark)
set_target_properties(rtcd_bench_compile PROPERTIES FOLDER "rtcd/apps")
rtcd_register_new_target(rtcd_bench_compile)

if(RTCD_HAVE_CHECKER)
  add_library(rtcd_bench_common STATIC bench_common.cpp bench_common.h)
  target_link_libraries(rtcd_bench_common PUBLIC rtcd rtcd_testsupport benchmark::benchmark)
  set_target_properties(rtcd_bench_common PROPERTIES FOLDER "rtcd/apps")
  rtcd_register_new_target(rtcd_bench_common)
  rtcd_add_app(rtcd_bench_discrete SOURCES bench_discrete.cpp LIBS rtcd_bench_common benchmark::benchmark)
  rtcd_add_app(rtcd_bench_curve SOURCES bench_curve.cpp LIBS rtcd_bench_common benchmark::benchmark)
else()
  message(STATUS "rtcd_bench_discrete / rtcd_bench_curve: compile-only (CollisionChecker not implemented in this tree)")
endif()
