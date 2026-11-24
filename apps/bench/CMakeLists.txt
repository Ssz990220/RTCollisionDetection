# New-stack benchmarks (Wave 1 e): rtcd_bench_discrete (old RTCD/Benchmark/Discrete) and rtcd_bench_curve (old
# RTCD/Benchmark/Curve), timing CollisionChecker::checkDevice. They need the CollisionChecker (Wave 2): until
# src/detector and the three backends are implemented they are compile-only (rtcd_bench_compile, part of AllNew);
# the executables appear automatically once rtcd_new_stack_status() reports RTCD_HAVE_CHECKER.
rtcd_new_stack_status()

add_library(rtcd_bench_compile OBJECT bench_common.cpp bench_discrete.cpp bench_curve.cpp)
target_link_libraries(rtcd_bench_compile PRIVATE rtcd rtcd_testsupport benchmark::benchmark)
set_target_properties(rtcd_bench_compile PROPERTIES FOLDER "rtcd/apps")
rtcd_register_new_target(rtcd_bench_compile)

if(RTCD_HAVE_CHECKER)
  add_library(rtcd_bench_common STATIC bench_common.cpp bench_common.h)
  target_link_libraries(rtcd_bench_common PUBLIC rtcd rtcd_testsupport)
  set_target_properties(rtcd_bench_common PROPERTIES FOLDER "rtcd/apps")
  rtcd_register_new_target(rtcd_bench_common)
  rtcd_add_app(rtcd_bench_discrete SOURCES bench_discrete.cpp LIBS rtcd_bench_common benchmark::benchmark)
  rtcd_add_app(rtcd_bench_curve SOURCES bench_curve.cpp LIBS rtcd_bench_common benchmark::benchmark)
else()
  message(STATUS "rtcd_bench_discrete / rtcd_bench_curve: compile-only (CollisionChecker not implemented yet)")
endif()
