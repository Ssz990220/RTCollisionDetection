# Benchmarks: rtcd_bench_discrete (poses: BM_IAS / BM_RAY) and rtcd_bench_curve (trajectories: BM_LINEAR, BM_QUAD,
# BM_MESH, BM_RAY), timing CollisionChecker::checkDevice. One binary per benchmark kind; the scene is a runtime
# option (--scene). Benchmark names and argument ranges are those of the old stack's executables, so results stay
# comparable (bench_common.h). Run by benchmark.sh / benchmark.bat; smoke-tested in tests/apps.
add_library(rtcd_bench_common STATIC bench_common.cpp bench_common.h)
target_link_libraries(rtcd_bench_common PUBLIC rtcd rtcd_testsupport benchmark::benchmark)
set_target_properties(rtcd_bench_common PROPERTIES FOLDER "rtcd/apps")
rtcd_register_new_target(rtcd_bench_common)
rtcd_add_app(rtcd_bench_discrete SOURCES bench_discrete.cpp LIBS rtcd_bench_common benchmark::benchmark)
rtcd_add_app(rtcd_bench_curve SOURCES bench_curve.cpp LIBS rtcd_bench_common benchmark::benchmark)
