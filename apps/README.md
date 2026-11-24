# apps/

Executables: benchmarks, demos and tools that use the public `CollisionChecker` API from `include/rtcd/` (link the
`rtcd` target). Each app, or group of related apps, lives in its own folder with a `CMakeLists.txt` that calls
`rtcd_add_app(<name> SOURCES ... LIBS rtcd)` (see `cmake/RTCDTargets.cmake`); folders are picked up automatically at
configure time and binaries go to `build/bin/apps`. Apps take their checker and scene settings from JSON files under
`configs/` instead of compile-time definitions.

| folder | executables |
|---|---|
| `example/` | `rtcd_example`: the minimal end-to-end use quoted in the top-level README |
| `regression/` | `rtcd_regression`: the golden regression (`configs/setups/regression_<mode>.json`, goldens in `tests/data/CCTest`) |
| `bench/` | `rtcd_bench_discrete`, `rtcd_bench_curve`: Google Benchmark apps, run by `benchmark.sh` / `benchmark.bat` |
| `demo/` | `rtcd_demo_discrete`, `rtcd_demo_curve`: demos on random poses / trajectories (`configs/setups/demo_<mode>.json`) |

The old stack's benchmarks and demos (`RTCD/Benchmark`, `RTCD/Demo`) were ported to `bench/` and `demo/` (same
benchmark names and parameters) and removed with the old stack (last commit: tag `old-stack-final`).
