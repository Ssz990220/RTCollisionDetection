# apps/

Executables of the new stack: benchmarks, demos and tools that use the public `CollisionChecker` API from
`include/rtcd/` (link the `rtcd` target). Each app, or group of related apps, lives in its own folder with a
`CMakeLists.txt` that calls `rtcd_add_app(<name> SOURCES ... LIBS rtcd)` (see `cmake/RTCDTargets.cmake`); folders
are picked up automatically at configure time and binaries go to `build/bin/apps`. Apps take their checker and
scene settings from JSON files under `configs/` instead of compile-time definitions; the old-stack benchmarks
stay under `RTCD/Benchmark/` until the new stack reaches parity.
