# New-stack executables (benchmarks, demos, tools). One folder per app (or group of apps) with its own
# CMakeLists.txt, using rtcd_add_app(<name> SOURCES ... LIBS rtcd); folders are picked up automatically at
# configure time.
rtcd_add_child_directories()
