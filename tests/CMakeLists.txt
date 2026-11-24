# Tests of the new stack (only configured with BUILD_TESTS=ON).
#
# ctest label convention:
#   new_unit    new stack, no GPU needed (parsing, validation, host-side logic)
#   new_gpu     new stack, needs a GPU (kernels, OptiX pipelines, collision flags vs. the goldens)
#   regression  the old stack's four golden tests (RTCD/CollisionDetector/CMakeLists.txt; unchanged)
# Run with `ctest -L <label>` (agents: agent_test.ps1 -Label <label>).
#
# One folder per module or topic (e.g. tests/config, tests/detect) with its own CMakeLists.txt calling
#   rtcd_add_new_test(<name> LABEL new_unit|new_gpu SOURCES ... LIBS ...)
# (cmake/RTCDTargets.cmake); folders are picked up automatically at configure time. Tests run from the repository
# root, so the shared data is ./data and configs are ./configs.
rtcd_add_child_directories()
