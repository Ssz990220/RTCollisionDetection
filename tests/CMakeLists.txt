# Tests of the new stack (only configured with BUILD_TESTS=ON; the agent build helper always sets it).
#
# ctest label convention:
#   new_unit    new stack, no GPU needed (parsing, validation, host-side logic)
#   new_gpu     new stack, needs a GPU (kernels, OptiX pipelines, collision flags vs. the goldens)
#   regression  the old stack's four golden tests (RTCD/CollisionDetector/CMakeLists.txt; unchanged)
# Run with `ctest -L <label>` (agents: agent_test.ps1 -Label <label>).
#
# One folder per library or topic (tests/core, tests/config, tests/json, tests/geometry, tests/scene,
# tests/robot, ...) with its own CMakeLists.txt calling
#   rtcd_add_new_test(<name> LABEL new_unit|new_gpu SOURCES ... LIBS ...)
# (cmake/RTCDTargets.cmake); folders are picked up automatically at configure time. tests/support (library
# rtcd_testsupport) is added by the top-level CMakeLists.txt instead. Tests run from the repository root, so the
# reference data is ./tests/data (tests/data/README.md) and configs are ./configs.

# ---- compile-only checks of the public headers (part of AllNew; nothing is linked or run) ----
# "Public headers are plain C++20": these targets see ONLY include/ (no CUDA, no OptiX, no other library) and
# compile with warnings as errors. A header that pulls in a CUDA / OptiX header or a private header fails here.
if(MSVC)
  set(rtcd_header_check_options /W4 /WX /permissive-)
else()
  set(rtcd_header_check_options -Wall -Wextra -Werror)
endif()

# 1. All headers together, plus usage of the declarations (tests/interfaces_compile.cpp).
add_library(rtcd_interfaces_compile OBJECT interfaces_compile.cpp)
target_include_directories(rtcd_interfaces_compile PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(rtcd_interfaces_compile PRIVATE ${rtcd_header_check_options})
set_target_properties(rtcd_interfaces_compile PROPERTIES FOLDER "rtcd/tests")
rtcd_register_new_target(rtcd_interfaces_compile)

# 2. Self-containment: one generated TU per header, including it first and twice (include guard).
file(GLOB_RECURSE rtcd_public_headers CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include/rtcd/*.h")
set(rtcd_selfcontained_sources "")
foreach(header IN LISTS rtcd_public_headers)
  string(MAKE_C_IDENTIFIER "${header}" tu_name)
  set(tu "${CMAKE_CURRENT_BINARY_DIR}/header_selfcontained/${tu_name}.cpp")
  file(CONFIGURE OUTPUT "${tu}" CONTENT "// Generated (tests/CMakeLists.txt): self-containment of <${header}>.\n#include <${header}>\n#include <${header}>\n")
  list(APPEND rtcd_selfcontained_sources "${tu}")
endforeach()
add_library(rtcd_headers_selfcontained OBJECT ${rtcd_selfcontained_sources})
target_include_directories(rtcd_headers_selfcontained PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(rtcd_headers_selfcontained PRIVATE ${rtcd_header_check_options})
set_target_properties(rtcd_headers_selfcontained PROPERTIES FOLDER "rtcd/tests")
rtcd_register_new_target(rtcd_headers_selfcontained)

rtcd_add_child_directories(EXCLUDE support)
