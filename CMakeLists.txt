cmake_minimum_required(VERSION 3.25)

project(RTCollisionChecker VERSION 1.0.1 LANGUAGES CXX CUDA C)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_VERBOSE_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")


find_package(CUDAToolkit 12.6 REQUIRED) # Need 12.6 or above to support CUB 2.5
find_package(OptiX REQUIRED)
find_package(urdfdom CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(benchmark REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
enable_testing()
find_package(GTest REQUIRED)

include(configCompilerFlags)
include(configure_tiny_obj_loader)
include(RTCDTargets) # rtcd_add_optix_ir, rtcd_add_new_test, rtcd_add_app, ...

set(BUILD_TESTS OFF CACHE BOOL "Enable or disable building tests")

# include/rtcd (public headers), src/<module> (libraries), apps/, tests/. The old header-only stack under RTCD/ was
# removed; its last commit is tag old-stack-final.
add_subdirectory(src)
# rtcd_testsupport (tests/support) is used by the tests AND by apps (regression app, benchmarks), so it is
# configured even with BUILD_TESTS=OFF; tests/CMakeLists.txt excludes it from its own folder scan.
add_subdirectory(tests/support)
add_subdirectory(apps)
if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

# AllNew: every target of the tree (every target registered with rtcd_register_new_target).
get_property(RTCD_NEW_TARGETS GLOBAL PROPERTY RTCD_NEW_TARGETS)
add_custom_target(AllNew)
if (RTCD_NEW_TARGETS)
  add_dependencies(AllNew ${RTCD_NEW_TARGETS})
endif()
