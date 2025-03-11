cmake_minimum_required(VERSION 3.25)

project(RTCollisionChecker VERSION 1.0.1 LANGUAGES CXX CUDA C)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_VERBOSE_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CUDA_OBJ_VERBOSE OFF CACHE BOOL "Enable verbose output from the OptiX IR compiler")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")


find_package(CUDAToolkit 12.6 REQUIRED) # Need 12.6 or above to support CUB 2.5
find_package(OptiX REQUIRED)
find_package(OpenGL REQUIRED)
find_package(urdfdom CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(benchmark REQUIRED)
enable_testing()
find_package(GTest REQUIRED)

include(configCompilerFlags)
include(configure_tiny_obj_loader)
include(CUDA_Macros)

include_directories(${CUDA_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_BINARY_DIR})

set(PROJECT_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")

if( WIN32 )
  string(REPLACE "/" "\\\\" PROJECT_PTX_DIR ${PROJECT_PTX_DIR})
endif( WIN32 )

configure_file(config.h.in config.h @ONLY)

set(BUILD_TESTS OFF CACHE BOOL "Enable or disable building tests")

add_subdirectory(RTCD)
