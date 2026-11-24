set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Newer MSVC STL versions reject CUDA toolkits older than the one they were released with
# (error STL1002). Allow the mismatch so CUDA 13.2 can be used with VS 2026 / MSVC 14.52+.
if(MSVC)
  add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>)
  # CUDA 13's CCCL (Thrust/CUB/libcu++) requires the standard-conforming MSVC preprocessor.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()


message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")

if(UNIX)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  # Check if CMAKE_BUILD_TYPE is Debug and add -DDEBUG to the compile flags
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
    # Add any other debug-specific flags here
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
  endif()
endif()

if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DDEBUG")
endif()
