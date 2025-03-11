if (TARGET OptiX::OptiX)
  return()
endif()

if (DEFINED ENV{OptiX_ROOT_DIR})
  set(OptiX_ROOT_DIR $ENV{OptiX_ROOT_DIR})
  message(STATUS "Using OptiX_ROOT_DIR from environment variable: ${OptiX_ROOT_DIR}")
endif()

if (OptiX_INSTALL_DIR)
  message(STATUS "Detected the OptiX_INSTALL_DIR variable (pointing to ${OptiX_INSTALL_DIR}; going to use this for finding optix.h")
  find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS ${OptiX_INSTALL_DIR})
elseif (DEFINED ENV{OptiX_INSTALL_DIR})
  message(STATUS "Detected the OptiX_INSTALL_DIR env variable (pointing to $ENV{OptiX_INSTALL_DIR}; going to use this for finding optix.h")
  find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS $ENV{OptiX_INSTALL_DIR})
else()
  if (WIN32)
    # Try to locate any OptiX SDK folders in ProgramData
    file(GLOB _optix_candidates "C:/ProgramData/NVIDIA Corporation/OptiX SDK*")
    if (_optix_candidates)
      message(STATUS "Searching for optix.h in ${_optix_candidates}")
      find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS ${_optix_candidates} NO_DEFAULT_PATH)
    endif()
  elseif (UNIX)
    message(WARNING "Please provide the OptiX_INSTALL_DIR or set the environment variable OptiX_INSTALL_DIR.")
  endif()
endif()

message(STATUS "Find Optix at ${OptiX_ROOT_DIR}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
  FOUND_VAR OptiX_FOUND
  REQUIRED_VARS
    OptiX_ROOT_DIR
)

add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE ${OptiX_ROOT_DIR}/include)
