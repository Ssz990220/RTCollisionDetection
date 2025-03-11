macro(cuda_compile_and_embed output_var format cuda_file_name)
  set(c_var_name ${output_var})
  
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set( BUILD_TYPE "$(ConfigurationName)" )
  else()
    set( BUILD_TYPE "$<CONFIG>")
  endif()


  # remove .cu from cuda_file
  string(REPLACE ".cu" "" cuda_file ${cuda_file_name})
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" cuda_file ${cuda_file})
  if(CUDA_OBJ_VERBOSE)
    message(STATUS "format: ${format}")
    message(STATUS "Build type: ${BUILD_TYPE}")
    message(STATUS "cuda_file_name: ${cuda_file_name}")
    message(STATUS "cuda_file: ${cuda_file}")
  endif()

  # extract the name of the file by extracting name after the last / symbol if there is one
  string(FIND ${cuda_file} "/" last_slash REVERSE)
  if(last_slash GREATER -1)
      math(EXPR last_slash "${last_slash} + 1")
      string(SUBSTRING ${cuda_file} ${last_slash} -1 lib_name)
  else()
      set(lib_name ${cuda_file})
  endif()
  # compile to ${format} file
  add_library(${lib_name} OBJECT ${cuda_file_name})
  if (${format} STREQUAL "optixir")
    set_target_properties(${lib_name} PROPERTIES FOLDER "OptixIR")
    target_link_libraries(${lib_name} PRIVATE OptiX::OptiX)
  elseif(${format} STREQUAL "ptx")
    set_target_properties(${lib_name} PROPERTIES FOLDER "PTX")
  endif()
  if(CUDA_OBJ_VERBOSE)
    message(STATUS "last_slash: ${last_slash}")
    message(STATUS "lib_name: ${lib_name}")
    message(STATUS "library name: ${lib_name}")
  endif()
  
  # if format is ${format}
  if(${format} STREQUAL "ptx")
    if(CUDA_OBJ_VERBOSE)
      message(STATUS "Build for ptx")
    endif()
    set_target_properties(${lib_name} PROPERTIES CUDA_PTX_COMPILATION_DEBUG ${BUILD_TYPE})
    set_target_properties(${lib_name} PROPERTIES CUDA_PTX_COMPILATION ON)
    set_target_properties(${lib_name} PROPERTIES CUDA_PTX_COMPILATION_OPTIONS -ptx)
  #else if format is optixir
  elseif(${format} STREQUAL "optixir")
    if(CUDA_OBJ_VERBOSE)
      message(STATUS "Build for optixir")
    endif()
    set_target_properties(${lib_name} PROPERTIES CUDA_OPTIX_COMPILATION_DEBUG ${BUILD_TYPE})
    set_target_properties(${lib_name} PROPERTIES CUDA_OPTIX_COMPILATION ON)
    set_target_properties(${lib_name} PROPERTIES CUDA_OPTIX_COMPILATION_OPTIONS -optixir)
  endif()
  target_compile_options(${lib_name} PRIVATE --generate-line-info -use_fast_math --relocatable-device-code=true $<$<CONFIG:Debug>: -G -g>)
  
  # find the name of the source_dir by extracting relative path of current project relavent to the root project
  string(REPLACE "${CMAKE_SOURCE_DIR}/" "" source_dir ${CMAKE_CURRENT_SOURCE_DIR})
  
  if (WIN32)
    set(output ${CMAKE_BINARY_DIR}/${source_dir}/${lib_name}.dir/${BUILD_TYPE}/${lib_name}.${format})
  elseif(UNIX)
    set(output ${CMAKE_BINARY_DIR}/${source_dir}/CMakeFiles/${lib_name}.dir/${cuda_file}.${format})
  endif()
  set(${format}_file ${CMAKE_BINARY_DIR}/lib/ptx/${BUILD_TYPE}/${lib_name}.${format})
  if(CUDA_OBJ_VERBOSE)
    message(STATUS "{CMAKE_CURRENT_SOURCE_DIR}: ${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "source_dir: ${source_dir}")
    message(STATUS "output: ${output}")
    message(STATUS "${format}_file: ${${format}_file}")
  endif()

  add_custom_command(
    OUTPUT ${${format}_file}
    COMMAND ${CMAKE_COMMAND} -E copy ${output} ${${format}_file}
    DEPENDS $<TARGET_OBJECTS:${lib_name}>
    COMMENT "compiling (and embedding ${format} from) ${cuda_file}"
    )
  set(${output_var} ${${format}_file})
endmacro()


macro(RTCD_add_source_groups)
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("OptixIR Files"  REGULAR_EXPRESSION ".+\\.optixir$")
  source_group("CUDA Header Files"  REGULAR_EXPRESSION ".+\\.cuh$")
  source_group("CUDA Source Files"  REGULAR_EXPRESSION ".+\\.cu$")
endmacro()