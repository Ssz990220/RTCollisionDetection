# rtcd_core: CUDA/OptiX context, streams, device buffers, errors, CUDA algorithms (scan, compaction, ...).
add_library(rtcd_core STATIC
  core_internal.h
  errors.cpp
  logger.cpp
  optix_function_table.cpp  # alone in its object file on purpose (see the file)
  device_buffer.cpp
  context.cpp
  algorithms.cu             # internal/algorithms.h (compiled without fast math, like the old BBCollisionDetection.cu)
  accel.cpp                 # internal/accel.h
)
target_include_directories(rtcd_core
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
# CUDA runtime/driver and the OptiX headers are PUBLIC: every module above needs them in its private code.
target_link_libraries(rtcd_core PUBLIC
  CUDA::cudart_static
  CUDA::cuda_driver
  OptiX::OptiX)
set_target_properties(rtcd_core PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_core)
