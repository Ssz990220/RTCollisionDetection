# rtcd_backend_curve (Wave 2h): createCurveBackend (linear / quadratic / cubic swept-sphere curves,
# curve kernels, refit). Declared in include/rtcd/backend.h; section 5 of docs/design/interfaces.md.
#   curve_backend.cpp   the Backend (build input, GAS build / refit, OBB culling path, ProgramSet)
#   curve_kernels.cu/.h private kernels: fused point generation, sphere centers, control points, segment OBBs,
#                       segment compaction (ports of RTCD/Robot/robot.cu and sweptVolume.cu, bit-identical)
#   curve_programs.cu   OptiX programs (__raygen__oneside, __anyhit__GAS), compiled to OptiX IR and embedded
# curve_kernels.cu is compiled with the project's default CUDA flags (no fast math, default FMA contraction), like
# the old rtcd_legacy_kernels, so the curve points stay bit-identical to the old stack.
rtcd_add_optix_ir(rtcd_backend_curve_ir
  SOURCES curve_programs.cu)

add_library(rtcd_backend_curve STATIC
  curve_kernels.h
  curve_kernels.cu
  curve_backend.cpp
)
target_include_directories(rtcd_backend_curve
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_backend_curve
  PUBLIC  rtcd_robot rtcd_scene rtcd_config
  PRIVATE rtcd_backend_curve_ir)
# optix_stubs.h pulls in windows.h; keep its min / max macros out of the way.
target_compile_definitions(rtcd_backend_curve PRIVATE NOMINMAX)
set_target_properties(rtcd_backend_curve PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_backend_curve)
