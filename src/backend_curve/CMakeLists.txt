# rtcd_backend_curve (Wave 2h): createCurveBackend (linear / quadratic / cubic swept-sphere curves,
# curve kernels, refit). Declared in include/rtcd/backend.h; section 5 of docs/design/interfaces.md.
# Its OptiX programs are compiled to OptiX IR and embedded with
#   rtcd_add_optix_ir(rtcd_backend_curve_ir SOURCES <programs>.cu)
#   target_link_libraries(rtcd_backend_curve PRIVATE rtcd_backend_curve_ir)
# (cmake/RTCDTargets.cmake; the bytes are returned in ProgramSet::moduleCode). Compile-time variants (e.g. curve
# degree) are templates explicitly instantiated inside this library's .cu files, never in a public header.
add_library(rtcd_backend_curve STATIC
  placeholder.cpp
)
target_include_directories(rtcd_backend_curve
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_backend_curve
  PUBLIC  rtcd_robot rtcd_scene rtcd_config)
set_target_properties(rtcd_backend_curve PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_backend_curve)
