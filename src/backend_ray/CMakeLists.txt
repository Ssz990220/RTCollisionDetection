# rtcd_backend_ray (Wave 2g): createRayBackend (RobotToObs: robot edge rays against the scene IAS,
# all RayHitModes). Declared in include/rtcd/backend.h; section 5 of docs/design/interfaces.md.
#   ray_backend.cpp   host side: construction (rays, link OBB template, per-stream buffers), prepare()
#   ray_programs.cu   OptiX programs (four raygen variants, closest-hit, miss), compiled to OptiX IR and embedded
#                     as rtcd::embedded::rtcd_backend_ray_ir_ray_programs (returned in ProgramSet::moduleCode).
rtcd_add_optix_ir(rtcd_backend_ray_ir SOURCES ray_programs.cu)

add_library(rtcd_backend_ray STATIC
  ray_backend.cpp
)
target_include_directories(rtcd_backend_ray
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_backend_ray
  PUBLIC  rtcd_robot rtcd_scene rtcd_config
  PRIVATE rtcd_backend_ray_ir)
set_target_properties(rtcd_backend_ray PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_backend_ray)
