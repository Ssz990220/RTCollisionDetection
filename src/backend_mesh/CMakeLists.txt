# rtcd_backend_mesh (Wave 2f): createMeshBackend (ObsToRobot: link GASes + per-batch IAS,
# OBB culling / compaction). Declared in include/rtcd/backend.h; section 5 of docs/design/interfaces.md.
# Its OptiX programs (mesh_programs.cu) are compiled to OptiX IR and embedded with rtcd_add_optix_ir
# (cmake/RTCDTargets.cmake); the bytes are returned in ProgramSet::moduleCode.
rtcd_add_optix_ir(rtcd_backend_mesh_ir
  SOURCES mesh_programs.cu)

add_library(rtcd_backend_mesh STATIC
  mesh_backend.cpp
  mesh_kernels.h
  mesh_kernels.cu
)
target_include_directories(rtcd_backend_mesh
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_backend_mesh
  PUBLIC  rtcd_robot rtcd_scene rtcd_config
  PRIVATE rtcd_backend_mesh_ir)
set_target_properties(rtcd_backend_mesh PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_backend_mesh)
