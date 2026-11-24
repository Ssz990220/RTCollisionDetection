# rtcd_exp_host_breakdown (Track G2): per-phase host-time breakdown of one check (DetectorCore + PhaseTrace,
# src/detector/phase_trace.h) for the benchmark configurations of mesh, ray, linear and quad.
# Build explicitly:  agent_build.ps1 -WT <root> -Targets rtcd_exp_host_breakdown
# Run from the repository root:  build/bin/experiments/rtcd_exp_host_breakdown.exe [--modes ...] [--n 1,16]
add_executable(rtcd_exp_host_breakdown EXCLUDE_FROM_ALL host_breakdown.cpp)
target_link_libraries(rtcd_exp_host_breakdown PRIVATE
  rtcd rtcd_testsupport rtcd_json rtcd_detector rtcd_backend_mesh rtcd_backend_ray rtcd_backend_curve CUDA::cudart)
target_include_directories(rtcd_exp_host_breakdown PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(rtcd_exp_host_breakdown PRIVATE NOMINMAX)
set_target_properties(rtcd_exp_host_breakdown PROPERTIES
  FOLDER "rtcd/experiments"
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/experiments)
