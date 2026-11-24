# rtcd_exp_graph_capture (Track G0): can the per-batch call sequence of the detector core (zero hits, staging copy,
# backend prepare() incl. optixAccelBuild, optixLaunch, result copy) be captured into a CUDA Graph and replayed with
# correct results, and how much host time does that save? See graph_capture.cpp for the experiments and options.
# Build explicitly:  agent_build.ps1 -WT <root> -Targets rtcd_exp_graph_capture
# Run from the repository root:  build/bin/experiments/rtcd_exp_graph_capture.exe --help
add_executable(rtcd_exp_graph_capture EXCLUDE_FROM_ALL graph_capture.cpp)
target_link_libraries(rtcd_exp_graph_capture PRIVATE
  rtcd rtcd_testsupport rtcd_json rtcd_detector rtcd_backend_mesh rtcd_backend_ray rtcd_backend_curve CUDA::cudart CUDA::cuda_driver)
target_include_directories(rtcd_exp_graph_capture PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(rtcd_exp_graph_capture PRIVATE NOMINMAX)
set_target_properties(rtcd_exp_graph_capture PROPERTIES
  FOLDER "rtcd/experiments"
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/experiments)
