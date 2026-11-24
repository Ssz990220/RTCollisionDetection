# Tests of rtcd_scene (Wave 1 c owns this folder). Both need a GPU (label new_gpu).
#
# test_scene_gpu     new stack only: shelf scene construction, edge rays / OBBs on the device, IAS + SBT sanity by
#                    tracing random segments with a small test pipeline (scene_trace_program.cu) against a
#                    brute-force host intersection, constructor errors.
# test_scene_parity  compares edge-ray buffers (both sets), ray counts / offsets and OBBs with the old RTCD::scene
#                    on the three shelf scenes. Uses OLD-STACK headers: delete it together with the old stack.

rtcd_add_optix_ir(test_scene_trace_ir SOURCES scene_trace_program.cu)

rtcd_add_new_test(test_scene_gpu LABEL new_gpu
  SOURCES test_scene_gpu.cpp
  LIBS rtcd_scene test_scene_trace_ir)

rtcd_add_new_test(test_scene_parity LABEL new_gpu
  SOURCES test_scene_parity.cpp
  LIBS rtcd_scene meshes)
target_include_directories(test_scene_parity PRIVATE
  ${PROJECT_SOURCE_DIR}/RTCD ${PROJECT_SOURCE_DIR}/3rdparty ${CMAKE_BINARY_DIR})
