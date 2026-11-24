# Tests of rtcd_scene. Both need a GPU (label new_gpu).
#
# test_scene_gpu     shelf scene construction, edge rays / OBBs on the device, IAS + SBT sanity by
#                    tracing random segments with a small test pipeline (scene_trace_program.cu) against a
#                    brute-force host intersection, constructor errors.
# test_scene_reference  edge-ray buffers (both sets), ray counts / offsets and OBBs of the three shelf scenes
#                       (configs/scenes) vs the stored reference tests/data/reference/scenes.txt (old RTCD::scene).

rtcd_add_optix_ir(test_scene_trace_ir SOURCES scene_trace_program.cu)

rtcd_add_new_test(test_scene_gpu LABEL new_gpu
  SOURCES test_scene_gpu.cpp
  LIBS rtcd_scene test_scene_trace_ir)

rtcd_add_new_test(test_scene_reference LABEL new_gpu
  SOURCES test_scene_reference.cpp
  LIBS rtcd_scene rtcd_json rtcd_testsupport)
