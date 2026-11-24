# Skeleton smoke tests: prove that the new tree links (every library), embeds OptiX IR and reaches the GPU.
# Delete once real tests exist for every library. rtcd_skeleton_ir is the one placeholder use of
# rtcd_add_optix_ir (the backends add their own once they have programs); delete it with this folder.
rtcd_add_optix_ir(rtcd_skeleton_ir
  SOURCES skeleton_program.cu
)
rtcd_add_new_test(test_skeleton_unit LABEL new_unit
  SOURCES test_skeleton_unit.cpp
  LIBS rtcd rtcd_testsupport rtcd_skeleton_ir)
rtcd_add_new_test(test_skeleton_gpu LABEL new_gpu
  SOURCES test_skeleton_gpu.cpp
  LIBS rtcd)
