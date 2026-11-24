# Skeleton smoke tests: prove that the new tree links (all modules), embeds OptiX IR and reaches the GPU.
# Delete once real tests exist for every module.
rtcd_add_new_test(test_skeleton_unit LABEL new_unit
  SOURCES test_skeleton_unit.cpp
  LIBS rtcd rtcd_programs)
rtcd_add_new_test(test_skeleton_gpu LABEL new_gpu
  SOURCES test_skeleton_gpu.cpp
  LIBS rtcd)
