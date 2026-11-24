# Tests of rtcd_backend_ray (Wave 2g owns this folder). All need a GPU (label new_gpu).
#
# test_backend_ray   ProgramSet per hit mode, LaunchPlan / RobotRayParams, link poses vs host FK, OBB-off mask,
#                    construction and prepare() errors, golden flags for non-default configurations (OBB culling
#                    off, skip-hit off, launch width caps, one stream, batch 1024) through ReferenceBatchRunner.
# rtcd_regression_ray_<hit mode>
#                    the backend gate (design doc section 8 g): rtcd_regression on regression_ray.json for the three
#                    non-default ray hit modes (closest_hit is tests/apps' rtcd_regression_ray), through the
#                    CollisionChecker. Labels new_gpu and regression (`ctest -L regression` covers every hit mode).

rtcd_add_new_test(test_backend_ray LABEL new_gpu
  SOURCES test_backend_ray.cpp
  LIBS rtcd_backend_ray rtcd_testsupport rtcd_json)

foreach(hit_mode first_hit_ch first_hit_miss traverse)
  add_test(NAME rtcd_regression_ray_${hit_mode}
    COMMAND rtcd_regression --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_ray.json
            --override ray.hit_mode=${hit_mode}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(rtcd_regression_ray_${hit_mode} PROPERTIES LABELS "new_gpu;regression")
endforeach()
