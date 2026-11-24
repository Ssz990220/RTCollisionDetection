# Tests of the apps (Wave 1 e owns this folder).

# Profile -> new CheckerConfig mapping of the side-by-side runner (validate / resolved through rtcd_config).
rtcd_add_new_test(test_sbs_profiles LABEL new_unit
  SOURCES test_sbs_profiles.cpp
  LIBS rtcd_sbs_common rtcd_config)

# OLD-STACK half of the side-by-side runner (deleted with the old stack): every reg_* profile, built on an
# rtcd::Context's OptiX device context, reproduces data/CCTest/golden_<mode>.bin on the inputs of support/inputs.h.
rtcd_add_new_test(test_sbs_old_half LABEL new_gpu
  SOURCES test_sbs_old_half.cpp
  LIBS rtcd_sbs_old_shelf rtcd_testsupport)

# The new-stack golden regression (apps/regression) as ctest entries, one per configs/setups/regression_<mode>.json,
# label new_gpu, registered only when an engine for that mode exists in this tree (the CollisionChecker, or the
# mode's backend through the reference engine; quad also runs with curve.geometry_flags=none, the second gate of
# design doc section 8 h).
rtcd_new_stack_status()
if(TARGET rtcd_regression)
  set(rtcd_reg_backend_ias MESH)
  set(rtcd_reg_backend_ray RAY)
  set(rtcd_reg_backend_linear CURVE)
  set(rtcd_reg_backend_quad CURVE)
  foreach(mode ias ray linear quad)
    if(RTCD_HAVE_CHECKER OR RTCD_HAVE_BACKEND_${rtcd_reg_backend_${mode}})
      add_test(NAME rtcd_regression_${mode}
        COMMAND rtcd_regression --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_${mode}.json
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(rtcd_regression_${mode} PROPERTIES LABELS new_gpu)
    endif()
  endforeach()
  if(RTCD_HAVE_CHECKER OR RTCD_HAVE_BACKEND_CURVE)
    add_test(NAME rtcd_regression_quad_geomflag_none
      COMMAND rtcd_regression --setup ${PROJECT_SOURCE_DIR}/configs/setups/regression_quad.json
              --override curve.geometry_flags=none
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(rtcd_regression_quad_geomflag_none PROPERTIES LABELS new_gpu)
  endif()
endif()
