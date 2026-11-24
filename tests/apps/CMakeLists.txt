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
