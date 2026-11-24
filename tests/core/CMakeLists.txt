# Tests of rtcd_core (Wave 1 a owns this folder).
#
# rtcd_core_abi_check: compile-only (OBJECT, part of AllNew, never linked). With the CUDA / OptiX include paths
# of rtcd_core it static_asserts that the handle aliases / mirrored flag values of the public headers equal the
# CUDA / OptiX ones (abi_check.cpp) and that rtcd/limits.h compiles as device code and matches the SBT constants
# (limits_device.cu). Wave 1 a may move these asserts into src/core once real sources exist.
add_library(rtcd_core_abi_check OBJECT abi_check.cpp limits_device.cu)
target_link_libraries(rtcd_core_abi_check PRIVATE rtcd_core)
set_target_properties(rtcd_core_abi_check PROPERTIES FOLDER "rtcd/tests")
rtcd_register_new_target(rtcd_core_abi_check)
