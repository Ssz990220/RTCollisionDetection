# Tests of rtcd_core (Wave 1 a owns this folder).
#
# rtcd_core_abi_check: compile-only (OBJECT, part of AllNew, never linked). With the CUDA / OptiX include paths
# of rtcd_core it static_asserts that the handle aliases / mirrored flag values of the public headers equal the
# CUDA / OptiX ones (abi_check.cpp) and that rtcd/limits.h compiles as device code and matches the SBT constants
# (limits_device.cu).
add_library(rtcd_core_abi_check OBJECT abi_check.cpp limits_device.cu)
target_link_libraries(rtcd_core_abi_check PRIVATE rtcd_core)
set_target_properties(rtcd_core_abi_check PROPERTIES FOLDER "rtcd/tests")
rtcd_register_new_target(rtcd_core_abi_check)

# Checking macros, error classification, Logger (no GPU needed).
rtcd_add_new_test(test_core_unit LABEL new_unit
  SOURCES test_core_unit.cpp
  LIBS rtcd_core)

# Context, streams, buffers, real CUDA failures.
rtcd_add_new_test(test_core_gpu LABEL new_gpu
  SOURCES test_core_gpu.cpp fault_kernels.cu
  LIBS rtcd_core)

# internal/algorithms.h against host references.
rtcd_add_new_test(test_core_algorithms LABEL new_gpu
  SOURCES test_core_algorithms.cpp
  LIBS rtcd_core)

# internal/accel.h: GAS / IAS builds, verified by tracing rays with a small embedded OptiX program.
rtcd_add_optix_ir(rtcd_core_test_ir SOURCES trace_program.cu)
rtcd_add_new_test(test_core_accel LABEL new_gpu
  SOURCES test_core_accel.cpp trace_params.h
  LIBS rtcd_core rtcd_core_test_ir)

# Parity with the OLD STACK (links rtcd_legacy_kernels; delete with the old stack): obbOverlapCounts vs
# RTCD::BBCDRobotMask, bit-identical counts.
rtcd_add_new_test(test_core_legacy_parity LABEL new_gpu
  SOURCES test_core_legacy_parity.cpp
  LIBS rtcd_core rtcd_legacy_kernels)
