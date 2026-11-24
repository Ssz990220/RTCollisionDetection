# Tests of rtcd_testsupport (Wave 1 e owns this folder).

# Host-only utilities: paths, inputs, goldens, legacy options, timing (no GPU).
rtcd_add_new_test(test_testsupport_unit LABEL new_unit
  SOURCES test_testsupport_unit.cpp
  LIBS rtcd_testsupport)

# Parity with the old harness: the inputs of support/inputs.h are byte-identical to what
# RTCD/Utils/Test/poseUtils.h (randomMotions, genCSpaceKNNTraj) produces from the same cache files, and goldens
# decode identically. Includes old-stack headers and links the old stack (deleted with the old stack). Reads the
# cache only (asserts the files exist first, so the old code never regenerates / writes them). No GPU work.
rtcd_add_new_test(test_testsupport_vs_old LABEL new_unit
  SOURCES test_testsupport_vs_old.cpp
  LIBS rtcd_testsupport)
rtcd_use_old_stack(test_testsupport_vs_old)

# ReferenceExecutor / ReferenceBatchRunner on the GPU with a trivial program module of its own (no backend, no
# Scene): raygen selection, params upload, SBT, any-hit through a single-triangle-pair GAS built in the test,
# batching / staging / zeroing / result mapping, launch == false, argument errors. Needs rtcd_core (1a).
rtcd_add_optix_ir(rtcd_testsupport_test_ir
  SOURCES reference_test_program.cu)
rtcd_add_new_test(test_reference_executor LABEL new_gpu
  SOURCES test_reference_executor.cpp
  LIBS rtcd_testsupport rtcd_testsupport_test_ir)
