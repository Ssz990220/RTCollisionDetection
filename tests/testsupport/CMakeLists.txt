# Tests of rtcd_testsupport.

# Host-only utilities: paths, inputs (vs the stored old-harness hashes), goldens, reference files, legacy options,
# timing (no GPU).
rtcd_add_new_test(test_testsupport_unit LABEL new_unit
  SOURCES test_testsupport_unit.cpp
  LIBS rtcd_testsupport)

# ReferenceExecutor / ReferenceBatchRunner on the GPU with a trivial program module of its own (no backend, no
# Scene): raygen selection, params upload, SBT, any-hit through a single-triangle-pair GAS built in the test,
# batching / staging / zeroing / result mapping, launch == false, argument errors. Needs rtcd_core (1a).
rtcd_add_optix_ir(rtcd_testsupport_test_ir
  SOURCES reference_test_program.cu)
rtcd_add_new_test(test_reference_executor LABEL new_gpu
  SOURCES test_reference_executor.cpp
  LIBS rtcd_testsupport rtcd_testsupport_test_ir)
