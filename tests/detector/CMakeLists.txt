# Tests of rtcd_detector (Wave 2e owns this folder). All of them use a fake backend (fake_backend.h) with its own
# trivial OptiX program module, so they do not need the mesh / ray / curve backends.
rtcd_add_optix_ir(rtcd_detector_test_ir
  SOURCES detector_test_program.cu)

# DetectorCore directly (the factory is injected): links only the core objects of rtcd_detector, never
# backend_factory.obj (the three backend factories).
rtcd_add_new_test(test_detector_core LABEL new_gpu
  SOURCES test_detector_core.cpp fake_backend.h detector_test_params.h
  LIBS rtcd_detector rtcd_testsupport rtcd_detector_test_ir)
target_include_directories(test_detector_core PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(test_detector_core PRIVATE NOMINMAX)

# The public CollisionChecker facade with the real Panda RobotModel and a real Scene, but with the fake backend:
# the test defines rtcd::detail::createBackendForMode itself (link seam), so the linker never pulls
# backend_factory.obj out of rtcd_detector. Remove the seam (or keep it) once the backends exist; the real factory
# is exercised by rtcd_regression --engine checker at the merge.
rtcd_add_new_test(test_detector_facade LABEL new_gpu
  SOURCES test_detector_facade.cpp fake_backend.h detector_test_params.h
  LIBS rtcd_detector rtcd_json rtcd_detector_test_ir)
target_include_directories(test_detector_facade PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(test_detector_facade PRIVATE NOMINMAX)
