# Tests of rtcd_scene (Wave 1 c owns this folder). Both need a GPU (label new_gpu).
#
# test_scene_gpu     new stack only: shelf scene construction, edge rays / OBBs on the device, IAS + SBT sanity by
#                    tracing random segments with a small test pipeline (scene_trace_program.cu) against a
#                    brute-force host intersection, constructor errors.
# test_scene_parity  compares edge-ray buffers (both sets), ray counts / offsets and OBBs with the old RTCD::scene
#                    on the three shelf scenes. Uses OLD-STACK headers: delete it together with the old stack.
#
# TEST-LOCAL STUB: rtcd::internal::buildTriangleGas / buildInstanceAs belong to rtcd_core (Wave 1 a). While
# src/core does not implement them yet (detected by the absence of "accel" in src/core/CMakeLists.txt), both tests
# link stub/accel_stub.cpp. Delete the stub and this block at the Wave-1 merge.
file(READ "${PROJECT_SOURCE_DIR}/src/core/CMakeLists.txt" rtcd_core_cmake_text)
string(FIND "${rtcd_core_cmake_text}" "accel" rtcd_core_has_accel)
set(scene_test_stub_sources "")
if(rtcd_core_has_accel EQUAL -1)
  message(STATUS "tests/scene: rtcd_core has no accel builders yet, using tests/scene/stub/accel_stub.cpp")
  set(scene_test_stub_sources stub/accel_stub.cpp)
endif()

rtcd_add_optix_ir(test_scene_trace_ir SOURCES scene_trace_program.cu)

rtcd_add_new_test(test_scene_gpu LABEL new_gpu
  SOURCES test_scene_gpu.cpp ${scene_test_stub_sources}
  LIBS rtcd_scene test_scene_trace_ir)

rtcd_add_new_test(test_scene_parity LABEL new_gpu
  SOURCES test_scene_parity.cpp ${scene_test_stub_sources}
  LIBS rtcd_scene meshes)
target_include_directories(test_scene_parity PRIVATE
  ${PROJECT_SOURCE_DIR}/RTCD ${PROJECT_SOURCE_DIR}/3rdparty ${CMAKE_BINARY_DIR})
