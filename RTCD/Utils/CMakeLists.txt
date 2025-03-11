file(GLOB SOURCES "*.h" "*.cuh" "cuda/*.cuh")
file(GLOB UTIL_CUDA_SRC "cuda/*.cu")
set(UTIL_CUDA_SRC ${UTIL_CUDA_SRC} PARENT_SCOPE)

add_library(utils INTERFACE)

target_sources(utils INTERFACE ${SOURCES})

set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(utils INTERFACE
  ${OptiX_INSTALL_DIR}/include
)

target_link_libraries(utils INTERFACE 
  OptiX::OptiX
  CUDA::cuda_driver
  CUDA::cudart_static
)

if (BUILD_TESTS)

  macro(add_util_test name)
    add_executable(${name} Test/src/${name}.cpp ${UTIL_CUDA_SRC})
    target_link_libraries(${name} PRIVATE utils)
    set_target_properties(${name} PROPERTIES FOLDER "Tests/Utils")
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)
    target_link_libraries(${name} PRIVATE GTest::gtest GTest::gtest_main)
  endmacro()

  add_util_test(testMotionGen)
  add_util_test(testStreamCompact)
  add_util_test(testExclusiveScan)
  add_util_test(testReduce)

  add_executable(testContext Test/src/testContext.cpp)
  target_link_libraries(testContext PRIVATE utils)
  set_target_properties(testContext PROPERTIES FOLDER "Tests/Utils")
  set_target_properties(testContext PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

  add_custom_target(AllUtilsTests DEPENDS testMotionGen testStreamCompact testExclusiveScan testTransform testContext testReduce)

  add_executable(testSelect Test/src/testSelect.cu)
  target_link_libraries(testSelect PRIVATE utils)

endif()