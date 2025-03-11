file(GLOB SOURCES "*.h" "*.cuh")

add_library(meshes INTERFACE ${SOURCES})
target_link_libraries(meshes INTERFACE
    utils 
    Eigen3::Eigen
    urdfdom::urdf_parser
    urdfdom::urdfdom_model
)
set_target_properties(meshes PROPERTIES FOLDER "RTCD/libs")

if (BUILD_TESTS)

  add_executable(testMeshModel Test/testModelLoader.cpp)
  target_link_libraries(testMeshModel meshes)
  set_target_properties(testMeshModel PROPERTIES FOLDER "Tests/Meshes")
  set_target_properties(testMeshModel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests
  )

  add_executable(testOBB Test/testOBB.cpp)
  target_link_libraries(testOBB meshes utils)
  set_target_properties(testOBB PROPERTIES FOLDER "Tests/Meshes")
  set_target_properties(testOBB PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests
  )

  add_custom_target(AllMeshTests
    DEPENDS testMeshModel
  )

endif()