
file(GLOB SCENE_CU_SRC "*.cu")
# cancat SCENE_CU_SRC to CU_SRC which is a global param
set(CU_SRC ${CU_SRC} ${SCENE_CU_SRC})

if (BUILD_TESTS)

  add_executable(testObstacle Test/testObstacle.cpp)
  target_link_libraries(testObstacle meshes)
  set_target_properties(testObstacle PROPERTIES FOLDER "Tests/CollisionScenes")
  set_target_properties(testObstacle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests
  )

  add_executable(testScene Test/testScene.cpp)
  target_link_libraries(testScene meshes)
  set_target_properties(testScene PROPERTIES FOLDER "Tests/CollisionScenes")
  set_target_properties(testScene PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests
  )

  add_executable(testBatchScene Test/testBatchScene.cpp ${CU_SRC})
  target_link_libraries(testBatchScene meshes)
  set_target_properties(testBatchScene PROPERTIES FOLDER "Tests/CollisionScenes")
  set_target_properties(testBatchScene PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests
  )

  add_custom_target(AllCollisionSceneTests DEPENDS testObstacle testScene testBatchScene)

endif()

set(CU_SRC ${CU_SRC} PARENT_SCOPE)