include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(Utils)
add_subdirectory(Robot)
add_subdirectory(Meshes)
add_subdirectory(CollisionScenes)
add_subdirectory(CollisionDetector)
add_subdirectory(Benchmark)
add_subdirectory(Planner)

add_subdirectory(Demo)

if (BUILD_TESTS)
  add_custom_target(AllTests
    DEPENDS
    AllRobotTests
    AllMeshTests
    AllVisualizerTests
    AllCollisionSceneTests
    AllCollisionCheckerTests
    AllUtilsTests
)
endif()