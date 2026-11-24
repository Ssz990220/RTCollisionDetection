# *.cu are compiled once in rtcd_legacy_kernels (RTCD/CMakeLists.txt)

if (BUILD_TESTS)

    add_executable(testRobot Test/robotTest.cpp)
    target_link_libraries(testRobot meshes rtcd_legacy_kernels)
    set_target_properties(testRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testBatchIASRobot Test/batchIASRobotTest.cpp)
    target_link_libraries(testBatchIASRobot meshes rtcd_legacy_kernels)
    set_target_properties(testBatchIASRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testBatchIASRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testBatchCurveRobot Test/batchCurveRobotTest.cpp)
    target_link_libraries(testBatchCurveRobot meshes rtcd_legacy_kernels)
    set_target_properties(testBatchCurveRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testBatchCurveRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testBatchRayRobot Test/batchRayRobotTest.cpp)
    target_link_libraries(testBatchRayRobot meshes rtcd_legacy_kernels)
    set_target_properties(testBatchRayRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testBatchRayRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testRobotFkine Test/robotFkineTest.cpp)
    target_link_libraries(testRobotFkine meshes rtcd_legacy_kernels)
    set_target_properties(testRobotFkine PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testRobotFkine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_custom_target(AllRobotTests
        DEPENDS
        testRobot
        testBatchIASRobot
        # testBatchGASRobot
        testBatchCurveRobot
        testBatchRayRobot
    )

endif()