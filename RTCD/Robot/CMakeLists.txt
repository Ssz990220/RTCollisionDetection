file(GLOB ROBO_CU "*.cu")
set(CU_SRC ${CU_SRC} ${ROBO_CU} PARENT_SCOPE)

if (BUILD_TESTS)

    add_executable(testRobot Test/robotTest.cpp robot.cu)
    target_link_libraries(testRobot meshes)
    target_compile_definitions(testRobot PUBLIC)
    set_target_properties(testRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testBatchIASRobot Test/batchIASRobotTest.cpp robot.cu ${UTIL_CUDA_SRC})
    target_link_libraries(testBatchIASRobot meshes)
    target_compile_definitions(testBatchIASRobot PUBLIC)
    set_target_properties(testBatchIASRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testBatchIASRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testBatchCurveRobot Test/batchCurveRobotTest.cpp robot.cu sweptVolume.cu ${UTIL_CUDA_SRC} ../Planner/KNN/knnCUDA.cu)
    target_link_libraries(testBatchCurveRobot meshes)
    set_target_properties(testBatchCurveRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testBatchCurveRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testBatchRayRobot Test/batchRayRobotTest.cpp robot.cu ${UTIL_CUDA_SRC})
    target_link_libraries(testBatchRayRobot meshes)
    set_target_properties(testBatchRayRobot PROPERTIES FOLDER "Tests/Robot")
    set_target_properties(testBatchRayRobot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    add_executable(testRobotFkine Test/robotFkineTest.cpp robot.cu ${UTIL_CUDA_SRC})
    target_link_libraries(testRobotFkine meshes)
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