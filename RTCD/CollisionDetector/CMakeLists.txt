file(GLOB CC_CU_SRC "BBCollisionDetection.cu") # exclude optixIR src files
set(CU_SRC ${CU_SRC} ${CC_CU_SRC})

# compile the optixIR first
cuda_compile_and_embed(collisionIR optixir CCCuda.cu)
cuda_compile_and_embed(rayCCIR optixir CCRay.cu)

# the CC Interface library
set(sourceFiles CollisionDetector.h launchParameters.h)
add_library(CollisionDetector INTERFACE ${sourceFiles} ${collisionIR} ${rayCCIR})
set_target_properties(CollisionDetector PROPERTIES FOLDER "RTCD/libs")
target_link_libraries(CollisionDetector INTERFACE meshes)
add_dependencies(CollisionDetector CCCuda CCRay)
RTCD_add_source_groups()

if (BUILD_TESTS)
    macro(addCCTest name src linktype)
        add_executable(${name} ${src} ${CU_SRC} ${UTIL_CUDA_SRC})
        set_target_properties(${name} PROPERTIES FOLDER "Tests/CollisionDetector")
        set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)
        target_link_libraries(${name} PRIVATE CollisionDetector)
        target_compile_definitions(${name} PRIVATE ${linktype})
        RTCD_add_source_groups()
    endmacro()


    addCCTest(testCCOBBIAS Test/testCollisionDetectionOBB.cpp IAS)
    target_sources(testCCOBBIAS PRIVATE ../Planner/KNN/knnCUDA.cu)
    addCCTest(testCCOBBRAYO Test/testCollisionDetectionOBB.cpp RayOnlineRobot)

    add_custom_target(AllCollisionCheckerTests DEPENDS
        testCCOBBIAS
        testCCOBBRAYO
    )

    add_executable(testCollisionCheckerCurve Test/testCollisionCheckerCurve.cpp ${CU_SRC} ${UTIL_CUDA_SRC} ../Planner/KNN/knnCUDA.cu)
    set_target_properties(testCollisionCheckerCurve PROPERTIES FOLDER "Tests/CollisionDetector")
    set_target_properties(testCollisionCheckerCurve PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)
    target_link_libraries(testCollisionCheckerCurve PRIVATE CollisionDetector)
    RTCD_add_source_groups()

    add_executable(testBinarySearch Test/testBinarySearch.cu)
    set_target_properties(testBinarySearch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

endif()

set(CU_SRC ${CU_SRC} PARENT_SCOPE)