# BBCollisionDetection.cu is compiled once in rtcd_legacy_kernels (RTCD/CMakeLists.txt)

# compile the optixIR first
cuda_compile_and_embed(collisionIR optixir CCCuda.cu)
cuda_compile_and_embed(rayCCIR optixir CCRay.cu)

# the CC Interface library
set(sourceFiles CollisionDetector.h OptixCompileConfig.h launchParameters.h)
add_library(CollisionDetector INTERFACE ${sourceFiles} ${collisionIR} ${rayCCIR})
set_target_properties(CollisionDetector PROPERTIES FOLDER "RTCD/libs")
target_link_libraries(CollisionDetector INTERFACE meshes rtcd_legacy_kernels)
add_dependencies(CollisionDetector CCCuda CCRay)
RTCD_add_source_groups()

if (BUILD_TESTS)
    macro(addCCTest name src linktype)
        add_executable(${name} ${src})
        set_target_properties(${name} PROPERTIES FOLDER "Tests/CollisionDetector")
        set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)
        target_link_libraries(${name} PRIVATE CollisionDetector)
        target_compile_definitions(${name} PRIVATE ${linktype})
        RTCD_add_source_groups()
    endmacro()


    addCCTest(testCCOBBIAS Test/testCollisionDetectionOBB.cpp IAS)
    addCCTest(testCCOBBRAYO Test/testCollisionDetectionOBB.cpp RayOnlineRobot)

    add_custom_target(AllCollisionCheckerTests DEPENDS
        testCCOBBIAS
        testCCOBBRAYO
    )

    add_executable(testCollisionCheckerCurve Test/testCollisionCheckerCurve.cpp)
    set_target_properties(testCollisionCheckerCurve PROPERTIES FOLDER "Tests/CollisionDetector")
    set_target_properties(testCollisionCheckerCurve PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)
    target_link_libraries(testCollisionCheckerCurve PRIVATE CollisionDetector)
    RTCD_add_source_groups()

    add_executable(testBinarySearch Test/testBinarySearch.cu)
    set_target_properties(testBinarySearch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)

    # Collision-result regression check: one executable per detector mode (see Test/ccRegression.cpp).
    # Run all of them with `ctest -L regression` (append --write-golden to an exe to record a new golden).
    set(REG_TARGETS "")
    foreach(mode IAS RAY LINEAR QUAD)
        string(TOLOWER ${mode} mode_lower)
        set(name ccRegression_${mode_lower})
        add_executable(${name} Test/ccRegression.cpp)
        set_target_properties(${name} PROPERTIES FOLDER "Tests/CollisionDetector")
        set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Tests)
        target_link_libraries(${name} PRIVATE CollisionDetector)
        target_compile_definitions(${name} PRIVATE REG_${mode})
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES LABELS regression)
        list(APPEND REG_TARGETS ${name})
    endforeach()
    add_custom_target(AllRegressionTests DEPENDS ${REG_TARGETS})
    RTCD_add_source_groups()

endif()