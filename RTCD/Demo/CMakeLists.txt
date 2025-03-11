macro(addDemo name src mode)
    add_executable(${name} ${src} ${CU_SRC} ${UTIL_CUDA_SRC} ../Planner/KNN/knnCUDA.cu)
    target_link_libraries(${name} PRIVATE CollisionDetector)
    set_target_properties(${name} PROPERTIES FOLDER "Demos")
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Demos)
    target_compile_definitions(${name} PRIVATE ${mode})
    RTCD_add_source_groups()
endmacro()


addDemo(demoIAS DemoCollisionDetectorDisc.cpp IAS)
addDemo(demoRAY DemoCollisionDetectorDisc.cpp RAYO)
addDemo(demoLinContinuous DemoCollisionDetectorCurve.cpp LINEAR)
addDemo(demoQuadContinuous DemoCollisionDetectorCurve.cpp QUAD)

add_custom_target(AllDemos DEPENDS demoIAS demoRAY demoLinContinuous demoQuadContinuous)