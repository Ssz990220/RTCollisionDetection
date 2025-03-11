macro(addBenchmark name src scene)
    add_executable(${name} ${src} ${CU_SRC} ${UTIL_CUDA_SRC})
    target_link_libraries(${name} benchmark::benchmark CollisionDetector)
    set_target_properties(${name} PROPERTIES FOLDER "Benchmark")
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Benchmarks)
    target_compile_definitions(${name} PRIVATE ${scene})
    RTCD_add_source_groups()
endmacro()


addBenchmark(benchmarkShelf benchmark.cpp SHELF)
addBenchmark(benchmarkShelfSimple benchmark.cpp SHELFSIMPLE)
addBenchmark(benchmarkDenseShelf benchmark.cpp DENSESHELF)