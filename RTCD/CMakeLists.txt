include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# All CUDA kernels of the old stack, compiled once and linked into every old-stack executable (they used to be
# recompiled per executable through CU_SRC / UTIL_CUDA_SRC). No .cu depends on per-executable compile definitions
# (SHELF, REG_*, IAS, ...): those apply to the .cpp files only. The OptiX IR programs (CCCuda.cu, CCRay.cu) are
# separate targets (cuda_compile_and_embed in CollisionDetector/).
add_library(rtcd_legacy_kernels STATIC
  Utils/cuda/cudaAlgorithms.cu
  Robot/robot.cu
  Robot/sweptVolume.cu
  Robot/iasRobotKernels.cu
  CollisionScenes/sceneKern.cu
  CollisionDetector/BBCollisionDetection.cu
  Planner/KNN/knnCUDA.cu
)
target_link_libraries(rtcd_legacy_kernels PUBLIC meshes)
set_target_properties(rtcd_legacy_kernels PROPERTIES FOLDER "RTCD/libs")

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