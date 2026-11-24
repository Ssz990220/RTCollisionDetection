// Skeleton smoke test (new_gpu): a CUDA device is reachable through rtcd_core's link dependencies.
#include <cuda_runtime.h>

#include <gtest/gtest.h>

TEST(Skeleton, CudaDeviceAvailable) {
    int n = 0;
    ASSERT_EQ(cudaGetDeviceCount(&n), cudaSuccess);
    EXPECT_GE(n, 1);
    EXPECT_EQ(cudaFree(nullptr), cudaSuccess);
}
