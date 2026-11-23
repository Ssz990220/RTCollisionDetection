#include <Robot/iasRobotKernels.cuh>
#include <optix.h>

namespace {
    __global__ void upInsVisibilityKer(
        OptixInstance* instances, const float4* transforms, const unsigned int* mask, const size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float4* tf  = transforms + idx * 3;
        float4* dst       = reinterpret_cast<float4*>(instances[idx].transform); // OptixInstance is 16-byte aligned
        dst[0]            = tf[0];
        dst[1]            = tf[1];
        dst[2]            = tf[2];
        const bool active = mask == nullptr || mask[idx] > 0;
        instances[idx].visibilityMask = active ? 255u : 0u;
    }
} // namespace

void upInsVisibility(CUdeviceptr instances, CUdeviceptr transforms, CUdeviceptr mask, size_t n, cudaStream_t stream) {
    if (n == 0) {
        return;
    }
    constexpr unsigned int numThreads = 128;
    const unsigned int numBlocks      = static_cast<unsigned int>((n + numThreads - 1) / numThreads);
    upInsVisibilityKer<<<numBlocks, numThreads, 0, stream>>>(reinterpret_cast<OptixInstance*>(instances),
        reinterpret_cast<const float4*>(transforms), reinterpret_cast<const unsigned int*>(mask), n);
}
