#include "sceneKern.cuh"
#include <Utils/optix7.h>
#include <array>

typedef unsigned int uint;
namespace RTCD {

    __constant__ float4 sceneSafeTf[3];
    void setSceneSafeTf() {
        constexpr std::array<float, 12> defaultTf_host = {0.001, 0, 0, 0, 0, 0.001, 0, 0, 0, 0, 0.001, -0.2};
        cudaMemcpyToSymbol(sceneSafeTf, defaultTf_host.data(), sizeof(float) * 12);
    }

    __constant__ float4 defaultTf[3];
    void setDefaultTf() {
        constexpr std::array<float, 12> defaultTf_host = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        cudaMemcpyToSymbol(defaultTf, defaultTf_host.data(), sizeof(float) * 12);
    }

    __global__ void upInsMaskedKer(OptixInstance* instances, uint* mask, size_t nMeshes) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nMeshes) {
            return;
        }
        if (mask[idx] == 0) {
            memcpy(instances[idx].transform, (void*) sceneSafeTf, sizeof(float) * 12);
        } else {
            memcpy(instances[idx].transform, (void*) defaultTf, sizeof(float) * 12);
        }
    }


    void moveSceneObj(CUdeviceptr instances, CUdeviceptr mask, size_t nMeshes, cudaStream_t stream) {
        const unsigned int numThreads = 128;
        const unsigned int numBlocks  = (nMeshes + numThreads - 1) / numThreads;
        upInsMaskedKer<<<numBlocks, numThreads, 0, stream>>>(
            reinterpret_cast<OptixInstance*>(instances), reinterpret_cast<uint*>(mask), nMeshes);
    }

    __forceinline__ __device__ float3 rotatePoint(const float* mat, const float3& p) {
        return make_float3(mat[0] * p.x + mat[1] * p.y + mat[2] * p.z, mat[4] * p.x + mat[5] * p.y + mat[6] * p.z,
            mat[8] * p.x + mat[9] * p.y + mat[10] * p.z);
    }

    // Compute the inverse of a 3x4 affine transformation matrix
    // 3 threads for each matrix
    __global__ void invTfKern(float* dst, const float* src, size_t nTfs) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // size_t workerId = threadIdx.y;
        if (idx >= nTfs) {
            return;
        }
        float* srcMat = (float*) src + idx * 12;
        float* dstMat = (float*) dst + idx * 12;

        dstMat[0]  = srcMat[0];
        dstMat[1]  = srcMat[4];
        dstMat[2]  = srcMat[8];
        dstMat[4]  = srcMat[1];
        dstMat[5]  = srcMat[5];
        dstMat[6]  = srcMat[9];
        dstMat[8]  = srcMat[2];
        dstMat[9]  = srcMat[6];
        dstMat[10] = srcMat[10];

        float3 t   = -rotatePoint(dstMat, make_float3(srcMat[3], srcMat[7], srcMat[11]));
        dstMat[3]  = t.x;
        dstMat[7]  = t.y;
        dstMat[11] = t.z;
    }


    void inverseTf(CUdeviceptr dst, const CUdeviceptr src, size_t nTfs, cudaStream_t stream) {
        const size_t numThreads = 128;
        const size_t numBlocks  = (nTfs + numThreads - 1) / numThreads;
        invTfKern<<<numBlocks, numThreads, 0, stream>>>(
            reinterpret_cast<float*>(dst), reinterpret_cast<float*>(src), nTfs);
    }

    __global__ void setInsTfKern(
        const float* tfs, OptixInstance* instances, const size_t batchSize, const size_t nObs) {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx > batchSize || idy > nObs) {
            return;
        }
        const float* tf          = tfs + idx * 12;
        const OptixInstance* ins = instances + idy * batchSize + idx;

        memcpy((void*) ins->transform, tf, sizeof(float) * 12);
    }

    void setInsTf(const CUdeviceptr tfs, CUdeviceptr instances, size_t batchsize, size_t nObs, cudaStream_t stream) {
        const dim3 numThreads{128, 8, 1};
        const dim3 numBlocks{static_cast<unsigned int>((batchsize + numThreads.x - 1) / numThreads.x),
            static_cast<unsigned int>((nObs + numThreads.y - 1) / numThreads.y), 1};
        setInsTfKern<<<numBlocks, numThreads, 0, stream>>>(
            reinterpret_cast<const float*>(tfs), reinterpret_cast<OptixInstance*>(instances), batchsize, nObs);
    }
} // namespace RTCD
