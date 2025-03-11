#pragma once

#include <Utils/cuUtils.cuh>
#include <array>
#include <cuda.h>
#include <vector>

namespace RTCD {
    void setSceneSafeTf();
    void setDefaultTf();
    void setInsTf(const CUdeviceptr tfs, CUdeviceptr instances, size_t batchsize, size_t nObs, cudaStream_t stream);
    void inverseTf(CUdeviceptr dst, const CUdeviceptr src, size_t nTfs, cudaStream_t stream);
    void moveSceneObj(CUdeviceptr instances, CUdeviceptr mask, size_t nMeshes, cudaStream_t stream);
} // namespace RTCD
