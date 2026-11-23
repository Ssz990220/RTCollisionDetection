#pragma once

#include <Meshes/meshTypes.h>
#include <cuda.h>

namespace RTCD {
    void BBCD(const CUdeviceptr robobOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        const CUdeviceptr robotMask, const CUdeviceptr sceneMask, const cudaStream_t stream);

    void BBCD(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        CUdeviceptr robotMask, CUdeviceptr sceneMask, CUdeviceptr rsMask, const cudaStream_t stream);

    // Robot mask only: robotMask[r] = number of scene OBBs colliding with robot OBB r, written for every r < nROBBs
    // (identical to BBCD's robotMask output on a zeroed buffer). No pre-zeroing needed; one kernel launch.
    void BBCDRobotMask(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs,
        const size_t nSOBBs, CUdeviceptr robotMask, const cudaStream_t stream);

    void BBCD2(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        CUdeviceptr rsMask, const cudaStream_t stream);

    void BBCD3(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        CUdeviceptr roMask, const cudaStream_t stream);
} // namespace RTCD
