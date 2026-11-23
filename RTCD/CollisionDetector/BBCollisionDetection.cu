#include <CollisionDetector/BBCollisionDetection.cuh>

namespace RTCD {
    __inline__ __device__ bool isSeparatingAxis(const OBB& obb1, const OBB& obb2, const float3& axis) {
        float3 u[3] = {obb1.orientation[0], obb1.orientation[1], obb1.orientation[2]};
        float3 v[3] = {obb2.orientation[0], obb2.orientation[1], obb2.orientation[2]};

        float ra = obb1.halfSize.x * fabsf(dot(axis, u[0])) + obb1.halfSize.y * fabsf(dot(axis, u[1]))
                 + obb1.halfSize.z * fabsf(dot(axis, u[2]));

        float rb = obb2.halfSize.x * fabsf(dot(axis, v[0])) + obb2.halfSize.y * fabsf(dot(axis, v[1]))
                 + obb2.halfSize.z * fabsf(dot(axis, v[2]));

        float3 t = obb2.center - obb1.center;
        float d  = fabsf(dot(t, axis));

        return d > ra + rb;
    }

    __device__ bool checkOBBCollision(const OBB& obb1, const OBB& obb2) {
        float3 u[3] = {obb1.orientation[0], obb1.orientation[1], obb1.orientation[2]};
        float3 v[3] = {obb2.orientation[0], obb2.orientation[1], obb2.orientation[2]};

        bool isColliding = true;

        for (int i = 0; i < 3; ++i) {
            if (isSeparatingAxis(obb1, obb2, u[i]) || isSeparatingAxis(obb1, obb2, v[i])
                || isSeparatingAxis(obb1, obb2, cross(u[i], v[0])) || isSeparatingAxis(obb1, obb2, cross(u[i], v[1]))
                || isSeparatingAxis(obb1, obb2, cross(u[i], v[2]))) {
                isColliding = false;
                break;
            }
        }

        // No separation found, OBBs are in collision.
        return isColliding;
    }

    __global__ void BBCDKern(const OBB* robotOBBs, const OBB* sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        uint* robotMask, uint* sceneMask) {
        const size_t robot_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t scene_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (robot_idx >= nROBBs || scene_idx >= nSOBBs) {
            return;
        }

        bool isColliding = checkOBBCollision(robotOBBs[robot_idx], sceneOBBs[scene_idx]);

        __syncthreads();

        if (isColliding) {
            // robotMask[robot_idx] = true;
            atomicAdd(&robotMask[robot_idx], 1);
            atomicAdd(&sceneMask[scene_idx], 1);
        }
    }

    void BBCD(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        CUdeviceptr robotMask, CUdeviceptr sceneMask, const cudaStream_t stream) {
        const dim3 blockSize(32, 16, 1);
        const dim3 grideSize((nROBBs + blockSize.x - 1) / blockSize.x, (nSOBBs + blockSize.y - 1) / blockSize.y, 1);

        BBCDKern<<<grideSize, blockSize, 0, stream>>>(reinterpret_cast<OBB*>(robotOBBs),
            reinterpret_cast<OBB*>(sceneOBBs), nROBBs, nSOBBs, reinterpret_cast<uint*>(robotMask),
            reinterpret_cast<uint*>(sceneMask));
    }


    __global__ void BBCDKern(const OBB* robotOBBs, const OBB* sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        uint* robotMask, uint* sceneMask, uint* rsMask) {
        const size_t robot_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t scene_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (robot_idx >= nROBBs || scene_idx >= nSOBBs) {
            return;
        }

        bool isColliding = checkOBBCollision(robotOBBs[robot_idx], sceneOBBs[scene_idx]);

        __syncthreads();

        if (isColliding) {
            // robotMask[robot_idx] = true;
            atomicAdd(&robotMask[robot_idx], 1);
            atomicAdd(&sceneMask[scene_idx], 1);
            atomicAdd(&rsMask[robot_idx + scene_idx * nROBBs], 1);
        }
    }
    void BBCD(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        CUdeviceptr robotMask, CUdeviceptr sceneMask, CUdeviceptr rsMask, const cudaStream_t stream) {
        const dim3 blockSize(32, 16, 1);
        const dim3 grideSize((nROBBs + blockSize.x - 1) / blockSize.x, (nSOBBs + blockSize.y - 1) / blockSize.y, 1);

        BBCDKern<<<grideSize, blockSize, 0, stream>>>(reinterpret_cast<OBB*>(robotOBBs),
            reinterpret_cast<OBB*>(sceneOBBs), nROBBs, nSOBBs, reinterpret_cast<uint*>(robotMask),
            reinterpret_cast<uint*>(sceneMask), reinterpret_cast<uint*>(rsMask));
    }

    // Robot-mask-only broad phase. Writes robotMask[r] = number of scene OBBs colliding with robot OBB r, for every
    // r < nROBBs (the same values BBCD accumulates with atomics into a zeroed mask), so no memset is needed.
    // Block: BBCD_RM_X robot OBBs x BBCD_RM_Y scene lanes; each lane strides over the scene OBBs, then the block
    // reduces the per-lane counts in shared memory.
    constexpr unsigned int BBCD_RM_X = 32;
    constexpr unsigned int BBCD_RM_Y = 8;
    __global__ void BBCDRobotMaskKern(
        const OBB* robotOBBs, const OBB* sceneOBBs, const size_t nROBBs, const size_t nSOBBs, uint* robotMask) {
        __shared__ uint counts[BBCD_RM_Y][BBCD_RM_X];
        const size_t robot_idx = blockIdx.x * BBCD_RM_X + threadIdx.x;

        uint cnt = 0;
        if (robot_idx < nROBBs) {
            const OBB robotOBB = robotOBBs[robot_idx];
            for (size_t s = threadIdx.y; s < nSOBBs; s += BBCD_RM_Y) {
                cnt += checkOBBCollision(robotOBB, sceneOBBs[s]) ? 1u : 0u;
            }
        }
        counts[threadIdx.y][threadIdx.x] = cnt;
        __syncthreads();
        if (threadIdx.y == 0 && robot_idx < nROBBs) {
            uint total = 0;
#pragma unroll
            for (unsigned int y = 0; y < BBCD_RM_Y; y++) {
                total += counts[y][threadIdx.x];
            }
            robotMask[robot_idx] = total;
        }
    }

    void BBCDRobotMask(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs,
        const size_t nSOBBs, CUdeviceptr robotMask, const cudaStream_t stream) {
        if (nROBBs == 0) {
            return;
        }
        const dim3 blockSize(BBCD_RM_X, BBCD_RM_Y, 1);
        const dim3 gridSize(static_cast<unsigned int>((nROBBs + BBCD_RM_X - 1) / BBCD_RM_X), 1, 1);
        BBCDRobotMaskKern<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<const OBB*>(robotOBBs),
            reinterpret_cast<const OBB*>(sceneOBBs), nROBBs, nSOBBs, reinterpret_cast<uint*>(robotMask));
    }

    __global__ void BBCDKern2(
        const OBB* robotOBBs, const OBB* sceneOBBs, const size_t nROBBs, const size_t nSOBBs, uint* rsMask) {
        const size_t robot_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t scene_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (robot_idx >= nROBBs || scene_idx >= nSOBBs) {
            return;
        }

        bool isColliding = checkOBBCollision(robotOBBs[robot_idx], sceneOBBs[scene_idx]);

        __syncthreads();

        if (isColliding) {
            // robotMask[robot_idx] = true;
            rsMask[robot_idx * nSOBBs + scene_idx] += 1;
        }
    }

    void BBCD2(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        CUdeviceptr rsMask, const cudaStream_t stream) {
        const dim3 blockSize(32, 16, 1);
        const dim3 grideSize((nROBBs + blockSize.x - 1) / blockSize.x, (nSOBBs + blockSize.y - 1) / blockSize.y, 1);

        BBCDKern2<<<grideSize, blockSize, 0, stream>>>(reinterpret_cast<OBB*>(robotOBBs),
            reinterpret_cast<OBB*>(sceneOBBs), nROBBs, nSOBBs, reinterpret_cast<uint*>(rsMask));
    }


    __global__ void BBCDKern3(
        const OBB* robotOBBs, const OBB* sceneOBBs, const size_t nROBBs, const size_t nSOBBs, uint* roMask) {
        const size_t robot_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t scene_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (robot_idx >= nROBBs || scene_idx >= nSOBBs) {
            return;
        }

        bool isColliding = checkOBBCollision(robotOBBs[robot_idx], sceneOBBs[scene_idx]);

        __syncthreads();

        if (isColliding) {
            // robotMask[robot_idx] = true;
            atomicAdd(&roMask[robot_idx], 1);
        }
    }

    void BBCD3(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs, const size_t nSOBBs,
        CUdeviceptr roMask, const cudaStream_t stream) {
        const dim3 blockSize(32, 16, 1);
        const dim3 grideSize((nROBBs + blockSize.x - 1) / blockSize.x, (nSOBBs + blockSize.y - 1) / blockSize.y, 1);

        BBCDKern3<<<grideSize, blockSize, 0, stream>>>(reinterpret_cast<OBB*>(robotOBBs),
            reinterpret_cast<OBB*>(sceneOBBs), nROBBs, nSOBBs, reinterpret_cast<uint*>(roMask));
    }
} // namespace RTCD
