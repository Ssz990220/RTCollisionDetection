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
