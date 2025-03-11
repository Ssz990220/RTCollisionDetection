#include "launchParameters.h"
#include <Utils/cuUtils.cuh>
#include <optix.h>
#include <optix_device.h>

extern "C" __constant__ LaunchParams optixLaunchParams;

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

static __forceinline__ __device__ float3 normallength(const float3& a, float& l) {
    l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);

    float invLen = 1.0f / l;
    return make_float3(a.x * invLen, a.y * invLen, a.z * invLen);
}

static __forceinline__ __device__ void transformPoint(const float* transform, const float3& point, float3& result) {
    result.x = transform[0] * point.x + transform[1] * point.y + transform[2] * point.z + transform[3];
    result.y = transform[4] * point.x + transform[5] * point.y + transform[6] * point.z + transform[7];
    result.z = transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11];
}

extern "C" __global__ void __closesthit__radiance() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__radiance() {
    optixSetPayload_0(0);
}

extern "C" __global__ void __anyhit__RayData() {
    const int ix  = optixGetLaunchIndex().x;
    const int iy  = optixGetLaunchIndex().y;
    const int idx = ix + iy * optixGetLaunchDimensions().x;
    (optixLaunchParams.hitBuffer)[idx] += 1;
}


__device__ uint binary_search(const uint* prefix_sum, uint size, const uint id) {
    // Boundary checks to avoid returning MAX_UINT when id is out of range
    if (size == 0) {
        return (1UL << 32) - 1;
    }
    if (id < prefix_sum[0]) {
        return 0;
    }
    if (id >= prefix_sum[size - 1]) {
        return size - 1;
    }

    int left  = 0;
    int right = size - 1;
    for (uint i = 0; i < (32 - __clz(size)); ++i) { // A fixed maximum of 32 iterations for safety
        uint mid = left + (right - left) / 2;

        // Condition checks converted into arithmetic expressions
        bool greater = (prefix_sum[mid] > id); // 1 if true, 0 if false
        bool within  = (prefix_sum[mid] <= id) && (mid == size - 1 || prefix_sum[mid + 1] > id);

        // Exit early if the target is found
        if (within) {
            return mid;
        }

        // Update boundaries based on conditions
        left  = left * greater + (mid + 1) * !greater;
        right = (mid - 1) * greater + right * !greater;
    }


    return (1UL << 32) - 1; // If not found, which should not happen with correct input
}

extern "C" __global__ void __raygen__robotRayCompact() {
    uint3 idx3 = optixGetLaunchIndex();
    uint idx   = optixGetLaunchIndex().x;

    // Decide which link this ray belongs to by a binary search
    uint linkIdx = binary_search(optixLaunchParams.robot.lkStarts, optixLaunchParams.robot.linkCnt, idx);
    __syncthreads();

    // get all related data based on the link idx
    uint poseIdx         = optixLaunchParams.robot.lkPosMap[linkIdx];
    float3* linkRays     = optixLaunchParams.robot.lkRays[linkIdx];
    uint linkStart       = optixLaunchParams.robot.lkStarts[linkIdx];
    float* linkTransform = optixLaunchParams.robot.lkTfs + linkIdx * 12;

    // get the ray info
    uint rayIdx         = idx - linkStart;
    const float3 origin = linkRays[rayIdx * 2];
    const float3 target = linkRays[rayIdx * 2 + 1];

    float3 newOrigin, newTarget;
    transformPoint(linkTransform, origin, newOrigin);
    transformPoint(linkTransform, target, newTarget);

    float distance;
    float3 rayDir = normallength(newTarget - newOrigin, distance);

    unsigned int u0 = 0;
    optixTrace(optixLaunchParams.traversable, newOrigin, rayDir,
        0.f, // tmin
        distance, // tmax
        0.0f, // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE, // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        SURFACE_RAY_TYPE, // missSBTIndex
        u0);

    if (u0 == 1) {
        atomicAdd((int*) optixLaunchParams.hitBuffer + poseIdx, 1);
        // optixLaunchParams.hitBuffer[poseIdx] += 1;
    }
}
