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


// RAY_ONLINE robot rays. (The entry name is kept for the pipeline; the rays are no longer compacted.)
// Launch index = poseIdx * raysPerRobot + rayIdx over the whole batch. A ray is traced only if its link's OBB hit an
// obstacle OBB (linkMask > 0): this is exactly the set of rays the former mask-compacted launch traced, with the same
// transform and endpoints, so per-pose results are identical, but no compaction or host read-back is needed.
// Note: no __syncthreads() here. OptiX programs have no CUDA block semantics, barriers are not supported
// (rejected by OptiX 9 / CUDA 13), and nothing here needs synchronization anyway.
extern "C" __global__ void __raygen__robotRayCompact() {
    const rayRobotCfg& cfg = optixLaunchParams.robot;
    const uint idx         = optixGetLaunchIndex().x;

    const uint poseIdx = idx / cfg.raysPerRobot;
    const uint rayIdx  = idx - poseIdx * cfg.raysPerRobot;
    const uint linkIdx = static_cast<uint>(cfg.rayLinkMap[rayIdx]);
    const uint glbLink = poseIdx * cfg.linksPerRobot + linkIdx;
    if (cfg.linkMask[glbLink] == 0) {
        return; // link OBB is collision free: its rays cannot hit anything
    }
    const float* linkTransform = cfg.lkTfs + glbLink * 12;

    // get the ray info
    const float3 origin = cfg.rays[rayIdx * 2];
    const float3 target = cfg.rays[rayIdx * 2 + 1];

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
