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
// Ray id = poseIdx * raysPerRobot + rayIdx over the whole batch (totalRayCount = nPoses * raysPerRobot ids). The launch
// width may be smaller than totalRayCount: each launch index walks the ids with a stride of the launch width, so a
// skipped ray costs a loop iteration instead of a launched thread, and consecutive launch indices stay on consecutive
// rays (same pose and link) in every iteration.
// A ray is traced only if its link's OBB hit an obstacle OBB (linkMask > 0) and its pose is not already known to be in
// collision. The first condition selects exactly the rays the former mask-compacted launch traced, with the same
// transform and endpoints; the second only skips rays of poses whose hit count is already > 0, so per-pose collision
// flags are identical (hit counts may be lower). No compaction or host read-back of a ray count is needed.
// Note: no __syncthreads() here. OptiX programs have no CUDA block semantics, barriers are not supported
// (rejected by OptiX 9 / CUDA 13), and nothing here needs synchronization anyway.
//
// Hit test variants (selected on the host by OptixCompileConfig::rayHitMode, which picks the raygen entry point).
// Each only needs yes/no per ray; any-hit is disabled for all of them (the scene triangles are also built with
// OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT), so every variant reports a hit exactly when some triangle intersects the
// segment [0, distance].
enum RayHitVariant {
    HIT_CLOSEST        = 0, // optixTrace, closest-hit search, CH sets payload 1 (miss program: none, payload stays 0)
    HIT_FIRST_CH       = 1, // + TERMINATE_ON_FIRST_HIT
    HIT_FIRST_MISS     = 2, // TERMINATE_ON_FIRST_HIT | DISABLE_CLOSESTHIT, payload starts at 1, miss clears it
    HIT_FIRST_TRAVERSE = 3, // optixTraverse + TERMINATE_ON_FIRST_HIT, optixHitObjectIsHit(); no CH / miss runs
};

template <int VARIANT>
static __forceinline__ __device__ bool traceHit(const float3& origin, const float3& dir, const float tmax) {
    if constexpr (VARIANT == HIT_FIRST_TRAVERSE) {
        optixTraverse(optixLaunchParams.traversable, origin, dir,
            0.f, // tmin
            tmax, // tmax
            0.0f, // rayTime
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            SURFACE_RAY_TYPE, // SBT offset
            RAY_TYPE_COUNT, // SBT stride
            SURFACE_RAY_TYPE); // missSBTIndex
        return optixHitObjectIsHit();
    } else {
        unsigned int flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
        unsigned int u0    = 0;
        if constexpr (VARIANT == HIT_FIRST_CH) {
            flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        } else if constexpr (VARIANT == HIT_FIRST_MISS) {
            flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
            u0 = 1;
        }
        optixTrace(optixLaunchParams.traversable, origin, dir,
            0.f, // tmin
            tmax, // tmax
            0.0f, // rayTime
            OptixVisibilityMask(255), flags,
            SURFACE_RAY_TYPE, // SBT offset
            RAY_TYPE_COUNT, // SBT stride
            SURFACE_RAY_TYPE, // missSBTIndex
            u0);
        return u0 == 1;
    }
}

template <int VARIANT>
static __forceinline__ __device__ void robotRays() {
    const rayRobotCfg& cfg = optixLaunchParams.robot;
    const uint total       = optixLaunchParams.totalRayCount;
    const uint stride      = optixGetLaunchDimensions().x;
    int* hitBuffer         = optixLaunchParams.hitBuffer;

    for (uint id = optixGetLaunchIndex().x; id < total; id += stride) {
        const uint poseIdx = id / cfg.raysPerRobot;
        const uint rayIdx  = id - poseIdx * cfg.raysPerRobot;
        const uint linkIdx = static_cast<uint>(cfg.rayLinkMap[rayIdx]);
        const uint glbLink = poseIdx * cfg.linksPerRobot + linkIdx;
        if (cfg.linkMask[glbLink] == 0) {
            continue; // link OBB is collision free: its rays cannot hit anything
        }
        if (cfg.skipHitPoses && *(volatile int*) (hitBuffer + poseIdx) != 0) {
            continue; // pose already in collision
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

        if (traceHit<VARIANT>(newOrigin, rayDir, distance)) {
            atomicAdd(hitBuffer + poseIdx, 1);
        }
    }
}

extern "C" __global__ void __raygen__robotRayCompact() {
    robotRays<HIT_CLOSEST>();
}
extern "C" __global__ void __raygen__robotRayFirstHitCH() {
    robotRays<HIT_FIRST_CH>();
}
extern "C" __global__ void __raygen__robotRayFirstHitMiss() {
    robotRays<HIT_FIRST_MISS>();
}
extern "C" __global__ void __raygen__robotRayFirstHitTraverse() {
    robotRays<HIT_FIRST_TRAVERSE>();
}
