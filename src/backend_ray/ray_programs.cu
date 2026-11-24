// OptiX programs of the ray backend (Mode::Ray, RobotToObs). Port of the old RTCD/CollisionDetector/CCRay.cu
// (robotRays<VARIANT>, the four raygen entry points, __closesthit__radiance, __miss__radiance) onto
// rtcd::RobotRayParams (rtcd/limits.h). Compiled to OptiX IR and embedded into rtcd_backend_ray.
//
// Ray id = item * raysPerRobot + rayInRobot over the whole batch (totalRayCount = itemCount * raysPerRobot ids).
// The launch width may be smaller than totalRayCount (RayOptions::launchWidthCap): each launch index walks the
// ids with a stride of the launch width, so a skipped ray costs a loop iteration instead of a launched thread, and
// consecutive launch indices stay on consecutive rays (same item and link) in every iteration.
// A ray is traced only if its link's OBB overlaps an obstacle OBB (linkMask > 0; an all-ones mask when OBB culling
// is off) and, with skipHitItems, its item is not already known to collide (volatile read of the hit count).
// No __syncthreads(): OptiX programs have no CUDA block semantics.
//
// Hit test variants (the host picks the raygen entry point from RayOptions::hitMode). Each only needs yes/no per
// ray; any-hit is disabled for all of them (the scene GASes are also built with DISABLE_ANYHIT), so every variant
// reports a hit exactly when some triangle intersects the segment [0, distance].
#include <rtcd/limits.h>

#include <optix.h>
#include <optix_device.h>

extern "C" __constant__ rtcd::RobotRayParams RTCD_LAUNCH_PARAMS_NAME;

namespace {

    enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

    enum RayHitVariant {
        HIT_CLOSEST        = 0, // optixTrace, closest-hit search, CH sets payload 1 (no miss program, payload stays 0)
        HIT_FIRST_CH       = 1, // + TERMINATE_ON_FIRST_HIT
        HIT_FIRST_MISS     = 2, // TERMINATE_ON_FIRST_HIT | DISABLE_CLOSESTHIT, payload starts at 1, miss clears it
        HIT_FIRST_TRAVERSE = 3, // optixTraverse + TERMINATE_ON_FIRST_HIT, optixHitObjectIsHit(); no CH / miss runs
    };

    __forceinline__ __device__ float3 sub(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __forceinline__ __device__ float3 normallength(const float3& a, float& l) {
        l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);

        float invLen = 1.0f / l;
        return make_float3(a.x * invLen, a.y * invLen, a.z * invLen);
    }

    __forceinline__ __device__ void transformPoint(const float* transform, const float3& point, float3& result) {
        result.x = transform[0] * point.x + transform[1] * point.y + transform[2] * point.z + transform[3];
        result.y = transform[4] * point.x + transform[5] * point.y + transform[6] * point.z + transform[7];
        result.z = transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11];
    }

    template <int VARIANT>
    __forceinline__ __device__ bool traceHit(const float3& origin, const float3& dir, const float tmax) {
        const OptixTraversableHandle handle = static_cast<OptixTraversableHandle>(RTCD_LAUNCH_PARAMS_NAME.traversable);
        if constexpr (VARIANT == HIT_FIRST_TRAVERSE) {
            optixTraverse(handle, origin, dir,
                0.f,  // tmin
                tmax, // tmax
                0.0f, // rayTime
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                SURFACE_RAY_TYPE,  // SBT offset
                RAY_TYPE_COUNT,    // SBT stride
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
            optixTrace(handle, origin, dir,
                0.f,  // tmin
                tmax, // tmax
                0.0f, // rayTime
                OptixVisibilityMask(255), flags,
                SURFACE_RAY_TYPE, // SBT offset
                RAY_TYPE_COUNT,   // SBT stride
                SURFACE_RAY_TYPE, // missSBTIndex
                u0);
            return u0 == 1;
        }
    }

    template <int VARIANT>
    __forceinline__ __device__ void robotRays() {
        const rtcd::RobotRayParams& p = RTCD_LAUNCH_PARAMS_NAME;
        const unsigned int total      = p.totalRayCount;
        const unsigned int stride     = optixGetLaunchDimensions().x;
        const unsigned int raysPer    = p.raysPerRobot;
        const unsigned int linksPer   = p.linksPerRobot;
        unsigned int* hitCounts       = reinterpret_cast<unsigned int*>(p.hitCounts);
        const unsigned int* linkMask  = reinterpret_cast<const unsigned int*>(p.linkMask);
        const int* rayLink            = reinterpret_cast<const int*>(p.rayLink);
        const float3* rays            = reinterpret_cast<const float3*>(p.rays);
        const float* linkPoses        = reinterpret_cast<const float*>(p.linkPoses);

        for (unsigned int id = optixGetLaunchIndex().x; id < total; id += stride) {
            const unsigned int item    = id / raysPer;
            const unsigned int rayIdx  = id - item * raysPer;
            const unsigned int linkIdx = static_cast<unsigned int>(rayLink[rayIdx]);
            const unsigned int glbLink = item * linksPer + linkIdx;
            if (linkMask[glbLink] == 0) {
                continue; // link OBB is collision free: its rays cannot hit anything
            }
            if (p.skipHitItems && *(volatile unsigned int*) (hitCounts + item) != 0) {
                continue; // item already in collision
            }
            const float* linkTransform = linkPoses + glbLink * 12;

            const float3 origin = rays[rayIdx * 2];
            const float3 target = rays[rayIdx * 2 + 1];

            float3 newOrigin, newTarget;
            transformPoint(linkTransform, origin, newOrigin);
            transformPoint(linkTransform, target, newTarget);

            float distance;
            float3 rayDir = normallength(sub(newTarget, newOrigin), distance);

            if (traceHit<VARIANT>(newOrigin, rayDir, distance)) {
                atomicAdd(hitCounts + item, 1u);
            }
        }
    }

} // namespace

extern "C" __global__ void __closesthit__radiance() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__radiance() {
    optixSetPayload_0(0);
}

// Entry names kept from the old CCRay.cu (RayHitMode -> entry: ClosestHit -> robotRayCompact, FirstHitCH ->
// robotRayFirstHitCH, FirstHitMiss -> robotRayFirstHitMiss, FirstHitTraverse -> robotRayFirstHitTraverse).
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
