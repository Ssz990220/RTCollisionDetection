// OptiX programs of the mesh backend (ObsToRobot): scene edge rays traced against the per-batch robot IAS.
// Ported from the old RTCD/CollisionDetector/CCCuda.cu (__raygen__oneside, __anyhit__IAS) with the launch
// parameters of rtcd::EdgeRayParams (rtcd/limits.h). Compiled to OptiX IR with the old flags (fast math, as the old
// cuda_compile_and_embed) and embedded into rtcd_backend_mesh.
#include <optix.h>

#include <rtcd/limits.h>

extern "C" __constant__ rtcd::EdgeRayParams RTCD_LAUNCH_PARAMS_NAME;

namespace {
    // Old normallength (CCCuda.cu): length and direction with the same float operations.
    __forceinline__ __device__ float3 normallength(const float3& a, float& l) {
        l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);

        const float invLen = 1.0f / l;
        return make_float3(a.x * invLen, a.y * invLen, a.z * invLen);
    }

    enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
} // namespace

// Old __raygen__oneside: ray idx runs from edge vertex 2 idx to 2 idx + 1, tmin 0, tmax = edge length, visibility
// mask 255 (instances culled by OBB get mask 0 on the masked path), any-hit enforced.
extern "C" __global__ void __raygen__edgeRays() {
    const rtcd::EdgeRayParams& p = RTCD_LAUNCH_PARAMS_NAME;
    const int ix                 = optixGetLaunchIndex().x;
    const int iy                 = optixGetLaunchIndex().y;
    const int idx                = ix + iy * optixGetLaunchDimensions().x;

    const float3* vertices = reinterpret_cast<const float3*>(p.edgeVertices);
    const float3 origin    = vertices[idx * 2];
    const float3 target    = vertices[idx * 2 + 1];

    float distance;
    const float3 d      = make_float3(target.x - origin.x, target.y - origin.y, target.z - origin.z);
    const float3 rayDir = normallength(d, distance);

    optixTrace(p.traversable, origin, rayDir,
        0.f,      // tmin
        distance, // tmax
        0.0f,     // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
        SURFACE_RAY_TYPE, // SBT offset
        RAY_TYPE_COUNT,   // SBT stride
        SURFACE_RAY_TYPE); // miss SBT index
}

// Old __anyhit__IAS: one hit for the item owning the instance, then keep traversing.
extern "C" __global__ void __anyhit__instanceItem() {
    const rtcd::EdgeRayParams& p = RTCD_LAUNCH_PARAMS_NAME;
    const unsigned int instance  = optixGetInstanceIndex();
    const unsigned int* map      = reinterpret_cast<const unsigned int*>(p.primToItem);
    atomicAdd(reinterpret_cast<unsigned int*>(p.hitCounts) + map[instance], 1u);
    optixIgnoreIntersection();
}
