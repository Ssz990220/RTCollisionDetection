// OptiX programs of the curve backend (compiled to OptiX IR and embedded, rtcd_add_optix_ir in CMakeLists.txt).
//
// Ports of __raygen__oneside and __anyhit__GAS of RTCD/CollisionDetector/CCCuda.cu, reading rtcd::EdgeRayParams
// (rtcd/limits.h) instead of the old LaunchParams union:
//   raygen: ray r = launch index x (+ y * width; launches are 1-D) from edgeVertices[2r] to edgeVertices[2r + 1],
//           tmin 0, tmax = edge length, ENFORCE_ANYHIT, SBT offset 0 / stride 1 / miss 0, no payload;
//   any-hit: hitCounts[primToItem[optixGetPrimitiveIndex()]] += 1, then ignore the intersection (every hit along
//           the ray is counted; collision flags = count > 0).
// The built-in curve intersection program comes from the core (ProgramSet::builtinIntersection).
#include <rtcd/limits.h>

#include <optix.h>
#include <optix_device.h>

extern "C" __constant__ rtcd::EdgeRayParams RTCD_LAUNCH_PARAMS_NAME;

namespace {
    // old normallength (CCCuda.cu)
    __forceinline__ __device__ float3 normallength(const float3& a, float& l) {
        l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);

        float invLen = 1.0f / l;
        return make_float3(a.x * invLen, a.y * invLen, a.z * invLen);
    }
} // namespace

extern "C" __global__ void __anyhit__GAS() {
    const rtcd::EdgeRayParams& p  = RTCD_LAUNCH_PARAMS_NAME;
    const unsigned int primID     = optixGetPrimitiveIndex();
    const unsigned int* primToItem = reinterpret_cast<const unsigned int*>(p.primToItem);
    atomicAdd(reinterpret_cast<unsigned int*>(p.hitCounts) + primToItem[primID], 1u);
    optixIgnoreIntersection();
}

extern "C" __global__ void __raygen__oneside() {
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
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
        0,  // SBT offset
        1,  // SBT stride
        0); // miss SBT index
}
