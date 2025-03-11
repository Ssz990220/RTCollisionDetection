#include "launchParameters.h"
#include <Utils/cuUtils.cuh>
#include <optix.h>
#include <optix_device.h>
// #define RENDER

extern "C" __constant__ LaunchParams optixLaunchParams;

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };


static __forceinline__ __device__ float3 normallength(const float3& a, float& l) {
    l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);

    float invLen = 1.0f / l;
    return make_float3(a.x * invLen, a.y * invLen, a.z * invLen);
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------


extern "C" __global__ void __anyhit__IAS() {
    unsigned int primID = optixGetInstanceIndex();
    atomicAdd((int*) optixLaunchParams.hitBuffer + optixLaunchParams.mesh.primIdxToPoseIdx[primID], 1);
    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__GAS() {
    unsigned int primID = optixGetPrimitiveIndex();
    atomicAdd((int*) optixLaunchParams.hitBuffer + optixLaunchParams.mesh.primIdxToPoseIdx[primID], 1);
    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__RayData() {
    const int ix  = optixGetLaunchIndex().x;
    const int iy  = optixGetLaunchIndex().y;
    const int idx = ix + iy * optixGetLaunchDimensions().x;
    atomicAdd((int*) optixLaunchParams.hitBuffer + idx, 1);
}


extern "C" __global__ void __miss__radiance() {}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__oneside() {
    // compute a test pattern based on pixel ID
    const int ix  = optixGetLaunchIndex().x;
    const int iy  = optixGetLaunchIndex().y;
    const int idx = ix + iy * optixGetLaunchDimensions().x;

    const float3 origin = optixLaunchParams.verticesBuffer[idx * 2];
    const float3 target = optixLaunchParams.verticesBuffer[idx * 2 + 1];

    float distance;
    const float3 rayDir = normallength(target - origin, distance);

    optixTrace(optixLaunchParams.traversable, origin, rayDir,
        0.f, // tmin
        distance, // tmax
        0.0f, // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_ENFORCE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE, // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        SURFACE_RAY_TYPE); //, // missSBTIndex
}

extern "C" __global__ void __raygen__doubleside() {
    // compute a test pattern based on pixel ID
    const int ix  = optixGetLaunchIndex().x;
    const int iy  = optixGetLaunchIndex().y;
    const int idx = ix + iy * optixGetLaunchDimensions().x;

    const int rayId = idx / 2;

    float3 origin, target;

    if (idx & 1 == 0) {
        // forward ray
        origin = optixLaunchParams.verticesBuffer[rayId * 2];
        target = optixLaunchParams.verticesBuffer[rayId * 2 + 1];
    } else {
        // backward ray
        origin = optixLaunchParams.verticesBuffer[rayId * 2 + 1];
        target = optixLaunchParams.verticesBuffer[rayId * 2];
    }

    float distance;
    const float3 rayDir = normallength(target - origin, distance);

    // uint u0 = 0;
    // uint u1 = 0;

    optixTrace(optixLaunchParams.traversable, origin, rayDir,
        0.f, // tmin
        distance, // tmax
        0.0f, // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_ENFORCE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE, // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        SURFACE_RAY_TYPE); //, // missSBTIndex
}
