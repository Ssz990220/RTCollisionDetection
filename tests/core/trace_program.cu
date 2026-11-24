// OptiX test program for test_core_accel: one ray per launch index, closest hit records instance / primitive.
#include <optix.h>

#include "trace_params.h"

extern "C" __constant__ TraceParams rtcdLaunchParams;

extern "C" __global__ void __raygen__trace() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (i >= rtcdLaunchParams.count) {
        return;
    }
    const TestRay r = reinterpret_cast<const TestRay*>(rtcdLaunchParams.rays)[i];
    unsigned int p0 = 0;
    optixTrace(rtcdLaunchParams.traversable, make_float3(r.origin[0], r.origin[1], r.origin[2]),
        make_float3(r.direction[0], r.direction[1], r.direction[2]), 0.0f, r.tmax, 0.0f, OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0);
    reinterpret_cast<unsigned int*>(rtcdLaunchParams.hits)[i] = p0;
}

extern "C" __global__ void __closesthit__trace() {
    optixSetPayload_0(0x80000000u | (optixGetInstanceIndex() << 16) | optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__trace() { optixSetPayload_0(0u); }
