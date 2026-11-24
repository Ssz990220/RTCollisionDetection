// Trivial OptiX program module for the detector-core tests (tests/detector). Not a backend.
#include <optix.h>

#include "detector_test_params.h"

using rtcd_detector_test::TestParams;

extern "C" __constant__ TestParams rtcdLaunchParams;

namespace {
    __device__ unsigned int* hits() { return reinterpret_cast<unsigned int*>(rtcdLaunchParams.hitCounts); }
    __device__ float item0(unsigned int i) {
        return reinterpret_cast<const float*>(rtcdLaunchParams.items)[i * rtcdLaunchParams.floatsPerItem];
    }
} // namespace

// hitCounts[i] += (uint) items[i * floatsPerItem] + addValue, for i < itemCount.
extern "C" __global__ void __raygen__copyItems() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (i >= rtcdLaunchParams.itemCount) {
        return;
    }
    atomicAdd(hits() + i, static_cast<unsigned int>(item0(i)) + rtcdLaunchParams.addValue);
}

// Same as copyItems, plus 1000 * (batchIndex + 1) (distinguishes raygen selection and batch mapping).
extern "C" __global__ void __raygen__copyItemsBatch() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (i >= rtcdLaunchParams.itemCount) {
        return;
    }
    atomicAdd(hits() + i, static_cast<unsigned int>(item0(i)) + 1000u * (rtcdLaunchParams.batchIndex + 1u));
}

// Busy-waits spinCycles, then behaves as copyItems (keeps a stream busy for the drain tests).
extern "C" __global__ void __raygen__spin() {
    const unsigned int i = optixGetLaunchIndex().x;
    const long long start = clock64();
    while (clock64() - start < rtcdLaunchParams.spinCycles) {
    }
    if (i >= rtcdLaunchParams.itemCount) {
        return;
    }
    atomicAdd(hits() + i, static_cast<unsigned int>(item0(i)) + rtcdLaunchParams.addValue);
}

// Ray i: origin (items[i * floatsPerItem], 0, -1), direction +z, tmax 2, against p.traversable; the any-hit counts
// every primitive crossed.
extern "C" __global__ void __raygen__trace() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (i >= rtcdLaunchParams.itemCount) {
        return;
    }
    const float3 origin = make_float3(item0(i), 0.0f, -1.0f);
    const float3 dir    = make_float3(0.0f, 0.0f, 1.0f);
    unsigned int p0     = 0;
    optixTrace(rtcdLaunchParams.traversable, origin, dir, 0.0f, 2.0f, 0.0f, OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0);
}

// Closest-hit / miss through the payload: hitCounts[i] += 5 on a hit (CH), += 3 on a miss (miss program). Any-hit is
// disabled so the closest-hit runs.
extern "C" __global__ void __raygen__traceChMiss() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (i >= rtcdLaunchParams.itemCount) {
        return;
    }
    const float3 origin = make_float3(item0(i), 0.0f, -1.0f);
    const float3 dir    = make_float3(0.0f, 0.0f, 1.0f);
    unsigned int p0     = 0;
    optixTrace(rtcdLaunchParams.traversable, origin, dir, 0.0f, 2.0f, 0.0f, OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0);
    atomicAdd(hits() + i, p0);
}

extern "C" __global__ void __anyhit__count() {
    atomicAdd(hits() + optixGetLaunchIndex().x, 1u);
    optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__mark() { optixSetPayload_0(5u); }

extern "C" __global__ void __miss__mark() { optixSetPayload_0(3u); }
