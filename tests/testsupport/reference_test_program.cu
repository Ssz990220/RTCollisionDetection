// Trivial OptiX program module for test_reference_executor (tests/testsupport). Not a backend.
#include <optix.h>

#include "reference_test_params.h"

extern "C" __constant__ rtcd_test::TestParams rtcdLaunchParams;

// hitCounts[i] += (uint) items[i * floatsPerItem] + addValue, for i < itemCount.
extern "C" __global__ void __raygen__copyItems() {
    const unsigned int i = optixGetLaunchIndex().x;
    const rtcd_test::TestParams& p = rtcdLaunchParams;
    if (i >= p.itemCount) {
        return;
    }
    const float* items  = reinterpret_cast<const float*>(p.items);
    unsigned int* hits  = reinterpret_cast<unsigned int*>(p.hitCounts);
    atomicAdd(hits + i, static_cast<unsigned int>(items[i * p.floatsPerItem]) + p.addValue);
}

// Ray i: origin (items[i * floatsPerItem], 0, -1), direction +z, tmax 2, against the GAS in p.traversable.
// The any-hit counts every triangle crossed.
extern "C" __global__ void __raygen__trace() {
    const unsigned int i = optixGetLaunchIndex().x;
    const rtcd_test::TestParams& p = rtcdLaunchParams;
    if (i >= p.itemCount) {
        return;
    }
    const float* items = reinterpret_cast<const float*>(p.items);
    const float3 origin = make_float3(items[i * p.floatsPerItem], 0.0f, -1.0f);
    const float3 dir    = make_float3(0.0f, 0.0f, 1.0f);
    optixTrace(p.traversable, origin, dir, 0.0f, 2.0f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0);
}

extern "C" __global__ void __anyhit__count() {
    const rtcd_test::TestParams& p = rtcdLaunchParams;
    unsigned int* hits             = reinterpret_cast<unsigned int*>(p.hitCounts);
    atomicAdd(hits + optixGetLaunchIndex().x, 1u);
    optixIgnoreIntersection();
}
