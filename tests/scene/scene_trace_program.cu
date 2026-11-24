// Test program for test_scene_gpu: traces segments (origin -> target, t in [0, 1]) against Scene::traversable()
// and records, per segment, the SBT record data and the instance index of the closest hit (0 = miss).
#include <optix.h>

#include "scene_trace_params.h"

extern "C" __constant__ SceneTraceParams rtcdLaunchParams;

extern "C" __global__ void __raygen__trace() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (i >= rtcdLaunchParams.count) {
        return;
    }
    const float* r = rtcdLaunchParams.rays + 6 * i;
    const float3 o = make_float3(r[0], r[1], r[2]);
    const float3 d = make_float3(r[3] - r[0], r[4] - r[1], r[5] - r[2]);
    unsigned int p0 = 0, p1 = 0;
    optixTrace(rtcdLaunchParams.handle, o, d, 0.0f, 1.0f, 0.0f, 255, OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1);
    rtcdLaunchParams.sbtData[i]  = p0;
    rtcdLaunchParams.instance[i] = p1;
}

extern "C" __global__ void __closesthit__trace() {
    const unsigned int data = *reinterpret_cast<const unsigned int*>(optixGetSbtDataPointer());
    optixSetPayload_0(data + 1);
    optixSetPayload_1(optixGetInstanceIndex() + 1);
}

extern "C" __global__ void __miss__trace() {}
