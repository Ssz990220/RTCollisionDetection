#pragma once
// Launch parameters of scene_trace_program.cu (test_scene_gpu only).
#include <cstdint>

struct SceneTraceParams {
    unsigned long long handle; // OptixTraversableHandle
    const float* rays;         // 6 floats per segment
    unsigned int count;
    unsigned int* sbtData;     // closest hit: SBT record data + 1, 0 = miss
    unsigned int* instance;    // closest hit: instance index + 1, 0 = miss
};
