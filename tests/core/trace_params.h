#pragma once
// Launch parameters of tests/core/trace_program.cu (shared by the OptiX program and test_core_accel.cpp).

struct TestRay {
    float origin[3];
    float direction[3]; // unit length
    float tmax;
    float pad;
};

struct TraceParams {
    unsigned long long traversable;
    unsigned long long rays; // const TestRay*
    unsigned long long hits; // unsigned int*: 0 = miss, else 0x80000000 | instanceIndex << 16 | primitiveIndex
    unsigned int count;
    unsigned int pad;
};
