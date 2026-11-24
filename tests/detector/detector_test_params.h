#pragma once
// Launch parameters of detector_test_program.cu (host and device). Not a backend layout.
namespace rtcd_detector_test {
    struct TestParams {
        unsigned long long traversable;
        unsigned long long items;     // const float*
        unsigned long long hitCounts; // unsigned int*
        unsigned int itemCount;
        unsigned int floatsPerItem;
        unsigned int addValue;
        unsigned int batchIndex;
        long long spinCycles;         // __raygen__spin: clock64 cycles to busy-wait before writing
    };
    static_assert(sizeof(TestParams) == 48, "TestParams layout");
} // namespace rtcd_detector_test
