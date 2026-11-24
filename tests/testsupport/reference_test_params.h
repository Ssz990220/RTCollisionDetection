#pragma once
// Launch parameters of reference_test_program.cu (host and device).
namespace rtcd_test {
    struct TestParams {
        unsigned long long traversable;
        unsigned long long items;     // const float*
        unsigned long long hitCounts; // unsigned int*
        unsigned int itemCount;
        unsigned int floatsPerItem;
        unsigned int addValue;
        unsigned int reserved;
    };
    static_assert(sizeof(TestParams) == 40, "TestParams layout");
} // namespace rtcd_test
