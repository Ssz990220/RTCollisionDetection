#include <Utils/CUDABuffer.h>
#include <Utils/cuda/cudaAlgorithms.cuh>
#include <algorithm>
#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace RTCD;

class StreamCompactTest : public ::testing::Test {
protected:
    CUDABuffer temp;
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen; // Seed the generator
    std::uniform_int_distribution<> distrib; // Define the range

    virtual void SetUp() {
        temp.alloc(8192 * sizeof(uint));
        gen.seed(rd());
        distrib = std::uniform_int_distribution<>(0, 3);
    }

    template <bool Flag, size_t nData>
    void runStreamCompactTest() {

        auto rand_num = [&]() { return distrib(gen); };
        std::vector<uint> streamData(nData);
        std::vector<uint> isInValid(nData);

        std::generate(streamData.begin(), streamData.end(), rand_num);
        std::generate(isInValid.begin(), isInValid.end(), rand_num);

        CUDABuffer d_streamData;
        CUDABuffer d_isInValid;

        d_streamData.alloc_and_upload(streamData);
        d_isInValid.alloc_and_upload(isInValid);

        CUDABuffer d_streamDataOut;
        CUDABuffer d_whereInDst;
        d_streamDataOut.alloc(streamData.size() * sizeof(uint));
        d_whereInDst.alloc(nData * sizeof(uint));

        streamCompact<Flag>(d_streamData.d_pointer(), d_isInValid.d_pointer(), temp.d_pointer(), sizeof(uint), nData,
            d_streamDataOut.d_pointer(), d_whereInDst.d_pointer(), 0);
        CUDA_SYNC_CHECK();

        std::vector<uint> result;
        d_streamDataOut.download(result);

        std::vector<uint> cpuResult;
        for (int i = 0; i < nData; i++) {
            if ((Flag && isInValid[i] > 0) || (!Flag && isInValid[i] == 0)) {
                cpuResult.push_back(streamData[i]);
            }
        }

        ASSERT_TRUE(std::equal(cpuResult.begin(), cpuResult.end(), result.begin()))
            << "Test failed for vary size " << Flag;
    }
};

TEST_F(StreamCompactTest, VarySizeUniformElements) {
    // std::vector<uint> lengths = {4, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2047, 2048, 2049, 4095, 4096, 4097};
    // for (const auto& length : lengths) {
    runStreamCompactTest<false, 4>();
    runStreamCompactTest<true, 4>();
    runStreamCompactTest<false, 7>();
    runStreamCompactTest<true, 7>();
    runStreamCompactTest<false, 8>();
    runStreamCompactTest<true, 8>();
    runStreamCompactTest<false, 16>();
    runStreamCompactTest<true, 16>();
    runStreamCompactTest<false, 32>();
    runStreamCompactTest<true, 32>();
    runStreamCompactTest<false, 64>();
    runStreamCompactTest<true, 64>();
    runStreamCompactTest<false, 128>();
    runStreamCompactTest<true, 128>();
    runStreamCompactTest<false, 256>();
    runStreamCompactTest<true, 256>();
    runStreamCompactTest<false, 512>();
    runStreamCompactTest<true, 512>();
    runStreamCompactTest<false, 1024>();
    runStreamCompactTest<true, 1024>();
    runStreamCompactTest<false, 2047>();
    runStreamCompactTest<true, 2047>();
    runStreamCompactTest<false, 2048>();
    runStreamCompactTest<true, 2048>();
    runStreamCompactTest<false, 2049>();
    runStreamCompactTest<true, 2049>();
    runStreamCompactTest<false, 4095>();
    runStreamCompactTest<true, 4095>();
    runStreamCompactTest<false, 4096>();
    runStreamCompactTest<true, 4096>();
    runStreamCompactTest<false, 4097>();
    runStreamCompactTest<true, 4097>();

    // }
}
