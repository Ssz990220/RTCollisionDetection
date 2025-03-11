#include <Utils/CUDABuffer.h>
#include <Utils/cuda/cudaAlgorithms.cuh>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace RTCD;
inline constexpr uint MAX_ELEMENT_CNT = 65536;

class ExclusiveScanTest : public ::testing::Test {
protected:
    CUDABuffer d_input;
    CUDABuffer d_output;
    CUDABuffer temp;
    // std::random_device rd; // Obtain a random number from hardware
    // std::mt19937 gen; // Seed the generator
    // std::uniform_int_distribution<> distrib; // Define the range

    virtual void SetUp() {
        temp.alloc(8192 * sizeof(uint));
        // gen.seed(rd());
        // distrib = std::uniform_int_distribution<>(0, 3);
    }

    template <size_t nData>
    void runExclusiveScan() {
        std::vector<uint> input(nData);

        // std::generate(input.begin(), input.end(), rand_num);
        std::fill(input.begin(), input.end(), 1);

        d_input.free();
        d_input.alloc_and_upload(input);

        d_output.free();

        // if (nData <= 4 * THREADBLOCK_SIZE) {
        d_output.alloc(roundUpLog2(nData) * sizeof(uint));
        // } else {
        //     d_output.alloc(roundToShortArrSize(nData) * sizeof(uint));
        // }

        exclusiveScan(d_output.d_pointer(), d_input.d_pointer(), temp.d_pointer(), nData, 0);
        CUDA_SYNC_CHECK();

        std::vector<uint> result;
        d_output.download(result);

        std::vector<uint> cpuResult;
        cpuResult.resize(nData);
        std::exclusive_scan(input.begin(), input.end(), cpuResult.begin(), 0);
        ASSERT_TRUE(std::equal(cpuResult.begin(), cpuResult.end(), result.begin()))
            << "Test failed for vary size " << nData;
    }
};

TEST_F(ExclusiveScanTest, VarySizeUniformElements) {
    // std::vector<uint> lengths = {4, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2047, 2048, 2049, 4095, 4096, 4097};
    // for (const auto& length : lengths) {
    runExclusiveScan<4>();
    runExclusiveScan<5>();
    runExclusiveScan<8>();
    runExclusiveScan<9>();
    runExclusiveScan<16>();
    runExclusiveScan<17>();
    runExclusiveScan<90>();
    runExclusiveScan<2049>();
    runExclusiveScan<4095>();
    runExclusiveScan<4096>();
    runExclusiveScan<4097>();
    runExclusiveScan<6145>();
    runExclusiveScan<8192>();
    runExclusiveScan<8191>();
    runExclusiveScan<8193>();
    runExclusiveScan<10239>();
    runExclusiveScan<10240>();
    runExclusiveScan<10241>();
    runExclusiveScan<41616>();

    // }
}
