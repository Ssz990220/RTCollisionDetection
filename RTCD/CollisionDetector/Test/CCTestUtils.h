#include <CollisionDetector/CCRobotWrapper.h>
#include <CollisionDetector/CollisionDetector.h>
// #include <CollisionScenes/Scene/pipeScene.h>
// #include <CollisionScenes/Scene/curoboTest.h>
// #include <Robot/models/jaka.h>
#include <Robot/models/panda.h>
#include <Utils/Test/testUtils.h>
#include <chrono>
inline constexpr size_t trajLength = 4096;

void save_cc_result(const std::string& filename, CUDABuffer& resultBuffer, const size_t trajLength) {
    // save the result to a .bin file, if folder does not exist, create one
    std::string folder = CONCAT_PATHS(PROJECT_BASE_DIR, "/data/CCTest");
    if (!std::filesystem::exists(folder)) {
        std::filesystem::create_directory(folder);
    }
    std::ofstream file(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/CCTest/" + filename), std::ios::binary);
    std::vector<int> result(trajLength);
    resultBuffer.download(result);
    file.write(reinterpret_cast<char*>(result.data()), trajLength * sizeof(int));
    file.close();
}

template <typename T>
void save_cc_result(const std::string& filename, std::vector<T>& result, const size_t trajLength) {
    // save the result to a .bin file, if folder does not exist, create one
    assert(result.size() == trajLength && "Result size does not match trajLength");
    std::string folder = CONCAT_PATHS(PROJECT_BASE_DIR, "/data/CCTest");
    if (!std::filesystem::exists(folder)) {
        std::filesystem::create_directory(folder);
    }
    std::ofstream file(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/CCTest/" + filename), std::ios::binary);
    file.write(reinterpret_cast<char*>(result.data()), trajLength * sizeof(T));
    file.close();
}

template <size_t NSTREAM, LinkType TYPE, int BATCHSIZE>
void benchmark(std::unique_ptr<CCWrappedRobot<NSTREAM, TYPE>>& ccRobot, CollisionDetector<NSTREAM, TYPE>& CC) {
    // Benchmark
    std::vector<size_t> trajLengths{32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (auto& tl : trajLengths) {
        size_t nB = tl / BATCHSIZE;
        if (tl < BATCHSIZE) {
            continue; // skip small trajLengths
        }

        auto elipsedTime = 0;
        for (int t = 0; t < 100; t++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < nB; ++i) {
                ccRobot->movePosesToStream(i);
                ccRobot->update(i);
                CC.detect(i);
            }
            CUDA_SYNC_CHECK();
            auto end_time = std::chrono::high_resolution_clock::now();
            elipsedTime += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        }
        std::cout << "Elapsed time: " << elipsedTime / 100 << " us for " << tl << " poses" << std::endl;
    }
}

template <size_t NSTREAM, LinkType TYPE, int BATCHSIZE>
void benchmark(std::unique_ptr<CCRayRobot<NSTREAM, TYPE>>& ccRobot, CollisionDetector<NSTREAM, TYPE>& CC) {
    // Benchmark
    std::vector<size_t> trajLengths{32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (auto& tl : trajLengths) {
        size_t nB = tl / BATCHSIZE;
        if (tl < BATCHSIZE) {
            continue; // skip small trajLengths
        }

        auto elipsedTime = 0;
        for (int t = 0; t < 100; t++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < nB; ++i) {
                ccRobot->movePosesToStream(i);
                ccRobot->update(i);
                CC.detect(i);
            }
            CUDA_SYNC_CHECK();
            auto end_time = std::chrono::high_resolution_clock::now();
            elipsedTime += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        }
        std::cout << "Elapsed time: " << elipsedTime / 100 << " us for " << tl << " poses" << std::endl;
    }
}
