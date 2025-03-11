#include <Robot/batchRayRobot.h>
#include <Robot/models/pandas.h>
#include <Utils/Test/testUtils.h>
#include <random>

inline constexpr size_t NSTREAM   = 4;
inline constexpr size_t BATCHSIZE = 1024;
inline constexpr batchRobotConfig batchConfig{NSTREAM, BATCHSIZE, Dim, BuildType::COMPACT, LinkType::RAY_ONLINE};

int main() {

    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();

    std::shared_ptr<robot<Dim>> r = std::make_shared<robot<Dim>>(Config);

    batchRayRobot bRR = batchRayRobot<batchConfig>(r, streams, optixContext);

    std::vector<std::array<float, Dim>> poses(4096);
    randomMotions(4096, poses, upperBound, lowerBound, r->name);

    CUDABuffer posesBuffer;
    posesBuffer.alloc_and_upload(poses);

    // test Ray Robot Update
    bRR.update(posesBuffer.d_pointer(), 0);
    CUDA_SYNC_CHECK();

    // test Ray Robot OBB updates
    bRR.fkine2(posesBuffer.d_pointer(), 0);
    bRR.updateOBBs2(0);
    CUDA_SYNC_CHECK();

    // test Ray Robot UpdateWithMask
    std::vector<uint> mask(BATCHSIZE * (Dim + 1), 0);
    std::random_device rd; // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> distrib(0, 4); // Define the range [0, 3]
    std::generate(mask.begin(), mask.end(), [&]() { return distrib(gen); });

    uint nValid = std::count_if(mask.begin(), mask.end(), [&](uint& m) { return m > 0; });
    std::cout << "Number of valid links: " << nValid << std::endl;

    CUDABuffer dMask;
    dMask.alloc_and_upload(mask);

    bRR.updateWithMask(dMask.d_pointer(), 0);
    CUDA_SYNC_CHECK();

    const meshRayInfo rayInfo = bRR.getRayInfo(0);

    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}
