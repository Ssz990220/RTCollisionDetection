#define PRINTIT
#include <Robot/batchIASRobot.h>
#include <Robot/models/pandas.h>
#include <Utils/Test/testUtils.h>
#include <random>

inline constexpr size_t NSTREAM   = 2;
inline constexpr size_t BATCHSIZE = 16;
inline constexpr batchRobotConfig batchConfig{
    NSTREAM, BATCHSIZE, Dim, BuildType::COMPACT, LinkType::SPHERES, false, true};

using namespace RTCD;
int main() {

    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();

    std::shared_ptr<robot<Dim>> r = std::make_shared<robot<Dim>>(Config);

    auto batchR = batchIASRobot<batchConfig>(r, streams, optixContext);

    // prepare poses
    std::vector<std::array<float, Dim>> traj;
    randomMotions(4096, traj, lowerBound, upperBound, r->name);
    CUDABuffer trajBuffer;
    trajBuffer.alloc_and_upload(traj);


#if defined(_DEBUG) || defined(DEBUG)
    batchR.fkine(trajBuffer.d_pointer(), 0);
    CUDA_SYNC_CHECK();
    // download the OBBs for visual check
    std::vector<RTCD::OBB> obbs;
    batchR.downloadOBB(obbs, 0);
    writeBin(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/obbs.bin"), obbs);
    std::cout << "OBBs saved to " << CONCAT_PATHS(PROJECT_BASE_DIR, "/data/obbs.bin") << std::endl;
#endif

    // test update with mask
    std::vector<uint> mask(BATCHSIZE, 0);
    std::random_device rd; // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> distrib(0, 4); // Define the range [0, 3]
    std::generate(mask.begin(), mask.end(), [&]() { return distrib(gen); });

    uint nValid = std::count_if(mask.begin(), mask.end(), [&](uint& m) { return m > 0; });
    std::cout << "Number of valid links: " << nValid << std::endl;

    CUDABuffer dMask;
    dMask.alloc_and_upload(mask);

    batchR.updateWithMask(dMask.d_pointer(), 0);
    CUDA_SYNC_CHECK();

    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}
