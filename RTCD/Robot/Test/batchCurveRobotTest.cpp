#include <Robot/batchCurveRobot.h>
#include <Robot/models/panda.h>
#include <Utils/Test/testUtils.h>
#include <random>

inline constexpr size_t NSTREAM       = 2;
inline constexpr size_t BATCHSIZE     = 16;
inline constexpr BuildType BUILD      = BuildType::FAST_BUILD;
inline constexpr LinkType TYPE        = LinkType::QUADRATIC_CURVE;
inline constexpr size_t nCPts         = 8;
inline constexpr size_t nTPts         = 16;
inline constexpr size_t nCSpaceSample = 8192;
inline constexpr int KNN_K            = 4;
inline constexpr std::array<float, Dim> weight{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
inline constexpr batchRobotConfig cfg{NSTREAM, BATCHSIZE, Dim, BUILD, TYPE, nTPts, nCPts};


using namespace RTCD;
int main() {
    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();
    // Robot Setup
    std::shared_ptr<robot<Dim>> r = std::make_shared<robot<Dim>>(Config);

    auto batchRobotPtr = std::make_shared<batchOptixCurveRobot<cfg>>(r, streams, optixContext);

    std::vector<std::array<float, Dim>> traj;
    traj.resize(nCSpaceSample * nTPts * KNN_K);
    genCSpaceKNNTraj<Dim, nCSpaceSample, KNN_K>(traj, nTPts, upperBound, lowerBound, r->name, weight);

    CUDABuffer trajs;
    trajs.alloc_and_upload(traj.data(), nTPts * BATCHSIZE);
    CUDA_SYNC_CHECK();

    // Test Fkine
    batchRobotPtr->fkine(trajs.d_pointer(), 0);
    CUDA_SYNC_CHECK();

    // Download OBBs for Viz
    std::vector<OBB> obbs;
    batchRobotPtr->downloadOBBs(0, obbs);
    // Write OBB to file
    std::string outputDir = CONCAT_PATHS(PROJECT_BASE_DIR, "/data/curve/test/");
    writeBin(std::format("{}curveOBBs.bin", outputDir), obbs);

    // Test Partial Build
    std::vector<uint> mask(BATCHSIZE * 62 * 4, 0);
    std::random_device rd; // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> distrib(0, 4); // Define the range [0, 3]
    std::generate(mask.begin(), mask.end(), [&]() { return distrib(gen); });
    CUDABuffer dMask;
    dMask.alloc_and_upload(mask.data(), BATCHSIZE * 62 * 4);

    batchRobotPtr->updateWithMask(dMask.d_pointer(), 0);
    CUDA_SYNC_CHECK();
}
