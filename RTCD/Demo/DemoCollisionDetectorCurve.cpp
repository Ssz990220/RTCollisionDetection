#if defined(_DEBUG) || defined(DEBUG)
#define PRINTIT
#else
#define USE_MULTI_THREADING
#endif
#include <CollisionDetector/CCRobotWrapper.h>
#include <CollisionDetector/CollisionDetector.h>
#include <CollisionDetector/Test/CCTestUtils.h>
#include <CollisionScenes/Scene/denseShelf.h>
#include <CollisionScenes/batchScene.h>
#include <Robot/batchCurveRobot.h>
#include <Robot/models/pandas.h>
#include <Utils/Test/testUtils.h>
#include <chrono>
#include <random>

inline constexpr size_t NSTREAM   = 4;
inline constexpr size_t BATCHSIZE = 128;
inline constexpr BuildType BUILD  = BuildType::FAST_BUILD;
#if defined(LINEAR)
inline constexpr LinkType TYPE    = LinkType::LINEAR_CURVE;
inline constexpr size_t nTrajPts  = 16;
inline constexpr size_t nCtrlPts  = 16;
#elif defined(QUAD)
inline constexpr LinkType TYPE    = LinkType::QUADRATIC_CURVE;
inline constexpr size_t nTrajPts  = 16;
inline constexpr size_t nCtrlPts  = 8;
#endif
inline constexpr batchRobotConfig cfg{NSTREAM, BATCHSIZE, Dim, BUILD, TYPE, nTrajPts, nCtrlPts, false};
inline constexpr batchSceneConfig sceneCfg(NSTREAM, BATCHSIZE);

inline constexpr std::array<float, Dim> weight{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

inline constexpr int nTrajs   = 4096;
inline constexpr size_t KNN_K = 4;
std::random_device rd;
std::mt19937 gen(rd());

using namespace RTCD;
template <int dim, int nTPts>
void sample_trajectories(
    const std::vector<std::array<float, dim>>& source, std::vector<std::array<float, dim>>& sampled, size_t M) {
    int total_trajectories = source.size() / nTPts;
    std::vector<int> indices(total_trajectories);
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, 2, ..., total_trajectories-1

    // Shuffle and pick the first M indices
    std::shuffle(indices.begin(), indices.end(), gen);

    sampled.clear();
    for (int i = 0; i < M; ++i) {
        int start_index = indices[i] * nTPts;
        for (int j = 0; j < nTPts; ++j) {
            sampled.push_back(source[start_index + j]);
        }
    }
}


int main() {


    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();

    // Robot Setup
    std::shared_ptr<robot<Dim>> r = std::make_shared<robot<Dim>>(Config);

    std::shared_ptr<baseBatchRobot<NSTREAM, TYPE>> batchRobotPtr =
        std::make_shared<batchOptixCurveRobot<cfg>>(r, streams, optixContext);

    auto ccRobot = std::make_shared<CCWrappedRobot<NSTREAM, TYPE>>(std::move(batchRobotPtr));

    std::vector<std::array<float, Dim>> traj(nTrajs * nTrajPts * KNN_K);
    std::vector<std::array<float, Dim>> sampledTrajs;
    genCSpaceKNNTraj<Dim, nTrajs, KNN_K>(traj, nTrajPts, upperBound, lowerBound, r->name, weight);

    ccRobot->prepareUpdateData((float*) traj.data(), nTrajs * KNN_K);
    // Scene Setup
    std::shared_ptr<scene> s                    = buildSharedScene(true);
    std::shared_ptr<baseBatchScene<NSTREAM>> bs = std::make_shared<batchScene<sceneCfg>>(s, streams, optixContext);

    // CC Setup
    CollisionDetector<NSTREAM, TYPE> CC(optixContext, streams);
    CC.setupScene(bs);
    CC.setupRobot(ccRobot);
    CC.setupOptix();
    CC.buildSBT();
    CC.setNCheck(nTrajs);


    CC.check();
    CUDA_SYNC_CHECK();

    std::vector<size_t> trajLengths{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (auto& tl : trajLengths) {
        CC.setNCheck(tl);
        sampledTrajs.clear();
        sample_trajectories<Dim, nTrajPts>(traj, sampledTrajs, tl);
        ccRobot->prepareUpdateData((float*) sampledTrajs.data(), tl);

        CC.check();
        CUDA_SYNC_CHECK();

        auto elipsedTime = 0;
        auto start_time  = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < 10; t++) {
            CC.check();
            CUDA_SYNC_CHECK();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        elipsedTime += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "Elapsed time: " << elipsedTime / 10 << " us for " << tl << " trajs" << std::endl;
    }
}
