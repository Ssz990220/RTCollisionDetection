#define USE_MULTI_THREADING

#define PRINTIT
#include <CollisionDetector/Test/CCTestUtils.h>
#include <CollisionScenes/batchScene.h>

inline constexpr size_t NSTREAM   = 4;
inline constexpr size_t BATCHSIZE = 1024; // Play with this number to see the performance difference
#if defined(IAS)
#include <Robot/batchIASRobot.h>
inline constexpr BuildType BUILD = BuildType::COMPACT;
inline constexpr LinkType TYPE   = LinkType::MESH;
#include <CollisionScenes/Scene/denseShelf.h>
#elif defined(RAYO)
#include <Robot/batchRayRobot.h>
inline constexpr BuildType BUILD = BuildType::NONE;
inline constexpr LinkType TYPE   = LinkType::RAY_ONLINE;
#include <CollisionScenes/Scene/denseShelf.h>
#endif

inline constexpr batchRobotConfig batchConfig(NSTREAM, BATCHSIZE, Dim, BUILD, TYPE, true);
inline constexpr batchSceneConfig sceneCfg(NSTREAM, BATCHSIZE);

std::random_device rd;
std::mt19937 gen(rd());

using namespace RTCD;
int main() {

    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();

// Robot Setup
#if defined(IAS)
    auto r = std::make_shared<robot<Dim>>(Config);
    std::shared_ptr<baseBatchRobot<NSTREAM, TYPE>> batchRobotPtr =
        std::make_shared<batchIASRobot<batchConfig>>(r, streams, optixContext);
    auto ccRobot = std::make_shared<CCWrappedRobot<NSTREAM, TYPE>>(batchRobotPtr);
#elif defined(RAYO)
    auto r = std::make_shared<robot<Dim>>(Config);
    std::shared_ptr<baseRayRobot<NSTREAM, TYPE>> batchRobotPtr =
        std::make_shared<batchRayRobot<batchConfig>>(r, streams, optixContext);
    auto ccRobot = std::make_shared<CCRayRobot<NSTREAM, TYPE>>(batchRobotPtr);
#endif

    // Prepare Trajectory
    std::vector<std::array<float, Dim>> traj(4096);
    randomMotions(4096, traj, lowerBound, upperBound, r->name);

    std::vector<std::array<float, Dim>> sampledTrajs;

    // Scene Setup
    std::shared_ptr<scene> s                    = buildSharedScene();
    std::shared_ptr<baseBatchScene<NSTREAM>> bs = std::make_shared<batchScene<sceneCfg>>(s, streams, optixContext);

    // CC Setup
    CollisionDetector<NSTREAM, TYPE> CC(optixContext, streams);
    CC.setupScene(bs);
    CC.setupOptix();

    int N = 100;
    int nCheck;
    std::cout << std::endl;
    for (int count : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
        CC.setupRobot(ccRobot);
        CC.buildSBT();
        CC.setNCheck(count);
        CUDA_SYNC_CHECK();

        ccRobot->prepareUpdateData((float*) traj.data(), count);

        CC.check();
        CUDA_SYNC_CHECK();

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < N; t++) {
            CC.check();
        }
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / N;
        std::cout << duration << " ";

        CUDA_SYNC_CHECK();
    }
    std::cout << std::endl;
}
