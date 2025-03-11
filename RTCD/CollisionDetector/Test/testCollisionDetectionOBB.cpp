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
#elif defined(RayOnlineRobot)
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
#elif defined(RayOnlineRobot)
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

    CC.clearResult();
    {
        // CC.setupRobot(ccRobot);
        CC.buildSBT();
        // CC.warmup(BATCHSIZE);
        nCheck = 4096;

        ccRobot->prepareUpdateData((float*) traj.data(), nCheck);
        CC.setNCheck(nCheck);
        CUDA_SYNC_CHECK();

        CC.check();
        CUDA_SYNC_CHECK();

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < N; t++) {
            CC.check();
        }
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / N;
        std::cout << "Time taken: " << duration << " microseconds" << std::endl;

        CUDA_SYNC_CHECK();
    }

    // CC.clearResult();

    CUDA_SYNC_CHECK();
    // Load the result
    std::vector<int> result_gt(trajLength);
    readBin(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/motions/Panda4096Result.bin"), result_gt);
    // result_gt = std::vector<int>(trajLength, 1);

    std::vector<uint> result(trajLength, 0);
    CC.downloadResult(result);
    int found  = std::count_if(result.begin(), result.begin() + nCheck, [](int i) { return i > 0; });
    int should = std::count_if(result_gt.begin(), result_gt.begin() + nCheck, [](int i) { return i > 0; });
    // find the number of false positive and false negative
    int falsePositive = 0;
    int falseNegative = 0;
    for (size_t i = 0; i < nCheck; i++) {
        if (result[i] > 0 && result_gt[i] == 0) {
            falsePositive++;
            std::cout << "[" << i << " " << result[i] << "]";
        }
        if (result[i] == 0 && result_gt[i] > 0) {
            falseNegative++;
            std::cout << "[" << i << " " << result[i] << "]";
        }
    }
    std::cout << std::endl << "Found: " << found << " Should: " << should << std::endl;
    std::cout << "False Positive: " << falsePositive << " False Negative: " << falseNegative << std::endl;


#if defined(IAS)
    save_cc_result("CCIAS_OBB_Result.bin", result, trajLength);
#elif defined(RayOnlineRobot)
    save_cc_result("CCRO_OBB_Result.bin", result, trajLength);
#endif

#if defined(IAS) // sphere check the start & end for trajs
    if constexpr (batchConfig.TYPE == LinkType::SPHERES) {
        std::vector<std::array<float, Dim>> traj2(64 * 4096 * 4);
        constexpr std::array<float, Dim> weight{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        genCSpaceKNNTraj<Dim, 4096, 4>(traj2, 2, upperBound, lowerBound, r->name, weight);

        CC.clearResult();
        {
            nCheck = 4096 * 2;

            ccRobot->prepareUpdateData((float*) traj2.data(), nCheck);
            CC.setNCheck(nCheck);
            CUDA_SYNC_CHECK();
            CC.check();
            CUDA_SYNC_CHECK();
        }

        std::vector<uint> resultSE(4096 * 2, 0);
        CC.downloadResult(resultSE);
        std::vector<int> resultTraj(4096);
        for (int i = 0; i < 4096; i++) {
            if (resultSE[i * 2] + resultSE[i * 2 + 1] > 0) {
                resultTraj[i] = 1;
            }
        }
        save_cc_result("CC_Traj_SE.bin", resultTraj, 4096);
    }
#endif
}
