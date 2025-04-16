#include <CollisionDetector/CCRobotWrapper.h>
#include <CollisionDetector/CollisionDetector.h>
#include <CollisionScenes/batchScene.h>
#include <Robot/batchGASRobot.h>
#include <Robot/batchIASRobot.h>
#include <Robot/batchRayRobot.h>
#include <Robot/models/panda.h>
#include <Utils/Test/testUtils.h>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>

using namespace RTCD;

inline constexpr size_t NSTREAM    = 4;
inline constexpr size_t batchSize  = 1024;
inline constexpr size_t trajLength = 8192;
inline constexpr batchSceneConfig sceneCfg(false, NSTREAM, 0);
inline constexpr batchRobotConfig RAYCFG(NSTREAM, batchSize, Dim, BuildType::NONE, LinkType::RAY_ONLINE, true);
inline constexpr batchRobotConfig MESHCFG(NSTREAM, batchSize, Dim, BuildType::COMPACT, LinkType::MESH, true);

extern std::shared_ptr<robot<Dim>> r;
extern std::array<cudaStream_t, NSTREAM> streams;
extern OptixDeviceContext optixContext;
extern std::vector<std::array<float, Dim>> traj;
extern std::vector<std::array<float, Dim>> sampledTrajs;
extern std::shared_ptr<CollisionDetector<NSTREAM, LinkType::MESH>> CCMesh;
extern std::shared_ptr<CollisionDetector<NSTREAM, LinkType::RAY_ONLINE>> CCRay;
extern std::shared_ptr<batchRayRobot<RAYCFG>> batchRayRobotPtr;
extern std::shared_ptr<CCRayRobot<NSTREAM, LinkType::RAY_ONLINE>> ccRayRobot;
extern std::shared_ptr<batchIASRobot<MESHCFG>> batchIASRobotPtr;
extern std::shared_ptr<CCWrappedRobot<NSTREAM, LinkType::MESH>> ccIASRobot;
// Random sequence generator
std::random_device rd;
std::mt19937 gen(rd());


void BM_RAY(benchmark::State& state) {
    for (auto _ : state) {
        // Pause timing while setting up the data
        state.PauseTiming();
        sampledTrajs.clear();
        std::sample(traj.begin(), traj.end(), std::back_inserter(sampledTrajs), state.range(0), gen);
        ccRayRobot->prepareUpdateData((float*) sampledTrajs.data(), state.range(0));
        CCRay->resetGraph();
        CCRay->setNCheck(state.range(0));
        CUDA_SYNC_CHECK();

        // Start the benchmark
        state.ResumeTiming();
        CCRay->check();
    }
}

void BM_IAS(benchmark::State& state) {
    for (auto _ : state) {
        // Pause timing while setting up the data
        state.PauseTiming();
        sampledTrajs.clear();
        std::sample(traj.begin(), traj.end(), std::back_inserter(sampledTrajs), state.range(0), gen);
        ccIASRobot->prepareUpdateData((float*) sampledTrajs.data(), state.range(0));
        CCMesh->resetGraph();
        CCMesh->setNCheck(state.range(0));
        CUDA_SYNC_CHECK();

        // Start the benchmark
        state.ResumeTiming();
        CCMesh->check();
    }
}
