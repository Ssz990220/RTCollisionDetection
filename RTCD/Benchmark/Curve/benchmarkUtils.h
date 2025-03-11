#include <CollisionDetector/CCRobotWrapper.h>
#include <CollisionDetector/CollisionDetector.h>
#include <CollisionScenes/batchScene.h>
#include <Robot/batchCurveRobot.h>
#include <Robot/batchRayRobot.h>
#include <Robot/models/pandas.h>
#include <Utils/Test/testUtils.h>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <nvtx3/nvToolsExt.h>
#include <random>

using namespace RTCD;

inline constexpr size_t NSTREAM = 16;

inline constexpr int KNN_K = 4;
inline constexpr std::array<float, Dim> weight{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

inline constexpr batchSceneConfig sceneCfg(false, NSTREAM, 0);
inline constexpr batchRobotConfig RAYCFG(NSTREAM, 1024, Dim, BuildType::NONE, LinkType::RAY_ONLINE, true);
inline constexpr batchRobotConfig MESHCFG(NSTREAM, 1024, Dim, BuildType::COMPACT, LinkType::MESH, true);

extern std::shared_ptr<robot<Dim>> r;
extern std::array<cudaStream_t, NSTREAM> streams;
extern OptixDeviceContext optixContext;
extern std::vector<std::array<float, Dim>> traj;
extern std::vector<std::array<float, Dim>> sampledTrajs;
extern std::shared_ptr<CollisionDetector<NSTREAM, LinkType::LINEAR_CURVE>> CC_LIN;
extern std::shared_ptr<CollisionDetector<NSTREAM, LinkType::QUADRATIC_CURVE>> CC_QUAD;
extern std::shared_ptr<CollisionDetector<NSTREAM, LinkType::MESH>> CCMesh;
extern std::shared_ptr<CollisionDetector<NSTREAM, LinkType::RAY_ONLINE>> CCRay;
extern std::shared_ptr<batchIASRobot<MESHCFG>> batchIASRobotPtr;
extern std::shared_ptr<CCWrappedRobot<NSTREAM, LinkType::MESH>> ccIASRobot;
extern std::shared_ptr<batchRayRobot<RAYCFG>> batchRayRobotPtr;
extern std::shared_ptr<CCRayRobot<NSTREAM, LinkType::RAY_ONLINE>> ccRayRobot;

inline constexpr size_t nCSpaceSample = 8192;

// Random sequence generator
std::random_device rd;
std::mt19937 gen(rd());

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


template <int nCPts, int nTPts, int batchSize>
void BM_LINEAR(benchmark::State& state) {
    static_assert(nCPts == nTPts, "nCPts and nTPts must be the same for linear case.");
    // Check if the condition to skip the benchmark is met
    if (nCSpaceSample * KNN_K < state.range(0)) {
        state.SkipWithError("Can not find more trajs in the sampled space.");
        return; // Skip the rest of this benchmark run
    }
    constexpr batchRobotConfig LINCFG{
        NSTREAM, batchSize, Dim, BuildType::FAST_BUILD, LinkType::LINEAR_CURVE, nTPts, nCPts, false};

    std::string marker =
        "BM_LINEAR" + std::to_string(nCPts) + "_" + std::to_string(nTPts) + "_" + std::to_string(state.range(0));
    nvtxRangePush(marker.c_str());

    // Robot Setup
    auto batchRobotPtr = std::make_shared<batchOptixCurveRobot<LINCFG>>(r, streams, optixContext);
    auto ccRobot       = std::make_shared<CCWrappedRobot<NSTREAM, LinkType::LINEAR_CURVE>>(batchRobotPtr);

    // Prepare trajectory data
    traj.resize(nCSpaceSample * nTPts * KNN_K);
    genCSpaceKNNTraj<Dim, nCSpaceSample, KNN_K>(traj, nTPts, upperBound, lowerBound, r->name, weight);

    CC_LIN->setupRobot(ccRobot);
    CC_LIN->buildSBT();
    CC_LIN->setNCheck(state.range(0));

    for (auto _ : state) {
        // Pause timing while setting up the data
        state.PauseTiming();
        sampledTrajs.clear();
        sample_trajectories<Dim, nTPts>(traj, sampledTrajs, state.range(0));
        ccRobot->prepareUpdateData((float*) sampledTrajs.data(), state.range(0));
        CUDA_SYNC_CHECK();

        // Start the benchmark
        state.ResumeTiming();
        CC_LIN->check();
    }
    nvtxRangePop();
    CC_LIN->resetRobot();
    CC_LIN->resetSBT();
}


template <int nCPts, int nTPts, int batchSize>
void BM_QUAD(benchmark::State& state) {
    static_assert(nCPts < nTPts, "nCPts and nTPts must be the same for linear case.");
    // Check if the condition to skip the benchmark is met
    if (nCSpaceSample * KNN_K < state.range(0)) {
        state.SkipWithError("Can not find more trajs in the sampled space.");
        return; // Skip the rest of this benchmark run
    }
    constexpr batchRobotConfig QUADCFG{
        NSTREAM, batchSize, Dim, BuildType::FAST_TRACE, LinkType::QUADRATIC_CURVE, nTPts, nCPts, false};
    std::string marker =
        "BM_QUAD_" + std::to_string(nCPts) + "_" + std::to_string(nTPts) + "_" + std::to_string(state.range(0));
    nvtxRangePush(marker.c_str());

    // Robot Setup
    auto batchRobotPtr = std::make_shared<batchOptixCurveRobot<QUADCFG>>(r, streams, optixContext);
    auto ccRobot       = std::make_shared<CCWrappedRobot<NSTREAM, LinkType::QUADRATIC_CURVE>>(batchRobotPtr);

    // Prepare trajectory data
    traj.resize(nCSpaceSample * nTPts * KNN_K);
    genCSpaceKNNTraj<Dim, nCSpaceSample, KNN_K>(traj, nTPts, upperBound, lowerBound, r->name, weight);

    CC_QUAD->setupRobot(ccRobot);
    CC_QUAD->buildSBT();
    CC_QUAD->setNCheck(state.range(0));

    for (auto _ : state) {
        // Pause timing while setting up the data
        state.PauseTiming();
        sampledTrajs.clear();
        sample_trajectories<Dim, nTPts>(traj, sampledTrajs, state.range(0));
        ccRobot->prepareUpdateData((float*) sampledTrajs.data(), state.range(0));
        CUDA_SYNC_CHECK();

        // Start the benchmark
        state.ResumeTiming();
        CC_QUAD->check();
    }
    nvtxRangePop();
    CC_QUAD->resetRobot();
    CC_QUAD->resetSBT();
}

template <int nTPts>
void BM_MESH(benchmark::State& state) {

    traj.resize(nCSpaceSample * nTPts * KNN_K);
    genCSpaceKNNTraj<Dim, nCSpaceSample, KNN_K>(traj, nTPts, upperBound, lowerBound, r->name, weight);
    CCMesh->setNCheck(state.range(0) * nTPts);

    for (auto _ : state) {
        // Pause timing while setting up the data
        state.PauseTiming();
        sampledTrajs.clear();
        sample_trajectories<Dim, nTPts>(traj, sampledTrajs, state.range(0));
        ccIASRobot->prepareUpdateData((float*) sampledTrajs.data(), state.range(0) * nTPts);
        CCMesh->resetGraph();
        CUDA_SYNC_CHECK();

        // Start the benchmark
        state.ResumeTiming();
        CCMesh->check();
    }
}


template <int nTPts>
void BM_RAY(benchmark::State& state) {

    traj.resize(nCSpaceSample * nTPts * KNN_K);
    genCSpaceKNNTraj<Dim, nCSpaceSample, KNN_K>(traj, nTPts, upperBound, lowerBound, r->name, weight);
    CCRay->setNCheck(state.range(0) * nTPts);

    for (auto _ : state) {
        // Pause timing while setting up the data
        state.PauseTiming();
        sampledTrajs.clear();
        sample_trajectories<Dim, nTPts>(traj, sampledTrajs, state.range(0));
        ccRayRobot->prepareUpdateData((float*) sampledTrajs.data(), state.range(0) * nTPts);
        CCRay->resetGraph();
        CUDA_SYNC_CHECK();

        // Start the benchmark
        state.ResumeTiming();
        CCRay->check();
    }
}
