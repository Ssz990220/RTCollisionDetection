#define USE_MULTI_THREADING

#include <Benchmark/Discrete/benchmarkUtils.h>
#if defined(DENSESHELF)
#include <CollisionScenes/Scene/denseShelf.h>
#elif defined(SHELF)
#include <CollisionScenes/Scene/shelf.h>
#elif defined(SHELFSIMPLE)
#include <CollisionScenes/Scene/shelfSimple.h>
#endif

using namespace RTCD;

// Global Setup
std::vector<std::array<float, Dim>> traj;
std::vector<std::array<float, Dim>> sampledTrajs;
std::shared_ptr<CollisionDetector<NSTREAM, LinkType::MESH>> CCMesh;
std::shared_ptr<CollisionDetector<NSTREAM, LinkType::RAY_ONLINE>> CCRay;
std::shared_ptr<robot<Dim>> r;
std::shared_ptr<batchRayRobot<RAYCFG>> batchRayRobotPtr;
std::shared_ptr<CCRayRobot<NSTREAM, LinkType::RAY_ONLINE>> ccRayRobot;
std::shared_ptr<batchIASRobot<MESHCFG>> batchIASRobotPtr;
std::shared_ptr<CCWrappedRobot<NSTREAM, LinkType::MESH>> ccIASRobot;
std::shared_ptr<scene> s;
std::shared_ptr<baseBatchScene<NSTREAM>> bs;
CUcontext cudaContext;
OptixDeviceContext optixContext;
std::array<cudaStream_t, NSTREAM> streams;


BENCHMARK(BM_IAS)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
#if defined(_DEBUG) || defined(DEBUG)
    ->Iterations(10)
#endif
    ->MinWarmUpTime(4);
BENCHMARK(BM_RAY)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
#if defined(_DEBUG) || defined(DEBUG)
    ->Iterations(100)
#endif
    ->MinWarmUpTime(4);

int main(int argc, char** argv) {
    std::tie(cudaContext, optixContext, streams) = createContextStream<NSTREAM>();

    CUDA_SYNC_CHECK();
    r = std::make_shared<robot<Dim>>(Config);
    // We generated 8192 random poses, in each benchmark, we sample a set randomly from this set
    traj.resize(trajLength);
    randomMotions(trajLength, traj, lowerBound, upperBound, r->name);

    std::shared_ptr<scene> s                    = buildSharedScene();
    std::shared_ptr<baseBatchScene<NSTREAM>> bs = std::make_shared<batchScene<sceneCfg>>(s, streams, optixContext);


    // Robot Setup
    batchRayRobotPtr = std::make_shared<batchRayRobot<RAYCFG>>(r, streams, optixContext);
    ccRayRobot       = std::make_shared<CCRayRobot<NSTREAM, LinkType::RAY_ONLINE>>(batchRayRobotPtr);
    ccRayRobot->allocPoseBuffer(8196);
    batchIASRobotPtr = std::make_shared<batchIASRobot<MESHCFG>>(r, streams, optixContext);
    ccIASRobot       = std::make_shared<CCWrappedRobot<NSTREAM, LinkType::MESH>>(batchIASRobotPtr);
    ccIASRobot->allocPoseBuffer(8196);
    // CCMesh Setup
    CCMesh = std::make_shared<CollisionDetector<NSTREAM, LinkType::MESH>>(optixContext, streams);
    CCRay  = std::make_shared<CollisionDetector<NSTREAM, LinkType::RAY_ONLINE>>(optixContext, streams);


    CCMesh->setupScene(bs);
    CCMesh->setupOptix();
    CCMesh->setupRobot(ccIASRobot);
    CCMesh->buildSBT();
    // CCMesh->warmup(1024);

    CCRay->setupScene(bs);
    CCRay->setupOptix();
    CCRay->setupRobot(ccRayRobot);
    CCRay->buildSBT();
    // CCRay->warmup(1024);

    // Initialize Google Benchmark
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    // // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
