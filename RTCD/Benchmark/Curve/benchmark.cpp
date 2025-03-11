#define USE_MULTI_THREADING
// #define NVTX_DISABLE
#include <Benchmark/Curve/benchmarkUtils.h>
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
std::shared_ptr<CollisionDetector<NSTREAM, LinkType::LINEAR_CURVE>> CC_LIN;
std::shared_ptr<CollisionDetector<NSTREAM, LinkType::QUADRATIC_CURVE>> CC_QUAD;
std::shared_ptr<CollisionDetector<NSTREAM, LinkType::MESH>> CCMesh;
std::shared_ptr<CollisionDetector<NSTREAM, LinkType::RAY_ONLINE>> CCRay;
std::shared_ptr<robot<Dim>> r;
std::shared_ptr<batchIASRobot<MESHCFG>> batchIASRobotPtr;
std::shared_ptr<CCWrappedRobot<NSTREAM, LinkType::MESH>> ccIASRobot;
std::shared_ptr<batchRayRobot<RAYCFG>> batchRayRobotPtr;
std::shared_ptr<CCRayRobot<NSTREAM, LinkType::RAY_ONLINE>> ccRayRobot;
CUcontext cudaContext;
OptixDeviceContext optixContext;
std::array<cudaStream_t, NSTREAM> streams;


// Linear 8 pts
BENCHMARK_TEMPLATE(BM_LINEAR, 8, 8, 16)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();

// Linear 16 Pts
BENCHMARK_TEMPLATE(BM_LINEAR, 16, 16, 16)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();


// Quad 4x8 Pts (4 Control Points from 8 Traj Points)
BENCHMARK_TEMPLATE(BM_QUAD, 4, 8, 16)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_QUAD, 8, 16, 16)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();


BENCHMARK_TEMPLATE(BM_MESH, 16)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_MESH, 32)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_RAY, 16)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_RAY, 32)
    ->RangeMultiplier(2)
    ->Range(1 << 0, 1 << 12)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(1)
    ->UseRealTime();

int main(int argc, char** argv) {
    std::tie(cudaContext, optixContext, streams) = createContextStream<NSTREAM>();

    CUDA_SYNC_CHECK();
    r = std::make_shared<robot<Dim>>(Config);

    std::shared_ptr<scene> s     = buildSharedScene(true); // curve scene, loop edge
    std::shared_ptr<scene> sMesh = buildSharedScene(false); // discrete poses scene, no loop edge

    std::shared_ptr<baseBatchScene<NSTREAM>> bs = std::make_shared<batchScene<sceneCfg>>(s, streams, optixContext);
    std::shared_ptr<baseBatchScene<NSTREAM>> bsMesh =
        std::make_shared<batchScene<sceneCfg>>(sMesh, streams, optixContext);


    // CCMesh Setup
    CC_LIN  = std::make_shared<CollisionDetector<NSTREAM, LinkType::LINEAR_CURVE>>(optixContext, streams);
    CC_QUAD = std::make_shared<CollisionDetector<NSTREAM, LinkType::QUADRATIC_CURVE>>(optixContext, streams);
    CCMesh  = std::make_shared<CollisionDetector<NSTREAM, LinkType::MESH>>(optixContext, streams);
    CCRay   = std::make_shared<CollisionDetector<NSTREAM, LinkType::RAY_ONLINE>>(optixContext, streams);

    CC_LIN->setupScene(bs);
    CC_LIN->setupOptix();

    CC_QUAD->setupScene(bs);
    CC_QUAD->setupOptix();

    batchRayRobotPtr = std::make_shared<batchRayRobot<RAYCFG>>(r, streams, optixContext);
    ccRayRobot       = std::make_shared<CCRayRobot<NSTREAM, LinkType::RAY_ONLINE>>(batchRayRobotPtr);
    ccRayRobot->allocPoseBuffer(4096 * 32);
    batchIASRobotPtr = std::make_shared<batchIASRobot<MESHCFG>>(r, streams, optixContext);
    ccIASRobot       = std::make_shared<CCWrappedRobot<NSTREAM, LinkType::MESH>>(batchIASRobotPtr);
    ccIASRobot->allocPoseBuffer(4096 * 32);

    CCMesh->setupScene(bsMesh);
    CCMesh->setupOptix();
    CCMesh->setupRobot(ccIASRobot);
    CCMesh->buildSBT();

    CCRay->setupScene(bsMesh);
    CCRay->setupOptix();
    CCRay->setupRobot(ccRayRobot);
    CCRay->buildSBT();

    // Initialize Google Benchmark
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    // // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
