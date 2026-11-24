// rtcd_bench_discrete: new-stack equivalent of RTCD/Benchmark/Discrete (benchmarkShelf / ...Simple / ...DenseShelf;
// the scene is --scene instead of a compile definition). BM_IAS and BM_RAY: 4 streams, batch 1024, OBB culling on,
// link GAS fast_trace_compact (old COMPACT); n poses sampled per iteration from the 8192 cached poses (std::sample,
// paused), timed region = checkDevice. Options: bench_common.h.
#include "bench_common.h"

#include <benchmark/benchmark.h>

using namespace rtcd;

namespace {
    CheckerConfig discreteConfig(Mode mode) {
        CheckerConfig c;
        c.mode              = mode;
        c.numStreams        = 4;      // NSTREAM
        c.maxBatchSize      = 1024;   // batchSize
        c.maxItemsPerCheck  = 8192;   // old allocPoseBuffer(8196) of an 8192 pool
        c.obbCulling        = Toggle::On;
        c.hostDispatch      = HostDispatch::ThreadPerStream; // USE_MULTI_THREADING
        c.mesh.linkGasBuild = BuildPreference::FastTraceCompact;
        return c;
    }

    const std::vector<float>& posePool() {
        static const std::vector<float> pool = support::benchmarkPosePool(bench::env().motionsDir);
        return pool;
    }

    void run(benchmark::State& state, bench::DeviceBench& b) {
        const std::size_t n   = static_cast<std::size_t>(state.range(0));
        const std::size_t dof = bench::env().robot->dof();
        for (auto _ : state) {
            state.PauseTiming();
            const std::vector<float> s = bench::samplePoses(posePool(), dof, n, bench::rng());
            b.upload(s, n);
            state.ResumeTiming();
            b.check();
        }
    }

    void BM_IAS(benchmark::State& state) {
        static bench::DeviceBench b(discreteConfig(Mode::Mesh));
        run(state, b);
    }
    void BM_RAY(benchmark::State& state) {
        static bench::DeviceBench b(discreteConfig(Mode::Ray));
        run(state, b);
    }
} // namespace

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
    bench::setup(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
