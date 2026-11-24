// rtcd_bench_discrete: new-stack equivalent of RTCD/Benchmark/Discrete (benchmarkShelf / ...Simple / ...DenseShelf;
// the scene is --scene instead of a compile definition). BM_IAS and BM_RAY: 4 streams, batch 1024, OBB culling on,
// link GAS fast_trace_compact (old COMPACT); n poses sampled per iteration from the 8192 cached poses (std::sample,
// paused), timed region = checkDevice. Options: bench_common.h. (MinWarmUpTime(4) is inert with Google Benchmark
// 1.10 unless MinTime() is set too, for the old executables alike; kept because it is part of the benchmark names.)
#include "bench_common.h"

#include <benchmark/benchmark.h>

#include <memory>
#include <string>

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

    // --rtcd_graph_bench: BM_RAY with CheckerConfig::graphPrepare = false (the default BM_RAY uses graph replay of
    // each batch's zero / staging copy / prepare()), to compare both in one process. Not an old-stack benchmark.
    void BM_RAY_NOGRAPH(benchmark::State& state) {
        static bench::DeviceBench b = [] {
            CheckerConfig c  = discreteConfig(Mode::Ray);
            c.graphPrepare   = false;
            return bench::DeviceBench(c);
        }();
        run(state, b);
    }} // namespace

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
    // --rtcd_graph_bench: also register BM_RAY_NOGRAPH (= BM_RAY with graph_prepare = false). Off by default so
    // benchmark.sh / benchmark.bat outputs keep exactly the old benchmark set.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--rtcd_graph_bench") {
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            --argc;
            benchmark::RegisterBenchmark("BM_RAY_NOGRAPH", BM_RAY_NOGRAPH)
                ->RangeMultiplier(2)
                ->Range(1 << 0, 1 << 12)
                ->Unit(benchmark::kMicrosecond)
                ->UseRealTime()
                ->MinWarmUpTime(4);
            break;
        }
    }
    bench::setup(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
