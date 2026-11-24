// rtcd_bench_curve: new-stack equivalent of RTCD/Benchmark/Curve (BM_CURVE_Shelf / ...Simple / ...DenseShelf; the
// scene is --scene). 16 streams. Benchmarks with the old names and template arguments:
//   BM_LINEAR<nCPts, nTPts, batch>  linear curves, gas fast_build            (old FAST_BUILD), OBB off
//   BM_QUAD<nCPts, nTPts, batch>    quadratic curves, gas fast_trace         (old FAST_TRACE), OBB off
//   BM_MESH<nTPts>, BM_RAY<nTPts>   n trajectories checked as n * nTPts discrete poses, batch 1024, OBB on
// n trajectories sampled per iteration (old sample_trajectories: shuffle, paused) from the 8192 * 4 KNN
// trajectories; timed region = checkDevice. The curve checkers are built per benchmark family, outside the timed
// loop (old: robot + SBT set up per family). Options: bench_common.h. (No nsys on this app: the old BM_CURVE_Shelf
// hangs under Nsight Systems.)
#include "bench_common.h"

#include <benchmark/benchmark.h>

#include <map>

using namespace rtcd;

namespace {
    constexpr std::size_t kStreams = 16;

    const std::vector<float>& trajectoryPool(std::uint32_t nTPts) {
        static std::map<std::uint32_t, std::vector<float>> pools;
        auto it = pools.find(nTPts);
        if (it == pools.end()) {
            it = pools.emplace(nTPts, support::benchmarkTrajectoryPool(bench::env().motionsDir, nTPts)).first;
        }
        return it->second;
    }

    CheckerConfig curveConfig(Mode mode, std::uint32_t nCPts, std::uint32_t nTPts, std::uint32_t batch, BuildPreference b) {
        CheckerConfig c;
        c.mode                   = mode;
        c.numStreams             = kStreams;
        c.maxBatchSize           = batch;
        c.maxItemsPerCheck       = 4096;
        c.obbCulling             = Toggle::Off;
        c.hostDispatch           = HostDispatch::ThreadPerStream;
        c.curve.trajectoryPoints = nTPts;
        c.curve.controlPoints    = nCPts;
        c.curve.gasBuild         = b;
        c.curve.splineWeightsDir = bench::env().splineWeightsDir;
        return c;
    }

    CheckerConfig discreteConfig(Mode mode, std::uint32_t nTPts) {
        CheckerConfig c;
        c.mode              = mode;
        c.numStreams        = kStreams;
        c.maxBatchSize      = 1024;
        c.maxItemsPerCheck  = 4096 * nTPts; // n trajectories of nTPts poses
        c.obbCulling        = Toggle::On;
        c.hostDispatch      = HostDispatch::ThreadPerStream;
        c.mesh.linkGasBuild = BuildPreference::FastTraceCompact;
        return c;
    }

    void runCurve(benchmark::State& state, bench::DeviceBench& b, std::uint32_t nTPts, std::size_t itemsPerTraj) {
        const std::size_t n          = static_cast<std::size_t>(state.range(0));
        const std::size_t unitFloats = std::size_t(nTPts) * bench::env().robot->dof();
        const auto& pool             = trajectoryPool(nTPts);
        if (pool.size() / unitFloats < n) {
            state.SkipWithError("Can not find more trajs in the sampled space.");
            return;
        }
        for (auto _ : state) {
            state.PauseTiming();
            const std::vector<float> s = bench::sampleTrajectories(pool, unitFloats, n, bench::rng());
            b.upload(s, n * itemsPerTraj);
            state.ResumeTiming();
            b.check();
        }
    }

    template <int nCPts, int nTPts, int batchSize>
    void BM_LINEAR(benchmark::State& state) {
        static_assert(nCPts == nTPts, "nCPts and nTPts must be the same for linear case.");
        bench::DeviceBench b(curveConfig(Mode::LinearCurve, nCPts, nTPts, batchSize, BuildPreference::FastBuild));
        runCurve(state, b, nTPts, 1);
    }

    template <int nCPts, int nTPts, int batchSize>
    void BM_QUAD(benchmark::State& state) {
        static_assert(nCPts < nTPts, "nCPts must be < nTPts");
        bench::DeviceBench b(curveConfig(Mode::QuadraticCurve, nCPts, nTPts, batchSize, BuildPreference::FastTrace));
        runCurve(state, b, nTPts, 1);
    }

    template <int nTPts>
    void BM_MESH(benchmark::State& state) {
        static bench::DeviceBench b(discreteConfig(Mode::Mesh, nTPts));
        runCurve(state, b, nTPts, nTPts);
    }

    template <int nTPts>
    void BM_RAY(benchmark::State& state) {
        static bench::DeviceBench b(discreteConfig(Mode::Ray, nTPts));
        runCurve(state, b, nTPts, nTPts);
    }
} // namespace

#define RTCD_CURVE_BM_OPTS ->RangeMultiplier(2)->Range(1 << 0, 1 << 12)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1)->UseRealTime()

BENCHMARK_TEMPLATE(BM_LINEAR, 8, 8, 16) RTCD_CURVE_BM_OPTS;
BENCHMARK_TEMPLATE(BM_LINEAR, 16, 16, 16) RTCD_CURVE_BM_OPTS;
BENCHMARK_TEMPLATE(BM_QUAD, 4, 8, 16) RTCD_CURVE_BM_OPTS;
BENCHMARK_TEMPLATE(BM_QUAD, 8, 16, 16) RTCD_CURVE_BM_OPTS;
BENCHMARK_TEMPLATE(BM_MESH, 16) RTCD_CURVE_BM_OPTS;
BENCHMARK_TEMPLATE(BM_MESH, 32) RTCD_CURVE_BM_OPTS;
BENCHMARK_TEMPLATE(BM_RAY, 16) RTCD_CURVE_BM_OPTS;
BENCHMARK_TEMPLATE(BM_RAY, 32) RTCD_CURVE_BM_OPTS;

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
