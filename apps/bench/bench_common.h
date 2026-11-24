#pragma once
// Shared setup of the new-stack Google-Benchmark apps (bench_discrete.cpp = old RTCD/Benchmark/Discrete,
// bench_curve.cpp = old RTCD/Benchmark/Curve). Equivalent timed region: CollisionChecker::checkDevice (old:
// CollisionDetector::check incl. its final cudaDeviceSynchronize); sampling and the host->device upload are paused,
// as in the old benchmarks.
//
// One executable serves every scene (the old stack built one binary per scene: benchmarkShelf / ...ShelfSimple /
// ...DenseShelf, BM_CURVE_Shelf / ...ShelfSimple / ...DenseShelf). Benchmark names, template arguments and argument
// ranges are exactly the old ones, so the JSON written with --benchmark_out=<file> is read unchanged by
// scripts/Benchmark/plots/plot_benchmarks.py (see benchmark.sh / benchmark.bat for the output file names).
//
// App options (removed from argv before benchmark::Initialize; all other arguments go to Google Benchmark):
//   --scene <name | scene.json>  a name is configs/scenes/<name>.json: shelf (default, old SHELF build),
//                                shelf_simple (old SHELFSIMPLE), shelf_dense (old DENSESHELF)
//   --robot <robot.json>         default configs/robots/panda.json
//   --data <dir>                 data folder (default: env RTCD_DATA_DIR, else <repo>/tests/data, the repository
//                                found from the working directory / executable)
//   --rtcd_help                  print these options and exit
// The JSON output's "context" gets rtcd_stack = "new", rtcd_scene, rtcd_robot and rtcd_device entries.
// Smoke test (tests/apps): --benchmark_dry_run=true --benchmark_filter=... runs one iteration, no warm-up.
// Environment (support/options.h): RTCD_OPTIX_CFG, RTCD_CURVE_GEOM_FLAG, RTCD_BM_REFIT_EVERY (old syntax), then
// RTCD_OVERRIDES (new syntax), applied to every checker the app builds.

#include <support/inputs.h>

#include <rtcd/checker.h>
#include <rtcd/config.h>
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace rtcd::bench {

    struct Environment {
        std::shared_ptr<Context> context;
        std::shared_ptr<const RobotModel> robot;
        std::shared_ptr<const Scene> scene;
        std::filesystem::path motionsDir;
        std::filesystem::path splineWeightsDir; // configs/spline_weights
        std::string sceneName;
    };

    // Parses and removes the app options from argv, loads robot and scene, creates the Context (ContextOptions
    // from RTCD_OPTIX_CFG validation / loglevel). Exits with code 2 on a usage error.
    Environment& setup(int& argc, char** argv);
    Environment& env();

    // Applies the environment knobs (see the header comment) to `config`.
    CheckerConfig withEnvironment(CheckerConfig config);

    // A checker plus caller-owned device buffers for checkDevice.
    class DeviceBench {
    public:
        explicit DeviceBench(const CheckerConfig& config);
        void upload(std::span<const float> items, std::size_t count); // untimed
        void check();                                                // timed
        CollisionChecker& checker() { return *checker_; }

    private:
        std::unique_ptr<CollisionChecker> checker_;
        DeviceBuffer items_, counts_;
        std::size_t count_ = 0;
    };

    // Old samplers: std::sample of `n` poses (order kept, BM_IAS / BM_RAY) and the shuffle-based
    // sample_trajectories (curve benchmarks), on a pool of units of `unitFloats` floats.
    std::vector<float> samplePoses(const std::vector<float>& pool, std::size_t unitFloats, std::size_t n, std::mt19937& gen);
    std::vector<float> sampleTrajectories(
        const std::vector<float>& pool, std::size_t unitFloats, std::size_t n, std::mt19937& gen);

    std::mt19937& rng(); // seeded from std::random_device, as the old benchmarks

} // namespace rtcd::bench
