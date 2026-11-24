#pragma once
// Shared part of the new-stack demos (rtcd_demo_discrete = old RTCD/Demo/DemoCollisionDetectorDisc.cpp,
// rtcd_demo_curve = old RTCD/Demo/DemoCollisionDetectorCurve.cpp). Public API only (<rtcd/...>, target `rtcd`):
// no old-stack includes, no test-support code, no data/ cache.
//
//   rtcd_demo_<kind> [--setup <setup.json>] [--override "path=value,..."] [--counts 1-4096] [--repeats R]
//                    [--seed S] [--step F]
//
//   --setup     setup JSON (rtcd::loadSetup: checker + scene + robot), or a bare name meaning
//               configs/setups/<name>.json of the source tree (found by walking up from the working directory,
//               then from the executable's directory). Default demo_ias (discrete) / demo_linear (curve). Any other
//               setup works too, e.g. configs/setups/regression_quad.json.
//   --override  rtcd::applyOverrides on the checker config (e.g. "streams=8,max_batch_size=256")
//   --counts    item counts to check: "1,16,256" or "a-b" (powers of two from a to b); default 1-4096, capped at
//               the checker's max_items_per_check
//   --repeats   timed checks per count (default: discrete 100, curve 10, as the old demos)
//   --seed      seed of the item generator (default 1; the old demos seeded from std::random_device)
//   --step      curve demo only: largest joint step from start to end of a trajectory, as a fraction of the
//               joint range (default 0.1)
//
// Flow: load setup -> Context -> RobotModel -> Scene -> CollisionChecker -> warmup(); then, per count N, one check
// of the first N generated items (prints the number of colliding items) followed by `repeats` timed checks of the
// same items. The time printed is the mean wall-clock time of CollisionChecker::check(HostItems, flags) INCLUDING
// the host->device upload and the result download, measured with std::chrono on whatever else the GPU is doing:
// it is INDICATIVE ONLY, not benchmark-grade (use rtcd_bench_discrete / rtcd_bench_curve for that).

#include <rtcd/checker.h>
#include <rtcd/json_config.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace rtcd::demo {

    enum class Kind { Discrete, Curve };

    struct Options {
        std::filesystem::path setup;
        std::string overrides;
        std::vector<std::size_t> counts;
        std::size_t repeats = 0;
        unsigned seed       = 1;
        float step          = 0.1f;
    };

    // Parses argv (exits with code 2 and a usage message on errors, 0 after --help).
    Options parseOptions(int argc, char** argv, Kind kind);

    // Generates `count` packed items for `layout` (poses or trajectories) within the robot's joint limits.
    using ItemGenerator = std::function<std::vector<float>(const RobotModel&, const ItemLayout&, std::size_t count,
        std::mt19937& gen)>;

    // Runs the demo (see the header comment). Returns the process exit code (0 ok, 3 error).
    int run(const Options& options, Kind kind, const ItemGenerator& generate);

} // namespace rtcd::demo
