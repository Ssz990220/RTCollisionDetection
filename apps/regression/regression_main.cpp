// rtcd_regression: the new stack's collision-result regression, the checks of
// RTCD/CollisionDetector/Test/ccRegression.cpp driven by a setup JSON instead of REG_* compile definitions.
//
//   rtcd_regression --setup configs/setups/regression_quad.json
//                   [--engine auto|checker|reference] [--override "path=value,..."] [--golden <file> | --no-golden]
//                   [--items 4096] [--data <dir>] [--dump-flags <file>]
//
// Checks (exit 0 = all passed, 1 = a check failed, 2 = usage, 3 = error):
//   1. determinism: two full runs of `items` items agree;
//   2. prefix consistency: the first N items alone give the same flags as the full run, N = 1, 37, B, B*S + 5,
//      1000, items - 1 (B = max_batch_size, S = streams);
//   3. golden diff: flags == tests/data/CCTest/golden_<ias|ray|linear|quad>.bin (by mode; --golden overrides,
//      --no-golden skips; cubic curves have no golden). Goldens are never written.
// Inputs: support/inputs.h regressionInputs (byte-identical to the old harness): poses for the discrete modes,
// KNN trajectories with curve.trajectory_points waypoints for the curve modes.
// Engines: "checker" = CollisionChecker::check (needs the Wave 2 detector), "reference" = ReferenceBatchRunner +
// create<Mode>Backend (needs only that backend); "auto" = checker if linked, else reference.
// Overrides: --override, then env RTCD_OVERRIDES (rtcd::applyOverrides syntax); the old env knobs RTCD_OPTIX_CFG,
// RTCD_CURVE_GEOM_FLAG, RTCD_BM_REFIT_EVERY are applied before them (support/options.h), so e.g.
// "--override curve.geometry_flags=none" replaces the old --curve-geom-flag none.
#include "engines.h"

#include <support/golden.h>
#include <support/inputs.h>
#include <support/options.h>
#include <support/paths.h>

#include <rtcd/errors.h>
#include <rtcd/json_config.h>

#include <iostream>
#include <optional>
#include <string>

namespace fs = std::filesystem;
using namespace rtcd;

namespace {
    [[noreturn]] void usage(const std::string& msg) {
        std::cerr << "rtcd_regression: " << msg
                  << "\nusage: rtcd_regression --setup <setup.json> [--engine auto|checker|reference] "
                     "[--override \"k=v,...\"] [--golden <file> | --no-golden] [--items N] [--data <dir>] "
                     "[--dump-flags <file>]\n";
        std::exit(2);
    }
} // namespace

int main(int argc, char** argv) {
    std::optional<std::string> setupPath, engineName, override_, goldenOpt, dataOpt, dumpOpt;
    bool noGolden     = false;
    std::size_t items = support::LegacyHarness::regressionItems;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value          = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage("missing value for " + a);
            }
            return argv[++i];
        };
        if (a == "--setup") setupPath = value();
        else if (a == "--engine") engineName = value();
        else if (a == "--override") override_ = value();
        else if (a == "--golden") goldenOpt = value();
        else if (a == "--no-golden") noGolden = true;
        else if (a == "--data") dataOpt = value();
        else if (a == "--dump-flags") dumpOpt = value();
        else if (a == "--items") {
            try {
                items = std::stoull(value());
            } catch (...) {
                usage("--items needs a number");
            }
        } else usage("unknown argument " + a);
    }
    if (!setupPath) {
        usage("--setup is required");
    }
    const std::string engineSel = engineName.value_or("auto");
    if (engineSel != "auto" && engineSel != "checker" && engineSel != "reference") {
        usage("--engine must be auto, checker or reference");
    }

    try {
        Setup setup = loadSetup(*setupPath);
        ContextOptions ctxOpt = ContextOptions::buildDefault();
        const support::EnvSettings env = support::readEnvSettings();
        support::applyLegacyEnv(env, &setup.checker, &ctxOpt);
        if (override_) {
            applyOverrides(setup.checker, *override_);
        }
        if (env.overrides) {
            applyOverrides(setup.checker, *env.overrides);
        }
        const CheckerConfig resolved = setup.checker.resolved();
        const std::string mode       = support::modeName(resolved.mode);
        const std::string tag        = "[" + mode + "] ";

        bool useChecker = engineSel == "checker" || (engineSel == "auto" && regression::checkerEngineAvailable());
        if (engineSel == "auto" && !useChecker && !regression::referenceEngineAvailable(resolved.mode)) {
            std::cerr << "rtcd_regression: neither the CollisionChecker nor the " << mode
                      << " backend is implemented in this build; nothing can run yet\n";
            return 2;
        }

        auto ctx   = Context::create(ctxOpt);
        auto robot = RobotModel::load(setup.robot, ctx->logger());
        if (robot->dof() != support::LegacyHarness::dof) {
            throw InvalidArgument("the regression inputs are " + std::to_string(support::LegacyHarness::dof)
                                  + "-DOF Panda poses; the setup's robot has " + std::to_string(robot->dof()));
        }
        auto scene = std::make_shared<const Scene>(ctx, setup.scene);
        std::unique_ptr<regression::Engine> engine = useChecker
                                                       ? regression::makeCheckerEngine(ctx, setup.checker, robot, scene)
                                                       : regression::makeReferenceEngine(ctx, setup.checker, robot, scene);
        const CheckerConfig& cfg = engine->resolvedConfig();
        std::cout << tag << "setup " << fs::absolute(*setupPath).string() << ", engine " << engine->name() << "\n"
                  << tag << support::describeConfig(cfg) << "\n";
        if (items > cfg.maxItemsPerCheck) {
            throw InvalidArgument("--items " + std::to_string(items) + " > max_items_per_check "
                                  + std::to_string(cfg.maxItemsPerCheck));
        }

        const fs::path data = support::dataDir(dataOpt ? std::optional<fs::path>(*dataOpt) : std::nullopt);
        const ItemLayout layout = cfg.itemLayout(robot->dof());
        const std::vector<float> inputs =
            support::regressionInputs(data / "motions", layout.pointsPerItem, items, robot->dof());

        bool ok = true;
        auto runFlags = [&](std::size_t n) { return support::flagsFromCounts(engine->run(inputs, n)); };

        // 1. determinism
        const support::Flags full  = runFlags(items);
        const support::Flags full2 = runFlags(items);
        const auto nondet          = support::diffFlags(full, full2, items);
        support::printFlagDiff(std::cout, nondet, full, full2);
        std::cout << tag << "full run: " << support::countSet(full) << "/" << items
                  << " in collision; repeat-run diffs: " << nondet.differing << "\n";
        ok &= nondet.equal();

        // 2. prefix consistency
        for (const std::size_t n : support::regressionPrefixCounts(cfg.maxBatchSize, cfg.numStreams, items)) {
            const support::Flags part = runFlags(n);
            const auto d              = support::diffFlags(part, full, n);
            support::printFlagDiff(std::cout, d, part, full);
            std::cout << tag << "prefix N=" << n << ": " << d.differing << " diffs vs full run\n";
            ok &= d.equal();
        }

        if (dumpOpt) {
            support::writeFlagsFile(*dumpOpt, full);
            std::cout << tag << "flags written to " << *dumpOpt << "\n";
        }

        // 3. golden
        std::optional<fs::path> golden;
        if (goldenOpt) {
            golden = fs::path(*goldenOpt);
        } else if (!noGolden) {
            golden = support::goldenPath(data, cfg.mode);
            if (!golden) {
                std::cout << tag << "no golden exists for this mode\n";
            }
        }
        if (golden) {
            const support::Flags g = support::readFlagsFile(*golden, items);
            const auto d           = support::diffFlags(full, g, items);
            support::printFlagDiff(std::cout, d, full, g);
            std::cout << tag << "golden: " << d.differing << " diffs (golden has " << support::countSet(g)
                      << " hits)\n";
            ok &= d.equal();
        }
        std::cout << tag << (ok ? "PASS" : "FAIL") << std::endl;
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "rtcd_regression: error: " << e.what() << std::endl;
        return 3;
    }
}
