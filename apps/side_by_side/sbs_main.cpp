// rtcd_sbs_<scene>: times the OLD stack and the NEW stack in the same process, on the same inputs, alternating per
// repetition, and prints both (main's tool for every parity timing; agents do not run it for timings).
//
//   rtcd_sbs_shelf --profile bm_ias [--n 1-4096] [--reps 200] [--warmup 20] [--seed 1]
//                  [--old-only | --new-only] [--golden] [--csv out.csv]
//                  [--optix "maxreg=32,rayhit=traverse"] [--refit-every K] [--curve-geom-flag none|single_anyhit]
//                  [--traj-pts 16|32] [--new-override "path=value,..."] [--data <dir>]
//                  [--scene <scene.json>] [--robot <robot.json>] [--list]
//
// Profiles (--list): exact copies of the old regression / benchmark setups (apps/side_by_side/old_profiles.cpp);
// the new checker config is derived from the same profile (profile_config.h), then the same old-syntax settings
// are applied to both stacks (env RTCD_OPTIX_CFG / RTCD_BM_REFIT_EVERY / RTCD_CURVE_GEOM_FLAG are read when the
// corresponding option is absent), then --new-override / env RTCD_OVERRIDES (new stack only).
//
// Timed regions: old CollisionDetector::check() (ends with cudaDeviceSynchronize) vs new
// CollisionChecker::checkDevice() (blocks until the counters are written). Item sampling and uploads are untimed,
// as in the old benchmarks: discrete profiles draw n poses without replacement from the 8192-pose pool (std::sample,
// old BM_IAS / BM_RAY), curve profiles n random trajectories of the 8192 * 4 KNN pool (old sample_trajectories),
// bm_curve_mesh / bm_curve_ray n trajectories of --traj-pts poses checked as n * traj-pts discrete poses (old
// BM_MESH / BM_RAY<nTPts>), reg_* profiles sample the 4096 regression inputs. Every repetition feeds the identical
// sample to both stacks and compares their flags (outside the timed region).
//
// --golden (reg_* profiles only): one full check of the 4096 regression inputs per stack against
// data/CCTest/golden_<mode>.bin, before any timing (exit code 1 on a mismatch). --reps 0 skips timing.
#include "new_runner.h"
#include "old_runner.h"
#include "profile_config.h"

#include <support/golden.h>
#include <support/inputs.h>
#include <support/options.h>
#include <support/paths.h>
#include <support/timing.h>

#include <rtcd/context.h>
#include <rtcd/errors.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace rtcd;
using namespace rtcd::sbs;

namespace {

    struct Args {
        std::string profile;
        std::optional<std::string> counts;
        std::size_t reps   = 200;
        std::size_t warmup = 20;
        unsigned seed      = 1;
        bool oldOnly = false, newOnly = false, golden = false, list = false;
        std::optional<std::string> csv, optix, refitEvery, geomFlag, newOverride, scene, robot, data;
        std::uint32_t trajPts = 16;
    };

    [[noreturn]] void usage(const std::string& msg) {
        std::cerr << "rtcd_sbs: " << msg << "\n(see the header of apps/side_by_side/sbs_main.cpp for the options)\n";
        std::exit(2);
    }

    Args parse(int argc, char** argv) {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) {
                    usage("missing value for " + s);
                }
                return argv[++i];
            };
            auto number = [&](const std::string& v) {
                try {
                    std::size_t pos = 0;
                    const unsigned long long n = std::stoull(v, &pos);
                    if (pos != v.size()) {
                        throw 0;
                    }
                    return static_cast<std::size_t>(n);
                } catch (...) {
                    usage("not a number: " + s + " " + v);
                }
            };
            if (s == "--profile") a.profile = value();
            else if (s == "--n") a.counts = value();
            else if (s == "--reps") a.reps = number(value());
            else if (s == "--warmup") a.warmup = number(value());
            else if (s == "--seed") a.seed = static_cast<unsigned>(number(value()));
            else if (s == "--old-only") a.oldOnly = true;
            else if (s == "--new-only") a.newOnly = true;
            else if (s == "--golden") a.golden = true;
            else if (s == "--list") a.list = true;
            else if (s == "--csv") a.csv = value();
            else if (s == "--optix") a.optix = value();
            else if (s == "--refit-every") a.refitEvery = value();
            else if (s == "--curve-geom-flag") a.geomFlag = value();
            else if (s == "--new-override") a.newOverride = value();
            else if (s == "--scene") a.scene = value();
            else if (s == "--robot") a.robot = value();
            else if (s == "--data") a.data = value();
            else if (s == "--traj-pts") a.trajPts = static_cast<std::uint32_t>(number(value()));
            else usage("unknown argument " + s);
        }
        if (a.oldOnly && a.newOnly) {
            usage("--old-only and --new-only exclude each other");
        }
        return a;
    }

    bool isRegression(const OldProfile& p) { return p.name.rfind("reg_", 0) == 0; }
    bool posesFromTrajectories(const OldProfile& p) { return p.name == "bm_curve_mesh" || p.name == "bm_curve_ray"; }

    // The input pool and how one "n" maps to items.
    struct Pool {
        std::vector<float> data;
        std::size_t unitFloats = 0;   // floats of one sampled unit (pose, trajectory)
        std::size_t units      = 0;
        std::size_t itemsPerUnit = 1; // bm_curve_mesh / ray: one unit (trajectory) = trajPts discrete items
        bool sampleKeepsOrder = true; // std::sample (discrete) vs shuffle (trajectories)
    };

    Pool makePool(const OldProfile& p, const fs::path& motions, std::uint32_t trajPts) {
        Pool pool;
        const std::uint32_t dof = support::LegacyHarness::dof;
        if (isRegression(p)) {
            pool.data       = support::regressionInputs(motions, static_cast<std::uint32_t>(p.trajPts));
            pool.unitFloats = p.trajPts * dof;
            pool.sampleKeepsOrder = p.trajPts == 1;
        } else if (posesFromTrajectories(p)) {
            pool.data         = support::benchmarkTrajectoryPool(motions, trajPts);
            pool.unitFloats   = std::size_t(trajPts) * dof;
            pool.itemsPerUnit = trajPts;
            pool.sampleKeepsOrder = false;
        } else if (p.trajPts == 1) {
            pool.data       = support::benchmarkPosePool(motions);
            pool.unitFloats = dof;
        } else {
            pool.data       = support::benchmarkTrajectoryPool(motions, static_cast<std::uint32_t>(p.trajPts));
            pool.unitFloats = p.trajPts * dof;
            pool.sampleKeepsOrder = false;
        }
        pool.units = pool.data.size() / pool.unitFloats;
        return pool;
    }

    std::vector<float> sample(const Pool& pool, std::size_t n, std::mt19937& gen) {
        std::vector<std::size_t> idx(pool.units);
        std::iota(idx.begin(), idx.end(), std::size_t(0));
        std::vector<std::size_t> pick;
        if (pool.sampleKeepsOrder) {
            std::sample(idx.begin(), idx.end(), std::back_inserter(pick), n, gen);
        } else {
            std::shuffle(idx.begin(), idx.end(), gen);
            pick.assign(idx.begin(), idx.begin() + std::ptrdiff_t(n));
        }
        std::vector<float> out(n * pool.unitFloats);
        for (std::size_t i = 0; i < n; ++i) {
            std::copy_n(pool.data.begin() + std::ptrdiff_t(pick[i] * pool.unitFloats), pool.unitFloats,
                out.begin() + std::ptrdiff_t(i * pool.unitFloats));
        }
        return out;
    }

    void printStats(const char* label, const support::TimingStats& s) {
        std::cout << label << " median " << std::setw(9) << s.median << "  mean " << std::setw(9) << s.mean << "  min "
                  << std::setw(9) << s.min << "  p90 " << std::setw(9) << s.p90;
    }

} // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    if (args.list || args.profile.empty()) {
        std::cout << "scene compiled in: " << oldSceneName() << "\nprofiles:\n";
        for (const auto& p : oldProfiles()) {
            std::cout << "  " << std::left << std::setw(14) << p.name << " streams " << p.streams << ", batch "
                      << p.batchSize << ", traj pts " << p.trajPts << ", ctrl pts " << p.ctrlPts << "  (" << p.origin
                      << ")\n";
        }
        std::cout << "new stack linked: " << (newStackAvailable() ? "yes" : "no (Wave 2 detector missing)") << "\n";
        return args.list ? 0 : 2;
    }
    const OldProfile* profile = findOldProfile(args.profile);
    if (!profile) {
        usage("unknown profile '" + args.profile + "' (--list)");
    }
    if (args.golden && !isRegression(*profile)) {
        usage("--golden needs a reg_* profile");
    }
    const bool runOld = !args.newOnly;
    const bool runNew = !args.oldOnly;
    if (runNew && !newStackAvailable()) {
        std::cerr << "rtcd_sbs: the new stack is not linked in this build (Wave 2 detector missing); use --old-only\n";
        return 2;
    }

    try {
        // ---- settings shared by both stacks (old syntax; option wins over env) ----
        const support::EnvSettings env = support::readEnvSettings();
        OldSettings os;
        if (auto v = args.optix ? args.optix : env.legacyOptixConfig) {
            os.optixConfig = *v;
        }
        if (auto v = args.refitEvery ? args.refitEvery : env.refitEvery) {
            (void) support::parseLegacyRefitEvery(*v); // validates (ConfigError)
            os.refitEvery = static_cast<std::size_t>(std::stoull(*v));
        }
        if (auto v = args.geomFlag ? args.geomFlag : env.curveGeometryFlag) {
            os.curveGeomFlag = *v;
        }
        const bool curveProfile = profile->mode == OldMode::Linear || profile->mode == OldMode::Quad;
        if (!curveProfile && (os.refitEvery || os.curveGeomFlag)) {
            usage("--refit-every / --curve-geom-flag (or RTCD_BM_REFIT_EVERY / RTCD_CURVE_GEOM_FLAG) only apply to "
                  "curve profiles");
        }

        // ---- context: one OptiX device context shared by both stacks (each has its own streams) ----
        ContextOptions ctxOpt = ContextOptions::buildDefault();
        CheckerConfig newCfg  = newConfigForProfile(*profile, support::configsDir() / "spline_weights");
        applyOldSettings(os, newCfg, ctxOpt);
        if (auto v = args.newOverride ? args.newOverride : env.overrides) {
            applyOverrides(newCfg, *v);
        }
        auto ctx = Context::create(ctxOpt);
        ctx->makeCurrent();
        std::cout << "[sbs] scene " << oldSceneName() << ", profile " << profile->name << " (" << profile->origin
                  << "), device " << ctx->deviceInfo().name << "\n";
        std::cout << "[sbs] old optix cfg: '" << os.optixConfig << "'"
                  << (os.refitEvery ? " refit-every=" + std::to_string(*os.refitEvery) : std::string())
                  << (os.curveGeomFlag ? " curve-geom-flag=" + *os.curveGeomFlag : std::string()) << "\n";

        std::unique_ptr<OldRunner> oldR;
        std::unique_ptr<NewRunner> newR;
        if (runOld) {
            oldR = createOldRunner(profile->name, ctx->optixContext(), os);
        }
        if (runNew) {
            const fs::path cfgs = support::configsDir();
            const SceneDesc sd  = loadSceneDesc(args.scene ? fs::path(*args.scene)
                                                           : cfgs / "scenes" / (std::string(oldSceneName()) + ".json"));
            const RobotDesc rd  = loadRobotDesc(args.robot ? fs::path(*args.robot) : cfgs / "robots" / "panda.json");
            auto robot          = RobotModel::load(rd, ctx->logger());
            auto scene          = std::make_shared<const Scene>(ctx, sd);
            newR                = createNewRunner(ctx, newCfg, robot, scene);
            std::cout << "[sbs] new config: " << support::describeConfig(newR->resolvedConfig()) << "\n";
        }

        const fs::path data = support::dataDir(args.data ? std::optional<fs::path>(*args.data) : std::nullopt);
        const Pool pool     = makePool(*profile, data / "motions", args.trajPts);
        bool ok             = true;

        // ---- golden check (regression profiles) ----
        if (args.golden) {
            const std::size_t n = pool.units; // 4096
            Mode m              = newConfigForProfile(*profile, {}).mode;
            const support::Flags golden = support::readFlagsFile(*support::goldenPath(data, m), n);
            auto checkStack = [&](const char* name, auto& r) {
                r->upload(pool.data.data(), n);
                r->check();
                const support::Flags f = support::flagsFromCounts(r->hitCounts(n));
                const auto d          = support::diffFlags(f, golden, n);
                std::cout << "[sbs] " << name << " vs golden: " << d.differing << " diffs (" << support::countSet(f)
                          << " / " << n << " in collision, golden " << support::countSet(golden) << ")\n";
                support::printFlagDiff(std::cout, d, f, golden);
                ok &= d.equal();
            };
            if (oldR) {
                checkStack("old", oldR);
            }
            if (newR) {
                checkStack("new", newR);
            }
        }

        // ---- timing ----
        std::vector<std::size_t> counts;
        const std::size_t maxUnits = std::min(pool.units, profile->maxItems / pool.itemsPerUnit);
        if (args.counts) {
            counts = support::parseCountList(*args.counts);
        } else {
            counts = support::powersOfTwo(std::min<std::size_t>(4096, maxUnits));
        }
        std::ofstream csv;
        if (args.csv) {
            csv.open(*args.csv);
            csv << "scene,profile,n,items,stack,samples,mean_us,median_us,min_us,p10_us,p90_us,max_us,stddev_us\n";
        }
        std::mt19937 gen(args.seed);
        for (const std::size_t n : args.reps == 0 ? std::vector<std::size_t>{} : counts) {
            if (n > maxUnits) {
                std::cout << "[sbs] n=" << n << " skipped (pool / capacity holds " << maxUnits << ")\n";
                continue;
            }
            const std::size_t items = n * pool.itemsPerUnit;
            std::vector<double> tOld, tNew;
            std::size_t mismatchReps = 0;
            support::Stopwatch sw;
            for (std::size_t rep = 0; rep < args.warmup + args.reps; ++rep) {
                const std::vector<float> in = sample(pool, n, gen);
                if (oldR) {
                    oldR->upload(in.data(), items);
                }
                if (newR) {
                    newR->upload(in.data(), items);
                }
                double o = 0, w = 0;
                auto timeOld = [&] { sw.start(); oldR->check(); o = sw.elapsedUs(); };
                auto timeNew = [&] { sw.start(); newR->check(); w = sw.elapsedUs(); };
                if (rep % 2 == 0) { // alternate the order to cancel order effects
                    if (oldR) timeOld();
                    if (newR) timeNew();
                } else {
                    if (newR) timeNew();
                    if (oldR) timeOld();
                }
                if (oldR && newR) {
                    const auto fo = support::flagsFromCounts(oldR->hitCounts(items));
                    const auto fn = support::flagsFromCounts(newR->hitCounts(items));
                    mismatchReps += support::diffFlags(fo, fn, items).equal() ? 0 : 1;
                }
                if (rep >= args.warmup) {
                    if (oldR) tOld.push_back(o);
                    if (newR) tNew.push_back(w);
                }
            }
            const auto so = support::summarize(tOld);
            const auto sn = support::summarize(tNew);
            std::cout << std::fixed << std::setprecision(1) << "[sbs] n=" << std::setw(5) << n << " items="
                      << std::setw(6) << items << " us |";
            if (oldR) printStats(" old", so);
            if (newR) {
                std::cout << " |";
                printStats(" new", sn);
            }
            if (oldR && newR) {
                std::cout << " | new/old median " << std::setprecision(3) << (sn.median / so.median)
                          << " | flag-mismatch reps " << mismatchReps;
                ok &= mismatchReps == 0;
            }
            std::cout << "\n";
            if (csv.is_open()) {
                auto row = [&](const char* stack, const support::TimingStats& s) {
                    csv << oldSceneName() << "," << profile->name << "," << n << "," << items << "," << stack << ","
                        << s.samples << "," << s.mean << "," << s.median << "," << s.min << "," << s.p10 << ","
                        << s.p90 << "," << s.max << "," << s.stddev << "\n";
                };
                if (oldR) row("old", so);
                if (newR) row("new", sn);
            }
        }
        std::cout << "[sbs] " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "rtcd_sbs: error: " << e.what() << "\n";
        return 3;
    }
}
