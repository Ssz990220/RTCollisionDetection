// tests/backend_curve: createCurveBackend end to end through the test-support ReferenceBatchRunner (new_gpu).
//
//   * ProgramSet per curve mode (built-in IS = GAS build flags, ALLOW_UPDATE iff OBB culling is off, SingleGas, ...)
//     and synchronizesHost();
//   * every refit policy gives the golden flags (refits never change collision flags, only build / trace cost);
//   * OBB culling on gives the same flags as off (design doc Q5; the old OBB path mapped compacted primitive indices
//     through the uncompacted segment -> pose map), linear and quadratic, on the regression inputs;
//   * cubic curves run, are deterministic and prefix-consistent (no golden exists);
//   * construction errors (wrong mode, missing / malformed spline weights, cubic + OBB).
#include <rtcd/backend.h>
#include <rtcd/context.h>
#include <rtcd/errors.h>
#include <rtcd/internal/reference_executor.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace rtcd;
namespace fs = std::filesystem;

namespace {

    constexpr std::size_t kItems = 4096;

    struct Env {
        std::shared_ptr<Context> ctx;
        Setup linear, quad;
        std::shared_ptr<const RobotModel> robot;
        std::shared_ptr<const Scene> scene;
        std::vector<float> inputs16; // regression trajectories, 16 points
        support::Flags goldenLinear, goldenQuad;
    };

    Env& env() {
        static Env e = [] {
            Env x;
            x.ctx    = Context::create();
            x.linear = loadSetup(support::configsDir() / "setups" / "regression_linear.json");
            x.quad   = loadSetup(support::configsDir() / "setups" / "regression_quad.json");
            x.robot  = RobotModel::load(x.linear.robot, x.ctx->logger());
            x.scene  = std::make_shared<const Scene>(x.ctx, x.linear.scene);
            const fs::path data = support::dataDir();
            x.inputs16     = support::regressionInputs(data / "motions", 16, kItems, x.robot->dof());
            x.goldenLinear = support::readFlagsFile(*support::goldenPath(data, Mode::LinearCurve), kItems);
            x.goldenQuad   = support::readFlagsFile(*support::goldenPath(data, Mode::QuadraticCurve), kItems);
            return x;
        }();
        return e;
    }

    bool haveGpu() {
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }

    // One backend + reference executor + runner for `cfg` (unresolved; resolved here as the checker would).
    class Harness {
    public:
        explicit Harness(const CheckerConfig& cfg)
            : resolved_(cfg.resolved()), lease_(env().ctx->acquireStreams(resolved_.numStreams)) {
            for (std::uint32_t i = 0; i < lease_.size(); ++i) {
                streams_.push_back(StreamContext{i, lease_[i]});
            }
            BackendCreateInfo info;
            info.context = env().ctx;
            info.config  = &resolved_;
            info.robot   = env().robot;
            info.scene   = env().scene;
            info.streams = streams_;
            backend_     = createCurveBackend(info);
            exec_        = std::make_unique<internal::ReferenceExecutor>(env().ctx, backend_->programs(), resolved_.optix);
            runner_      = std::make_unique<internal::ReferenceBatchRunner>(
                env().ctx, *backend_, *exec_, resolved_, streams_, env().robot->dof());
        }
        support::Flags run(const std::vector<float>& items, std::size_t n) {
            return support::flagsFromCounts(runner_->runAll(items, static_cast<std::uint32_t>(n)));
        }
        Backend& backend() { return *backend_; }

    private:
        CheckerConfig resolved_;
        StreamLease lease_;
        std::vector<StreamContext> streams_;
        std::unique_ptr<Backend> backend_;
        std::unique_ptr<internal::ReferenceExecutor> exec_;
        std::unique_ptr<internal::ReferenceBatchRunner> runner_;
    };

    class CurveBackendTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!haveGpu()) {
                GTEST_SKIP() << "no CUDA device";
            }
            ASSERT_NO_THROW(env());
        }
        static CheckerConfig config(Mode m) {
            CheckerConfig c = m == Mode::LinearCurve ? env().linear.checker : env().quad.checker;
            c.mode          = m;
            return c;
        }
    };

    std::size_t diffCount(const support::Flags& a, const support::Flags& b, std::size_t n) {
        return support::diffFlags(a, b, n).differing;
    }

} // namespace

TEST_F(CurveBackendTest, ProgramSetMatchesModeAndGasFlags) {
    struct Case {
        Mode mode;
        BuiltinIntersection is;
        std::uint32_t prim;
    };
    for (const Case c : {Case{Mode::LinearCurve, BuiltinIntersection::RoundLinear, kPrimitiveRoundLinear},
             Case{Mode::QuadraticCurve, BuiltinIntersection::RoundQuadratic, kPrimitiveRoundQuadratic},
             Case{Mode::CubicCurve, BuiltinIntersection::RoundCubic, kPrimitiveRoundCubic}}) {
        for (bool obb : {false, true}) {
            if (obb && c.mode == Mode::CubicCurve) {
                continue;
            }
            SCOPED_TRACE(std::to_string(static_cast<int>(c.mode)) + (obb ? " obb" : ""));
            CheckerConfig cfg = config(c.mode == Mode::LinearCurve ? Mode::LinearCurve : Mode::QuadraticCurve);
            cfg.mode          = c.mode;
            cfg.obbCulling    = obb ? Toggle::On : Toggle::Off;
            Harness h(cfg);
            const ProgramSet& p = h.backend().programs();
            EXPECT_EQ(h.backend().mode(), c.mode);
            EXPECT_EQ(p.builtinIntersection, c.is);
            EXPECT_EQ(p.primitiveTypes, c.prim);
            EXPECT_EQ(p.builtinIsBuildFlags,
                toBuildFlags(BuildPreference::FastBuild) | (obb ? 0u : BuildFlags{kBuildFlagAllowUpdate}));
            EXPECT_EQ(p.traversableGraph, TraversableGraph::SingleGas);
            EXPECT_EQ(p.maxTraversableDepth, 1u);
            EXPECT_EQ(p.hitGroupRecordCount, 1u);
            EXPECT_EQ(p.numPayloadValues, 0u);
            ASSERT_EQ(p.raygens.size(), 1u);
            EXPECT_EQ(p.raygens[0], "__raygen__oneside");
            EXPECT_EQ(p.anyHit, "__anyhit__GAS");
            EXPECT_TRUE(p.closestHit.empty());
            EXPECT_TRUE(p.miss.empty());
            EXPECT_FALSE(p.moduleCode.empty());
            EXPECT_EQ(h.backend().synchronizesHost(1), obb);
            EXPECT_EQ(h.backend().synchronizesHost(128), obb);
            EXPECT_FALSE(h.backend().synchronizesHost(0));
        }
    }
}

// Refits do NOT leave the flags exactly unchanged: a refitted curve GAS can report a grazing hit that a fresh build
// misses (or the reverse). Measured: quad with {"every": 3} flips item 4065 (last batch, a refit) to "collides"; the
// OLD stack does exactly the same (rtcd_sbs_shelf --profile reg_quad --old-only --golden --reps 0 --refit-every 3:
// "1 diffs ... diff at item 4065: 1 vs 0"), so this is OptiX behaviour, not a porting error. Policies that never
// refit must match the golden exactly; the others within a couple of grazing items.
TEST_F(CurveBackendTest, EveryRefitPolicyGivesTheGoldenFlags) {
    constexpr std::size_t kRefitTolerance = 2;
    for (Mode m : {Mode::LinearCurve, Mode::QuadraticCurve}) {
        const support::Flags& golden = m == Mode::LinearCurve ? env().goldenLinear : env().goldenQuad;
        for (int policy = 0; policy < 5; ++policy) {
            SCOPED_TRACE(std::string(m == Mode::LinearCurve ? "linear" : "quad") + " policy " + std::to_string(policy));
            CheckerConfig cfg = config(m);
            switch (policy) {
            case 0: cfg.curve.refit = {RefitPolicy::Kind::Always, 0}; break;
            case 1: cfg.curve.refit = {RefitPolicy::Kind::Never, 0}; break;
            case 2: cfg.curve.refit = {RefitPolicy::Kind::Every, 3}; break;
            case 3: cfg.curve.autoRefitMaxBatches = 1000; break; // auto: refit for every check here
            default: cfg.hostDispatch = HostDispatch::SingleThread; cfg.numStreams = 1; break; // one stream: refits chain
            }
            const std::size_t tol = policy == 1 ? 0 : kRefitTolerance;
            Harness h(cfg);
            auto check = [&](const support::Flags& f, std::size_t n, const char* what) {
                const auto d = support::diffFlags(f, golden, n);
                EXPECT_LE(d.differing, tol) << what << ": first differing item "
                                            << (d.firstIndices.empty() ? -1 : (long long) d.firstIndices[0]);
                if (d.differing > 0) {
                    std::cout << "[ info ] " << (m == Mode::LinearCurve ? "linear" : "quad") << " policy " << policy << " "
                              << what << ": " << d.differing << " grazing item(s) differ from the golden, first "
                              << d.firstIndices[0] << "\n";
                }
            };
            check(h.run(env().inputs16, kItems), kItems, "first check");
            // a second check refits on top of the first one's GAS (same item counts)
            check(h.run(env().inputs16, kItems), kItems, "second check");
            // small checks (one partial batch, then a full one): rebuild on count change, refit after
            check(h.run(env().inputs16, 37), 37, "37 items");
            check(h.run(env().inputs16, 128), 128, "128 items");
        }
    }
}

TEST_F(CurveBackendTest, ObbCullingGivesTheSameFlagsAsWithout) {
    for (Mode m : {Mode::LinearCurve, Mode::QuadraticCurve}) {
        for (GeometryFlags gf : {GeometryFlags{kGeometryFlagNone}, GeometryFlags{kGeometryFlagRequireSingleAnyHitCall}}) {
            SCOPED_TRACE(std::string(m == Mode::LinearCurve ? "linear" : "quad") + " geometry flags " + std::to_string(gf));
            const support::Flags& golden = m == Mode::LinearCurve ? env().goldenLinear : env().goldenQuad;
            CheckerConfig off            = config(m);
            off.curve.geometryFlags      = gf;
            off.obbCulling               = Toggle::Off;
            CheckerConfig on             = off;
            on.obbCulling                = Toggle::On;
            Harness hOff(off), hOn(on);
            const support::Flags fOff = hOff.run(env().inputs16, kItems);
            const support::Flags fOn  = hOn.run(env().inputs16, kItems);
            EXPECT_EQ(diffCount(fOff, golden, kItems), 0u) << "OBB off vs golden";
            const auto d = support::diffFlags(fOn, fOff, kItems);
            EXPECT_EQ(d.differing, 0u) << "OBB on vs off; first differing item "
                                       << (d.firstIndices.empty() ? -1 : static_cast<long long>(d.firstIndices[0]))
                                       << ", on: " << support::countSet(fOn) << " hits, off: " << support::countSet(fOff);
            // prefix consistency and a second run on the OBB path
            const support::Flags p = hOn.run(env().inputs16, 517);
            EXPECT_EQ(diffCount(p, fOn, 517), 0u);
            const support::Flags again = hOn.run(env().inputs16, kItems);
            EXPECT_EQ(diffCount(again, fOn, kItems), 0u);
        }
    }
}

TEST_F(CurveBackendTest, CubicCurvesRunDeterministicallyAndPrefixConsistent) {
    CheckerConfig cfg = config(Mode::QuadraticCurve);
    cfg.mode          = Mode::CubicCurve;
    cfg.curve.geometryFlags.reset(); // auto: none for cubic
    Harness h(cfg);
    const support::Flags a = h.run(env().inputs16, kItems);
    const support::Flags b = h.run(env().inputs16, kItems);
    EXPECT_EQ(diffCount(a, b, kItems), 0u);
    for (std::size_t n : {std::size_t(1), std::size_t(37), std::size_t(517), std::size_t(4095)}) {
        const support::Flags p = h.run(env().inputs16, n);
        EXPECT_EQ(diffCount(p, a, n), 0u) << "prefix " << n;
    }
    // Same swept spheres, fitted with a different spline: the flags must be close to the quadratic golden.
    const std::size_t hits = support::countSet(a);
    EXPECT_GT(hits, 2500u);
    EXPECT_LT(hits, 3200u);
    EXPECT_LT(diffCount(a, env().goldenQuad, kItems), 300u);
}

TEST_F(CurveBackendTest, ConstructionErrors) {
    {
        CheckerConfig cfg = config(Mode::LinearCurve);
        cfg.mode          = Mode::Mesh;
        EXPECT_THROW(Harness h(cfg), InvalidArgument);
    }
    {
        CheckerConfig cfg          = config(Mode::QuadraticCurve);
        cfg.curve.splineWeightsDir = fs::temp_directory_path() / "rtcd_no_such_spline_weights_dir";
        EXPECT_THROW(Harness h(cfg), IoError);
    }
    {
        CheckerConfig cfg     = config(Mode::QuadraticCurve);
        cfg.curve.endPoints   = CurveEndPoints::Phantom; // no phantom weight files exist (design doc Q15)
        EXPECT_THROW(Harness h(cfg), IoError);
    }
    {
        const fs::path dir = fs::temp_directory_path() / "rtcd_backend_curve_bad_weights";
        fs::create_directories(dir / "quad" / "normal");
        {
            std::ofstream(dir / "quad" / "normal" / "8_16.txt") << "0.5\n0.25\n"; // 2 of 128 weights
        }
        CheckerConfig cfg          = config(Mode::QuadraticCurve);
        cfg.curve.splineWeightsDir = dir;
        EXPECT_THROW(Harness h(cfg), IoError);
        {
            std::ofstream f(dir / "quad" / "normal" / "8_16.txt");
            for (int i = 0; i < 127; ++i) f << "0.1\n";
            f << "abc\n";
        }
        EXPECT_THROW(Harness h(cfg), IoError);
        fs::remove_all(dir);
    }
    {
        // cubic + OBB culling: rejected by validate(); the backend rejects it too if handed such a config
        CheckerConfig cfg = config(Mode::QuadraticCurve);
        cfg.mode          = Mode::CubicCurve;
        cfg.obbCulling    = Toggle::On;
        EXPECT_THROW(cfg.resolved(), ConfigError);
        CheckerConfig r = config(Mode::QuadraticCurve).resolved();
        r.mode          = Mode::CubicCurve;
        r.obbCulling    = Toggle::On;
        StreamLease lease = env().ctx->acquireStreams(r.numStreams);
        std::vector<StreamContext> sc;
        for (std::uint32_t i = 0; i < lease.size(); ++i) sc.push_back({i, lease[i]});
        BackendCreateInfo info{env().ctx, &r, env().robot, env().scene, sc};
        EXPECT_THROW(createCurveBackend(info), UnsupportedError);
    }
}
