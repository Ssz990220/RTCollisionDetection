// tests/backend_curve: refit-policy behaviour of the curve backend vs the stored OLD-detector behaviour (new_gpu).
//
// Reference tests/data/reference/refit_flags.txt, written from the OLD CollisionDetector (profiles reg_linear /
// reg_quad) before the old stack was removed: for every item count N of a list, three consecutive checks of the
// first N regression trajectories (the 2nd and 3rd can take the refit path: same item count per stream), per refit
// policy (auto = default, always, never, every 3) and geometry flag, the items whose flag differs from the golden.
// A refitted curve GAS can report a grazing hit a fresh build misses; the old detector did so on exactly the same
// checks and items (e.g. quad, every 3, N = 4096: item 4065 on check 1, 3528 on check 3). The new backend
// (ReferenceBatchRunner) must deviate from the golden in exactly the stored way: same checks, same items.
#include <rtcd/backend.h>
#include <rtcd/context.h>
#include <rtcd/internal/reference_executor.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>
#include <support/reference.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace rtcd;

namespace {

    constexpr std::size_t kItems = 4096;

    struct Policy {
        const char* name;
        RefitPolicy policy;
    };

    void runCase(Mode mode, const Policy& pol, std::optional<GeometryFlags> geom, const char* geomName) {
        static const support::ReferenceFile ref =
            support::ReferenceFile::load(support::referenceDir() / "refit_flags.txt");
        const bool linear  = mode == Mode::LinearCurve;
        const char* setupF = linear ? "regression_linear.json" : "regression_quad.json";
        Setup setup        = loadSetup(support::configsDir() / "setups" / setupF);
        setup.checker.curve.refit = pol.policy;
        if (geom) setup.checker.curve.geometryFlags = geom;
        const CheckerConfig cfg = setup.checker.resolved();

        auto ctx   = Context::create();
        auto robot = RobotModel::load(setup.robot, ctx->logger());
        auto scene = std::make_shared<const Scene>(ctx, setup.scene);
        const std::vector<float> in = support::regressionInputs(support::dataDir() / "motions", 16, kItems, robot->dof());
        const support::Flags golden = support::readFlagsFile(*support::goldenPath(support::dataDir(), mode), kItems);

        StreamLease lease = ctx->acquireStreams(cfg.numStreams);
        std::vector<StreamContext> sc;
        for (std::uint32_t i = 0; i < lease.size(); ++i) sc.push_back({i, lease[i]});
        BackendCreateInfo info{ctx, &cfg, robot, scene, sc};
        auto backend = createCurveBackend(info);
        internal::ReferenceExecutor exec(ctx, backend->programs(), cfg.optix);
        internal::ReferenceBatchRunner runner(ctx, *backend, exec, cfg, sc, robot->dof());

        for (const std::size_t n : {std::size_t(1), std::size_t(16), std::size_t(32), std::size_t(37), std::size_t(128),
                 std::size_t(200), std::size_t(256), std::size_t(517), std::size_t(4096)}) {
            for (int rep = 0; rep < 3; ++rep) {
                const std::string key = std::string(linear ? "linear" : "quad") + "." + geomName + "." + pol.name + "."
                                      + std::to_string(n) + "." + std::to_string(rep + 1);
                const support::Flags f = support::flagsFromCounts(runner.runAll(in, static_cast<std::uint32_t>(n)));
                const auto d           = support::diffFlags(f, golden, n, n);
                const auto want        = ref.integers(key);
                ASSERT_FALSE(want.empty()) << key;
                EXPECT_EQ(d.differing, want[0]) << key << ": items differing from the golden";
                const std::vector<std::uint64_t> wantIdx(want.begin() + 1, want.end());
                const std::vector<std::uint64_t> gotIdx(d.firstIndices.begin(), d.firstIndices.end());
                EXPECT_EQ(gotIdx, wantIdx) << key << ": which items differ from the golden";
            }
        }
        cudaDeviceSynchronize();
    }

    class RefitReference : public ::testing::Test {
    protected:
        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) GTEST_SKIP() << "no CUDA device";
        }
    };

    const Policy kPolicies[] = {
        {"auto", {RefitPolicy::Kind::Auto, 0}},
        {"always", {RefitPolicy::Kind::Always, 0}},
        {"never", {RefitPolicy::Kind::Never, 0}},
        {"every3", {RefitPolicy::Kind::Every, 3}},
    };

} // namespace

TEST_F(RefitReference, Linear) {
    for (const Policy& p : kPolicies) runCase(Mode::LinearCurve, p, std::nullopt, "auto");
}

TEST_F(RefitReference, QuadSingleAnyHit) {
    for (const Policy& p : kPolicies) runCase(Mode::QuadraticCurve, p, std::nullopt, "auto");
}

TEST_F(RefitReference, QuadGeometryFlagNone) {
    for (const Policy& p : kPolicies) runCase(Mode::QuadraticCurve, p, GeometryFlags{kGeometryFlagNone}, "none");
}
