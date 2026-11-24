// tests/backend_curve: refit-policy behaviour of the curve backend vs the OLD stack (new_gpu).
//
// OLD-STACK PARITY TEST (delete with the old stack): the old CollisionDetector of the regression profiles reg_linear /
// reg_quad (through the side-by-side runner's old half, apps/side_by_side/old_runner.h) and the new curve backend
// (ReferenceBatchRunner) get the SAME sequence of checks: for every item count N of a list, three consecutive checks
// of the first N regression trajectories (the 2nd and 3rd can take the refit path: same item count per stream).
// Per policy (auto = default, always, never, every 3) and geometry flag, the per-item flags of old and new must be
// identical for every check; every flag that differs from the golden is printed with its direction.
#include <old_runner.h>

#include <rtcd/backend.h>
#include <rtcd/context.h>
#include <rtcd/internal/reference_executor.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace rtcd;

namespace {

    constexpr std::size_t kItems = 4096;

    struct Policy {
        const char* name;
        RefitPolicy newPolicy;
        std::optional<std::size_t> oldRefitEvery; // old setRefitPolicy(K); nullopt = REFIT_AUTO (default)
    };

    void runCase(Mode mode, const Policy& pol, std::optional<GeometryFlags> geom, const char* geomName) {
        const bool linear  = mode == Mode::LinearCurve;
        const char* setupF = linear ? "regression_linear.json" : "regression_quad.json";
        Setup setup        = loadSetup(support::configsDir() / "setups" / setupF);
        setup.checker.curve.refit = pol.newPolicy;
        if (geom) setup.checker.curve.geometryFlags = geom;
        const CheckerConfig cfg = setup.checker.resolved();

        auto ctx   = Context::create();
        auto robot = RobotModel::load(setup.robot, ctx->logger());
        auto scene = std::make_shared<const Scene>(ctx, setup.scene);
        const std::vector<float> in = support::regressionInputs(support::dataDir() / "motions", 16, kItems, robot->dof());
        const support::Flags golden = support::readFlagsFile(*support::goldenPath(support::dataDir(), mode), kItems);

        // new
        StreamLease lease = ctx->acquireStreams(cfg.numStreams);
        std::vector<StreamContext> sc;
        for (std::uint32_t i = 0; i < lease.size(); ++i) sc.push_back({i, lease[i]});
        BackendCreateInfo info{ctx, &cfg, robot, scene, sc};
        auto backend = createCurveBackend(info);
        internal::ReferenceExecutor exec(ctx, backend->programs(), cfg.optix);
        internal::ReferenceBatchRunner runner(ctx, *backend, exec, cfg, sc, robot->dof());

        // old (same OptiX context and device)
        sbs::OldSettings os;
        os.refitEvery = pol.oldRefitEvery;
        if (geom) os.curveGeomFlag = *geom == kGeometryFlagNone ? "none" : "single_anyhit";
        auto old = sbs::createOldRunner(linear ? "reg_linear" : "reg_quad", ctx->optixContext(), os);

        std::size_t mismatchChecks = 0;
        for (const std::size_t n : {std::size_t(1), std::size_t(16), std::size_t(32), std::size_t(37), std::size_t(128),
                 std::size_t(200), std::size_t(256), std::size_t(517), std::size_t(4096)}) {
            for (int rep = 0; rep < 3; ++rep) {
                const support::Flags fNew = support::flagsFromCounts(runner.runAll(in, static_cast<std::uint32_t>(n)));
                old->upload(in.data(), n);
                old->check();
                const support::Flags fOld = support::flagsFromCounts(old->hitCounts(n));
                const auto dOldNew        = support::diffFlags(fOld, fNew, n);
                mismatchChecks += !dOldNew.equal();
                EXPECT_EQ(dOldNew.differing, 0u) << "old vs new, N " << n << " check " << rep + 1;
                for (int side = 0; side < 2; ++side) {
                    const support::Flags& f = side == 0 ? fOld : fNew;
                    const auto d            = support::diffFlags(f, golden, n, 64);
                    if (d.differing == 0) continue;
                    std::ostringstream s;
                    s << "[ flip ] " << (linear ? "linear" : "quad") << " geom=" << geomName << " policy=" << pol.name
                      << " N=" << n << " check " << rep + 1 << " " << (side == 0 ? "OLD" : "NEW") << ": "
                      << d.differing << " item(s) differ from golden:";
                    for (std::size_t i : d.firstIndices) {
                        s << " " << i << (f[i] ? " (free->collision)" : " (collision->free)");
                    }
                    std::cout << s.str() << "\n";
                }
            }
        }
        std::cout << "[ case ] " << (linear ? "linear" : "quad") << " geom=" << geomName << " policy=" << pol.name
                  << ": 27 checks, old/new mismatching checks: " << mismatchChecks << "\n";
        cudaDeviceSynchronize();
    }

    class RefitParityVsOld : public ::testing::Test {
    protected:
        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) GTEST_SKIP() << "no CUDA device";
        }
    };

    const Policy kPolicies[] = {
        {"auto(default)", {RefitPolicy::Kind::Auto, 0}, std::nullopt},
        {"always", {RefitPolicy::Kind::Always, 0}, std::size_t(0)},
        {"never", {RefitPolicy::Kind::Never, 0}, std::size_t(1)},
        {"every3", {RefitPolicy::Kind::Every, 3}, std::size_t(3)},
    };

} // namespace

TEST_F(RefitParityVsOld, Linear) {
    for (const Policy& p : kPolicies) runCase(Mode::LinearCurve, p, std::nullopt, "auto(none)");
}

TEST_F(RefitParityVsOld, QuadSingleAnyHit) {
    for (const Policy& p : kPolicies) runCase(Mode::QuadraticCurve, p, std::nullopt, "auto(single_anyhit)");
}

TEST_F(RefitParityVsOld, QuadGeometryFlagNone) {
    for (const Policy& p : kPolicies) runCase(Mode::QuadraticCurve, p, GeometryFlags{kGeometryFlagNone}, "none");
}
