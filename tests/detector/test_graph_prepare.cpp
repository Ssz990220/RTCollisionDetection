// GPU test of CheckerConfig::graphPrepare (default on) with the real backends: Backend::prepareCapturable of each
// backend (ray: true, mesh / curves: false, capturable implies not synchronizing), and with the ray backend a
// checker that replays the batch's CUDA part
// (zero hits, staging copy, prepare()) as CUDA graphs must give the same collision flags as a checker without it,
// and the goldens, for every item count (partial batches -> several cached graphs per stream, eviction), for host
// and device items, and for device items at changing addresses (the staging-copy node is re-pointed).
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <rtcd/backend.h>
#include <rtcd/checker.h>
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace rtcd;

namespace {

    struct Env {
        std::shared_ptr<Context> ctx;
        Setup setup;
        std::shared_ptr<const RobotModel> robot;
        std::shared_ptr<const Scene> scene;
        std::vector<float> inputs;
        support::Flags golden;
    };
    Env& env() {
        static Env e = [] {
            Env x;
            x.ctx    = Context::create();
            x.setup  = loadSetup(support::configsDir() / "setups" / "regression_ray.json");
            x.robot  = RobotModel::load(x.setup.robot, x.ctx->logger());
            x.scene  = std::make_shared<const Scene>(x.ctx, x.setup.scene);
            const auto data = support::dataDir();
            x.inputs = support::regressionInputs(data / "motions", 1, support::LegacyHarness::regressionItems, x.robot->dof());
            x.golden = support::readFlagsFile(*support::goldenPath(data, Mode::Ray), support::LegacyHarness::regressionItems);
            return x;
        }();
        return e;
    }

    std::unique_ptr<CollisionChecker> makeChecker(bool graph, std::uint32_t streams, HostDispatch dispatch) {
        CheckerConfig c = env().setup.checker;
        c.numStreams    = streams;
        c.hostDispatch  = dispatch;
        c.graphPrepare  = graph;
        return std::make_unique<CollisionChecker>(env().ctx, c, env().robot, env().scene);
    }

} // namespace

TEST(GraphPrepare, RayFlagsEqualDirectAndGolden) {
    const std::uint32_t dof = env().robot->dof();
    for (std::uint32_t streams : {1u, 4u}) {
        for (HostDispatch dispatch : {HostDispatch::SingleThread, HostDispatch::ThreadPerStream}) {
            auto direct = makeChecker(false, streams, dispatch);
            auto graph  = makeChecker(true, streams, dispatch);
            // Item counts: one item, partial / full / multiple batches; more distinct last-batch sizes than the
            // per-stream graph cache holds (eviction), then the first ones again.
            const std::vector<std::size_t> counts{1, 37, 128, 129, 517, 1000, 3, 5, 7, 9, 11, 13, 15, 17, 19, 4095, 4096, 1, 37};
            for (int rep = 0; rep < 2; ++rep) {
                for (std::size_t n : counts) {
                    const HostItems items{env().inputs, n, 0};
                    const CheckResult a = direct->check(items);
                    const CheckResult b = graph->check(items);
                    const auto d        = support::diffFlags(a.collides, b.collides, n);
                    ASSERT_EQ(d.differing, 0u) << "streams " << streams << ", n " << n << ", rep " << rep;
                    const auto g = support::diffFlags(b.collides, env().golden, n);
                    ASSERT_EQ(g.differing, 0u) << "vs golden: streams " << streams << ", n " << n;
                }
            }
        }
    }
    // Device items at two different addresses (re-pointed staging-copy node), 4 streams, workers.
    auto direct = makeChecker(false, 4, HostDispatch::ThreadPerStream);
    auto graph  = makeChecker(true, 4, HostDispatch::ThreadPerStream);
    const std::size_t N = support::LegacyHarness::regressionItems;
    DeviceBuffer itemsA = DeviceBuffer::fromHost(std::span<const float>(env().inputs));
    std::vector<float> rotated(env().inputs.size());
    for (std::size_t i = 0; i < N; ++i) {
        std::copy_n(env().inputs.begin() + ((i + 1000) % N) * dof, dof, rotated.begin() + i * dof);
    }
    DeviceBuffer itemsB = DeviceBuffer::fromHost(std::span<const float>(rotated));
    DeviceBuffer hitsA(N * sizeof(std::uint32_t)), hitsB(N * sizeof(std::uint32_t));
    for (int rep = 0; rep < 3; ++rep) {
        for (const DeviceBuffer* items : {&itemsA, &itemsB}) {
            for (std::size_t n : {std::size_t(200), N}) {
                const DeviceItems di{items->ptr(), n, 0};
                direct->checkDevice(di, DeviceSpan<std::uint32_t>{hitsA.ptr(), n});
                graph->checkDevice(di, DeviceSpan<std::uint32_t>{hitsB.ptr(), n});
                std::vector<std::uint32_t> a(n), b(n);
                hitsA.download(a.data(), n * sizeof(std::uint32_t));
                hitsB.download(b.data(), n * sizeof(std::uint32_t));
                const auto d = support::diffFlags(support::flagsFromCounts(a), support::flagsFromCounts(b), n);
                ASSERT_EQ(d.differing, 0u) << "device items, n " << n << ", rep " << rep;
            }
        }
    }
}

// Backend::prepareCapturable per backend and configuration; it must imply !synchronizesHost.
TEST(GraphPrepare, PrepareCapturablePerBackend) {
    struct Case {
        const char* setup;
        std::string overrides;
        bool expectCapturable;
    };
    const std::vector<Case> cases{
        {"regression_ray.json", "", true},
        {"regression_ray.json", "obb_culling=false", true},
        {"regression_ray.json", "ray.hit_mode=first_hit_miss", true},
        {"regression_ray.json", "ray.hit_mode=traverse", true},
        {"regression_ias.json", "", false},
        {"regression_ias.json", "obb_culling=false", false},
        {"regression_linear.json", "", false},
        {"regression_quad.json", "", false},
    };
    for (const Case& k : cases) {
        CheckerConfig c = loadSetup(support::configsDir() / "setups" / k.setup).checker;
        if (!k.overrides.empty()) {
            applyOverrides(c, k.overrides);
        }
        c.numStreams          = 1;
        const CheckerConfig r = c.resolved();
        StreamLease lease     = env().ctx->acquireStreams(1);
        const std::vector<StreamContext> streams{StreamContext{0, lease[0]}};
        BackendCreateInfo info{env().ctx, &r, env().robot, env().scene, streams};
        std::unique_ptr<Backend> b = r.mode == Mode::Ray    ? createRayBackend(info)
                                   : r.mode == Mode::Mesh ? createMeshBackend(info)
                                                          : createCurveBackend(info);
        for (std::uint32_t n : {1u, 2u, 127u, 128u, r.maxBatchSize / 2, r.maxBatchSize}) {
            if (n > r.maxBatchSize) {
                continue;
            }
            EXPECT_EQ(b->prepareCapturable(n), k.expectCapturable) << k.setup << " " << k.overrides << " n " << n;
            if (b->prepareCapturable(n)) {
                EXPECT_FALSE(b->synchronizesHost(n)) << k.setup << " n " << n;
            }
            // prepareSplit (G3): mesh below mesh.compaction_min_poses (masked and OBB-off paths), nobody else
            const bool split = r.mode == Mode::Mesh && n < r.mesh.compactionMinPoses;
            EXPECT_EQ(b->prepareSplit(n), split) << k.setup << " " << k.overrides << " n " << n;
            if (b->prepareSplit(n)) {
                EXPECT_FALSE(b->synchronizesHost(n)) << k.setup << " n " << n;
                EXPECT_FALSE(b->prepareCapturable(n)) << k.setup << " n " << n;
            }
        }
        if (r.mode == Mode::Ray) {
            EXPECT_FALSE(b->prepareCapturable(0));
            EXPECT_FALSE(b->prepareCapturable(r.maxBatchSize + 1));
        }
    }
}