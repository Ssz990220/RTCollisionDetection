// GPU test of the graph-prepare experiment of DetectorCore (Track G0; env RTCD_GRAPH_PREPARE=1 at checker
// construction, off by default): with the real ray backend, a checker that replays the batch's CUDA part
// (zero hits, staging copy, prepare()) as CUDA graphs must give the same collision flags as a checker without it,
// and the goldens, for every item count (partial batches -> several cached graphs per stream, eviction), for host
// and device items, and for device items at changing addresses (the staging-copy node is re-pointed).
#ifndef NOMINMAX
#define NOMINMAX
#endif
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

    void setGraphPrepareEnv(bool on) {
#ifdef _WIN32
        _putenv_s("RTCD_GRAPH_PREPARE", on ? "1" : "0");
#else
        setenv("RTCD_GRAPH_PREPARE", on ? "1" : "0", 1);
#endif
    }

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
        setGraphPrepareEnv(graph);
        auto checker = std::make_unique<CollisionChecker>(env().ctx, c, env().robot, env().scene);
        setGraphPrepareEnv(false);
        return checker;
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

// The switch really reaches the core (the facade gives no other way to observe it): the checker logs at Info.
TEST(GraphPrepare, EnabledOnlyForRayAndOnlyWithTheSwitch) {
    std::vector<std::string> messages;
    ContextOptions o = ContextOptions::buildDefault();
    o.logger.level   = LogLevel::Debug;
    o.logger.sink    = [&](LogLevel, std::string_view m) { messages.emplace_back(m); };
    auto ctx         = Context::create(o);
    auto scene       = std::make_shared<const Scene>(ctx, env().setup.scene);
    auto count       = [&](const char* text) {
        std::size_t k = 0;
        for (const auto& m : messages) k += m.find(text) != std::string::npos;
        return k;
    };
    CheckerConfig c = env().setup.checker;
    c.verbosity     = LogLevel::Info;
    { CollisionChecker off(ctx, c, env().robot, scene); }
    EXPECT_EQ(count("RTCD_GRAPH_PREPARE"), 0u);
    setGraphPrepareEnv(true);
    { CollisionChecker on(ctx, c, env().robot, scene); }
    CheckerConfig mesh = loadSetup(support::configsDir() / "setups" / "regression_ias.json").checker;
    mesh.verbosity     = LogLevel::Info;
    { CollisionChecker m(ctx, mesh, env().robot, scene); }
    setGraphPrepareEnv(false);
    EXPECT_EQ(count("replayed as CUDA graphs"), 1u);
    EXPECT_EQ(count("ignored (only the ray backend"), 1u);
}
