// new_gpu: the public CollisionChecker (rtcd/checker.h) with the real Panda RobotModel and a real Scene but a FAKE
// backend. Link seam: this file defines rtcd::detail::createBackendForMode (src/detector/backend_factory.h), so
// the linker resolves the facade's factory call here and never pulls backend_factory.obj (and with it the three
// backend factories) out of the static rtcd_detector library. The real factory is exercised at the merge
// (rtcd_regression --engine checker).
#include <rtcd/checker.h>

#include "detector/backend_factory.h"

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/json_config.h>

#include "fake_backend.h"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using namespace rtcd;
using namespace rtcd_detector_test;

namespace {
    FakeOptions gFakeOptions;          // options of the next backend the facade creates
    FakeBackend* gLastFake   = nullptr; // the backend the facade created last
    int gFactoryCalls        = 0;
} // namespace

namespace rtcd::detail {
    std::unique_ptr<Backend> createBackendForMode(const BackendCreateInfo& info) {
        ++gFactoryCalls;
        EXPECT_NE(info.robot, nullptr);
        EXPECT_NE(info.scene, nullptr);
        EXPECT_EQ(info.config->numStreams, info.streams.size());
        auto b    = std::make_unique<FakeBackend>(info, gFakeOptions);
        gLastFake = b.get();
        return b;
    }
} // namespace rtcd::detail

namespace {

    class FacadeTest : public ::testing::Test {
    protected:
        static std::shared_ptr<Context> ctx;
        static std::shared_ptr<const RobotModel> robot;
        static std::shared_ptr<const Scene> scene;

        static void SetUpTestSuite() {
            ctx   = Context::create();
            robot = RobotModel::load(loadRobotDesc("configs/robots/panda.json"));
            scene = std::make_shared<Scene>(ctx, loadSceneDesc("configs/scenes/shelf_simple.json"));
        }
        static void TearDownTestSuite() {
            scene.reset();
            robot.reset();
            ctx.reset();
        }
        void SetUp() override {
            gFakeOptions     = FakeOptions{};
            gFakeOptions.dof = robot->dof();
            gLastFake        = nullptr;
        }

        static CheckerConfig config(HostDispatch d = HostDispatch::ThreadPerStream) {
            CheckerConfig c;
            c.mode             = Mode::Mesh;
            c.numStreams       = 2;
            c.hostDispatch     = d;
            c.maxBatchSize     = 8;
            c.maxItemsPerCheck = 100;
            c.optix            = OptixOptions::release();
            return c;
        }
        // item i: joint 0 = value(i) (the fake's copy program adds (uint) joint 0), the rest 0.1
        static std::vector<float> items(std::uint32_t n, std::uint32_t dof) {
            std::vector<float> v(std::size_t(n) * dof, 0.1f);
            for (std::uint32_t i = 0; i < n; ++i) {
                v[std::size_t(i) * dof] = float(i % 3); // items with i % 3 == 0 are "free"
            }
            return v;
        }
    };
    std::shared_ptr<Context> FacadeTest::ctx;
    std::shared_ptr<const RobotModel> FacadeTest::robot;
    std::shared_ptr<const Scene> FacadeTest::scene;

} // namespace

TEST_F(FacadeTest, CheckOverloadsDeviceWarmupAndStats) {
    for (HostDispatch d : {HostDispatch::SingleThread, HostDispatch::ThreadPerStream}) {
        CollisionChecker checker(ctx, config(d), robot, scene);
        ASSERT_NE(gLastFake, nullptr);
        EXPECT_EQ(checker.config().mode, Mode::Mesh);
        EXPECT_EQ(checker.config().obbCulling, Toggle::On); // resolved copy
        EXPECT_TRUE(checker.config().sceneEdges.has_value());
        EXPECT_EQ(checker.itemLayout().dof, robot->dof());
        EXPECT_EQ(checker.itemLayout().pointsPerItem, 1u);
        EXPECT_EQ(&checker.robot(), robot.get());
        EXPECT_EQ(&checker.scene(), scene.get());

        const std::uint32_t dof = robot->dof();
        const std::uint32_t n   = 37;
        const auto in           = items(n, dof);

        const CheckResult flagsOnly = checker.check(HostItems{in, n, 0});
        ASSERT_EQ(flagsOnly.collides.size(), n);
        EXPECT_TRUE(flagsOnly.hitCounts.empty());
        const CheckResult r = checker.check(HostItems{in, n, 0}, ResultDetail::FlagsAndHitCounts);
        ASSERT_EQ(r.hitCounts.size(), n);
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(r.hitCounts[i], i % 3) << "item " << i;
            EXPECT_EQ(r.collides[i], i % 3 != 0 ? 1 : 0);
            EXPECT_EQ(flagsOnly.collides[i], r.collides[i]);
        }
        EXPECT_EQ(r.numColliding(), 24u);
        const CheckStats st = checker.lastStats();
        EXPECT_EQ(st.items, n);
        EXPECT_EQ(st.batches, 5u);
        EXPECT_EQ(st.launches, 5u);

        std::vector<std::uint8_t> flags(n);
        std::vector<std::uint32_t> counts(n);
        checker.check(HostItems{in, n, 0}, flags, counts);
        EXPECT_EQ(flags, r.collides);
        EXPECT_EQ(counts, r.hitCounts);

        DeviceBuffer dItems = DeviceBuffer::fromHost(std::span<const float>(in));
        DeviceBuffer dHits(n * sizeof(std::uint32_t));
        checker.checkDevice(DeviceItems{dItems.ptr(), n, 0}, dHits.span<std::uint32_t>());
        std::vector<std::uint32_t> dev(n);
        dHits.download(dev.data(), n * sizeof(std::uint32_t));
        EXPECT_EQ(dev, r.hitCounts);

        EXPECT_TRUE(checker.check(HostItems{in, 0, 0}).collides.empty());
        EXPECT_THROW(checker.check(HostItems{in, 101, 0}), CapacityError);
        EXPECT_THROW(checker.check(HostItems{std::span<const float>(in.data(), 5), 2, 0}), InvalidArgument);

        checker.warmup();
        EXPECT_EQ(checker.lastStats().items, 16u); // numStreams * maxBatchSize
        EXPECT_EQ(gLastFake->begins.back().itemCount, 16u);
    }
}

TEST_F(FacadeTest, MoveKeepsTheCheckerWorking) {
    const std::uint32_t dof = robot->dof();
    const auto in           = items(20, dof);
    CollisionChecker a(ctx, config(), robot, scene);
    CollisionChecker b(std::move(a));
    EXPECT_EQ(b.check(HostItems{in, 20, 0}, ResultDetail::FlagsAndHitCounts).hitCounts[5], 2u);
    CollisionChecker c(ctx, config(HostDispatch::SingleThread), robot, scene);
    c = std::move(b);
    EXPECT_EQ(c.check(HostItems{in, 20, 0}, ResultDetail::FlagsAndHitCounts).hitCounts[7], 1u);
    EXPECT_EQ(c.config().hostDispatch, HostDispatch::ThreadPerStream);
}

TEST_F(FacadeTest, ConstructionErrors) {
    gFactoryCalls = 0;
    EXPECT_THROW(CollisionChecker(nullptr, config(), robot, scene), InvalidArgument);
    EXPECT_THROW(CollisionChecker(ctx, config(), nullptr, scene), InvalidArgument);
    EXPECT_THROW(CollisionChecker(ctx, config(), robot, nullptr), InvalidArgument);
    auto otherCtx = Context::create();
    EXPECT_THROW(CollisionChecker(otherCtx, config(), robot, scene), InvalidArgument); // scene on another Context
    CheckerConfig bad = config();
    bad.numStreams    = 0;
    EXPECT_THROW(CollisionChecker(ctx, bad, robot, scene), ConfigError);
    EXPECT_EQ(gFactoryCalls, 0); // all rejected before the backend is created

    // A backend failure at construction propagates unchanged.
    gFakeOptions.editPrograms = [](ProgramSet& ps) { ps.raygens = {"__raygen__doesNotExist"}; };
    EXPECT_THROW(CollisionChecker(ctx, config(), robot, scene), OptixError);
}
