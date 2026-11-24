// GPU test of the relaxed BatchView address rules (2026-11-24, rtcd/backend.h) with the REAL backends of every
// mode, end to end through CollisionChecker against the goldens: the core reads packed items in place (mesh,
// curves; ray replays its graph from the staging buffer), counts into the result range when a stream's previous
// batch had the same target (ray: graph keyed by the hit-count address), and runs single-batch checks on the
// calling thread. Covered: repeated checks into the SAME result buffer, alternating between TWO result buffers and
// TWO input buffers, single- and multi-batch checks (streams with one and with several batches), strided device
// items (staging), host check() between device checks (also strided host items), both HostDispatch modes, and
// CheckerConfig::mappedHostTransfer on (zero-copy host checks) and off (upload + download).
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
#include <memory>
#include <string>
#include <vector>

using namespace rtcd;

namespace {

    struct ModeEnv {
        Setup setup;
        std::shared_ptr<const RobotModel> robot;
        std::shared_ptr<const Scene> scene;
        std::size_t fpi = 0;
        std::vector<float> inputs; // 4096 regression items, packed
        support::Flags golden;
    };

    std::shared_ptr<Context> ctx() {
        static std::shared_ptr<Context> c = Context::create();
        return c;
    }

    ModeEnv makeEnv(const char* setupFile) {
        ModeEnv e;
        e.setup = loadSetup(support::configsDir() / "setups" / setupFile);
        e.robot = RobotModel::load(e.setup.robot, ctx()->logger());
        e.scene = std::make_shared<const Scene>(ctx(), e.setup.scene);
        const CheckerConfig r = e.setup.checker.resolved();
        const ItemLayout l    = r.itemLayout(e.robot->dof());
        e.fpi                 = l.floatsPerItem();
        const auto data       = support::dataDir();
        e.inputs = support::regressionInputs(data / "motions", l.pointsPerItem, support::LegacyHarness::regressionItems,
            e.robot->dof());
        e.golden = support::readFlagsFile(*support::goldenPath(data, r.mode), support::LegacyHarness::regressionItems);
        return e;
    }

    class AddressRules : public ::testing::TestWithParam<const char*> {};

} // namespace

TEST_P(AddressRules, FlagsMatchGoldenForEveryAddressPattern) {
    const ModeEnv e    = makeEnv(GetParam());
    const std::size_t N = support::LegacyHarness::regressionItems;
    // Input B = the regression items rotated by 1000: its golden is the rotated golden.
    std::vector<float> rotated(e.inputs.size());
    support::Flags goldenB(N);
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t j = (i + 1000) % N;
        std::copy_n(e.inputs.begin() + std::ptrdiff_t(j * e.fpi), e.fpi, rotated.begin() + std::ptrdiff_t(i * e.fpi));
        goldenB[i] = e.golden[j];
    }
    // strided copy of input A (stride fpi + 1) -> staging path
    const std::size_t stride = e.fpi + 1;
    std::vector<float> strided(N * stride, -7.0f);
    for (std::size_t i = 0; i < N; ++i) {
        std::copy_n(e.inputs.begin() + std::ptrdiff_t(i * e.fpi), e.fpi, strided.begin() + std::ptrdiff_t(i * stride));
    }
    ctx()->makeCurrent();
    DeviceBuffer inA = DeviceBuffer::fromHost(std::span<const float>(e.inputs));
    DeviceBuffer inB = DeviceBuffer::fromHost(std::span<const float>(rotated));
    DeviceBuffer inS = DeviceBuffer::fromHost(std::span<const float>(strided));
    DeviceBuffer r0(N * sizeof(std::uint32_t)), r1(N * sizeof(std::uint32_t));

    for (int variant = 0; variant < 4; ++variant) {
        const HostDispatch dispatch = variant % 2 == 0 ? HostDispatch::SingleThread : HostDispatch::ThreadPerStream;
        const bool mapped           = variant < 2; // CheckerConfig::mappedHostTransfer (host checks only)
        SCOPED_TRACE(std::string(mapped ? "mapped host transfer" : "copy host transfer") + ", dispatch "
                     + std::to_string(int(dispatch)));
        CheckerConfig c       = e.setup.checker;
        c.hostDispatch        = dispatch;
        c.mappedHostTransfer  = mapped;
        CollisionChecker checker(ctx(), c, e.robot, e.scene);
        const std::uint32_t B = checker.config().maxBatchSize;
        const std::uint32_t S = checker.config().numStreams;
        // single batch; one batch per stream; several batches per stream; everything
        const std::vector<std::size_t> counts{1, std::min<std::size_t>(B, N), std::min<std::size_t>(std::size_t(B) * S, N),
            std::min<std::size_t>(std::size_t(B) * S + 3, N), N};
        auto device = [&](const DeviceBuffer& in, std::size_t st, DeviceBuffer& r, std::size_t n,
                          const support::Flags& gold, const std::string& what) {
            checker.checkDevice(DeviceItems{in.ptr(), n, st}, DeviceSpan<std::uint32_t>{r.ptr(), n});
            std::vector<std::uint32_t> h(n);
            r.download(h.data(), n * sizeof(std::uint32_t));
            const auto d = support::diffFlags(support::flagsFromCounts(h), gold, n);
            EXPECT_EQ(d.differing, 0u) << what << ": n " << n << ", dispatch " << int(dispatch);
        };
        for (std::size_t n : counts) {
            for (int rep = 0; rep < 3; ++rep) { // same result buffer, same input: direct counting from rep 1 on
                device(inA, 0, r0, n, e.golden, "same buffer rep " + std::to_string(rep));
            }
            for (int rep = 0; rep < 4; ++rep) { // alternating result and input buffers
                const bool a = rep % 2 == 0;
                device(a ? inA : inB, 0, (rep / 2) % 2 == 0 ? r1 : r0, n, a ? e.golden : goldenB,
                    "alternating rep " + std::to_string(rep));
            }
            device(inS, stride, r0, n, e.golden, "strided");
            device(inS, stride, r0, n, e.golden, "strided again");
            for (int rep = 0; rep < 2; ++rep) {
                const CheckResult h = checker.check(HostItems{e.inputs, n, 0});
                EXPECT_EQ(support::diffFlags(h.collides, e.golden, n).differing, 0u) << "host check n " << n;
                const CheckResult hb = checker.check(HostItems{rotated, n, 0});
                EXPECT_EQ(support::diffFlags(hb.collides, goldenB, n).differing, 0u) << "host check B n " << n;
            }
            const CheckResult hs = checker.check(HostItems{strided, n, stride});
            EXPECT_EQ(support::diffFlags(hs.collides, e.golden, n).differing, 0u) << "strided host check n " << n;
            device(inB, 0, r0, n, goldenB, "after host check");
        }
    }
}

INSTANTIATE_TEST_SUITE_P(AllModes, AddressRules,
    ::testing::Values("regression_ias.json", "regression_ray.json", "regression_linear.json", "regression_quad.json"),
    [](const ::testing::TestParamInfo<const char*>& i) {
        std::string s = i.param;
        s             = s.substr(std::string("regression_").size());
        return s.substr(0, s.find('.'));
    });
