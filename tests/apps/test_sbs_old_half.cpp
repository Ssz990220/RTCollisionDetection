// new_gpu, OLD-STACK test (deleted with the old stack): the old half of the side-by-side runner reproduces the four
// goldens with the regression profiles, using an rtcd::Context's OptiX context and support/inputs.h inputs. This is
// what makes its timings comparable: same detector setup as ccRegression, same inputs as the new stack gets.
#include <old_runner.h>

#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <rtcd/context.h>

#include <gtest/gtest.h>

using namespace rtcd;

namespace {
    struct Case {
        const char* profile;
        Mode mode;
        std::uint32_t points;
    };
} // namespace

TEST(SbsOldHalf, RegressionProfilesMatchGoldens) {
    EXPECT_STREQ(sbs::oldSceneName(), "shelf");
    auto ctx = Context::create();
    ctx->makeCurrent();
    const auto data = support::dataDir();
    for (const Case c : {Case{"reg_ias", Mode::Mesh, 1}, Case{"reg_ray", Mode::Ray, 1},
             Case{"reg_linear", Mode::LinearCurve, 16}, Case{"reg_quad", Mode::QuadraticCurve, 16}}) {
        const std::vector<float> in = support::regressionInputs(data / "motions", c.points);
        const support::Flags golden = support::readFlagsFile(*support::goldenPath(data, c.mode), 4096);
        auto r                      = sbs::createOldRunner(c.profile, ctx->optixContext(), {});
        ASSERT_EQ(r->profile().trajPts, c.points);
        r->upload(in.data(), 4096);
        r->check();
        const support::Flags f = support::flagsFromCounts(r->hitCounts(4096));
        const auto d           = support::diffFlags(f, golden, 4096);
        EXPECT_TRUE(d.equal()) << c.profile << ": " << d.differing << " diffs vs golden";
        // a prefix (partial batch on a later stream) agrees with the full run
        r->upload(in.data(), 517);
        r->check();
        const support::Flags p = support::flagsFromCounts(r->hitCounts(517));
        EXPECT_TRUE(support::diffFlags(p, golden, 517).equal()) << c.profile << " prefix 517";
    }
}

TEST(SbsOldHalf, UnknownProfileThrows) {
    auto ctx = Context::create();
    EXPECT_THROW(sbs::createOldRunner("nope", ctx->optixContext(), {}), std::runtime_error);
    EXPECT_EQ(sbs::findOldProfile("nope"), nullptr);
    ASSERT_NE(sbs::findOldProfile("bm_quad_8_16"), nullptr);
    EXPECT_EQ(sbs::oldProfiles().size(), 12u);
}
