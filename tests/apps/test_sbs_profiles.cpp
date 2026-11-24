// new_unit: the side-by-side runner's profile -> CheckerConfig mapping gives valid configs whose resolved values are
// the old setups' (mode, streams, batch, OBB, build preference, curve points, the per-mode defaults).
#include <old_runner.h>
#include <profile_config.h>

#include <rtcd/errors.h>

#include <gtest/gtest.h>

using namespace rtcd;
using namespace rtcd::sbs;

namespace {
    // Copy of the profile table entries checked here (old_profiles.cpp is the old-stack TU and is not linked into
    // this host-only test).
    OldProfile prof(std::string name, OldMode m, std::size_t s, std::size_t b, std::size_t t, std::size_t c, OldBuild bd,
        bool obb, std::size_t cap) {
        return OldProfile{std::move(name), "", m, s, b, t, c, bd, obb, cap};
    }
} // namespace

TEST(SbsProfiles, DiscreteProfiles) {
    const CheckerConfig ias = newConfigForProfile(prof("bm_ias", OldMode::Ias, 4, 1024, 1, 0, OldBuild::Compact, true, 8196), {});
    const CheckerConfig r   = ias.resolved();
    EXPECT_EQ(r.mode, Mode::Mesh);
    EXPECT_EQ(r.numStreams, 4u);
    EXPECT_EQ(r.maxBatchSize, 1024u);
    EXPECT_EQ(r.maxItemsPerCheck, 8196u);
    EXPECT_TRUE(r.obbCullingEnabled());
    EXPECT_EQ(r.hostDispatch, HostDispatch::ThreadPerStream);
    EXPECT_EQ(r.mesh.linkGasBuild, BuildPreference::FastTraceCompact);
    EXPECT_EQ(r.mesh.compactionMinPoses, 128u); // old compactMinPoses default
    EXPECT_EQ(r.sceneEdges, std::optional<EdgeSet>(EdgeSet::Unique));
    EXPECT_TRUE(r.collapseRootLinkObb);

    const CheckerConfig ray = newConfigForProfile(prof("bm_ray", OldMode::Ray, 4, 128, 1, 0, OldBuild::None, true, 4096), {}).resolved();
    EXPECT_EQ(ray.mode, Mode::Ray);
    EXPECT_EQ(ray.ray.launchWidthCap, 262144u); // old batchRayRobot default
    EXPECT_TRUE(ray.ray.skipHitItems);
    EXPECT_EQ(ray.ray.hitMode, RayHitMode::ClosestHit);
}

TEST(SbsProfiles, CurveProfiles) {
    const auto quad = newConfigForProfile(
        prof("bm_quad_4_8", OldMode::Quad, 16, 16, 8, 4, OldBuild::FastTrace, false, 4096), "C:/weights").resolved();
    EXPECT_EQ(quad.mode, Mode::QuadraticCurve);
    EXPECT_EQ(quad.numStreams, 16u);
    EXPECT_EQ(quad.maxBatchSize, 16u);
    EXPECT_EQ(quad.curve.trajectoryPoints, 8u);
    EXPECT_EQ(quad.curve.controlPoints, 4u);
    EXPECT_EQ(quad.curve.gasBuild, BuildPreference::FastTrace);
    EXPECT_FALSE(quad.obbCullingEnabled());
    EXPECT_EQ(quad.curve.geometryFlags, std::optional<GeometryFlags>(kGeometryFlagRequireSingleAnyHitCall));
    EXPECT_EQ(quad.sceneEdges, std::optional<EdgeSet>(EdgeSet::Loop));
    EXPECT_EQ(quad.curve.splineWeightsDir, std::filesystem::path("C:/weights"));

    const auto lin = newConfigForProfile(
        prof("reg_linear", OldMode::Linear, 4, 128, 16, 16, OldBuild::FastBuild, false, 4096), {}).resolved();
    EXPECT_EQ(lin.mode, Mode::LinearCurve);
    EXPECT_EQ(lin.curve.gasBuild, BuildPreference::FastBuild);
    EXPECT_EQ(lin.curve.geometryFlags, std::optional<GeometryFlags>(kGeometryFlagNone));
    EXPECT_EQ(lin.itemLayout(7).floatsPerItem(), 16u * 7u);
}

TEST(SbsProfiles, OldSettingsApplyToTheNewStructs) {
    CheckerConfig c = newConfigForProfile(prof("q", OldMode::Quad, 4, 128, 16, 8, OldBuild::FastBuild, false, 4096), "w");
    ContextOptions ctx;
    OldSettings s;
    s.optixConfig   = "maxreg=32,rayhit=traverse,validation=1";
    s.refitEvery    = 3;
    s.curveGeomFlag = "none";
    applyOldSettings(s, c, ctx);
    EXPECT_EQ(c.optix.maxRegisterCount, 32u);
    EXPECT_EQ(c.ray.hitMode, RayHitMode::FirstHitTraverse);
    EXPECT_TRUE(ctx.optixValidation);
    EXPECT_EQ(c.curve.refit.kind, RefitPolicy::Kind::Every);
    EXPECT_EQ(c.curve.refit.every, 3u);
    EXPECT_EQ(c.curve.geometryFlags, std::optional<GeometryFlags>(kGeometryFlagNone));
    EXPECT_NO_THROW(c.validate());
    s.optixConfig = "bogus=1";
    EXPECT_THROW(applyOldSettings(s, c, ctx), ConfigError);
}

TEST(SbsProfiles, BuildNoneOnlyForRay) {
    EXPECT_THROW(newConfigForProfile(prof("x", OldMode::Ias, 4, 128, 1, 0, OldBuild::None, true, 4096), {}), InvalidArgument);
}
