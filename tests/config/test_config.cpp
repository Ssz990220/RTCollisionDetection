// rtcd_config: CheckerConfig defaults (docs/design/interfaces.md 7.1), every validate() rule (3.7) and resolved()
// per mode. new_unit (no GPU).
#include <rtcd/config.h>
#include <rtcd/errors.h>
#include <rtcd/limits.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace rtcd;

namespace {

    constexpr Mode kAllModes[] = {Mode::Mesh, Mode::Ray, Mode::LinearCurve, Mode::QuadraticCurve, Mode::CubicCurve};

    // A config of `mode` that passes validate() with every other field at its default.
    CheckerConfig validFor(Mode mode) {
        CheckerConfig c;
        c.mode = mode;
        if (curveDegree(mode) >= 2) {
            c.curve.splineWeightsDir = "spline_weights";
        }
        return c;
    }

    // The ConfigError thrown by c.validate() (fails the test if none).
    ConfigError validationError(const CheckerConfig& c) {
        try {
            c.validate();
        } catch (const ConfigError& e) {
            return e;
        }
        ADD_FAILURE() << "validate() did not throw";
        return ConfigError("", "");
    }

    std::vector<std::string> lines(const std::string& s) {
        std::vector<std::string> out;
        std::size_t start = 0;
        while (start <= s.size()) {
            const std::size_t nl = s.find('\n', start);
            out.push_back(s.substr(start, nl == std::string::npos ? std::string::npos : nl - start));
            if (nl == std::string::npos) {
                break;
            }
            start = nl + 1;
        }
        return out;
    }

    // Expects exactly one problem, on `field`, whose message contains `fragment`.
    void expectSingleProblem(const CheckerConfig& c, const std::string& field, const std::string& fragment = {}) {
        const ConfigError e = validationError(c);
        EXPECT_EQ(e.field(), field);
        EXPECT_EQ(e.kind(), ErrorKind::Config);
        const auto l = lines(e.what());
        ASSERT_EQ(l.size(), 1u) << e.what();
        EXPECT_EQ(l[0].rfind(field + ": ", 0), 0u) << l[0];
        EXPECT_NE(l[0].find(fragment), std::string::npos) << l[0];
    }

} // namespace

// ---- defaults ----------------------------------------------------------------------------------------------------

TEST(ConfigDefaults, MatchTheSchemaTable) {
    const CheckerConfig c;
    EXPECT_EQ(c.mode, Mode::Mesh);
    EXPECT_EQ(c.numStreams, 4u);
    EXPECT_EQ(c.hostDispatch, HostDispatch::ThreadPerStream);
    EXPECT_EQ(c.maxBatchSize, 1024u);
    EXPECT_EQ(c.maxItemsPerCheck, 8192u);
    EXPECT_EQ(c.obbCulling, Toggle::Auto);
    EXPECT_TRUE(c.collapseRootLinkObb);
    EXPECT_FALSE(c.sceneEdges.has_value());
    EXPECT_EQ(c.verbosity, LogLevel::Warning);
#if !defined(_DEBUG) && !defined(DEBUG)
    EXPECT_EQ(c.optix.optLevel, OptixOptions::OptLevel::Default);
    EXPECT_EQ(c.optix.debugLevel, OptixOptions::DebugLevel::Default);
    EXPECT_EQ(c.optix.exceptionFlags, 0u);
#else
    EXPECT_EQ(c.optix.optLevel, OptixOptions::OptLevel::O0);
    EXPECT_EQ(c.optix.debugLevel, OptixOptions::DebugLevel::Full);
    EXPECT_EQ(c.optix.exceptionFlags, OptixOptions::kExceptionTraceDepth | OptixOptions::kExceptionStackOverflow);
#endif
    EXPECT_EQ(c.optix.maxRegisterCount, 50u);
    EXPECT_EQ(c.optix.maxTraceDepth, 2u);
    EXPECT_FALSE(c.optix.printPipelineLog);
    EXPECT_EQ(c.mesh.linkGasBuild, BuildPreference::FastTraceCompact);
    EXPECT_EQ(c.mesh.compactionMinPoses, 128u);
    const Transform park{{0.001f, 0, 0, 0, 0, 0.001f, 0, 0, 0, 0, 0.001f, -0.1f}};
    EXPECT_EQ(c.mesh.parkTransform.m, park.m);
    EXPECT_EQ(c.ray.launchWidthCap, 262144u);
    EXPECT_TRUE(c.ray.skipHitItems);
    EXPECT_EQ(c.ray.hitMode, RayHitMode::ClosestHit);
    EXPECT_EQ(c.curve.trajectoryPoints, 16u);
    EXPECT_EQ(c.curve.controlPoints, 8u);
    EXPECT_EQ(c.curve.endPoints, CurveEndPoints::Normal);
    EXPECT_EQ(c.curve.gasBuild, BuildPreference::FastBuild);
    EXPECT_EQ(c.curve.refit.kind, RefitPolicy::Kind::Auto);
    EXPECT_EQ(c.curve.autoRefitMaxBatches, -1);
    EXPECT_FALSE(c.curve.geometryFlags.has_value());
    EXPECT_TRUE(c.curve.splineWeightsDir.empty());
}

TEST(ConfigDefaults, ValidForDiscreteAndLinearModes) {
    for (Mode m : {Mode::Mesh, Mode::Ray, Mode::LinearCurve}) {
        CheckerConfig c;
        c.mode = m;
        EXPECT_NO_THROW(c.validate()) << static_cast<int>(m);
    }
}

TEST(ConfigDefaults, SplineCurvesNeedTheWeightsDir) {
    for (Mode m : {Mode::QuadraticCurve, Mode::CubicCurve}) {
        CheckerConfig c;
        c.mode = m;
        expectSingleProblem(c, "curve.spline_weights_dir", "must be set");
        EXPECT_NO_THROW(validFor(m).validate());
    }
}

// ---- validate(): one test per rule ---------------------------------------------------------------------------------

TEST(ConfigValidate, CountsMustBePositive) {
    for (Mode m : kAllModes) {
        CheckerConfig c = validFor(m);
        c.numStreams    = 0;
        expectSingleProblem(c, "streams", ">= 1");
        c               = validFor(m);
        c.maxBatchSize  = 0;
        expectSingleProblem(c, "max_batch_size", ">= 1");
        c                  = validFor(m);
        c.maxItemsPerCheck = 0;
        expectSingleProblem(c, "max_items_per_check", ">= 1");
        c                     = validFor(m);
        c.optix.maxTraceDepth = 0;
        expectSingleProblem(c, "optix.max_trace_depth", ">= 1");
    }
    CheckerConfig c   = validFor(Mode::Mesh);
    c.numStreams      = 1;
    c.maxBatchSize    = 1;
    c.maxItemsPerCheck = 1;
    c.optix.maxTraceDepth = 1;
    EXPECT_NO_THROW(c.validate());
}

TEST(ConfigValidate, CompactionMinPosesAnyValue) {
    for (std::uint32_t v : {0u, 1u, 128u, 1024u, 1025u, 0xffffffffu}) {
        CheckerConfig c            = validFor(Mode::Mesh);
        c.mesh.compactionMinPoses = v;
        EXPECT_NO_THROW(c.validate()) << v;
    }
}

TEST(ConfigValidate, TrajectoryPointsRange) {
    for (Mode m : {Mode::LinearCurve, Mode::QuadraticCurve, Mode::CubicCurve}) {
        CheckerConfig c          = validFor(m);
        c.curve.trajectoryPoints = 1;
        c.curve.controlPoints    = 0; // keep the control-point rule out of the way for linear (ignored there)
        const ConfigError e      = validationError(c);
        EXPECT_EQ(e.field(), "curve.trajectory_points") << e.what();

        c                        = validFor(m);
        c.curve.trajectoryPoints = kMaxTrajectoryPoints + 1;
        const ConfigError e2     = validationError(c);
        EXPECT_EQ(e2.field(), "curve.trajectory_points") << e2.what();
    }
    CheckerConfig c          = validFor(Mode::LinearCurve);
    c.curve.trajectoryPoints = 2;
    EXPECT_NO_THROW(c.validate());
    c.curve.trajectoryPoints = kMaxTrajectoryPoints;
    EXPECT_NO_THROW(c.validate()); // linear: control points and spline weights do not apply
}

TEST(ConfigValidate, ControlPointsBetweenDegreeAndTrajectoryPoints) {
    for (Mode m : {Mode::QuadraticCurve, Mode::CubicCurve}) {
        const std::uint32_t degree = curveDegree(m);
        CheckerConfig c            = validFor(m);
        c.curve.controlPoints      = degree; // must be > degree
        expectSingleProblem(c, "curve.control_points", "must be >");
        c.curve.controlPoints = degree + 1;
        EXPECT_NO_THROW(c.validate());
        c.curve.controlPoints = c.curve.trajectoryPoints; // must be < trajectory points
        expectSingleProblem(c, "curve.control_points", "< curve.trajectory_points = 16");
        c.curve.controlPoints = c.curve.trajectoryPoints - 1;
        EXPECT_NO_THROW(c.validate());
    }
    // Linear curves ignore control points.
    CheckerConfig c       = validFor(Mode::LinearCurve);
    c.curve.controlPoints = 0;
    EXPECT_NO_THROW(c.validate());
    c.curve.controlPoints = 1000;
    EXPECT_NO_THROW(c.validate());
}

TEST(ConfigValidate, SplineWeightCapacity) {
    CheckerConfig c          = validFor(Mode::QuadraticCurve);
    c.curve.trajectoryPoints = 64;
    c.curve.controlPoints    = 8; // 512 = kMaxSplineWeights: ok
    EXPECT_NO_THROW(c.validate());
    c.curve.controlPoints = 9; // 576
    expectSingleProblem(c, "curve.control_points", "kMaxSplineWeights = 512");
    // Phantom end points add two control points per curve: 6 + 2 = 8 -> 512 ok, 7 + 2 -> 576 too many.
    c.curve.endPoints     = CurveEndPoints::Phantom;
    c.curve.controlPoints = 6;
    EXPECT_NO_THROW(c.validate());
    c.curve.controlPoints = 7;
    expectSingleProblem(c, "curve.control_points", "phantom");
}

TEST(ConfigValidate, RefitEveryNeedsAtLeastTwo) {
    for (Mode m : {Mode::LinearCurve, Mode::QuadraticCurve, Mode::CubicCurve}) {
        CheckerConfig c = validFor(m);
        for (std::uint32_t k : {0u, 1u}) {
            c.curve.refit = RefitPolicy{RefitPolicy::Kind::Every, k};
            expectSingleProblem(c, "curve.refit", "every >= 2");
        }
        c.curve.refit = RefitPolicy{RefitPolicy::Kind::Every, 2};
        EXPECT_NO_THROW(c.validate());
        for (auto kind : {RefitPolicy::Kind::Auto, RefitPolicy::Kind::Always, RefitPolicy::Kind::Never}) {
            c.curve.refit = RefitPolicy{kind, 0};
            EXPECT_NO_THROW(c.validate());
        }
    }
}

TEST(ConfigValidate, ObbCullingOnRejectedForCubicOnly) {
    for (Mode m : kAllModes) {
        CheckerConfig c = validFor(m);
        c.obbCulling    = Toggle::On;
        if (m == Mode::CubicCurve) {
            expectSingleProblem(c, "obb_culling", "cubic_curve");
        } else {
            EXPECT_NO_THROW(c.validate()) << static_cast<int>(m);
        }
        c.obbCulling = Toggle::Off;
        EXPECT_NO_THROW(c.validate());
    }
}

TEST(ConfigValidate, CurveGeometryFlags) {
    for (Mode m : {Mode::LinearCurve, Mode::QuadraticCurve, Mode::CubicCurve}) {
        CheckerConfig c = validFor(m);
        for (GeometryFlags ok : {kGeometryFlagNone, kGeometryFlagRequireSingleAnyHitCall}) {
            c.curve.geometryFlags = ok;
            EXPECT_NO_THROW(c.validate());
        }
        for (GeometryFlags bad : {GeometryFlags{kGeometryFlagDisableAnyHit},
                 GeometryFlags{kGeometryFlagDisableAnyHit | kGeometryFlagRequireSingleAnyHitCall}, GeometryFlags{8}}) {
            c.curve.geometryFlags = bad;
            expectSingleProblem(c, "curve.geometry_flags", "not allowed");
        }
    }
}

TEST(ConfigValidate, CurveRulesDoNotApplyToDiscreteModes) {
    for (Mode m : {Mode::Mesh, Mode::Ray}) {
        CheckerConfig c          = validFor(m);
        c.curve.trajectoryPoints = 0;
        c.curve.controlPoints    = 1000;
        c.curve.refit            = RefitPolicy{RefitPolicy::Kind::Every, 1};
        c.curve.geometryFlags    = kGeometryFlagDisableAnyHit;
        EXPECT_NO_THROW(c.validate());
    }
}

TEST(ConfigValidate, SceneEdgesAnyValueForEveryMode) {
    for (Mode m : kAllModes) {
        for (EdgeSet s : {EdgeSet::Unique, EdgeSet::Loop}) {
            CheckerConfig c = validFor(m);
            c.sceneEdges    = s;
            EXPECT_NO_THROW(c.validate()); // ignored by Ray, no error
        }
    }
}

TEST(ConfigValidate, InvalidEnumValues) {
    CheckerConfig c = validFor(Mode::Mesh);
    c.mode          = static_cast<Mode>(9);
    expectSingleProblem(c, "mode", "invalid value 9");
    c                = validFor(Mode::Mesh);
    c.hostDispatch   = static_cast<HostDispatch>(5);
    expectSingleProblem(c, "host_dispatch");
    c            = validFor(Mode::Mesh);
    c.obbCulling = static_cast<Toggle>(3);
    expectSingleProblem(c, "obb_culling");
    c            = validFor(Mode::Mesh);
    c.sceneEdges = static_cast<EdgeSet>(2);
    expectSingleProblem(c, "scene_edges");
    c           = validFor(Mode::Mesh);
    c.verbosity = static_cast<LogLevel>(4);
    expectSingleProblem(c, "verbosity");
    c                = validFor(Mode::Mesh);
    c.optix.optLevel = static_cast<OptixOptions::OptLevel>(5);
    expectSingleProblem(c, "optix.opt_level");
    c                  = validFor(Mode::Mesh);
    c.optix.debugLevel = static_cast<OptixOptions::DebugLevel>(5);
    expectSingleProblem(c, "optix.debug_level");
    c                      = validFor(Mode::Mesh);
    c.optix.exceptionFlags = 1u << 3;
    expectSingleProblem(c, "optix.exceptions", "unknown exception bits 8");
    c                   = validFor(Mode::Mesh);
    c.mesh.linkGasBuild = static_cast<BuildPreference>(3);
    expectSingleProblem(c, "mesh.link_gas_build");
    c             = validFor(Mode::Mesh);
    c.ray.hitMode = static_cast<RayHitMode>(4);
    expectSingleProblem(c, "ray.hit_mode");
    c                 = validFor(Mode::Mesh);
    c.curve.endPoints = static_cast<CurveEndPoints>(3);
    expectSingleProblem(c, "curve.end_points");
    c                = validFor(Mode::Mesh);
    c.curve.gasBuild = static_cast<BuildPreference>(7);
    expectSingleProblem(c, "curve.gas_build");
    c                  = validFor(Mode::Mesh);
    c.curve.refit.kind = static_cast<RefitPolicy::Kind>(4);
    expectSingleProblem(c, "curve.refit");

    // every valid exception combination is accepted
    c                      = validFor(Mode::Mesh);
    c.optix.exceptionFlags = OptixOptions::kExceptionStackOverflow | OptixOptions::kExceptionTraceDepth
                           | OptixOptions::kExceptionUser;
    EXPECT_NO_THROW(c.validate());
}

TEST(ConfigValidate, ReportsEveryProblemAtOnce) {
    CheckerConfig c          = validFor(Mode::CubicCurve);
    c.numStreams             = 0;
    c.optix.maxTraceDepth    = 0;
    c.curve.controlPoints    = 2;
    c.curve.splineWeightsDir.clear();
    c.obbCulling = Toggle::On;
    const ConfigError e = validationError(c);
    EXPECT_EQ(e.field(), "streams"); // the first offending path
    const auto l = lines(e.what());
    ASSERT_EQ(l.size(), 5u) << e.what();
    const std::vector<std::string> expectedFields{
        "streams", "optix.max_trace_depth", "curve.control_points", "curve.spline_weights_dir", "obb_culling"};
    for (std::size_t i = 0; i < l.size(); ++i) {
        EXPECT_EQ(l[i].rfind(expectedFields[i] + ": ", 0), 0u) << l[i];
    }
}

TEST(ConfigValidate, MessagesNameTheValues) {
    CheckerConfig c       = validFor(Mode::QuadraticCurve);
    c.curve.controlPoints = 20;
    const ConfigError e   = validationError(c);
    // 20 * 16 = 320 <= 512: only the range rule fires
    EXPECT_STREQ(e.what(),
        "curve.control_points: 20 must be > 2 (the curve degree of quadratic_curve) and < curve.trajectory_points = 16");
}

// ---- resolved() --------------------------------------------------------------------------------------------------

TEST(ConfigResolved, AutoValuesPerMode) {
    struct Expect {
        Mode mode;
        Toggle obb;
        EdgeSet edges;
        std::int32_t autoRefit;
        GeometryFlags geometry;
        RefitPolicy::Kind refit;
    };
    const Expect table[] = {
        {Mode::Mesh, Toggle::On, EdgeSet::Unique, 1, kGeometryFlagNone, RefitPolicy::Kind::Auto},
        {Mode::Ray, Toggle::On, EdgeSet::Unique, 1, kGeometryFlagNone, RefitPolicy::Kind::Auto},
        {Mode::LinearCurve, Toggle::Off, EdgeSet::Loop, 2, kGeometryFlagNone, RefitPolicy::Kind::Auto},
        {Mode::QuadraticCurve, Toggle::Off, EdgeSet::Loop, 1, kGeometryFlagRequireSingleAnyHitCall,
            RefitPolicy::Kind::Auto},
        {Mode::CubicCurve, Toggle::Off, EdgeSet::Loop, 1, kGeometryFlagNone, RefitPolicy::Kind::Auto},
    };
    for (const Expect& x : table) {
        const CheckerConfig r = validFor(x.mode).resolved();
        EXPECT_EQ(r.mode, x.mode);
        EXPECT_EQ(r.obbCulling, x.obb) << static_cast<int>(x.mode);
        ASSERT_TRUE(r.sceneEdges.has_value());
        EXPECT_EQ(*r.sceneEdges, x.edges) << static_cast<int>(x.mode);
        EXPECT_EQ(r.curve.autoRefitMaxBatches, x.autoRefit) << static_cast<int>(x.mode);
        ASSERT_TRUE(r.curve.geometryFlags.has_value());
        EXPECT_EQ(*r.curve.geometryFlags, x.geometry) << static_cast<int>(x.mode);
        EXPECT_EQ(r.curve.refit.kind, x.refit);
        EXPECT_EQ(r.obbCullingEnabled(), x.obb == Toggle::On);
        // resolving twice changes nothing
        const CheckerConfig rr = r.resolved();
        EXPECT_EQ(rr.obbCulling, r.obbCulling);
        EXPECT_EQ(rr.sceneEdges, r.sceneEdges);
        EXPECT_EQ(rr.curve.autoRefitMaxBatches, r.curve.autoRefitMaxBatches);
        EXPECT_EQ(rr.curve.geometryFlags, r.curve.geometryFlags);
        EXPECT_EQ(rr.curve.refit.kind, r.curve.refit.kind);
    }
}

TEST(ConfigResolved, ExplicitValuesAreKept) {
    for (Mode m : kAllModes) {
        CheckerConfig c             = validFor(m);
        c.obbCulling                = Toggle::Off;
        c.sceneEdges                = isContinuous(m) ? EdgeSet::Unique : EdgeSet::Loop;
        c.curve.autoRefitMaxBatches = 7;
        c.curve.geometryFlags       = m == Mode::QuadraticCurve ? kGeometryFlagNone : kGeometryFlagRequireSingleAnyHitCall;
        c.curve.refit               = RefitPolicy{RefitPolicy::Kind::Every, 3};
        const CheckerConfig r       = c.resolved();
        EXPECT_EQ(r.obbCulling, Toggle::Off);
        EXPECT_EQ(r.sceneEdges, c.sceneEdges);
        EXPECT_EQ(r.curve.autoRefitMaxBatches, 7);
        EXPECT_EQ(r.curve.geometryFlags, c.curve.geometryFlags);
        EXPECT_EQ(r.curve.refit.kind, RefitPolicy::Kind::Every);
        EXPECT_EQ(r.curve.refit.every, 3u);
        // everything else is copied unchanged
        EXPECT_EQ(r.numStreams, c.numStreams);
        EXPECT_EQ(r.maxBatchSize, c.maxBatchSize);
        EXPECT_EQ(r.curve.splineWeightsDir, c.curve.splineWeightsDir);
        EXPECT_EQ(r.mesh.parkTransform.m, c.mesh.parkTransform.m);
    }
}

TEST(ConfigResolved, ObbCullingOnForcesNoRefitForCurves) {
    for (Mode m : {Mode::LinearCurve, Mode::QuadraticCurve}) {
        for (auto kind : {RefitPolicy::Kind::Auto, RefitPolicy::Kind::Always, RefitPolicy::Kind::Every}) {
            CheckerConfig c = validFor(m);
            c.obbCulling    = Toggle::On;
            c.curve.refit   = RefitPolicy{kind, kind == RefitPolicy::Kind::Every ? 4u : 0u};
            const CheckerConfig r = c.resolved();
            EXPECT_EQ(r.curve.refit.kind, RefitPolicy::Kind::Never);
            EXPECT_EQ(r.obbCulling, Toggle::On);
        }
    }
    // discrete modes keep their (unused) curve refit policy
    CheckerConfig c = validFor(Mode::Mesh);
    c.obbCulling    = Toggle::On;
    c.curve.refit   = RefitPolicy{RefitPolicy::Kind::Always, 0};
    EXPECT_EQ(c.resolved().curve.refit.kind, RefitPolicy::Kind::Always);
}

TEST(ConfigResolved, ValidatesFirst) {
    CheckerConfig c = validFor(Mode::Mesh);
    c.numStreams    = 0;
    EXPECT_THROW((void)c.resolved(), ConfigError);
}

TEST(ConfigAccessors, ItemLayout) {
    CheckerConfig c          = validFor(Mode::QuadraticCurve);
    c.curve.trajectoryPoints = 32;
    EXPECT_EQ(c.itemLayout(7).pointsPerItem, 32u);
    EXPECT_EQ(c.itemLayout(7).floatsPerItem(), 7u * 32u);
    c.mode = Mode::Ray;
    EXPECT_EQ(c.itemLayout(7).pointsPerItem, 1u);
}
