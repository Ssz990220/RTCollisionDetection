// rtcd_json: the JSON module (include/rtcd/json_config.h, schemas in docs/design/interfaces.md section 7).
// Covers: every file under configs/ (values ported from the old stack), defaults, unknown keys, wrong types and
// enum strings, syntax errors, path resolution, setups (inline and referenced), overrides, dump round trips.
// new_unit (no GPU). nlohmann-json is used here only to inspect dumps.
#include <rtcd/errors.h>
#include <rtcd/json_config.h>

#include <support/reference.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace rtcd;
namespace fs = std::filesystem;

namespace {

    fs::path norm(const fs::path& p) {
        fs::path n = fs::absolute(p).lexically_normal();
        n.make_preferred();
        return n;
    }

    const fs::path& repo() {
        static const fs::path root = norm(RTCD_TEST_SOURCE_DIR);
        return root;
    }
    fs::path configs() { return repo() / "configs"; }

    std::string slurp(const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

    // Unique scratch directory, removed at the end of the test.
    class TempDir {
    public:
        TempDir() {
            std::random_device rd;
            dir_ = norm(fs::temp_directory_path() / ("rtcd_json_test_" + std::to_string(rd()) + std::to_string(rd())));
            fs::create_directories(dir_);
        }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(dir_, ec);
        }
        const fs::path& path() const { return dir_; }
        fs::path write(const fs::path& rel, const std::string& text) const {
            const fs::path p = dir_ / rel;
            fs::create_directories(p.parent_path());
            std::ofstream(p, std::ios::binary) << text;
            return p;
        }

    private:
        fs::path dir_;
    };

    template <typename E>
    E thrown(const std::function<void()>& f) {
        try {
            f();
        } catch (const E& e) {
            return e;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "unexpected exception type: " << e.what();
            throw;
        }
        ADD_FAILURE() << "no exception thrown";
        throw std::runtime_error("no exception");
    }

    ConfigError checkerError(const std::string& json) {
        return thrown<ConfigError>([&] { (void)parseCheckerConfig(json, repo()); });
    }

    bool sameBits(const Transform& a, const Transform& b) {
        return std::memcmp(a.m.data(), b.m.data(), sizeof(a.m)) == 0;
    }

    void expectSameConfig(const CheckerConfig& a, const CheckerConfig& b) {
        EXPECT_EQ(a.mode, b.mode);
        EXPECT_EQ(a.numStreams, b.numStreams);
        EXPECT_EQ(a.hostDispatch, b.hostDispatch);
        EXPECT_EQ(a.maxBatchSize, b.maxBatchSize);
        EXPECT_EQ(a.maxItemsPerCheck, b.maxItemsPerCheck);
        EXPECT_EQ(a.obbCulling, b.obbCulling);
        EXPECT_EQ(a.collapseRootLinkObb, b.collapseRootLinkObb);
        EXPECT_EQ(a.sceneEdges, b.sceneEdges);
        EXPECT_EQ(a.verbosity, b.verbosity);
        EXPECT_EQ(a.graphPrepare, b.graphPrepare);
        EXPECT_EQ(a.optix.optLevel, b.optix.optLevel);
        EXPECT_EQ(a.optix.debugLevel, b.optix.debugLevel);
        EXPECT_EQ(a.optix.maxRegisterCount, b.optix.maxRegisterCount);
        EXPECT_EQ(a.optix.exceptionFlags, b.optix.exceptionFlags);
        EXPECT_EQ(a.optix.maxTraceDepth, b.optix.maxTraceDepth);
        EXPECT_EQ(a.optix.printPipelineLog, b.optix.printPipelineLog);
        EXPECT_EQ(a.mesh.linkGasBuild, b.mesh.linkGasBuild);
        EXPECT_EQ(a.mesh.compactionMinPoses, b.mesh.compactionMinPoses);
        EXPECT_TRUE(sameBits(a.mesh.parkTransform, b.mesh.parkTransform));
        EXPECT_EQ(a.ray.launchWidthCap, b.ray.launchWidthCap);
        EXPECT_EQ(a.ray.skipHitItems, b.ray.skipHitItems);
        EXPECT_EQ(a.ray.hitMode, b.ray.hitMode);
        EXPECT_EQ(a.curve.trajectoryPoints, b.curve.trajectoryPoints);
        EXPECT_EQ(a.curve.controlPoints, b.curve.controlPoints);
        EXPECT_EQ(a.curve.endPoints, b.curve.endPoints);
        EXPECT_EQ(a.curve.gasBuild, b.curve.gasBuild);
        EXPECT_EQ(a.curve.refit.kind, b.curve.refit.kind);
        EXPECT_EQ(a.curve.refit.every, b.curve.refit.every);
        EXPECT_EQ(a.curve.autoRefitMaxBatches, b.curve.autoRefitMaxBatches);
        EXPECT_EQ(a.curve.geometryFlags, b.curve.geometryFlags);
        EXPECT_EQ(a.curve.splineWeightsDir, b.curve.splineWeightsDir);
    }

    void expectSameScene(const SceneDesc& a, const SceneDesc& b) {
        EXPECT_EQ(a.name, b.name);
        ASSERT_EQ(a.obstacles.size(), b.obstacles.size());
        for (std::size_t i = 0; i < a.obstacles.size(); ++i) {
            EXPECT_EQ(a.obstacles[i].name, b.obstacles[i].name);
            EXPECT_EQ(a.obstacles[i].mesh, b.obstacles[i].mesh);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(a.obstacles[i].scale), std::bit_cast<std::uint32_t>(b.obstacles[i].scale));
            EXPECT_TRUE(sameBits(a.obstacles[i].pose, b.obstacles[i].pose)) << "obstacle " << i;
        }
    }

    void expectSameSource(const GeometrySource& a, const GeometrySource& b) {
        EXPECT_EQ(a.dir, b.dir);
        EXPECT_EQ(a.format, b.format);
        EXPECT_EQ(a.files, b.files);
    }

    void expectSameRobot(const RobotDesc& a, const RobotDesc& b) {
        EXPECT_EQ(a.description, b.description);
        expectSameSource(a.mesh, b.mesh);
        EXPECT_EQ(a.meshSource, b.meshSource);
        ASSERT_EQ(a.spheres.has_value(), b.spheres.has_value());
        if (a.spheres) {
            expectSameSource(*a.spheres, *b.spheres);
        }
        EXPECT_TRUE(sameBits(a.basePose, b.basePose));
    }

    CheckerConfig roundTrip(const CheckerConfig& c) { return parseCheckerConfig(dumpCheckerConfig(c), "/"); }

} // namespace

// =====================================================================================================================
// configs/ files
// =====================================================================================================================

TEST(JsonConfigs, CheckerIas) {
    const CheckerConfig c = loadCheckerConfig(configs() / "checker_ias.json");
    EXPECT_EQ(c.mode, Mode::Mesh);
    EXPECT_EQ(c.numStreams, 4u);
    EXPECT_EQ(c.hostDispatch, HostDispatch::ThreadPerStream);
    EXPECT_EQ(c.maxBatchSize, 128u);
    EXPECT_EQ(c.maxItemsPerCheck, 4096u);
    EXPECT_EQ(c.mesh.linkGasBuild, BuildPreference::FastTraceCompact); // old BuildType::COMPACT
    EXPECT_EQ(c.mesh.compactionMinPoses, 128u);
    EXPECT_TRUE(c.collapseRootLinkObb);
    const CheckerConfig r = c.resolved();
    EXPECT_EQ(r.obbCulling, Toggle::On); // old CFG(..., useOBB = true)
    EXPECT_EQ(r.sceneEdges, EdgeSet::Unique); // old buildSharedScene(false)
}

TEST(JsonConfigs, CheckerRay) {
    const CheckerConfig c = loadCheckerConfig(configs() / "checker_ray.json");
    EXPECT_EQ(c.mode, Mode::Ray);
    EXPECT_EQ(c.numStreams, 4u);
    EXPECT_EQ(c.maxBatchSize, 128u);
    EXPECT_EQ(c.maxItemsPerCheck, 4096u);
    EXPECT_EQ(c.ray.launchWidthCap, 262144u);
    EXPECT_TRUE(c.ray.skipHitItems);
    EXPECT_EQ(c.ray.hitMode, RayHitMode::ClosestHit);
    EXPECT_EQ(c.resolved().obbCulling, Toggle::On);
}

TEST(JsonConfigs, CheckerLinear) {
    const CheckerConfig c = loadCheckerConfig(configs() / "checker_linear.json");
    EXPECT_EQ(c.mode, Mode::LinearCurve);
    EXPECT_EQ(c.maxBatchSize, 128u);
    EXPECT_EQ(c.maxItemsPerCheck, 4096u);
    EXPECT_EQ(c.curve.trajectoryPoints, 16u);
    EXPECT_EQ(c.curve.gasBuild, BuildPreference::FastBuild);
    const CheckerConfig r = c.resolved();
    EXPECT_EQ(r.obbCulling, Toggle::Off);
    EXPECT_EQ(r.sceneEdges, EdgeSet::Loop);
    EXPECT_EQ(r.curve.autoRefitMaxBatches, 2);
    EXPECT_EQ(r.curve.geometryFlags, GeometryFlags{kGeometryFlagNone});
}

TEST(JsonConfigs, CheckerQuadAndItsSplineWeights) {
    const CheckerConfig c = loadCheckerConfig(configs() / "checker_quad.json");
    EXPECT_EQ(c.mode, Mode::QuadraticCurve);
    EXPECT_EQ(c.curve.trajectoryPoints, 16u);
    EXPECT_EQ(c.curve.controlPoints, 8u);
    EXPECT_EQ(c.curve.endPoints, CurveEndPoints::Normal);
    EXPECT_EQ(c.curve.gasBuild, BuildPreference::FastBuild);
    EXPECT_EQ(c.curve.refit.kind, RefitPolicy::Kind::Auto);
    EXPECT_FALSE(c.curve.geometryFlags.has_value());
    EXPECT_EQ(c.curve.splineWeightsDir, configs() / "spline_weights");
    const CheckerConfig r = c.resolved();
    EXPECT_EQ(r.curve.geometryFlags, GeometryFlags{kGeometryFlagRequireSingleAnyHitCall});
    EXPECT_EQ(r.curve.autoRefitMaxBatches, 1);
    EXPECT_EQ(r.sceneEdges, EdgeSet::Loop);

    // The weight file the curve backend will read: <dir>/quad/normal/<control>_<trajectory>.txt with
    // control * trajectory floats.
    const fs::path weights = c.curve.splineWeightsDir / "quad" / "normal" / "8_16.txt";
    ASSERT_TRUE(fs::exists(weights)) << weights;
    std::ifstream in(weights);
    std::size_t n = 0;
    float v       = 0;
    while (in >> v) {
        ++n;
    }
    EXPECT_EQ(n, 8u * 16u);
}

namespace {
    // text content with CR removed (git may check text files out with CRLF line ends)
    std::string slurpLf(const fs::path& p) {
        std::string s = slurp(p);
        std::erase(s, '\r');
        return s;
    }
} // namespace

// configs/spline_weights equals the stored manifest (tests/data/reference/spline_weights.txt: every file of the old
// stack's RTCD/data/weights, relative path, bytes and FNV-1a of the content with CR removed).
TEST(JsonConfigs, SplineWeightsMatchTheStoredManifest) {
    const auto ref        = rtcd::support::ReferenceFile::load(rtcd::support::referenceDir() / "spline_weights.txt");
    const fs::path newDir = configs() / "spline_weights";
    std::size_t files     = 0;
    for (const auto& e : fs::recursive_directory_iterator(newDir)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const std::string rel = fs::relative(e.path(), newDir).generic_string();
        ASSERT_TRUE(ref.has(rel)) << rel << " is not in the manifest";
        const std::string text = slurpLf(e.path());
        EXPECT_EQ(text.size(), ref.integer(rel, 0)) << rel;
        EXPECT_EQ(rtcd::support::fnv1a(text.data(), text.size()), ref.hash(rel, 1)) << rel;
        ++files;
    }
    EXPECT_EQ(files, ref.keys().size()) << "every file of the manifest exists";
}

namespace {
    struct ExpectedObstacle {
        const char* name;
        const char* mesh; // relative to the repository root
        float x, y, z;
    };

    // Values of the old RTCD/CollisionScenes/Scene/*.h (Eigen::Affine3f::Identity().translate(Vector3f(x, y, z)),
    // scale 1.0).
    void expectScene(const fs::path& file, const char* name, const std::vector<ExpectedObstacle>& expected) {
        const SceneDesc d = loadSceneDesc(file);
        EXPECT_EQ(d.name, name);
        ASSERT_EQ(d.obstacles.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const ObstacleDesc& o = d.obstacles[i];
            EXPECT_EQ(o.name, expected[i].name);
            EXPECT_EQ(o.mesh, norm(repo() / expected[i].mesh));
            EXPECT_TRUE(fs::exists(o.mesh)) << o.mesh;
            EXPECT_EQ(o.scale, 1.0f);
            const Transform t{{1, 0, 0, expected[i].x, 0, 1, 0, expected[i].y, 0, 0, 1, expected[i].z}};
            EXPECT_TRUE(sameBits(o.pose, t)) << expected[i].name;
        }
    }
} // namespace

TEST(JsonConfigs, SceneShelf) {
    expectScene(configs() / "scenes" / "shelf.json", "shelf",
        {{"shelf", "models/curobo/shelves.obj", 0.45f, 0, 0.4f}, {"bin1", "models/curobo/bin.obj", 0, -0.6f, 0},
            {"bin2", "models/curobo/bin.obj", 0, 0.6f, 0}, {"bob", "models/curobo/Bob.obj", -0.4f, 0, 0},
            {"cow", "models/curobo/Cow.obj", -0.4f, 0, 0}, {"fish", "models/curobo/Fish.obj", -0.4f, 0, 0},
            {"sheep", "models/curobo/Sheep.obj", -0.4f, 0, 0},
            {"snakeboard", "models/curobo/Snakeboard.obj", -0.4f, 0, 0}});
}

TEST(JsonConfigs, SceneShelfDense) {
    expectScene(configs() / "scenes" / "shelf_dense.json", "shelf_dense",
        {{"shelf", "models/curobo/shelves.obj", 0.45f, 0, 0.4f}, {"bin1", "models/curobo/bin_dense.obj", 0, -0.6f, 0},
            {"bin2", "models/curobo/bin_dense.obj", 0, 0.6f, 0}, {"bob", "models/curobo/Bob.obj", -0.4f, 0, 0},
            {"face1", "models/curobo/Face1.obj", -0.4f, 0, 0}, {"face2", "models/curobo/Face2.obj", -0.4f, 0, 0},
            {"face3", "models/curobo/Face3.obj", -0.4f, 0, 0}, {"cow", "models/curobo/Cow.obj", -0.4f, 0, 0},
            {"fish", "models/curobo/Fish.obj", -0.4f, 0, 0}, {"sheep", "models/curobo/Sheep.obj", -0.4f, 0, 0},
            {"snakeboard", "models/curobo/Snakeboard.obj", -0.4f, 0, 0}});
}

TEST(JsonConfigs, SceneShelfSimple) {
    expectScene(configs() / "scenes" / "shelf_simple.json", "shelf_simple",
        {{"shelf", "models/curobo/shelves.obj", 0.45f, 0, 0.4f}, {"bin1", "models/curobo/bin.obj", 0, -0.6f, 0},
            {"bin2", "models/curobo/bin.obj", 0, 0.6f, 0}});
}

TEST(JsonConfigs, RobotPanda) {
    const RobotDesc d = loadRobotDesc(configs() / "robots" / "panda.json");
    const fs::path fr = repo() / "models" / "franka_description";
    EXPECT_EQ(d.description, norm(fr / "panda.urdf"));
    EXPECT_TRUE(fs::exists(d.description));
    EXPECT_EQ(d.mesh.dir, norm(fr / "meshes" / "Simplified" / "watertight"));
    EXPECT_TRUE(fs::is_directory(d.mesh.dir));
    EXPECT_EQ(d.mesh.format, "obj");
    EXPECT_TRUE(d.mesh.files.empty());
    EXPECT_EQ(d.meshSource, MeshSource::Visual);
    ASSERT_TRUE(d.spheres.has_value());
    EXPECT_EQ(d.spheres->dir, norm(fr / "spheres"));
    EXPECT_TRUE(fs::exists(d.spheres->dir / "panda_link0.txt"));
    EXPECT_EQ(d.spheres->format, "xyzr");
    EXPECT_TRUE(sameBits(d.basePose, Transform::identity()));
}

TEST(JsonConfigs, RegressionSetups) {
    for (const char* mode : {"ias", "ray", "linear", "quad"}) {
        const fs::path file = configs() / "setups" / (std::string("regression_") + mode + ".json");
        const rtcd::Setup s       = loadSetup(file);
        EXPECT_EQ(s.sourceFile, norm(file));
        expectSameConfig(s.checker, loadCheckerConfig(configs() / (std::string("checker_") + mode + ".json")));
        expectSameScene(s.scene, loadSceneDesc(configs() / "scenes" / "shelf.json"));
        expectSameRobot(s.robot, loadRobotDesc(configs() / "robots" / "panda.json"));
    }
}

TEST(JsonConfigs, EveryFileLoadsAndRoundTrips) {
    for (const auto& e : fs::recursive_directory_iterator(configs())) {
        if (e.path().extension() != ".json") {
            continue;
        }
        const std::string stem = e.path().stem().string();
        const std::string dir  = e.path().parent_path().filename().string();
        SCOPED_TRACE(e.path().string());
        if (dir == "scenes") {
            const SceneDesc d = loadSceneDesc(e.path());
            expectSameScene(parseSceneDesc(dumpSceneDesc(d), "/"), d);
        } else if (dir == "robots") {
            const RobotDesc d = loadRobotDesc(e.path());
            expectSameRobot(parseRobotDesc(dumpRobotDesc(d), "/"), d);
        } else if (dir == "setups") {
            EXPECT_NO_THROW((void)loadSetup(e.path()));
        } else {
            const CheckerConfig c = loadCheckerConfig(e.path());
            expectSameConfig(roundTrip(c), c);
        }
    }
}

TEST(JsonConfigs, DocumentationExampleEqualsCheckerQuad) {
    // docs/design/interfaces.md 7.1 shows configs/checker_quad.json verbatim.
    const std::string doc = slurp(repo() / "docs" / "design" / "interfaces.md");
    const std::size_t at  = doc.find("Example (`configs/checker_quad.json`");
    ASSERT_NE(at, std::string::npos);
    const std::size_t open  = doc.find("```json", at) + 7;
    const std::size_t close = doc.find("```", open);
    const CheckerConfig fromDoc = parseCheckerConfig(doc.substr(open, close - open), configs(), "interfaces.md");
    expectSameConfig(fromDoc, loadCheckerConfig(configs() / "checker_quad.json"));
}

// =====================================================================================================================
// checker config: defaults, types, enums, unknown keys, syntax
// =====================================================================================================================

TEST(JsonChecker, EmptyObjectGivesTheStructDefaults) {
    expectSameConfig(parseCheckerConfig("{}", repo()), CheckerConfig{});
}

TEST(JsonChecker, EveryKeyIsRead) {
    const std::string json = R"({
        "mode": "cubic_curve", "streams": 3, "host_dispatch": "single_thread", "max_batch_size": 64,
        "max_items_per_check": 1000, "obb_culling": false, "collapse_root_link_obb": false, "scene_edges": "unique",
        "verbosity": "debug", "graph_prepare": false,
        "optix": { "opt_level": 2, "debug_level": "moderate", "max_register_count": 0,
                   "exceptions": ["user", "stack_overflow"], "max_trace_depth": 1, "print_pipeline_log": true },
        "mesh": { "link_gas_build": "fast_trace", "compaction_min_poses": 0,
                  "park_transform": [1, 0, 0, 0.5, 0, 1, 0, -2, 0, 0, 1, 1e-3] },
        "ray": { "launch_width_cap": 0, "skip_hit_items": false, "hit_mode": "first_hit_miss" },
        "curve": { "trajectory_points": 32, "control_points": 16, "end_points": "repeated", "gas_build": "fast_trace_compact",
                   "refit": {"every": 5}, "auto_refit_max_batches": 3, "geometry_flags": "single_anyhit",
                   "spline_weights_dir": "w" } })";
    const CheckerConfig c = parseCheckerConfig(json, "/base");
    EXPECT_EQ(c.mode, Mode::CubicCurve);
    EXPECT_EQ(c.numStreams, 3u);
    EXPECT_EQ(c.hostDispatch, HostDispatch::SingleThread);
    EXPECT_EQ(c.maxBatchSize, 64u);
    EXPECT_EQ(c.maxItemsPerCheck, 1000u);
    EXPECT_EQ(c.obbCulling, Toggle::Off);
    EXPECT_FALSE(c.collapseRootLinkObb);
    EXPECT_EQ(c.sceneEdges, EdgeSet::Unique);
    EXPECT_EQ(c.verbosity, LogLevel::Debug);
    EXPECT_FALSE(c.graphPrepare);
    EXPECT_EQ(c.optix.optLevel, OptixOptions::OptLevel::O2);
    EXPECT_EQ(c.optix.debugLevel, OptixOptions::DebugLevel::Moderate);
    EXPECT_EQ(c.optix.maxRegisterCount, 0u);
    EXPECT_EQ(c.optix.exceptionFlags, OptixOptions::kExceptionUser | OptixOptions::kExceptionStackOverflow);
    EXPECT_EQ(c.optix.maxTraceDepth, 1u);
    EXPECT_TRUE(c.optix.printPipelineLog);
    EXPECT_EQ(c.mesh.linkGasBuild, BuildPreference::FastTrace);
    EXPECT_EQ(c.mesh.compactionMinPoses, 0u);
    EXPECT_TRUE(sameBits(c.mesh.parkTransform, Transform{{1, 0, 0, 0.5f, 0, 1, 0, -2, 0, 0, 1, 1e-3f}}));
    EXPECT_EQ(c.ray.launchWidthCap, 0u);
    EXPECT_FALSE(c.ray.skipHitItems);
    EXPECT_EQ(c.ray.hitMode, RayHitMode::FirstHitMiss);
    EXPECT_EQ(c.curve.trajectoryPoints, 32u);
    EXPECT_EQ(c.curve.controlPoints, 16u);
    EXPECT_EQ(c.curve.endPoints, CurveEndPoints::Repeated);
    EXPECT_EQ(c.curve.gasBuild, BuildPreference::FastTraceCompact);
    EXPECT_EQ(c.curve.refit.kind, RefitPolicy::Kind::Every);
    EXPECT_EQ(c.curve.refit.every, 5u);
    EXPECT_EQ(c.curve.autoRefitMaxBatches, 3);
    EXPECT_EQ(c.curve.geometryFlags, GeometryFlags{kGeometryFlagRequireSingleAnyHitCall});
    EXPECT_EQ(c.curve.splineWeightsDir, norm("/base/w"));
}

TEST(JsonChecker, EnumSpellings) {
    const std::pair<const char*, Mode> modes[] = {{"mesh", Mode::Mesh}, {"ray", Mode::Ray},
        {"linear_curve", Mode::LinearCurve}, {"quadratic_curve", Mode::QuadraticCurve}, {"cubic_curve", Mode::CubicCurve}};
    for (const auto& [name, m] : modes) {
        const CheckerConfig c = parseCheckerConfig(
            std::string(R"({"mode": ")") + name + R"(", "curve": {"spline_weights_dir": "w"}})", repo());
        EXPECT_EQ(c.mode, m) << name;
    }
    const std::pair<const char*, RayHitMode> hits[] = {{"closest_hit", RayHitMode::ClosestHit},
        {"first_hit_ch", RayHitMode::FirstHitCH}, {"first_hit_miss", RayHitMode::FirstHitMiss},
        {"traverse", RayHitMode::FirstHitTraverse}};
    for (const auto& [name, h] : hits) {
        EXPECT_EQ(parseCheckerConfig(std::string(R"({"ray": {"hit_mode": ")") + name + "\"}}", repo()).ray.hitMode, h);
    }
    for (const auto& [text, t] : {std::pair<const char*, Toggle>{"\"auto\"", Toggle::Auto}, {"true", Toggle::On},
             {"false", Toggle::Off}}) {
        EXPECT_EQ(parseCheckerConfig(std::string(R"({"obb_culling": )") + text + "}", repo()).obbCulling, t);
    }
    EXPECT_FALSE(parseCheckerConfig(R"({"scene_edges": "auto"})", repo()).sceneEdges.has_value());
    EXPECT_EQ(parseCheckerConfig(R"({"scene_edges": "loop"})", repo()).sceneEdges, EdgeSet::Loop);
    EXPECT_FALSE(parseCheckerConfig(R"({"curve": {"geometry_flags": "auto"}})", repo()).curve.geometryFlags.has_value());
    EXPECT_EQ(parseCheckerConfig(R"({"curve": {"geometry_flags": "none"}})", repo()).curve.geometryFlags,
        GeometryFlags{kGeometryFlagNone});
    EXPECT_EQ(parseCheckerConfig(R"({"optix": {"opt_level": "default"}})", repo()).optix.optLevel,
        OptixOptions::OptLevel::Default);
    EXPECT_EQ(parseCheckerConfig(R"({"optix": {"opt_level": 0}})", repo()).optix.optLevel, OptixOptions::OptLevel::O0);
    EXPECT_EQ(parseCheckerConfig(R"({"optix": {"opt_level": 3}})", repo()).optix.optLevel, OptixOptions::OptLevel::O3);
    EXPECT_EQ(parseCheckerConfig(R"({"optix": {"exceptions": ["none"]}})", repo()).optix.exceptionFlags, 0u);
    EXPECT_EQ(parseCheckerConfig(R"({"optix": {"exceptions": ["trace_depth"]}})", repo()).optix.exceptionFlags,
        std::uint32_t{OptixOptions::kExceptionTraceDepth});
    EXPECT_EQ(parseCheckerConfig(R"({"curve": {"refit": "always"}})", repo()).curve.refit.kind, RefitPolicy::Kind::Always);
    EXPECT_EQ(parseCheckerConfig(R"({"curve": {"refit": "never"}})", repo()).curve.refit.kind, RefitPolicy::Kind::Never);
    EXPECT_EQ(parseCheckerConfig(R"({"curve": {"auto_refit_max_batches": "auto"}})", repo()).curve.autoRefitMaxBatches, -1);
    EXPECT_EQ(parseCheckerConfig(R"({"curve": {"auto_refit_max_batches": 0}})", repo()).curve.autoRefitMaxBatches, 0);
    EXPECT_TRUE(parseCheckerConfig(R"({"curve": {"spline_weights_dir": null}})", repo()).curve.splineWeightsDir.empty());
}

TEST(JsonChecker, UnknownKeysAreErrors) {
    struct Case {
        const char* json;
        const char* field;
    };
    for (const Case& c : {Case{R"({"stream": 4})", "stream"}, Case{R"({"optix": {"maxreg": 4}})", "optix.maxreg"},
             Case{R"({"mesh": {"compact_min_poses": 4}})", "mesh.compact_min_poses"},
             Case{R"({"ray": {"hitmode": "traverse"}})", "ray.hitmode"},
             Case{R"({"curve": {"refit": {"every": 3, "x": 1}}})", "curve.refit.x"},
             Case{R"({"curve": {"degree": 2}})", "curve.degree"}}) {
        const ConfigError e = checkerError(c.json);
        EXPECT_EQ(e.field(), c.field) << c.json;
        EXPECT_EQ(std::string(e.what()).rfind(std::string("<string>: ") + c.field + ": unknown key", 0), 0u) << e.what();
    }
}

TEST(JsonChecker, WrongTypesAndValues) {
    struct Case {
        const char* json;
        const char* field;
        const char* fragment;
    };
    const Case cases[] = {
        {R"({"mode": "meshes"})", "mode", "is not one of \"mesh\""},
        {R"({"mode": 0})", "mode", "is not one of"},
        {R"({"streams": "4"})", "streams", "non-negative integer"},
        {R"({"streams": -1})", "streams", ">= 0"},
        {R"({"streams": 2.5})", "streams", "non-negative integer"},
        {R"({"streams": 4.0})", "streams", "non-negative integer"},
        {R"({"max_batch_size": 4294967296})", "max_batch_size", "too large"},
        {R"({"host_dispatch": "threads"})", "host_dispatch", "is not one of"},
        {R"({"obb_culling": "on"})", "obb_culling", "is not one of \"auto\", true, false"},
        {R"({"obb_culling": 1})", "obb_culling", "is not one of"},
        {R"({"collapse_root_link_obb": 1})", "collapse_root_link_obb", "true or false"},
        {R"({"graph_prepare": "yes"})", "graph_prepare", "true or false"},
        {R"({"scene_edges": "all"})", "scene_edges", "is not one of \"auto\", \"unique\", \"loop\""},
        {R"({"verbosity": "trace"})", "verbosity", "is not one of"},
        {R"({"optix": []})", "optix", "must be an object"},
        {R"({"optix": {"opt_level": 4}})", "optix.opt_level", "\"default\", 0, 1, 2, 3"},
        {R"({"optix": {"opt_level": "O3"}})", "optix.opt_level", "is not one of"},
        {R"({"optix": {"debug_level": 1}})", "optix.debug_level", "is not one of"},
        {R"({"optix": {"exceptions": "user"}})", "optix.exceptions", "list of exception names"},
        {R"({"optix": {"exceptions": ["debug"]}})", "optix.exceptions[0]", "is not one of"},
        {R"({"optix": {"exceptions": ["none", "user"]}})", "optix.exceptions", "cannot be combined"},
        {R"({"mesh": {"link_gas_build": "compact"}})", "mesh.link_gas_build", "is not one of"},
        {R"({"mesh": {"park_transform": [1, 0, 0]}})", "mesh.park_transform", "12 numbers"},
        {R"({"mesh": {"park_transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, "z"]}})", "mesh.park_transform[11]", "number"},
        {R"({"mesh": {"park_transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1e300]}})", "mesh.park_transform[11]",
            "float range"},
        {R"({"ray": {"skip_hit_items": "yes"}})", "ray.skip_hit_items", "true or false"},
        {R"({"curve": {"end_points": "phantoms"}})", "curve.end_points", "is not one of"},
        {R"({"curve": {"refit": "every"}})", "curve.refit", "is not one of"},
        {R"({"curve": {"refit": 3}})", "curve.refit", "{\"every\": K}"},
        {R"({"curve": {"refit": {}}})", "curve.refit.every", "required key is missing"},
        {R"({"curve": {"auto_refit_max_batches": -1}})", "curve.auto_refit_max_batches", ">= 0"},
        {R"({"curve": {"auto_refit_max_batches": "two"}})", "curve.auto_refit_max_batches", "\"auto\" or an integer"},
        {R"({"curve": {"geometry_flags": "disable_anyhit"}})", "curve.geometry_flags", "is not one of \"auto\""},
        {R"({"curve": {"spline_weights_dir": ""}})", "curve.spline_weights_dir", "non-empty path"},
        {R"({"curve": {"spline_weights_dir": 3}})", "curve.spline_weights_dir", "must be a string"},
        {R"([])", "", "must be an object"},
    };
    for (const Case& c : cases) {
        const ConfigError e = checkerError(c.json);
        EXPECT_EQ(e.field(), c.field) << c.json;
        EXPECT_NE(std::string(e.what()).find(c.fragment), std::string::npos) << c.json << " -> " << e.what();
        EXPECT_EQ(std::string(e.what()).rfind("<string>: ", 0), 0u) << e.what();
    }
}

TEST(JsonChecker, LoadedConfigsAreValidatedButNotResolved) {
    const ConfigError e = checkerError(R"({"mode": "quadratic_curve", "streams": 0})");
    EXPECT_EQ(e.field(), "streams");
    EXPECT_STREQ(e.what(), "<string>: streams: 0 must be >= 1\n"
                           "<string>: curve.spline_weights_dir: must be set for quadratic_curve");
    const CheckerConfig c = parseCheckerConfig(R"({"mode": "linear_curve"})", repo());
    EXPECT_EQ(c.obbCulling, Toggle::Auto);
    EXPECT_FALSE(c.sceneEdges.has_value());
    EXPECT_FALSE(c.curve.geometryFlags.has_value());
    EXPECT_EQ(c.curve.autoRefitMaxBatches, -1);
}

TEST(JsonChecker, SourceNameAndFileNamesInMessages) {
    const ConfigError e = thrown<ConfigError>([] { (void)parseCheckerConfig(R"({"streams": true})", "/", "my.json"); });
    EXPECT_EQ(std::string(e.what()).rfind("my.json: streams: ", 0), 0u) << e.what();

    TempDir tmp;
    const fs::path f     = tmp.write("bad.json", R"({"optix": {"max_trace_depth": 0}})");
    const ConfigError e2 = thrown<ConfigError>([&] { (void)loadCheckerConfig(f); });
    const std::u8string u = norm(f).generic_u8string();
    const std::string expectedPrefix = std::string(u.begin(), u.end()) + ": optix.max_trace_depth: ";
    EXPECT_EQ(std::string(e2.what()).rfind(expectedPrefix, 0), 0u) << e2.what();
    EXPECT_EQ(e2.field(), "optix.max_trace_depth");
}

TEST(JsonChecker, SyntaxErrorsAreIoErrorsWithLineAndColumn) {
    const IoError e = thrown<IoError>([] { (void)parseCheckerConfig("{\n  \"mode\": ,\n}", "/", "cfg.json"); });
    EXPECT_EQ(e.line(), 2u);
    EXPECT_GT(e.column(), 0u);
    EXPECT_EQ(e.path(), fs::path("cfg.json"));
    EXPECT_EQ(std::string(e.what()).rfind("cfg.json:2:" + std::to_string(e.column()) + ": JSON syntax error", 0), 0u)
        << e.what();

    const IoError e2 = thrown<IoError>([] { (void)parseCheckerConfig("", "/"); });
    EXPECT_EQ(e2.line(), 1u);
    const IoError e3 = thrown<IoError>([] { (void)parseCheckerConfig("{\"a\": 1} trailing", "/"); });
    EXPECT_EQ(e3.line(), 1u);
    // comments are not JSON
    (void)thrown<IoError>([] { (void)parseCheckerConfig("{ // c\n}", "/"); });
}

TEST(JsonChecker, DuplicateKeysAreErrors) {
    const ConfigError e = checkerError(R"({"streams": 1, "streams": 2})");
    EXPECT_NE(std::string(e.what()).find("duplicate key"), std::string::npos);
    EXPECT_EQ(e.field(), "streams");
    (void)checkerError(R"({"optix": {"opt_level": 0, "opt_level": 1}})");
    // the same key in different objects is fine
    EXPECT_NO_THROW((void)parseCheckerConfig(R"({"ray": {"skip_hit_items": true}, "optix": {"print_pipeline_log": true}})", "/"));
    EXPECT_NO_THROW((void)parseSceneDesc(
        R"({"obstacles": [{"mesh": "a.obj", "name": "a"}, {"mesh": "b.obj", "name": "b"}]})", "/"));
}

TEST(JsonChecker, MissingFileIsIoError) {
    const fs::path missing = repo() / "configs" / "does_not_exist.json";
    const IoError e        = thrown<IoError>([&] { (void)loadCheckerConfig(missing); });
    EXPECT_EQ(e.path(), norm(missing));
    EXPECT_EQ(e.line(), 0u);
    EXPECT_NE(std::string(e.what()).find("cannot open file"), std::string::npos);
    (void)thrown<IoError>([&] { (void)loadSceneDesc(missing); });
    (void)thrown<IoError>([&] { (void)loadRobotDesc(missing); });
    (void)thrown<IoError>([&] { (void)loadSetup(missing); });
}

// =====================================================================================================================
// paths
// =====================================================================================================================

TEST(JsonPaths, RelativeToBaseDirAbsoluteKept) {
    TempDir tmp;
    const CheckerConfig a = parseCheckerConfig(R"({"curve": {"spline_weights_dir": "x/../w"}})", tmp.path());
    EXPECT_EQ(a.curve.splineWeightsDir, tmp.path() / "w");
    const std::u8string abs = (tmp.path() / "elsewhere" / "w").generic_u8string();
    const CheckerConfig b   = parseCheckerConfig(
        R"({"curve": {"spline_weights_dir": ")" + std::string(abs.begin(), abs.end()) + "\"}}", repo());
    EXPECT_EQ(b.curve.splineWeightsDir, tmp.path() / "elsewhere" / "w");
    // a relative baseDir of the text overload is taken as given by the caller (relative to the working directory)
    const CheckerConfig c = parseCheckerConfig(R"({"curve": {"spline_weights_dir": "w"}})", "rel/dir");
    EXPECT_EQ(c.curve.splineWeightsDir, norm(fs::current_path() / "rel" / "dir" / "w"));
    // a file's paths are relative to that file, not to the working directory
    const fs::path f = tmp.write("sub/checker.json", R"({"curve": {"spline_weights_dir": "../weights"}})");
    EXPECT_EQ(loadCheckerConfig(f).curve.splineWeightsDir, tmp.path() / "weights");
}

TEST(JsonPaths, NonAsciiPathsAreUtf8) {
    TempDir tmp;
    const std::string json = "{\"name\": \"s\", \"obstacles\": [{\"mesh\": \"m\xC3\xA9sh/caf\xC3\xA9.obj\"}]}";
    const SceneDesc d      = parseSceneDesc(json, tmp.path());
    ASSERT_EQ(d.obstacles.size(), 1u);
    EXPECT_EQ(d.obstacles[0].mesh, tmp.path() / fs::path(u8"mésh") / fs::path(u8"café.obj"));
    EXPECT_EQ(d.obstacles[0].name, "caf\xC3\xA9");
    expectSameScene(parseSceneDesc(dumpSceneDesc(d), "/"), d);
}

// =====================================================================================================================
// scene
// =====================================================================================================================

TEST(JsonScene, ObstacleDefaultsAndPoses) {
    const std::string json = R"({
      "name": "test",
      "obstacles": [
        { "mesh": "meshes/box.obj" },
        { "name": "t", "mesh": "a.obj", "scale": 2.5, "pose": { "translation": [1, 2, 3] } },
        { "name": "r", "mesh": "a.obj", "pose": { "translation": [1, 2, 3], "rpy": [0.1, -0.2, 0.3] } },
        { "name": "q", "mesh": "a.obj", "pose": { "quaternion": [0, 0, 1, 1] } },
        { "name": "m", "mesh": "a.obj", "pose": { "matrix": [0, -1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 3] } },
        { "name": "e", "mesh": "a.obj", "pose": {} }
      ] })";
    const SceneDesc d = parseSceneDesc(json, "/scenes");
    EXPECT_EQ(d.name, "test");
    ASSERT_EQ(d.obstacles.size(), 6u);
    EXPECT_EQ(d.obstacles[0].name, "box"); // default: mesh file stem
    EXPECT_EQ(d.obstacles[0].mesh, norm("/scenes/meshes/box.obj"));
    EXPECT_EQ(d.obstacles[0].scale, 1.0f);
    EXPECT_TRUE(sameBits(d.obstacles[0].pose, Transform::identity()));
    EXPECT_EQ(d.obstacles[1].scale, 2.5f);
    EXPECT_TRUE(sameBits(d.obstacles[1].pose, Transform::translation(1, 2, 3)));
    EXPECT_TRUE(sameBits(d.obstacles[2].pose, Transform::fromXyzRpy(1, 2, 3, 0.1, -0.2, 0.3)));
    EXPECT_TRUE(sameBits(d.obstacles[3].pose, Transform::fromXyzQuat(0, 0, 0, 0, 0, 1, 1)));
    // 90 degrees about z
    EXPECT_NEAR(d.obstacles[3].pose.m[1], -1.0f, 1e-6f);
    EXPECT_TRUE(sameBits(d.obstacles[4].pose, Transform{{0, -1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 3}}));
    EXPECT_TRUE(sameBits(d.obstacles[5].pose, Transform::identity()));
    EXPECT_TRUE(parseSceneDesc("{}", "/").obstacles.empty());
}

TEST(JsonScene, Errors) {
    struct Case {
        const char* json;
        const char* field;
        const char* fragment;
    };
    const Case cases[] = {
        {R"({"obstacles": {}})", "obstacles", "array"},
        {R"({"obstacles": [{}]})", "obstacles[0].mesh", "required"},
        {R"({"obstacles": [{"mesh": "a.obj"}, {"mesh": "a.obj", "colour": 1}]})", "obstacles[1].colour", "unknown key"},
        {R"({"obstacles": [{"mesh": "a.obj", "scale": 0}]})", "obstacles[0].scale", "> 0"},
        {R"({"obstacles": [{"mesh": "a.obj", "scale": -1}]})", "obstacles[0].scale", "> 0"},
        {R"({"obstacles": [{"mesh": "a.obj", "pose": {"rpy": [0, 0, 0], "quaternion": [0, 0, 0, 1]}}]})",
            "obstacles[0].pose", "at most one"},
        {R"({"obstacles": [{"mesh": "a.obj", "pose": {"translation": [0, 0, 0], "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}}]})",
            "obstacles[0].pose.translation", "not allowed together"},
        {R"({"obstacles": [{"mesh": "a.obj", "pose": {"translation": [0, 0]}}]})", "obstacles[0].pose.translation",
            "3 numbers"},
        {R"({"obstacles": [{"mesh": "a.obj", "pose": {"quaternion": [0, 0, 0, 0]}}]})", "obstacles[0].pose.quaternion",
            "zero quaternion"},
        {R"({"obstacles": [{"mesh": "a.obj", "pose": {"rotation": [0, 0, 0]}}]})", "obstacles[0].pose.rotation",
            "unknown key"},
        {R"({"name": 3})", "name", "string"},
        {R"({"scene": "x"})", "scene", "unknown key"},
    };
    for (const Case& c : cases) {
        const ConfigError e = thrown<ConfigError>([&] { (void)parseSceneDesc(c.json, "/"); });
        EXPECT_EQ(e.field(), c.field) << c.json;
        EXPECT_NE(std::string(e.what()).find(c.fragment), std::string::npos) << c.json << " -> " << e.what();
    }
}

TEST(JsonScene, DumpRoundTripIsBitExact) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uni(-10.f, 10.f);
    std::uniform_int_distribution<std::uint32_t> bits;
    SceneDesc d;
    d.name = "random \"quoted\" name";
    for (int i = 0; i < 200; ++i) {
        ObstacleDesc o;
        o.name  = "o" + std::to_string(i);
        o.mesh  = fs::path("/meshes") / ("m" + std::to_string(i) + ".obj");
        o.scale = std::abs(uni(rng)) + 1e-3f;
        for (float& v : o.pose.m) {
            if (i % 2 == 0) {
                v = uni(rng);
            } else { // arbitrary finite bit patterns (denormals, huge and tiny magnitudes)
                float f;
                do {
                    f = std::bit_cast<float>(bits(rng));
                } while (!std::isfinite(f));
                v = f;
            }
        }
        d.obstacles.push_back(o);
    }
    d.obstacles.push_back(ObstacleDesc{"shelf", "/m/shelves.obj", Transform::translation(0.45f, 0, 0.4f), 1.0f});
    const SceneDesc back = parseSceneDesc(dumpSceneDesc(d), "/");
    ASSERT_EQ(back.obstacles.size(), d.obstacles.size());
    for (std::size_t i = 0; i < d.obstacles.size(); ++i) {
        EXPECT_EQ(back.obstacles[i].mesh, norm(d.obstacles[i].mesh));
        EXPECT_TRUE(sameBits(back.obstacles[i].pose, d.obstacles[i].pose)) << i;
        EXPECT_EQ(std::bit_cast<std::uint32_t>(back.obstacles[i].scale), std::bit_cast<std::uint32_t>(d.obstacles[i].scale));
        EXPECT_EQ(back.obstacles[i].name, d.obstacles[i].name);
    }
    EXPECT_EQ(back.name, d.name);
    // Short decimals stay short: 0.45f is written as 0.45.
    EXPECT_NE(dumpSceneDesc(d).find("0.45,"), std::string::npos);
}

// =====================================================================================================================
// robot
// =====================================================================================================================

TEST(JsonRobot, BlockWithOverridesAndBasePose) {
    const std::string json = R"({ "robot": {
        "description": "urdf/robot.urdf",
        "geometry": {
          "mesh": { "dir": "meshes", "source": "collision",
                    "files": { "link7": "special/l7.obj", "link0": "/abs/base.obj" } },
          "spheres": { "dir": "spheres", "files": { "link3": "l3.txt" } } },
        "base_pose": { "translation": [0, 0, 0.5] } } })";
    const RobotDesc d = parseRobotDesc(json, "/robots");
    EXPECT_EQ(d.description, norm("/robots/urdf/robot.urdf"));
    EXPECT_EQ(d.mesh.dir, norm("/robots/meshes"));
    EXPECT_EQ(d.mesh.format, "obj");
    EXPECT_EQ(d.meshSource, MeshSource::Collision);
    ASSERT_EQ(d.mesh.files.size(), 2u);
    EXPECT_EQ(d.mesh.files.at("link7"), norm("/robots/meshes/special/l7.obj")); // relative to mesh.dir
    EXPECT_EQ(d.mesh.files.at("link0"), norm("/abs/base.obj"));
    ASSERT_TRUE(d.spheres.has_value());
    EXPECT_EQ(d.spheres->dir, norm("/robots/spheres"));
    EXPECT_EQ(d.spheres->format, "xyzr"); // default sphere format
    EXPECT_EQ(d.spheres->files.at("link3"), norm("/robots/spheres/l3.txt"));
    EXPECT_TRUE(sameBits(d.basePose, Transform::translation(0, 0, 0.5f)));
    expectSameRobot(parseRobotDesc(dumpRobotDesc(d), "/"), d);
}

TEST(JsonRobot, SpheresAreOptional) {
    const RobotDesc d = parseRobotDesc(R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m"}}}})", "/x");
    EXPECT_FALSE(d.spheres.has_value());
    EXPECT_EQ(d.meshSource, MeshSource::Visual);
    expectSameRobot(parseRobotDesc(dumpRobotDesc(d), "/"), d);
    const RobotDesc n = parseRobotDesc(
        R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m"}, "spheres": null}}})", "/x");
    EXPECT_FALSE(n.spheres.has_value());
    // sphere overrides without a dir resolve against the file's directory
    const RobotDesc o = parseRobotDesc(
        R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m"}, "spheres": {"files": {"l": "s/l.txt"}}}}})",
        "/x");
    ASSERT_TRUE(o.spheres.has_value());
    EXPECT_TRUE(o.spheres->dir.empty());
    EXPECT_EQ(o.spheres->files.at("l"), norm("/x/s/l.txt"));
    expectSameRobot(parseRobotDesc(dumpRobotDesc(o), "/"), o);
}

TEST(JsonRobot, Errors) {
    struct Case {
        const char* json;
        const char* field;
        const char* fragment;
    };
    const Case cases[] = {
        {R"({})", "robot", "required"},
        {R"({"robot": {}, "extra": 1})", "extra", "unknown key"},
        {R"({"robot": {"geometry": {"mesh": {"dir": "m"}}}})", "robot.description", "required"},
        {R"({"robot": {"description": "r.urdf"}})", "robot.geometry", "required"},
        {R"({"robot": {"description": "r.urdf", "geometry": {}}})", "robot.geometry.mesh", "required"},
        {R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {}}}})", "robot.geometry.mesh.dir", "required"},
        {R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m", "format": "stl"}}}})",
            "robot.geometry.mesh.format", "only \"obj\""},
        {R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m", "source": "both"}}}})",
            "robot.geometry.mesh.source", "is not one of"},
        {R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m"}, "spheres": {"format": "obj"}}}})",
            "robot.geometry.spheres.format", "only \"xyzr\""},
        {R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m"}, "spheres": {"source": "visual"}}}})",
            "robot.geometry.spheres.source", "unknown key"},
        {R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m", "files": {"a": 1}}}}})",
            "robot.geometry.mesh.files.a", "string"},
        {R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m"}}, "base_pose": {"rpy": [0]}}})",
            "robot.base_pose.rpy", "3 numbers"},
    };
    for (const Case& c : cases) {
        const ConfigError e = thrown<ConfigError>([&] { (void)parseRobotDesc(c.json, "/"); });
        EXPECT_EQ(e.field(), c.field) << c.json;
        EXPECT_NE(std::string(e.what()).find(c.fragment), std::string::npos) << c.json << " -> " << e.what();
    }
}

// =====================================================================================================================
// setup files
// =====================================================================================================================

TEST(JsonSetup, ReferencedFilesResolveAgainstTheirOwnDirectory) {
    TempDir tmp;
    tmp.write("B/checker.json", R"({"mode": "quadratic_curve", "curve": {"spline_weights_dir": "w"}})");
    tmp.write("C/robot.json", R"({"robot": {"description": "r.urdf", "geometry": {"mesh": {"dir": "m"}}}})");
    const fs::path setup = tmp.write("A/setup.json", R"({
        "checker": "../B/checker.json",
        "scene": { "name": "inline", "obstacles": [ { "mesh": "m.obj" } ] },
        "robot": "../C/robot.json" })");
    const rtcd::Setup s = loadSetup(setup);
    EXPECT_EQ(s.sourceFile, tmp.path() / "A" / "setup.json");
    EXPECT_EQ(s.checker.mode, Mode::QuadraticCurve);
    EXPECT_EQ(s.checker.curve.splineWeightsDir, tmp.path() / "B" / "w");
    ASSERT_EQ(s.scene.obstacles.size(), 1u);
    EXPECT_EQ(s.scene.obstacles[0].mesh, tmp.path() / "A" / "m.obj"); // inline: relative to the setup file
    EXPECT_EQ(s.robot.description, tmp.path() / "C" / "r.urdf");
    EXPECT_EQ(s.robot.mesh.dir, tmp.path() / "C" / "m");
}

TEST(JsonSetup, InlineObjects) {
    TempDir tmp;
    const fs::path setup = tmp.write("s.json", R"({
        "checker": { "mode": "ray", "max_batch_size": 16 },
        "scene": { "obstacles": [] },
        "robot": { "description": "r.urdf", "geometry": { "mesh": { "dir": "m" } } } })");
    const rtcd::Setup s = loadSetup(setup);
    EXPECT_EQ(s.checker.mode, Mode::Ray);
    EXPECT_EQ(s.checker.maxBatchSize, 16u);
    EXPECT_EQ(s.robot.description, tmp.path() / "r.urdf");
}

TEST(JsonSetup, Errors) {
    TempDir tmp;
    const fs::path missingKey = tmp.write("a.json", R"({"checker": {}, "scene": {}})");
    EXPECT_EQ(thrown<ConfigError>([&] { (void)loadSetup(missingKey); }).field(), "robot");

    const fs::path unknown = tmp.write("b.json", R"({"checker": {}, "scene": {}, "robot": "r.json", "notes": ""})");
    EXPECT_EQ(thrown<ConfigError>([&] { (void)loadSetup(unknown); }).field(), "notes");

    const fs::path inlineBad = tmp.write("c.json",
        R"({"checker": {"streams": 0}, "scene": {}, "robot": {"description": "r", "geometry": {"mesh": {"dir": "m"}}}})");
    const ConfigError e = thrown<ConfigError>([&] { (void)loadSetup(inlineBad); });
    EXPECT_EQ(e.field(), "checker.streams");
    const std::u8string u = inlineBad.generic_u8string();
    EXPECT_EQ(std::string(e.what()), std::string(u.begin(), u.end()) + ": checker.streams: 0 must be >= 1");

    const fs::path inlineType = tmp.write("d.json",
        R"({"checker": {}, "scene": {"obstacles": [{"mesh": 1}]}, "robot": {"description": "r", "geometry": {"mesh": {"dir": "m"}}}})");
    EXPECT_EQ(thrown<ConfigError>([&] { (void)loadSetup(inlineType); }).field(), "scene.obstacles[0].mesh");

    // an error inside a referenced file names that file
    const fs::path badChecker = tmp.write("sub/bad_checker.json", R"({"streams": "x"})");
    const fs::path refBad     = tmp.write("e.json",
        R"({"checker": "sub/bad_checker.json", "scene": {}, "robot": {"description": "r", "geometry": {"mesh": {"dir": "m"}}}})");
    const ConfigError e2      = thrown<ConfigError>([&] { (void)loadSetup(refBad); });
    const std::u8string u2    = badChecker.generic_u8string();
    EXPECT_EQ(std::string(e2.what()).rfind(std::string(u2.begin(), u2.end()) + ": streams: ", 0), 0u) << e2.what();

    const fs::path refMissing = tmp.write("f.json",
        R"({"checker": "nope.json", "scene": {}, "robot": {"description": "r", "geometry": {"mesh": {"dir": "m"}}}})");
    EXPECT_EQ(thrown<IoError>([&] { (void)loadSetup(refMissing); }).path(), tmp.path() / "nope.json");
}

// =====================================================================================================================
// overrides
// =====================================================================================================================

TEST(JsonOverrides, HeaderExample) {
    CheckerConfig c;
    applyOverrides(c, R"(mode=ray,optix.max_register_count=32,ray.hit_mode=traverse,curve.refit={"every":4})");
    EXPECT_EQ(c.mode, Mode::Ray);
    EXPECT_EQ(c.optix.maxRegisterCount, 32u);
    EXPECT_EQ(c.ray.hitMode, RayHitMode::FirstHitTraverse);
    EXPECT_EQ(c.curve.refit.kind, RefitPolicy::Kind::Every);
    EXPECT_EQ(c.curve.refit.every, 4u);
}

TEST(JsonOverrides, ValuesWithCommasBareWordsAndJson) {
    CheckerConfig c;
    applyOverrides(c, R"( mesh.park_transform=[1,0,0,0, 0,1,0,0, 0,0,1,-5] , optix.exceptions=["user","trace_depth"],)"
                      R"(obb_culling=false, scene_edges=loop, curve.geometry_flags="single_anyhit", verbosity=info,)"
                      R"(optix.opt_level=default, curve.auto_refit_max_batches=3 ,, )");
    EXPECT_TRUE(sameBits(c.mesh.parkTransform, Transform::translation(0, 0, -5)));
    EXPECT_EQ(c.optix.exceptionFlags, OptixOptions::kExceptionUser | OptixOptions::kExceptionTraceDepth);
    EXPECT_EQ(c.obbCulling, Toggle::Off);
    EXPECT_EQ(c.sceneEdges, EdgeSet::Loop);
    EXPECT_EQ(c.curve.geometryFlags, GeometryFlags{kGeometryFlagRequireSingleAnyHitCall});
    EXPECT_EQ(c.verbosity, LogLevel::Info);
    EXPECT_EQ(c.optix.optLevel, OptixOptions::OptLevel::Default);
    EXPECT_EQ(c.curve.autoRefitMaxBatches, 3);

    applyOverrides(c, "scene_edges=auto,curve.geometry_flags=auto,curve.auto_refit_max_batches=auto,obb_culling=true");
    EXPECT_FALSE(c.sceneEdges.has_value());
    EXPECT_FALSE(c.curve.geometryFlags.has_value());
    EXPECT_EQ(c.curve.autoRefitMaxBatches, -1);
    EXPECT_EQ(c.obbCulling, Toggle::On);
    applyOverrides(c, ""); // nothing to do
    EXPECT_EQ(c.obbCulling, Toggle::On);
}

TEST(JsonOverrides, ObjectValuesMergeIntoObjects) {
    CheckerConfig c;
    c.optix.maxRegisterCount = 77;
    applyOverrides(c, R"(optix={"opt_level": 3, "print_pipeline_log": true})");
    EXPECT_EQ(c.optix.optLevel, OptixOptions::OptLevel::O3);
    EXPECT_TRUE(c.optix.printPipelineLog);
    EXPECT_EQ(c.optix.maxRegisterCount, 77u); // untouched by the merge
}

TEST(JsonOverrides, KeepsOtherFieldsAndPathsExact) {
    CheckerConfig c          = loadCheckerConfig(configs() / "checker_quad.json");
    const CheckerConfig orig = c;
    applyOverrides(c, "streams=2");
    EXPECT_EQ(c.numStreams, 2u);
    c.numStreams = orig.numStreams;
    expectSameConfig(c, orig);
    EXPECT_TRUE(c.graphPrepare); // default
    applyOverrides(c, "graph_prepare=false");
    EXPECT_FALSE(c.graphPrepare);
    c.graphPrepare = orig.graphPrepare;
    expectSameConfig(c, orig);
    // a relative path in an override is resolved against the working directory
    applyOverrides(c, "curve.spline_weights_dir=some/dir");
    EXPECT_EQ(c.curve.splineWeightsDir, norm(fs::current_path() / "some" / "dir"));
}

TEST(JsonOverrides, ErrorsNameTheEntryAndLeaveTheConfigUnchanged) {
    struct Case {
        const char* overrides;
        const char* field;
        const char* fragment;
    };
    const Case cases[] = {
        {"mode=ray,optix.maxreg=3", "optix.maxreg", "overrides entry \"optix.maxreg=3\": optix.maxreg: unknown key"},
        {"nothing", "", "expected <dotted.path>=<value>"},
        {"=3", "", "expected <dotted.path>=<value>"},
        {"streams=many", "streams", "overrides entry \"streams=many\": streams: must be a non-negative integer"},
        {"ray.hit_mode=fast", "ray.hit_mode", "is not one of"},
        {"curve.refit.every=3", "curve.refit.every", "unknown key"},
        {"curve..refit=3", "curve..refit", "malformed"},
        {R"(optix={"bogus": 1})", "optix.bogus", "unknown key"},
        {"streams=0", "streams", "overrides: streams: 0 must be >= 1"},
        {"streams=-1", "streams", "-1 must be >= 0"},
        {"curve.auto_refit_max_batches=-2", "curve.auto_refit_max_batches", "-2 must be >= 0"},
        {"mode=quadratic_curve", "curve.spline_weights_dir", "overrides: curve.spline_weights_dir: must be set"},
    };
    for (const Case& x : cases) {
        CheckerConfig c;
        c.maxBatchSize      = 99;
        const ConfigError e = thrown<ConfigError>([&] { applyOverrides(c, x.overrides); });
        EXPECT_EQ(e.field(), x.field) << x.overrides;
        EXPECT_NE(std::string(e.what()).find(x.fragment), std::string::npos) << x.overrides << " -> " << e.what();
        EXPECT_EQ(c.mode, Mode::Mesh) << x.overrides; // unchanged
        EXPECT_EQ(c.maxBatchSize, 99u);
    }
}

// =====================================================================================================================
// dumps
// =====================================================================================================================

TEST(JsonDump, CheckerHasEveryFieldInSchemaOrder) {
    CheckerConfig c;
    c.curve.splineWeightsDir = "relative/w"; // dumps write absolute paths
    const auto j             = nlohmann::ordered_json::parse(dumpCheckerConfig(c));
    std::vector<std::string> top;
    for (const auto& item : j.items()) {
        top.push_back(item.key());
    }
    EXPECT_EQ(top, (std::vector<std::string>{"mode", "streams", "host_dispatch", "max_batch_size", "max_items_per_check",
                       "obb_culling", "collapse_root_link_obb", "scene_edges", "verbosity", "graph_prepare", "optix", "mesh", "ray",
                       "curve"}));
    EXPECT_EQ(j["optix"].size(), 6u);
    EXPECT_EQ(j["mesh"].size(), 3u);
    EXPECT_EQ(j["ray"].size(), 3u);
    EXPECT_EQ(j["curve"].size(), 8u);
    EXPECT_EQ(j["mesh"]["park_transform"].size(), 12u);
    const std::string w = j["curve"]["spline_weights_dir"].get<std::string>();
    EXPECT_TRUE(fs::path(std::u8string(w.begin(), w.end())).is_absolute()) << w;
    EXPECT_EQ(j["obb_culling"], "auto");
    EXPECT_EQ(j["curve"]["refit"], "auto");
    EXPECT_EQ(j["curve"]["geometry_flags"], "auto");
    EXPECT_EQ(j["curve"]["auto_refit_max_batches"], "auto");
    // empty path -> null
    EXPECT_TRUE(nlohmann::json::parse(dumpCheckerConfig(CheckerConfig{}))["curve"]["spline_weights_dir"].is_null());
}

TEST(JsonDump, CheckerRoundTrips) {
    std::vector<CheckerConfig> configs{CheckerConfig{}};
    for (Mode m : {Mode::Mesh, Mode::Ray, Mode::LinearCurve, Mode::QuadraticCurve, Mode::CubicCurve}) {
        CheckerConfig c;
        c.mode                   = m;
        c.curve.splineWeightsDir = repo() / "w"; // absolute, so it compares equal after the round trip
        configs.push_back(c);
        configs.push_back(c.resolved()); // resolved configs (explicit edges / flags / refit) round-trip too
    }
    CheckerConfig all;
    all.mode                      = Mode::QuadraticCurve;
    all.numStreams                = 7;
    all.hostDispatch              = HostDispatch::SingleThread;
    all.maxBatchSize              = 33;
    all.maxItemsPerCheck          = 12345;
    all.obbCulling                = Toggle::On;
    all.collapseRootLinkObb       = false;
    all.sceneEdges                = EdgeSet::Unique;
    all.verbosity                 = LogLevel::Error;
    all.graphPrepare              = false;
    all.optix                     = OptixOptions::debug();
    all.optix.optLevel            = OptixOptions::OptLevel::O1;
    all.optix.exceptionFlags      = OptixOptions::kExceptionUser | OptixOptions::kExceptionStackOverflow;
    all.optix.maxRegisterCount    = 0;
    all.optix.maxTraceDepth       = 5;
    all.optix.printPipelineLog    = true;
    all.mesh.linkGasBuild         = BuildPreference::FastBuild;
    all.mesh.compactionMinPoses   = 0;
    all.mesh.parkTransform        = Transform::fromXyzRpy(0.1, 0.2, 0.3, 1.1, -0.7, 2.9);
    all.ray.launchWidthCap        = 1;
    all.ray.skipHitItems          = false;
    all.ray.hitMode               = RayHitMode::FirstHitCH;
    all.curve.trajectoryPoints    = 32;
    all.curve.controlPoints       = 13;
    all.curve.endPoints           = CurveEndPoints::Phantom;
    all.curve.gasBuild            = BuildPreference::FastTrace;
    all.curve.refit               = RefitPolicy{RefitPolicy::Kind::Every, 9};
    all.curve.autoRefitMaxBatches = 0;
    all.curve.geometryFlags       = kGeometryFlagNone;
    all.curve.splineWeightsDir    = repo() / "configs" / "spline_weights";
    configs.push_back(all);
    for (const CheckerConfig& c : configs) {
        const std::string dump = dumpCheckerConfig(c);
        const CheckerConfig back = parseCheckerConfig(dump, "/");
        expectSameConfig(back, c);
        EXPECT_EQ(dumpCheckerConfig(back), dump); // stable text
    }
    // relative struct paths come back absolute
    CheckerConfig rel;
    rel.mode                   = Mode::CubicCurve;
    rel.curve.splineWeightsDir = "w";
    EXPECT_EQ(roundTrip(rel).curve.splineWeightsDir, norm(fs::current_path() / "w"));
}

TEST(JsonDump, RobotDumpLoadsAsARobotFile) {
    const RobotDesc d = loadRobotDesc(configs() / "robots" / "panda.json");
    const auto j      = nlohmann::ordered_json::parse(dumpRobotDesc(d));
    ASSERT_TRUE(j.contains("robot"));
    EXPECT_EQ(j["robot"]["geometry"]["mesh"]["source"], "visual");
    EXPECT_EQ(j["robot"]["base_pose"]["matrix"].size(), 12u);
    TempDir tmp;
    const fs::path f = tmp.write("dumped/robot.json", dumpRobotDesc(d));
    expectSameRobot(loadRobotDesc(f), d); // absolute paths: independent of the dump's location
}
