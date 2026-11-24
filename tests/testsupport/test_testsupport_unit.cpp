// new_unit: host utilities of rtcd_testsupport (support/*.h). Runs from the repository root (data/ junction).
#include <support/golden.h>
#include <support/inputs.h>
#include <support/options.h>
#include <support/paths.h>
#include <support/timing.h>

#include <rtcd/errors.h>

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace rtcd;
using namespace rtcd::support;
namespace fs = std::filesystem;

namespace {
    fs::path motions() { return dataDir() / "motions"; }
    fs::path tempDir() {
        const fs::path p = fs::temp_directory_path() / "rtcd_testsupport_unit";
        fs::create_directories(p);
        return p;
    }
} // namespace

TEST(Paths, FindsRepoRootAndData) {
    const fs::path root = findRepoRoot();
    EXPECT_TRUE(fs::is_directory(root / "data"));
    EXPECT_TRUE(fs::is_directory(dataDir() / "CCTest"));
    EXPECT_EQ(dataDir(fs::path("x/y")), fs::absolute("x/y"));
    EXPECT_FALSE(executableDir().empty());
}

TEST(Inputs, WeightString) {
    const float w[] = {1, 1, 1, 1, 1, 1, 1};
    EXPECT_EQ(knnWeightString(w), "1111111");
    const float w2[] = {2.9f, 0.5f, 10.0f};
    EXPECT_EQ(knnWeightString(w2), "2010");
    EXPECT_EQ(knnWeightString(w), LegacyHarness::weightString);
}

TEST(Inputs, DiscreteRegressionInputsAreTheCachedPoses) {
    const std::vector<float> in = regressionInputs(motions(), 1);
    ASSERT_EQ(in.size(), 4096u * 7u);
    const std::vector<float> raw = readFloatFile(motions() / "Panda4096.bin");
    ASSERT_EQ(raw.size(), in.size());
    EXPECT_EQ(std::memcmp(raw.data(), in.data(), raw.size() * sizeof(float)), 0);
}

TEST(Inputs, KnnTrajectoryArithmetic) {
    // 3 samples of dof 2, k = 2: trajectory t = i * k + j from sample i to sample idx[t]
    const std::vector<float> s = {0.f, 1.f, 3.f, -1.f, 0.1f, 0.7f};
    const std::vector<std::int32_t> idx = {1, 2, 0, 2, 1, 0};
    const auto tr = knnTrajectories(s, 2, idx, 2, 4, 100);
    ASSERT_EQ(tr.size(), 6u * 4u * 2u);
    for (std::size_t t = 0; t < 6; ++t) {
        const float* a = &s[(t / 2) * 2];
        const float* b = &s[std::size_t(idx[t]) * 2];
        for (std::uint32_t p = 0; p < 4; ++p) {
            for (int d = 0; d < 2; ++d) {
                const float span = (b[d] - a[d]) / 3.0f;
                const float prod = span * float(p);
                EXPECT_EQ(tr[(t * 4 + p) * 2 + d], a[d] + prod);
            }
        }
        // waypoint 0 is exactly the start sample
        EXPECT_EQ(tr[t * 8], a[0]);
    }
    EXPECT_EQ(knnTrajectories(s, 2, idx, 2, 4, 5).size(), 5u * 4u * 2u);
    EXPECT_THROW(knnTrajectories(s, 2, idx, 2, 1, 5), InvalidArgument);
    EXPECT_THROW(knnTrajectories(s, 2, std::span<const std::int32_t>(idx.data(), 5), 2, 4, 5), InvalidArgument);
}

TEST(Inputs, CurveRegressionInputsShapeAndEndpoints) {
    const std::vector<float> in = regressionInputs(motions(), 16);
    ASSERT_EQ(in.size(), 4096u * 16u * 7u);
    const std::vector<float> samples = loadCachedPoses(motions(), "Panda", 4096, 7);
    const std::vector<std::int32_t> idx = loadKnnIndices(motions(), "Panda", "1111111", 4096, 4);
    // trajectory t starts at sample t / 4
    for (std::size_t t : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(4095)}) {
        for (int d = 0; d < 7; ++d) {
            EXPECT_EQ(in[t * 16 * 7 + d], samples[(t / 4) * 7 + d]);
            // last waypoint is close to (not necessarily equal to) the neighbour sample
            EXPECT_NEAR(in[(t * 16 + 15) * 7 + d], samples[std::size_t(idx[t]) * 7 + d], 1e-5f);
        }
    }
}

TEST(Inputs, BenchmarkPools) {
    EXPECT_EQ(benchmarkPosePool(motions()).size(), 8192u * 7u);
    EXPECT_EQ(benchmarkTrajectoryPool(motions(), 8).size(), 8192u * 4u * 8u * 7u);
}

TEST(Inputs, MissingOrMalformedCacheThrowsIoError) {
    EXPECT_THROW(loadCachedPoses(motions(), "Panda", 123, 7), IoError); // no such file
    EXPECT_THROW(loadCachedPoses(motions(), "Panda", 4096, 6), IoError); // wrong size for dof 6
    EXPECT_THROW(loadKnnIndices(motions(), "Panda", "1111111", 4096, 5), IoError);
    const fs::path dir = tempDir() / "motions";
    fs::create_directories(dir / "R" / "1");
    {
        std::ofstream f(dir / "R" / "1" / "2_knn1_idx.bin", std::ios::binary);
        const std::int32_t v[2] = {0, 7}; // 7 out of range
        f.write(reinterpret_cast<const char*>(v), sizeof(v));
    }
    EXPECT_THROW(loadKnnIndices(dir, "R", "1", 2, 1), IoError);
}

TEST(Golden, ReadsTheFourGoldens) {
    for (Mode m : {Mode::Mesh, Mode::Ray, Mode::LinearCurve, Mode::QuadraticCurve}) {
        const auto p = goldenPath(dataDir(), m);
        ASSERT_TRUE(p.has_value());
        const Flags g = readFlagsFile(*p, 4096);
        EXPECT_EQ(g.size(), 4096u);
        const std::size_t hits = countSet(g);
        EXPECT_GT(hits, 0u);
        EXPECT_LT(hits, 4096u);
    }
    EXPECT_FALSE(goldenPath(dataDir(), Mode::CubicCurve).has_value());
    EXPECT_EQ(goldenPath(dataDir(), Mode::Ray)->filename(), "golden_ray.bin");
    EXPECT_THROW(readFlagsFile(*goldenPath(dataDir(), Mode::Ray), 4000), IoError);
}

TEST(Golden, WriteReadDiff) {
    const fs::path f = tempDir() / "flags.bin";
    const Flags a    = {0, 1, 1, 0, 1};
    writeFlagsFile(f, a);
    EXPECT_EQ(readFlagsFile(f, 5), a);
    EXPECT_THROW(writeFlagsFile(tempDir() / "golden_ias.bin", a), InvalidArgument);
    {
        std::ofstream o(tempDir() / "bad.bin", std::ios::binary);
        o.put(0);
        o.put(2);
    }
    EXPECT_THROW(readFlagsFile(tempDir() / "bad.bin"), IoError);

    const std::uint32_t counts[] = {0, 3, 1, 0, 0};
    const Flags b                = flagsFromCounts(counts);
    EXPECT_EQ(b, (Flags{0, 1, 1, 0, 0}));
    const FlagDiff d = diffFlags(a, b, 5);
    EXPECT_EQ(d.differing, 1u);
    ASSERT_EQ(d.firstIndices.size(), 1u);
    EXPECT_EQ(d.firstIndices[0], 4u);
    std::ostringstream os;
    printFlagDiff(os, d, a, b);
    EXPECT_EQ(os.str(), "    diff at item 4: 1 vs 0\n");
    EXPECT_TRUE(diffFlags(a, b, 4).equal());
    EXPECT_THROW(diffFlags(a, b, 6), InvalidArgument);
    const FlagDiff d2 = diffFlags(Flags(20, 0), Flags(20, 1), 20, 3);
    EXPECT_EQ(d2.differing, 20u);
    EXPECT_EQ(d2.firstIndices.size(), 3u);
}

TEST(Golden, PrefixCounts) {
    EXPECT_EQ(regressionPrefixCounts(128, 4, 4096), (std::vector<std::size_t>{1, 37, 128, 517, 1000, 4095}));
    // B * S + 5 = 4101 > 4096 is dropped
    EXPECT_EQ(regressionPrefixCounts(1024, 4, 4096), (std::vector<std::size_t>{1, 37, 1024, 1000, 4095}));
    // duplicates (B = 1) and 1000 > 37 are dropped
    EXPECT_EQ(regressionPrefixCounts(1, 1, 37), (std::vector<std::size_t>{1, 37, 6, 36}));
}

TEST(Options, LegacyOptixConfig) {
    CheckerConfig c;
    ContextOptions ctx;
    applyLegacyOptixConfig("opt=2,debug=moderate,maxreg=32,exceptions=debug,tracedepth=3,rayhit=traverse,"
                           "pipelinelog=1,validation=1,loglevel=4",
        &c, &ctx);
    EXPECT_EQ(c.optix.optLevel, OptixOptions::OptLevel::O2);
    EXPECT_EQ(c.optix.debugLevel, OptixOptions::DebugLevel::Moderate);
    EXPECT_EQ(c.optix.maxRegisterCount, 32u);
    EXPECT_EQ(c.optix.exceptionFlags, OptixOptions::kExceptionTraceDepth | OptixOptions::kExceptionStackOverflow);
    EXPECT_EQ(c.optix.maxTraceDepth, 3u);
    EXPECT_EQ(c.ray.hitMode, RayHitMode::FirstHitTraverse);
    EXPECT_TRUE(c.optix.printPipelineLog);
    EXPECT_TRUE(ctx.optixValidation);
    EXPECT_EQ(ctx.optixLogLevel, 4);

    applyLegacyOptixConfig("opt=default,debug=none,exceptions=4,rayhit=1", &c, nullptr);
    EXPECT_EQ(c.optix.optLevel, OptixOptions::OptLevel::Default);
    EXPECT_EQ(c.optix.debugLevel, OptixOptions::DebugLevel::None);
    EXPECT_EQ(c.optix.exceptionFlags, OptixOptions::kExceptionUser);
    EXPECT_EQ(c.ray.hitMode, RayHitMode::FirstHitCH);

    const CheckerConfig before = c;
    EXPECT_THROW(applyLegacyOptixConfig("maxreg=16,bogus=1", &c, nullptr), ConfigError);
    EXPECT_EQ(c.optix.maxRegisterCount, before.optix.maxRegisterCount); // all-or-nothing
    EXPECT_THROW(applyLegacyOptixConfig("opt=4", &c, nullptr), ConfigError);
    EXPECT_THROW(applyLegacyOptixConfig("exceptions=8", &c, nullptr), ConfigError);
    EXPECT_THROW(applyLegacyOptixConfig("tracedepth=0", &c, nullptr), ConfigError);
    EXPECT_THROW(applyLegacyOptixConfig("maxreg", &c, nullptr), ConfigError);
    applyLegacyOptixConfig("", &c, nullptr);
    applyLegacyOptixConfig(" maxreg = 40 , ", &c, nullptr);
    EXPECT_EQ(c.optix.maxRegisterCount, 40u);
}

TEST(Options, LegacyCurveKnobs) {
    EXPECT_EQ(parseLegacyCurveGeometryFlag("none"), kGeometryFlagNone);
    EXPECT_EQ(parseLegacyCurveGeometryFlag("single_anyhit"), kGeometryFlagRequireSingleAnyHitCall);
    EXPECT_THROW(parseLegacyCurveGeometryFlag("x"), ConfigError);
    EXPECT_EQ(parseLegacyRefitEvery("0").kind, RefitPolicy::Kind::Always);
    EXPECT_EQ(parseLegacyRefitEvery("1").kind, RefitPolicy::Kind::Never);
    const RefitPolicy p = parseLegacyRefitEvery("4");
    EXPECT_EQ(p.kind, RefitPolicy::Kind::Every);
    EXPECT_EQ(p.every, 4u);
    EXPECT_THROW(parseLegacyRefitEvery("-1"), ConfigError);

    EnvSettings env;
    env.curveGeometryFlag = "none";
    env.refitEvery        = "1";
    env.legacyOptixConfig = "maxreg=24";
    CheckerConfig c;
    applyLegacyEnv(env, &c, nullptr);
    EXPECT_EQ(c.curve.geometryFlags, std::optional<GeometryFlags>(kGeometryFlagNone));
    EXPECT_EQ(c.curve.refit.kind, RefitPolicy::Kind::Never);
    EXPECT_EQ(c.optix.maxRegisterCount, 24u);
}

TEST(Options, CountLists) {
    EXPECT_EQ(parseCountList("1,37,128"), (std::vector<std::size_t>{1, 37, 128}));
    EXPECT_EQ(parseCountList("1-16"), (std::vector<std::size_t>{1, 2, 4, 8, 16}));
    EXPECT_EQ(parseCountList("3-20,5"), (std::vector<std::size_t>{3, 6, 12, 5}));
    EXPECT_THROW(parseCountList(""), InvalidArgument);
    EXPECT_THROW(parseCountList("0"), InvalidArgument);
    EXPECT_THROW(parseCountList("8-2"), InvalidArgument);
    EXPECT_EQ(powersOfTwo(4096).size(), 13u);
    EXPECT_EQ(modeName(Mode::QuadraticCurve), "quadratic_curve");
    EXPECT_FALSE(describeConfig(CheckerConfig{}).empty());
}

TEST(Timing, Summary) {
    const TimingStats s = summarize({5, 1, 3, 2, 4});
    EXPECT_EQ(s.samples, 5u);
    EXPECT_DOUBLE_EQ(s.median, 3);
    EXPECT_DOUBLE_EQ(s.mean, 3);
    EXPECT_DOUBLE_EQ(s.min, 1);
    EXPECT_DOUBLE_EQ(s.max, 5);
    EXPECT_DOUBLE_EQ(s.p10, 1.4);
    EXPECT_DOUBLE_EQ(s.p90, 4.6);
    EXPECT_EQ(summarize({}).samples, 0u);
    EXPECT_DOUBLE_EQ(summarize({2}).p90, 2);
}
