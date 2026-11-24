// GPU tests of the ray backend (createRayBackend, src/backend_ray), label new_gpu. Without the detector core:
// the backend is driven by ReferenceExecutor / ReferenceBatchRunner (rtcd/internal/reference_executor.h).
//   * ProgramSet per RayHitMode (raygen, miss, closest-hit, record count, pipeline options), synchronizesHost;
//   * prepare(): LaunchPlan (launch width = min(total rays, cap), cap 0 = no cap) and RobotRayParams fields,
//     link poses on the device vs. the host reference FK, the constant all-ones mask with OBB culling off;
//   * construction / prepare errors (mode, unresolved config, streams, the 2^30 launch-size capacity, batch size,
//     stream index, short spans);
//   * golden flags (data/CCTest/golden_ray.bin) for configurations the rtcd_regression ctest entries do not
//     cover: OBB culling off, skip-hit off, no width cap, a small width cap (many grid-stride rounds), one stream,
//     batch 1024.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <rtcd/backend.h>
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/reference_executor.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
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
        std::vector<float> inputs; // 4096 regression poses
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
            x.inputs = support::regressionInputs(data / "motions", 1, support::LegacyHarness::regressionItems,
                x.robot->dof());
            x.golden = support::readFlagsFile(*support::goldenPath(data, Mode::Ray),
                support::LegacyHarness::regressionItems);
            return x;
        }();
        return e;
    }

    std::uint32_t expectedRaysPerRobot() {
        std::uint32_t n = 0;
        for (const LinkGeometry& l : env().robot->links()) {
            n += static_cast<std::uint32_t>(l.mesh.uniqueEdges.size());
        }
        return n;
    }

    // A backend with its own stream lease (members in construction order; destroyed in reverse).
    struct Rig {
        CheckerConfig cfg;
        StreamLease lease;
        std::vector<StreamContext> streams;
        std::unique_ptr<Backend> backend;

        explicit Rig(const CheckerConfig& c, bool resolve = true) : cfg(resolve ? c.resolved() : c) {
            lease = env().ctx->acquireStreams(cfg.numStreams);
            for (std::uint32_t i = 0; i < lease.size(); ++i) {
                streams.push_back(StreamContext{i, lease[i]});
            }
            BackendCreateInfo info;
            info.context = env().ctx;
            info.config  = &cfg;
            info.robot   = env().robot;
            info.scene   = env().scene;
            info.streams = streams;
            backend      = createRayBackend(info);
        }
    };

    CheckerConfig baseConfig() { return env().setup.checker; }

    // Device items / hit counters for one batch taken from the regression inputs.
    struct BatchBuffers {
        DeviceBuffer items, hits;
        BatchView view;
        BatchBuffers(std::uint32_t n, std::uint32_t capacity) {
            const std::uint32_t dof = env().robot->dof();
            items = DeviceBuffer::fromHost(std::span<const float>(env().inputs.data(), std::size_t(n) * dof));
            hits.allocate(std::size_t(capacity) * sizeof(std::uint32_t));
            view.batchIndex = 0;
            view.firstItem  = 0;
            view.itemCount  = n;
            view.items      = DeviceSpan<const float>{items.ptr(), std::size_t(n) * dof};
            view.hitCounts  = DeviceSpan<std::uint32_t>{hits.ptr(), n};
        }
    };

    support::Flags runFlags(const CheckerConfig& c, std::size_t items = support::LegacyHarness::regressionItems) {
        Rig rig(c);
        internal::ReferenceExecutor exec(env().ctx, rig.backend->programs(), rig.cfg.optix);
        internal::ReferenceBatchRunner run(env().ctx, *rig.backend, exec, rig.cfg, rig.streams, env().robot->dof());
        return support::flagsFromCounts(run.runAll(env().inputs, static_cast<std::uint32_t>(items)));
    }

    void expectGolden(const CheckerConfig& c, const std::string& what) {
        const support::Flags f = runFlags(c);
        const auto d = support::diffFlags(f, env().golden, support::LegacyHarness::regressionItems);
        EXPECT_EQ(d.differing, 0u) << what << ": " << d.differing << " flag diffs vs golden_ray.bin";
        EXPECT_EQ(support::countSet(f), 1383u) << what;
    }

} // namespace

// ---- programs ----------------------------------------------------------------------------------------------

TEST(RayBackend, ProgramSetPerHitMode) {
    struct Case {
        RayHitMode mode;
        const char* raygen;
        const char* miss;
        const char* ch;
    };
    const Case cases[] = {
        {RayHitMode::ClosestHit, "__raygen__robotRayCompact", "", "__closesthit__radiance"},
        {RayHitMode::FirstHitCH, "__raygen__robotRayFirstHitCH", "", "__closesthit__radiance"},
        {RayHitMode::FirstHitMiss, "__raygen__robotRayFirstHitMiss", "__miss__radiance", ""},
        {RayHitMode::FirstHitTraverse, "__raygen__robotRayFirstHitTraverse", "", ""},
    };
    for (const Case& c : cases) {
        CheckerConfig cfg = baseConfig();
        cfg.ray.hitMode   = c.mode;
        Rig rig(cfg);
        const ProgramSet& p = rig.backend->programs();
        EXPECT_EQ(rig.backend->mode(), Mode::Ray);
        EXPECT_FALSE(p.moduleCode.empty());
        ASSERT_EQ(p.raygens.size(), 1u);
        EXPECT_EQ(p.raygens[0], c.raygen);
        EXPECT_EQ(p.miss, c.miss);
        EXPECT_EQ(p.closestHit, c.ch);
        EXPECT_TRUE(p.anyHit.empty());
        EXPECT_EQ(p.builtinIntersection, BuiltinIntersection::None);
        EXPECT_EQ(p.traversableGraph, TraversableGraph::SingleLevelInstancing);
        EXPECT_EQ(p.primitiveTypes, std::uint32_t(kPrimitiveDefault));
        EXPECT_EQ(p.numPayloadValues, 1u);
        EXPECT_EQ(p.numAttributeValues, 2u);
        EXPECT_EQ(p.maxTraversableDepth, 2u);
        EXPECT_EQ(p.hitGroupRecordCount, env().scene->numObstacles());
        for (std::uint32_t n : {1u, 37u, 127u, 128u}) {
            EXPECT_FALSE(rig.backend->synchronizesHost(n));
        }
        // the program set must build a pipeline
        EXPECT_NO_THROW(internal::ReferenceExecutor(env().ctx, p, rig.cfg.optix));
    }
}

// ---- prepare -------------------------------------------------------------------------------------------------

TEST(RayBackend, PlanParamsAndLinkPoses) {
    const std::uint32_t rays = expectedRaysPerRobot();
    ASSERT_GT(rays, 0u);
    struct Case {
        std::uint32_t cap;
        bool skip;
        std::uint32_t n;
    };
    const Case cases[] = {{262144, true, 37}, {262144, true, 128}, {1000, false, 37}, {0, true, 128}};
    for (const Case& c : cases) {
        CheckerConfig cfg        = baseConfig();
        cfg.ray.launchWidthCap   = c.cap;
        cfg.ray.skipHitItems     = c.skip;
        Rig rig(cfg);
        BatchBuffers b(c.n, cfg.maxBatchSize);
        const StreamContext sc = rig.streams[1 % rig.streams.size()];
        b.hits.memsetAsync(0, std::size_t(c.n) * sizeof(std::uint32_t), sc.stream);
        const LaunchPlan plan = rig.backend->prepare(sc, b.view);
        ASSERT_EQ(cudaStreamSynchronize(sc.stream), cudaSuccess);

        const std::uint32_t total = c.n * rays;
        EXPECT_TRUE(plan.launch);
        EXPECT_EQ(plan.width, (c.cap != 0 && total > c.cap) ? c.cap : total) << "cap " << c.cap << " n " << c.n;
        EXPECT_EQ(plan.height, 1u);
        EXPECT_EQ(plan.depth, 1u);
        EXPECT_EQ(plan.raygen, 0u);
        EXPECT_EQ(plan.traversable, env().scene->traversable());
        ASSERT_EQ(plan.paramsBytes, sizeof(RobotRayParams));
        ASSERT_NE(plan.params, nullptr);
        RobotRayParams p;
        std::memcpy(&p, plan.params, sizeof(p));
        EXPECT_EQ(p.traversable, env().scene->traversable());
        EXPECT_EQ(p.hitCounts, b.hits.ptr());
        EXPECT_EQ(p.totalRayCount, total);
        EXPECT_EQ(p.raysPerRobot, rays);
        EXPECT_EQ(p.linksPerRobot, env().robot->numLinks());
        EXPECT_EQ(p.skipHitItems, c.skip ? 1u : 0u);
        EXPECT_NE(p.rays, 0u);
        EXPECT_NE(p.rayLink, 0u);
        EXPECT_NE(p.linkMask, 0u);
        EXPECT_NE(p.linkPoses, 0u);

        // ray -> link map: non-decreasing link indices, ray count per link = unique edges of that link
        std::vector<std::int32_t> rayLink(rays);
        ASSERT_EQ(cudaMemcpy(rayLink.data(), reinterpret_cast<const void*>(p.rayLink), rays * sizeof(std::int32_t),
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        std::vector<std::uint32_t> perLink(env().robot->numLinks(), 0);
        for (std::uint32_t r = 0; r < rays; ++r) {
            ASSERT_GE(rayLink[r], 0);
            ASSERT_LT(std::uint32_t(rayLink[r]), env().robot->numLinks());
            if (r > 0) {
                ASSERT_LE(rayLink[r - 1], rayLink[r]);
            }
            ++perLink[rayLink[r]];
        }
        for (std::uint32_t l = 0; l < env().robot->numLinks(); ++l) {
            EXPECT_EQ(perLink[l], env().robot->links()[l].mesh.uniqueEdges.size()) << "link " << l;
        }

        // link poses on the device vs. the host reference FK (std::sin/cos vs __sincosf: tolerance)
        const std::uint32_t L = env().robot->numLinks();
        std::vector<float> dev(std::size_t(c.n) * L * 12);
        ASSERT_EQ(cudaMemcpy(dev.data(), reinterpret_cast<const void*>(p.linkPoses), dev.size() * sizeof(float),
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        std::vector<float> ref(std::size_t(L) * 12);
        const std::uint32_t dof = env().robot->dof();
        for (std::uint32_t i = 0; i < c.n; ++i) {
            env().robot->linkPoses(std::span<const float>(env().inputs.data() + std::size_t(i) * dof, dof), ref);
            for (std::size_t k = 0; k < ref.size(); ++k) {
                ASSERT_NEAR(dev[std::size_t(i) * L * 12 + k], ref[k], 1e-4f) << "item " << i << " float " << k;
            }
        }
    }
}

TEST(RayBackend, ObbCullingOffUsesAllOnesMask) {
    CheckerConfig cfg = baseConfig();
    cfg.obbCulling    = Toggle::Off;
    Rig rig(cfg);
    BatchBuffers b(cfg.maxBatchSize, cfg.maxBatchSize);
    const StreamContext sc = rig.streams[0];
    const LaunchPlan plan  = rig.backend->prepare(sc, b.view);
    ASSERT_EQ(cudaStreamSynchronize(sc.stream), cudaSuccess);
    RobotRayParams p;
    std::memcpy(&p, plan.params, sizeof(p));
    std::vector<std::uint32_t> mask(std::size_t(cfg.maxBatchSize) * env().robot->numLinks());
    ASSERT_EQ(cudaMemcpy(mask.data(), reinterpret_cast<const void*>(p.linkMask), mask.size() * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (std::size_t i = 0; i < mask.size(); ++i) {
        ASSERT_EQ(mask[i], 1u) << i;
    }
}

TEST(RayBackend, ObbMaskHasCulledAndKeptLinks) {
    // With OBB culling on, the mask of the regression's first 128 poses has both zero and non-zero entries (the
    // broad phase culls some links and keeps others); every entry is an overlap count <= numObstacles.
    CheckerConfig cfg = baseConfig();
    Rig rig(cfg);
    BatchBuffers b(cfg.maxBatchSize, cfg.maxBatchSize);
    const StreamContext sc = rig.streams[0];
    const LaunchPlan plan  = rig.backend->prepare(sc, b.view);
    ASSERT_EQ(cudaStreamSynchronize(sc.stream), cudaSuccess);
    RobotRayParams p;
    std::memcpy(&p, plan.params, sizeof(p));
    std::vector<std::uint32_t> mask(std::size_t(cfg.maxBatchSize) * env().robot->numLinks());
    ASSERT_EQ(cudaMemcpy(mask.data(), reinterpret_cast<const void*>(p.linkMask), mask.size() * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    std::size_t zero = 0, nonzero = 0;
    for (std::uint32_t v : mask) {
        (v == 0 ? zero : nonzero)++;
        EXPECT_LE(v, env().scene->numObstacles());
    }
    EXPECT_GT(zero, 0u);
    EXPECT_GT(nonzero, 0u);
}

// ---- errors ------------------------------------------------------------------------------------------------

TEST(RayBackend, ConstructionErrors) {
    {
        CheckerConfig cfg = baseConfig();
        cfg.mode          = Mode::Mesh;
        EXPECT_THROW(Rig{cfg}, InvalidArgument);
    }
    {
        CheckerConfig cfg = baseConfig(); // not resolved: obb_culling auto
        cfg.obbCulling    = Toggle::Auto;
        EXPECT_THROW(Rig(cfg, false), InvalidArgument);
    }
    {
        // 2^30 launch-size capacity: max_batch_size * rays per robot must be < 2^30 (checked before allocating)
        CheckerConfig cfg = baseConfig();
        cfg.maxBatchSize  = static_cast<std::uint32_t>(kMaxLaunchSize / expectedRaysPerRobot()) + 1;
        try {
            Rig rig(cfg);
            FAIL() << "expected CapacityError";
        } catch (const CapacityError& e) {
            EXPECT_GE(e.requested(), kMaxLaunchSize);
        }
    }
    {
        // stream span size != config.streams
        CheckerConfig cfg = baseConfig().resolved();
        StreamLease lease = env().ctx->acquireStreams(1);
        std::vector<StreamContext> streams{StreamContext{0, lease[0]}};
        BackendCreateInfo info{env().ctx, &cfg, env().robot, env().scene, streams};
        EXPECT_THROW(createRayBackend(info), InvalidArgument);
        info.config = nullptr;
        EXPECT_THROW(createRayBackend(info), InvalidArgument);
    }
}

TEST(RayBackend, PrepareErrors) {
    CheckerConfig cfg = baseConfig();
    Rig rig(cfg);
    BatchBuffers b(cfg.maxBatchSize, cfg.maxBatchSize);
    const StreamContext sc = rig.streams[0];

    BatchView v = b.view;
    v.itemCount = 0;
    EXPECT_THROW(rig.backend->prepare(sc, v), InvalidArgument);

    v           = b.view;
    v.itemCount = cfg.maxBatchSize + 1;
    EXPECT_THROW(rig.backend->prepare(sc, v), CapacityError);

    StreamContext bad = sc;
    bad.index         = static_cast<std::uint32_t>(rig.streams.size());
    EXPECT_THROW(rig.backend->prepare(bad, b.view), InvalidArgument);

    v             = b.view;
    v.items.count = std::size_t(v.itemCount) * env().robot->dof() - 1;
    EXPECT_THROW(rig.backend->prepare(sc, v), InvalidArgument);

    v                 = b.view;
    v.hitCounts.count = v.itemCount - 1;
    EXPECT_THROW(rig.backend->prepare(sc, v), InvalidArgument);

    EXPECT_NO_THROW(rig.backend->prepare(sc, b.view));
    EXPECT_EQ(cudaStreamSynchronize(sc.stream), cudaSuccess);
}

// ---- golden flags for non-default configurations -------------------------------------------------------------

TEST(RayBackendGolden, ObbCullingOff) {
    for (RayHitMode m : {RayHitMode::ClosestHit, RayHitMode::FirstHitTraverse}) {
        CheckerConfig cfg = baseConfig();
        cfg.obbCulling    = Toggle::Off;
        cfg.ray.hitMode   = m;
        expectGolden(cfg, "obb off, hit mode " + std::to_string(int(m)));
    }
}

TEST(RayBackendGolden, SkipHitOff) {
    CheckerConfig cfg    = baseConfig();
    cfg.ray.skipHitItems = false;
    expectGolden(cfg, "skip_hit_items off");
}

TEST(RayBackendGolden, LaunchWidthCaps) {
    for (std::uint32_t cap : {0u, 1000u, 65536u}) {
        CheckerConfig cfg      = baseConfig();
        cfg.ray.launchWidthCap = cap;
        expectGolden(cfg, "launch_width_cap " + std::to_string(cap));
    }
}

TEST(RayBackendGolden, OneStreamAndBatch1024) {
    {
        CheckerConfig cfg = baseConfig();
        cfg.numStreams    = 1;
        expectGolden(cfg, "1 stream");
    }
    {
        CheckerConfig cfg = baseConfig();
        cfg.maxBatchSize  = 1024;
        expectGolden(cfg, "batch 1024");
    }
}

TEST(RayBackendGolden, SkipHitOffCountsDeterministic) {
    // Without skip-hit and with closest-hit, hit counts are exact per item, so two runs give identical counts.
    CheckerConfig cfg    = baseConfig();
    cfg.ray.skipHitItems = false;
    Rig rig(cfg);
    internal::ReferenceExecutor exec(env().ctx, rig.backend->programs(), rig.cfg.optix);
    internal::ReferenceBatchRunner run(env().ctx, *rig.backend, exec, rig.cfg, rig.streams, env().robot->dof());
    const auto a = run.runAll(env().inputs, support::LegacyHarness::regressionItems);
    const auto b = run.runAll(env().inputs, support::LegacyHarness::regressionItems);
    EXPECT_EQ(a, b);
}
