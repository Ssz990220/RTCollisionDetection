// new_gpu: rtcd_backend_mesh (createMeshBackend) through the test-support ReferenceExecutor / ReferenceBatchRunner
// (no detector core): program declaration, synchronizesHost, the golden on every prepare path (masked, compaction,
// OBB off), equal hit counts across those paths, the launch == false path, and argument errors.
#include "mesh_test_fixture.h"

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

using namespace rtcd;
using rtcd::mesh_test::MeshBackendTest;

namespace {
    constexpr std::uint32_t kItems = 4096;
} // namespace

TEST_F(MeshBackendTest, ProgramSetMatchesTheOldMeshDetector) {
    auto h               = makeHarness("", nullptr, false);
    const ProgramSet& ps = h->backend->programs();
    EXPECT_EQ(h->backend->mode(), Mode::Mesh);
    EXPECT_FALSE(ps.moduleCode.empty());
    ASSERT_EQ(ps.raygens.size(), 1u);
    EXPECT_EQ(ps.raygens[0], "__raygen__edgeRays");
    EXPECT_TRUE(ps.miss.empty());
    EXPECT_EQ(ps.anyHit, "__anyhit__instanceItem");
    EXPECT_TRUE(ps.closestHit.empty());
    EXPECT_EQ(ps.builtinIntersection, BuiltinIntersection::None);
    EXPECT_EQ(ps.traversableGraph, TraversableGraph::SingleLevelInstancing);
    EXPECT_EQ(ps.primitiveTypes, std::uint32_t(kPrimitiveDefault));
    EXPECT_EQ(ps.numPayloadValues, 0u);
    EXPECT_EQ(ps.numAttributeValues, 2u);
    EXPECT_EQ(ps.maxTraversableDepth, 2u);
    // old getSBTSize(): BATCHSIZE * (DOF + 1)
    EXPECT_EQ(ps.hitGroupRecordCount, h->config.maxBatchSize * robot_->numLinks());
    EXPECT_EQ(ps.hitGroupRecordCount, 128u * 8u);
}

TEST_F(MeshBackendTest, SynchronizesHostExactlyOnTheCompactionPath) {
    {
        auto h = makeHarness("", nullptr, false); // OBB on, compaction_min_poses 128
        EXPECT_FALSE(h->backend->synchronizesHost(1));
        EXPECT_FALSE(h->backend->synchronizesHost(127));
        EXPECT_TRUE(h->backend->synchronizesHost(128));
    }
    {
        auto h = makeHarness("mesh.compaction_min_poses=0", nullptr, false);
        EXPECT_TRUE(h->backend->synchronizesHost(1));
        EXPECT_TRUE(h->backend->synchronizesHost(128));
    }
    {
        auto h = makeHarness("mesh.compaction_min_poses=129", nullptr, false);
        EXPECT_FALSE(h->backend->synchronizesHost(128));
    }
    {
        auto h = makeHarness("obb_culling=false", nullptr, false);
        EXPECT_FALSE(h->backend->synchronizesHost(1));
        EXPECT_FALSE(h->backend->synchronizesHost(128));
    }
}

// The golden on every prepare path, and identical per-item hit counts across them: an instance whose link OBB
// overlaps no obstacle OBB cannot be hit by an obstacle edge, so culling (masked or compacted) never changes a count.
TEST_F(MeshBackendTest, GoldenOnEveryPathWithIdenticalHitCounts) {
    struct Variant {
        const char* name;
        const char* overrides;
    };
    const Variant variants[] = {
        {"default (masked for partial batches, compaction for full ones)", ""},
        {"always compact", "mesh.compaction_min_poses=0"},
        {"never compact (masked only)", "mesh.compaction_min_poses=100000"},
        {"OBB culling off (refit + park transform)", "obb_culling=false"},
        {"batch 37, compaction from 20, 3 streams", "max_batch_size=37,mesh.compaction_min_poses=20,streams=3"},
    };
    std::vector<std::uint32_t> reference;
    for (const Variant& v : variants) {
        SCOPED_TRACE(v.name);
        auto h                                  = makeHarness(v.overrides);
        const std::vector<std::uint32_t> counts = h->runner->runAll(inputs_, kItems);
        const support::Flags flags              = support::flagsFromCounts(counts);
        const auto d                            = support::diffFlags(flags, golden_, kItems);
        EXPECT_TRUE(d.equal()) << d.differing << " diffs vs golden";
        EXPECT_EQ(support::countSet(flags), 1365u);
        // deterministic on repeat
        EXPECT_EQ(h->runner->runAll(inputs_, kItems), counts);
        // a prefix (partial last batch) agrees
        const std::vector<std::uint32_t> prefix = h->runner->runAll(inputs_, 517);
        EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), counts.begin()));
        if (reference.empty()) {
            reference = counts;
        } else {
            EXPECT_EQ(counts, reference) << "hit counts differ from the default variant";
        }
    }
}

// Scene far away from the robot: on the compaction path no instance survives, prepare() returns launch == false
// after its host sync; on the masked path every instance is masked out and the launch reports no hits.
TEST_F(MeshBackendTest, NoSurvivorsGivesNoLaunchOnCompactionPathAndZeroHitsOnMaskedPath) {
    SceneDesc far = setup_->scene;
    far.obstacles.resize(1);
    far.obstacles[0].pose = Transform::translation(20.0f, 20.0f, 20.0f);
    auto farScene         = std::make_shared<const Scene>(ctx_, far);

    auto h                  = makeHarness("", farScene);
    const std::uint32_t dof = robot_->dof();
    const std::uint32_t B   = h->config.maxBatchSize;
    DeviceBuffer items      = DeviceBuffer::fromHost(std::span<const float>(inputs_.data(), std::size_t(B) * dof));
    DeviceBuffer hits(std::size_t(B) * sizeof(std::uint32_t));
    const StreamContext& sc = h->streams[0];

    for (const std::uint32_t n : {B, 5u}) {
        SCOPED_TRACE(n);
        hits.memsetAsync(0, std::size_t(B) * sizeof(std::uint32_t), sc.stream);
        BatchView view;
        view.batchIndex      = 0;
        view.firstItem       = 0;
        view.itemCount       = n;
        view.items           = DeviceSpan<const float>{items.ptr(), std::size_t(n) * dof};
        view.hitCounts       = DeviceSpan<std::uint32_t>{hits.ptr(), n};
        const LaunchPlan plan = h->backend->prepare(sc, view);
        EXPECT_EQ(plan.launch, n >= h->config.mesh.compactionMinPoses ? false : true);
        EXPECT_EQ(plan.paramsBytes, sizeof(EdgeRayParams));
        EXPECT_EQ(plan.width, farScene->edgeRays(EdgeSet::Unique).count);
        h->exec->launch(sc.stream, plan);
        ASSERT_EQ(cudaStreamSynchronize(sc.stream), cudaSuccess);
        std::vector<std::uint32_t> host(n, 1);
        hits.download(host.data(), host.size() * sizeof(std::uint32_t));
        EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](std::uint32_t c) { return c == 0; }));
    }

    // Same scene, full run through the runner: everything collision free.
    auto r                                  = makeHarness("", farScene);
    const std::vector<std::uint32_t> counts = r->runner->runAll(inputs_, kItems);
    EXPECT_TRUE(std::all_of(counts.begin(), counts.end(), [](std::uint32_t c) { return c == 0; }));
}

TEST_F(MeshBackendTest, LaunchParamsDescribeTheBatch) {
    auto h                  = makeHarness("", nullptr, false);
    const std::uint32_t dof = robot_->dof();
    DeviceBuffer items      = DeviceBuffer::fromHost(std::span<const float>(inputs_.data(), std::size_t(128) * dof));
    DeviceBuffer hits(128 * sizeof(std::uint32_t));
    const StreamContext& sc = h->streams[1];
    const EdgeRays& edges   = scene_->edgeRays(EdgeSet::Unique);
    for (const std::uint32_t n : {128u, 7u}) {
        hits.memsetAsync(0, 128 * sizeof(std::uint32_t), sc.stream);
        BatchView view;
        view.itemCount        = n;
        view.items            = DeviceSpan<const float>{items.ptr(), std::size_t(n) * dof};
        view.hitCounts        = DeviceSpan<std::uint32_t>{hits.ptr(), n};
        const LaunchPlan plan = h->backend->prepare(sc, view);
        ASSERT_EQ(cudaStreamSynchronize(sc.stream), cudaSuccess);
        ASSERT_TRUE(plan.launch);
        ASSERT_EQ(plan.paramsBytes, sizeof(EdgeRayParams));
        EdgeRayParams p;
        std::memcpy(&p, plan.params, sizeof(p));
        EXPECT_EQ(p.traversable, plan.traversable);
        EXPECT_NE(p.traversable, 0u);
        EXPECT_EQ(p.edgeVertices, edges.vertices.ptr);
        EXPECT_EQ(p.edgeCount, edges.count);
        EXPECT_EQ(p.hitCounts, hits.ptr());
        EXPECT_NE(p.primToItem, 0u);
        EXPECT_EQ(plan.width, edges.count);
        EXPECT_EQ(plan.height, 1u);
        EXPECT_EQ(plan.depth, 1u);
        EXPECT_EQ(plan.raygen, 0u);
    }
}

TEST_F(MeshBackendTest, ConstructionErrors) {
    {
        mesh_test::Harness h;
        h.config = config();
        h.lease  = ctx_->acquireStreams(h.config.numStreams);
        for (std::uint32_t i = 0; i < h.lease.size(); ++i) {
            h.streams.push_back(StreamContext{i, h.lease[i]});
        }
        BackendCreateInfo info = createInfo(h);

        CheckerConfig ray = config("mode=ray");
        info.config       = &ray;
        EXPECT_THROW(createMeshBackend(info), InvalidArgument);

        CheckerConfig unresolved = setup_->checker; // obb_culling auto, scene_edges auto
        info.config              = &unresolved;
        EXPECT_THROW(createMeshBackend(info), InvalidArgument);

        info.config = &h.config;
        info.streams = std::span<const StreamContext>(h.streams.data(), h.streams.size() - 1);
        EXPECT_THROW(createMeshBackend(info), InvalidArgument);

        std::vector<StreamContext> swapped = h.streams;
        std::swap(swapped[0], swapped[1]);
        info.streams = swapped;
        EXPECT_THROW(createMeshBackend(info), InvalidArgument);

        info.streams = h.streams;
        info.robot   = nullptr;
        EXPECT_THROW(createMeshBackend(info), InvalidArgument);
        info.robot = robot_;
        info.scene = nullptr;
        EXPECT_THROW(createMeshBackend(info), InvalidArgument);

        // scene built on another Context
        auto other      = Context::create();
        auto otherScene = std::make_shared<const Scene>(other, setup_->scene);
        info.scene      = otherScene;
        EXPECT_THROW(createMeshBackend(info), InvalidArgument);
        ctx_->makeCurrent();
    }
}

TEST_F(MeshBackendTest, PrepareArgumentErrors) {
    auto h                  = makeHarness("", nullptr, false);
    const std::uint32_t dof = robot_->dof();
    DeviceBuffer items      = DeviceBuffer::fromHost(std::span<const float>(inputs_.data(), std::size_t(129) * dof));
    DeviceBuffer hits(129 * sizeof(std::uint32_t));
    BatchView view;
    view.items     = DeviceSpan<const float>{items.ptr(), std::size_t(129) * dof};
    view.hitCounts = DeviceSpan<std::uint32_t>{hits.ptr(), 129};

    view.itemCount = 0;
    EXPECT_THROW(h->backend->prepare(h->streams[0], view), CapacityError);
    view.itemCount = 129; // > max_batch_size 128
    EXPECT_THROW(h->backend->prepare(h->streams[0], view), CapacityError);

    view.itemCount = 4;
    EXPECT_THROW(h->backend->prepare(StreamContext{99, h->streams[0].stream}, view), InvalidArgument);
    EXPECT_THROW(h->backend->prepare(StreamContext{0, h->streams[1].stream}, view), InvalidArgument);
    view.items.count = 3 * dof; // fewer floats than items
    EXPECT_THROW(h->backend->prepare(h->streams[0], view), InvalidArgument);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
