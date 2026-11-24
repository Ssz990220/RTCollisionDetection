// new_gpu: ReferenceExecutor / ReferenceBatchRunner (rtcd/internal/reference_executor.h) with a trivial program
// module (reference_test_program.cu) and a fake Backend. Verifies what the Wave 2 backend agents rely on:
// raygen selection by LaunchPlan::raygen, params upload, SBT (raygen / miss / hit records), traversal + any-hit,
// the batch loop of rtcd/backend.h (batch -> stream mapping, staging copy, zeroing, result copy at firstItem,
// beginCheck), launch == false, and argument checks.
#include <rtcd/internal/reference_executor.h>

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>

#include <rtcd_testsupport_test_ir_embedded.h>

#include "reference_test_params.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using namespace rtcd;
using rtcd::internal::ReferenceBatchRunner;
using rtcd::internal::ReferenceExecutor;
using rtcd_test::TestParams;

namespace {

    constexpr std::uint32_t kRaygenCopy  = 0;
    constexpr std::uint32_t kRaygenTrace = 1;

    ProgramSet testPrograms() {
        ProgramSet ps;
        ps.moduleCode = std::span<const unsigned char>(rtcd::embedded::rtcd_testsupport_test_ir_reference_test_program,
            rtcd::embedded::rtcd_testsupport_test_ir_reference_test_program_size);
        ps.moduleName          = "reference_test_program";
        ps.raygens             = {"__raygen__copyItems", "__raygen__trace"};
        ps.anyHit              = "__anyhit__count";
        ps.traversableGraph    = TraversableGraph::SingleGas;
        ps.maxTraversableDepth = 1;
        ps.numPayloadValues    = 0;
        ps.numAttributeValues  = 2;
        ps.hitGroupRecordCount = 1;
        return ps;
    }

    // Two triangles covering x in [0, 1], y in [-1, 1] at z = 0 (diagonal (0,-1)-(1,1) crosses y = 0 at x = 0.5).
    struct TestGas {
        DeviceBuffer vertices, output;
        TraversableHandle handle = 0;
        explicit TestGas(const Context& ctx) {
            const float v[] = {0, -1, 0, 1, -1, 0, 1, 1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0};
            vertices        = DeviceBuffer::fromHost(std::span<const float>(v));
            CUdeviceptr vp  = vertices.ptr();
            unsigned int geomFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
            OptixBuildInput bi{};
            bi.type                                  = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            bi.triangleArray.vertexBuffers           = &vp;
            bi.triangleArray.numVertices             = 6;
            bi.triangleArray.vertexFormat            = OPTIX_VERTEX_FORMAT_FLOAT3;
            bi.triangleArray.vertexStrideInBytes     = 3 * sizeof(float);
            bi.triangleArray.flags                   = geomFlags;
            bi.triangleArray.numSbtRecords           = 1;
            OptixAccelBuildOptions ao{};
            ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            ao.operation  = OPTIX_BUILD_OPERATION_BUILD;
            OptixAccelBufferSizes sizes{};
            RTCD_OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx.optixContext(), &ao, &bi, 1, &sizes));
            DeviceBuffer temp(sizes.tempSizeInBytes);
            output.allocate(sizes.outputSizeInBytes);
            RTCD_OPTIX_CHECK(optixAccelBuild(ctx.optixContext(), nullptr, &ao, &bi, 1, temp.ptr(), temp.size(),
                output.ptr(), output.size(), &handle, nullptr, 0));
            RTCD_CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    // Backend whose prepare() only fills TestParams for the batch (no GPU work of its own).
    class FakeBackend final : public Backend {
    public:
        FakeBackend(std::uint32_t streams, std::uint32_t floatsPerItem) : params_(streams), fpi_(floatsPerItem) {
            programs_ = testPrograms();
        }
        Mode mode() const noexcept override { return Mode::LinearCurve; }
        const ProgramSet& programs() const noexcept override { return programs_; }
        void beginCheck(const CheckInfo& info) override { begins.push_back(info); }
        LaunchPlan prepare(const StreamContext& stream, const BatchView& batch) override {
            calls.push_back({stream.index, batch.batchIndex, batch.firstItem, batch.itemCount});
            EXPECT_EQ(batch.items.count, std::size_t(batch.itemCount) * fpi_);
            EXPECT_EQ(batch.hitCounts.count, batch.itemCount);
            TestParams& p = params_.at(stream.index);
            p             = {};
            p.traversable   = traversable;
            p.items         = batch.items.ptr;
            p.hitCounts     = batch.hitCounts.ptr;
            p.itemCount     = batch.itemCount;
            p.floatsPerItem = fpi_;
            p.addValue      = addValue;
            LaunchPlan plan;
            plan.launch      = batch.batchIndex != skipBatch;
            plan.traversable = traversable;
            plan.width       = batch.itemCount;
            plan.params      = &p;
            plan.paramsBytes = sizeof(TestParams);
            plan.raygen      = raygen;
            return plan;
        }
        bool synchronizesHost(std::uint32_t) const noexcept override { return false; }

        struct Call {
            std::uint32_t stream, batch, first, count;
        };
        std::vector<CheckInfo> begins;
        std::vector<Call> calls;
        std::uint32_t raygen    = kRaygenCopy;
        std::uint32_t addValue  = 0;
        std::uint32_t skipBatch = ~0u;
        TraversableHandle traversable = 0;

    private:
        ProgramSet programs_;
        std::vector<TestParams> params_;
        std::uint32_t fpi_;
    };

    struct Fixture : ::testing::Test {
        static constexpr std::uint32_t kStreams = 3;
        static constexpr std::uint32_t kDof     = 2;
        std::shared_ptr<Context> ctx;
        StreamLease lease;
        std::vector<StreamContext> streams;
        CheckerConfig cfg;
        std::uint32_t fpi = 0;

        void SetUp() override {
            ctx   = Context::create();
            lease = ctx->acquireStreams(kStreams);
            for (std::uint32_t i = 0; i < kStreams; ++i) {
                streams.push_back(StreamContext{i, lease[i]});
            }
            cfg.mode                   = Mode::LinearCurve; // items are trajectories: floatsPerItem = 3 * 2
            cfg.curve.trajectoryPoints = 3;
            cfg.numStreams             = kStreams;
            cfg.maxBatchSize           = 5;
            cfg.maxItemsPerCheck       = 64;
            cfg.optix                  = OptixOptions::release();
            fpi                        = static_cast<std::uint32_t>(cfg.itemLayout(kDof).floatsPerItem());
        }
        // item i's first float = first(i), the other floats are noise
        std::vector<float> items(std::uint32_t n, float (*first)(std::uint32_t)) const {
            std::vector<float> v(std::size_t(n) * fpi, 0.5f);
            for (std::uint32_t i = 0; i < n; ++i) {
                v[std::size_t(i) * fpi] = first(i);
                v[std::size_t(i) * fpi + 1] = -3.0f;
            }
            return v;
        }
    };

} // namespace

TEST_F(Fixture, BatchLoopCopiesStagesZeroesAndMapsResults) {
    FakeBackend backend(kStreams, fpi);
    backend.addValue = 7;
    ReferenceExecutor exec(ctx, backend.programs(), cfg.optix);
    ReferenceBatchRunner runner(ctx, backend, exec, cfg, streams, kDof);

    const std::uint32_t n = 23; // 5 batches: 4 full + 3 items, streams 0 1 2 0 1
    const auto in         = items(n, [](std::uint32_t i) { return float(100 + i); });
    for (int rep = 0; rep < 2; ++rep) { // second run: counters must have been zeroed again
        const std::vector<std::uint32_t> counts = runner.runAll(in, n);
        ASSERT_EQ(counts.size(), n);
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(counts[i], 107 + i) << "item " << i << " rep " << rep;
        }
    }
    ASSERT_EQ(backend.begins.size(), 2u);
    EXPECT_EQ(backend.begins[0].itemCount, n);
    EXPECT_EQ(backend.begins[0].batchCount, 5u);
    ASSERT_EQ(backend.calls.size(), 10u);
    for (std::uint32_t b = 0; b < 5; ++b) {
        EXPECT_EQ(backend.calls[b].stream, b % kStreams);
        EXPECT_EQ(backend.calls[b].batch, b);
        EXPECT_EQ(backend.calls[b].first, b * 5);
        EXPECT_EQ(backend.calls[b].count, b < 4 ? 5u : 3u);
    }
}

TEST_F(Fixture, PartialChecksAndEmptyCheck) {
    FakeBackend backend(kStreams, fpi);
    ReferenceExecutor exec(ctx, backend.programs(), cfg.optix);
    ReferenceBatchRunner runner(ctx, backend, exec, cfg, streams, kDof);
    const auto in = items(64, [](std::uint32_t i) { return float(3 * i + 1); });
    for (std::uint32_t n : {1u, 4u, 5u, 6u, 16u, 64u}) {
        const auto counts = runner.runAll(in, n);
        ASSERT_EQ(counts.size(), n);
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(counts[i], 3 * i + 1) << "n " << n << " item " << i;
        }
    }
    EXPECT_TRUE(runner.runAll(in, 0).empty());
}

TEST_F(Fixture, TraceRaygenAndAnyHitThroughGas) {
    TestGas gas(*ctx);
    FakeBackend backend(kStreams, fpi);
    backend.raygen      = kRaygenTrace;
    backend.traversable = gas.handle;
    ReferenceExecutor exec(ctx, backend.programs(), cfg.optix);
    ReferenceBatchRunner runner(ctx, backend, exec, cfg, streams, kDof);
    // x = 0.25 / 0.75 hit one triangle each, 1.5 and -0.5 miss
    const auto in = items(17, [](std::uint32_t i) {
        const float xs[] = {0.25f, 1.5f, 0.75f, -0.5f};
        return xs[i % 4];
    });
    const auto counts = runner.runAll(in, 17);
    for (std::uint32_t i = 0; i < 17; ++i) {
        EXPECT_EQ(counts[i], (i % 4 == 0 || i % 4 == 2) ? 1u : 0u) << "item " << i;
    }
}

TEST_F(Fixture, LaunchFalseLeavesZeroCounts) {
    FakeBackend backend(kStreams, fpi);
    backend.skipBatch = 1; // items 5 .. 9
    ReferenceExecutor exec(ctx, backend.programs(), cfg.optix);
    ReferenceBatchRunner runner(ctx, backend, exec, cfg, streams, kDof);
    const auto in     = items(12, [](std::uint32_t i) { return float(i + 1); });
    const auto counts = runner.runAll(in, 12);
    for (std::uint32_t i = 0; i < 12; ++i) {
        EXPECT_EQ(counts[i], (i >= 5 && i < 10) ? 0u : i + 1) << "item " << i;
    }
}

TEST_F(Fixture, ArgumentErrors) {
    FakeBackend backend(kStreams, fpi);
    ReferenceExecutor exec(ctx, backend.programs(), cfg.optix);
    ReferenceBatchRunner runner(ctx, backend, exec, cfg, streams, kDof);
    const auto in = items(64, [](std::uint32_t i) { return float(i); });

    EXPECT_THROW(runner.runAll(in, 65), CapacityError);
    EXPECT_THROW(runner.runAll(std::span<const float>(in.data(), 10), 5), InvalidArgument);

    TestParams p{};
    LaunchPlan plan;
    plan.width       = 1;
    plan.params      = &p;
    plan.paramsBytes = sizeof(p);
    plan.raygen      = 2; // only 2 raygens
    EXPECT_THROW(exec.launch(streams[0].stream, plan), InvalidArgument);
    plan.raygen      = 0;
    plan.paramsBytes = kMaxLaunchParamsBytes + 1;
    EXPECT_THROW(exec.launch(streams[0].stream, plan), InvalidArgument);
    plan.paramsBytes = sizeof(p);
    plan.width       = 0;
    EXPECT_THROW(exec.launch(streams[0].stream, plan), InvalidArgument);
    plan.launch = false; // nothing is checked or launched
    exec.launch(streams[0].stream, plan);

    ProgramSet bad = testPrograms();
    bad.raygens.clear();
    EXPECT_THROW(ReferenceExecutor(ctx, bad, cfg.optix), InvalidArgument);

    std::vector<StreamContext> wrongIndex = streams;
    wrongIndex[1].index                   = 2;
    EXPECT_THROW(ReferenceBatchRunner(ctx, backend, exec, cfg, wrongIndex, kDof), InvalidArgument);
}

TEST_F(Fixture, BadEntryPointThrowsOptixError) {
    ProgramSet bad = testPrograms();
    bad.raygens    = {"__raygen__doesNotExist"};
    EXPECT_THROW(ReferenceExecutor(ctx, bad, cfg.optix), OptixError);
}
