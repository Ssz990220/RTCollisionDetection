// new_gpu: the detector core (src/detector, DetectorCore) with a fake backend (fake_backend.h) and a trivial
// program module (detector_test_program.cu). Covers: program-set handling (raygen selection, miss / closest-hit /
// any-hit, built-in sphere IS), batch loop order, stream mapping, worker threads and current CUDA context,
// param-upload cache, launch == false, stats, error propagation and stream draining, capacity / argument errors,
// HostDispatch both ways, checkDevice, strided input, warmup, and a cross-check against ReferenceBatchRunner.
#include "detector/detector_core.h"

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/reference_executor.h>

#include "fake_backend.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

using namespace rtcd;
using namespace rtcd_detector_test;
using rtcd::detail::DetectorCore;
using rtcd::detail::DetectorCoreDesc;

namespace {

    // Two triangles covering x in [0, 1], y in [-1, 1] at z = 0 (as in test_reference_executor).
    struct TriangleGas {
        DeviceBuffer vertices, output;
        TraversableHandle handle = 0;
        explicit TriangleGas(const Context& ctx) {
            const float v[] = {0, -1, 0, 1, -1, 0, 1, 1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0};
            vertices        = DeviceBuffer::fromHost(std::span<const float>(v));
            CUdeviceptr vp  = vertices.ptr();
            unsigned int geomFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
            OptixBuildInput bi{};
            bi.type                              = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            bi.triangleArray.vertexBuffers       = &vp;
            bi.triangleArray.numVertices         = 6;
            bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            bi.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
            bi.triangleArray.flags               = geomFlags;
            bi.triangleArray.numSbtRecords       = 1;
            build(ctx, bi, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        }
        void build(const Context& ctx, const OptixBuildInput& bi, unsigned int flags) {
            OptixAccelBuildOptions ao{};
            ao.buildFlags = flags;
            ao.operation  = OPTIX_BUILD_OPERATION_BUILD;
            OptixAccelBufferSizes sizes{};
            RTCD_OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx.optixContext(), &ao, &bi, 1, &sizes));
            DeviceBuffer temp(sizes.tempSizeInBytes);
            output.allocate(sizes.outputSizeInBytes);
            RTCD_OPTIX_CHECK(optixAccelBuild(ctx.optixContext(), nullptr, &ao, &bi, 1, temp.ptr(), temp.size(),
                output.ptr(), output.size(), &handle, nullptr, 0));
            RTCD_CUDA_CHECK(cudaDeviceSynchronize());
        }
        TriangleGas() = default;
    };

    // One sphere at (0.5, 0, 0), radius 0.25, built with kSphereGasFlags (the IS module must use the same flags).
    constexpr unsigned int kSphereGasFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    struct SphereGas : TriangleGas {
        DeviceBuffer centers, radii;
        explicit SphereGas(const Context& ctx) {
            const float c[] = {0.5f, 0.0f, 0.0f};
            const float r[] = {0.25f};
            centers         = DeviceBuffer::fromHost(std::span<const float>(c));
            radii           = DeviceBuffer::fromHost(std::span<const float>(r));
            CUdeviceptr cp  = centers.ptr();
            CUdeviceptr rp  = radii.ptr();
            unsigned int geomFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
            OptixBuildInput bi{};
            bi.type                          = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            bi.sphereArray.vertexBuffers     = &cp;
            bi.sphereArray.numVertices       = 1;
            bi.sphereArray.radiusBuffers     = &rp;
            bi.sphereArray.flags             = geomFlags;
            bi.sphereArray.numSbtRecords     = 1;
            build(ctx, bi, kSphereGasFlags);
        }
    };

    class CoreTest : public ::testing::TestWithParam<HostDispatch> {
    protected:
        static constexpr std::uint32_t kStreams = 3;
        static constexpr std::uint32_t kDof     = 2;
        static constexpr std::uint32_t kBatch   = 5;
        static constexpr std::uint32_t kMaxItems = 64;
        std::shared_ptr<Context> ctx;
        CheckerConfig cfg;
        std::uint32_t fpi = 0;

        void SetUp() override {
            ctx                        = Context::create();
            CheckerConfig c;
            c.mode                     = Mode::LinearCurve; // items are trajectories: floatsPerItem = 3 * 2
            c.curve.trajectoryPoints   = 3;
            c.numStreams               = kStreams;
            c.hostDispatch             = GetParam();
            c.maxBatchSize             = kBatch;
            c.maxItemsPerCheck         = kMaxItems;
            c.optix                    = OptixOptions::release();
            cfg                        = c.resolved();
            fpi                        = static_cast<std::uint32_t>(cfg.itemLayout(kDof).floatsPerItem());
        }

        std::unique_ptr<DetectorCore> makeCore(FakeOptions opts, FakeBackend** fake, const CheckerConfig* config = nullptr) {
            opts.dof = kDof;
            DetectorCoreDesc d;
            d.context       = ctx;
            d.config        = config ? *config : cfg;
            d.dof           = kDof;
            d.lowerLimits   = {-1.0f, 0.5f};
            d.upperLimits   = {1.0f, 2.5f};
            d.createBackend = makeFactory(opts, fake);
            return std::make_unique<DetectorCore>(std::move(d));
        }

        // item i: first float = first(i), the others noise
        std::vector<float> items(std::uint32_t n, float (*first)(std::uint32_t), std::size_t stride = 0) const {
            const std::size_t st = stride ? stride : fpi;
            std::vector<float> v(std::size_t(n) * st, 0.5f);
            for (std::uint32_t i = 0; i < n; ++i) {
                v[std::size_t(i) * st]     = first(i);
                v[std::size_t(i) * st + 1] = -3.0f;
            }
            return v;
        }

        static std::vector<std::uint32_t> counts(DetectorCore& core, const std::vector<float>& in, std::uint32_t n,
            std::size_t stride = 0) {
            std::vector<std::uint8_t> flags(n, 7);
            std::vector<std::uint32_t> c(n, 12345);
            core.checkHost(HostItems{in, n, stride}, flags, c);
            for (std::uint32_t i = 0; i < n; ++i) {
                EXPECT_EQ(flags[i], c[i] > 0 ? 1 : 0) << "item " << i;
            }
            return c;
        }
        static float plus100(std::uint32_t i) { return float(100 + i); }
        static float times3(std::uint32_t i) { return float(3 * i + 1); }
    };

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Batch loop, stream mapping, threads, current context
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, BatchLoopOrderStreamMappingAndResults) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.addValue = 7;
    auto core  = makeCore(o, &fake);
    ASSERT_NE(fake, nullptr);
    ASSERT_EQ(fake->streams.size(), kStreams);
    for (std::uint32_t s = 0; s < kStreams; ++s) {
        EXPECT_EQ(fake->streams[s].index, s);
        EXPECT_EQ(fake->streams[s].stream, core->streams()[s].stream);
    }

    const std::uint32_t n = 23; // 5 batches: 4 full + 3 items, streams 0 1 2 0 1
    const auto in         = items(n, plus100);
    for (int rep = 0; rep < 2; ++rep) { // second run: counters must be zeroed again
        const auto c = counts(*core, in, n);
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(c[i], 107 + i) << "item " << i << " rep " << rep;
        }
    }
    ASSERT_EQ(fake->begins.size(), 2u);
    EXPECT_EQ(fake->begins[0].itemCount, n);
    EXPECT_EQ(fake->begins[0].batchCount, 5u);
    ASSERT_EQ(fake->calls.size(), 10u);

    const std::thread::id caller = std::this_thread::get_id();
    std::vector<std::vector<std::uint32_t>> perStream(kStreams);
    for (std::size_t k = 0; k < 5; ++k) { // first check
        const auto& c = fake->calls[k];
        perStream[c.streamIndex].push_back(c.batch);
        EXPECT_EQ(c.streamIndex, c.batch % kStreams);
        EXPECT_EQ(c.stream, core->streams()[c.streamIndex].stream);
        EXPECT_EQ(c.first, c.batch * kBatch);
        EXPECT_EQ(c.count, c.batch < 4 ? kBatch : 3u);
        EXPECT_EQ(c.itemsCount, std::size_t(c.count) * fpi);
        EXPECT_EQ(c.hitCountsCount, c.count);
        EXPECT_TRUE(c.contextCurrent) << "batch " << c.batch << ": the Context was not current in prepare()";
        if (GetParam() == HostDispatch::SingleThread) {
            EXPECT_EQ(c.thread, caller);
            EXPECT_EQ(c.batch, k); // strictly in batch order
        } else {
            EXPECT_NE(c.thread, caller);
            EXPECT_EQ(c.thread, core->workerThreadId(c.streamIndex)); // worker s runs stream s
        }
    }
    // per stream: batches s, s + S, ... in order
    EXPECT_EQ(perStream[0], (std::vector<std::uint32_t>{0, 3}));
    EXPECT_EQ(perStream[1], (std::vector<std::uint32_t>{1, 4}));
    EXPECT_EQ(perStream[2], (std::vector<std::uint32_t>{2}));
    // Packed input: every batch reads its items in place, at the batch's first item of one packed buffer.
    for (std::size_t k = 0; k < 10; ++k) {
        const auto& c   = fake->calls[k];
        const std::size_t check = k / 5; // calls of one check are calls[5 * check .. 5 * check + 4]
        DevicePtr base  = 0;
        for (std::size_t j = 5 * check; j < 5 * check + 5; ++j) {
            if (fake->calls[j].batch == 0) {
                base = fake->calls[j].items;
            }
        }
        EXPECT_EQ(c.items - base, DevicePtr(c.first) * fpi * sizeof(float)) << "call " << k;
    }
    // Hit counters of a host check: its results go to a pinned host buffer, which is never counted into directly,
    // so every batch uses its stream's own counters: one address per stream in both checks. (Device checks and
    // counting into the result range: AddressRulesItemsInPlaceAndDirectHitCounts.)
    for (std::size_t k = 5; k < 10; ++k) {
        const auto& c = fake->calls[k];
        for (std::size_t j = 0; j < 5; ++j) {
            if (fake->calls[j].streamIndex == c.streamIndex) {
                EXPECT_EQ(fake->calls[j].hitCounts, c.hitCounts);
            }
        }
    }

    const CheckStats st = core->lastStats();
    EXPECT_EQ(st.items, n);
    EXPECT_EQ(st.batches, 5u);
    EXPECT_EQ(st.launches, 5u);
    EXPECT_EQ(st.hostSyncs, 0u);
}

TEST_P(CoreTest, PartialAndEmptyChecks) {
    FakeBackend* fake = nullptr;
    auto core         = makeCore({}, &fake);
    const auto in     = items(kMaxItems, times3);
    for (std::uint32_t n : {1u, 4u, 5u, 6u, 15u, 16u, 37u, 64u}) {
        const auto c = counts(*core, in, n);
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(c[i], 3 * i + 1) << "n " << n << " item " << i;
        }
        EXPECT_EQ(core->lastStats().batches, (n + kBatch - 1) / kBatch);
    }
    const std::size_t beginsBefore = fake->begins.size();
    core->checkHost(HostItems{in, 0, 0}, {}, {});
    EXPECT_EQ(fake->begins.size(), beginsBefore); // nothing happens for an empty check
    EXPECT_EQ(core->lastStats().items, 0u);
    EXPECT_EQ(core->lastStats().batches, 0u);
}

// ---------------------------------------------------------------------------------------------------------------
// Program-set handling
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, RaygenSelectionPerPlan) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.addValue          = 1;
    o.includeBatchIndex = true;
    o.raygen            = [](std::uint32_t b) { return b % 2 == 0 ? kRaygenCopy : kRaygenCopyBatch; };
    auto core  = makeCore(o, &fake);
    const std::uint32_t n = 23;
    const auto in         = items(n, plus100);
    const auto c          = counts(*core, in, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t b = i / kBatch;
        EXPECT_EQ(c[i], b % 2 == 0 ? 101 + i : 100 + i + 1000 * (b + 1)) << "item " << i;
    }
}

TEST_P(CoreTest, AnyHitClosestHitAndMissPrograms) {
    TriangleGas gas(*ctx);
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.traversable = gas.handle;
    o.raygen      = [](std::uint32_t b) { return b % 2 == 0 ? kRaygenTrace : kRaygenTraceChMiss; };
    auto core     = makeCore(o, &fake);
    // x = 0.25 / 0.75 hit one triangle each, 1.5 and -0.5 miss
    const std::uint32_t n = 17;
    const auto in         = items(n, [](std::uint32_t i) {
        const float xs[] = {0.25f, 1.5f, 0.75f, -0.5f};
        return xs[i % 4];
    });
    const auto c = counts(*core, in, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const bool hit = i % 4 == 0 || i % 4 == 2;
        if ((i / kBatch) % 2 == 0) {
            EXPECT_EQ(c[i], hit ? 1u : 0u) << "any-hit, item " << i;
        } else {
            EXPECT_EQ(c[i], hit ? 5u : 3u) << "closest-hit / miss, item " << i;
        }
    }
}

TEST_P(CoreTest, BuiltinSphereIntersection) {
    SphereGas gas(*ctx);
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.traversable  = gas.handle;
    o.raygen       = [](std::uint32_t) { return kRaygenTrace; };
    o.editPrograms = [](ProgramSet& ps) {
        ps.builtinIntersection = BuiltinIntersection::Sphere;
        ps.builtinIsBuildFlags = kSphereGasFlags;
        ps.primitiveTypes      = kPrimitiveSphere;
    };
    auto core             = makeCore(o, &fake);
    const std::uint32_t n = 12;
    const auto in         = items(n, [](std::uint32_t i) {
        const float xs[] = {0.5f, 0.1f, 0.6f, 0.9f};
        return xs[i % 4];
    });
    const auto c = counts(*core, in, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const bool hit = i % 4 == 0 || i % 4 == 2;
        EXPECT_EQ(c[i] > 0, hit) << "item " << i << " count " << c[i];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Param-upload cache, launch == false, host-sync stats
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, ParamUploadCache) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.addValue = 2;
    auto core  = makeCore(o, &fake);
    const auto in = items(kMaxItems, times3);
    // The fake's params hold the items pointer. Packed items are read in place (a different pointer per batch),
    // so this test uses STRIDED device items: they go through each stream's staging buffer (one pointer per
    // stream). Two batches per stream into different result targets keep the hit counters in the stream's own
    // buffer too, so the params are identical per stream.
    const std::size_t stride = fpi + 1;
    const auto strided       = items(kMaxItems, times3, stride);
    DeviceBuffer dItems      = DeviceBuffer::fromHost(std::span<const float>(strided));
    DeviceBuffer dHits(std::size_t(kMaxItems) * sizeof(std::uint32_t));
    auto expectOk = [&](std::uint32_t n) {
        core->checkDevice(DeviceItems{dItems.ptr(), n, stride}, DeviceSpan<std::uint32_t>{dHits.ptr(), n});
        std::vector<std::uint32_t> c(n);
        dHits.download(c.data(), c.size() * sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(c[i], 3 * i + 3) << "n " << n << " item " << i;
        }
    };
    // 30 items = 6 full batches, 2 per stream; per stream the params are identical for all of them.
    expectOk(30);
    EXPECT_EQ(core->lastStats().paramUploads, kStreams);
    expectOk(30);
    EXPECT_EQ(core->lastStats().paramUploads, 0u); // the cache survives across checks
    expectOk(28); // batch 5 (stream 2) has 3 items: one new upload
    EXPECT_EQ(core->lastStats().paramUploads, 1u);
    expectOk(30); // stream 2 batch 2 differs from the cached 3-item params again; batch 5 then matches
    EXPECT_EQ(core->lastStats().paramUploads, 1u);
    EXPECT_EQ(core->lastStats().launches, 6u);

    FakeBackend* fake2 = nullptr;
    FakeOptions o2;
    o2.includeBatchIndex = true; // every batch's params differ
    o2.raygen            = [](std::uint32_t) { return kRaygenCopyBatch; };
    auto core2           = makeCore(o2, &fake2);
    for (int rep = 0; rep < 2; ++rep) {
        const auto c = counts(*core2, in, 30);
        for (std::uint32_t i = 0; i < 30; ++i) {
            EXPECT_EQ(c[i], 3 * i + 1 + 1000 * (i / kBatch + 1)) << "item " << i;
        }
        EXPECT_EQ(core2->lastStats().paramUploads, 6u);
    }
}

TEST_P(CoreTest, LaunchFalseAndHostSyncStats) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.skipBatch     = 1;      // items 5 .. 9
    o.syncThreshold = kBatch; // full batches "synchronize the host"
    auto core       = makeCore(o, &fake);
    const std::uint32_t n = 28;
    const auto in         = items(n, [](std::uint32_t i) { return float(i + 1); });
    for (int rep = 0; rep < 2; ++rep) {
        const auto c = counts(*core, in, n);
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(c[i], (i >= 5 && i < 10) ? 0u : i + 1) << "item " << i;
        }
        const CheckStats st = core->lastStats();
        EXPECT_EQ(st.batches, 6u);
        EXPECT_EQ(st.launches, 5u);
        EXPECT_EQ(st.hostSyncs, 5u); // the 5 full batches
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, CapacityAndArgumentErrorsBeforeGpuWork) {
    FakeBackend* fake = nullptr;
    auto core         = makeCore({}, &fake);
    const auto in     = items(kMaxItems + 1, times3);
    std::vector<std::uint8_t> flags(kMaxItems + 1);
    std::vector<std::uint32_t> hc(kMaxItems + 1);

    EXPECT_THROW(core->checkHost(HostItems{in, kMaxItems + 1, 0}, flags, hc), CapacityError);
    EXPECT_THROW(core->checkHost(HostItems{std::span<const float>(in.data(), 10), 5, 0}, flags, hc), InvalidArgument);
    EXPECT_THROW(core->checkHost(HostItems{in, 5, fpi - 1}, flags, hc), InvalidArgument); // stride < floatsPerItem
    EXPECT_THROW(core->checkHost(HostItems{{}, 5, 0}, flags, hc), InvalidArgument);          // null data
    EXPECT_THROW(core->checkHost(HostItems{in, 5, 0}, std::span<std::uint8_t>(flags.data(), 4), hc), InvalidArgument);
    EXPECT_THROW(core->checkHost(HostItems{in, 5, 0}, flags, std::span<std::uint32_t>(hc.data(), 4)), InvalidArgument);
    // strided view one float short
    EXPECT_THROW(core->checkHost(HostItems{std::span<const float>(in.data(), 4 * (fpi + 1) + fpi - 1), 5, fpi + 1}, flags, hc),
        InvalidArgument);

    DeviceBuffer dItems(std::size_t(kMaxItems + 1) * fpi * sizeof(float));
    DeviceBuffer dHits(std::size_t(kMaxItems + 1) * sizeof(std::uint32_t));
    EXPECT_THROW(core->checkDevice(DeviceItems{dItems.ptr(), kMaxItems + 1, 0}, dHits.span<std::uint32_t>()), CapacityError);
    EXPECT_THROW(core->checkDevice(DeviceItems{0, 5, 0}, dHits.span<std::uint32_t>()), InvalidArgument);
    EXPECT_THROW(core->checkDevice(DeviceItems{dItems.ptr() + 2, 5, 0}, dHits.span<std::uint32_t>()), InvalidArgument);
    EXPECT_THROW(core->checkDevice(DeviceItems{dItems.ptr(), 5, 0}, DeviceSpan<std::uint32_t>{dHits.ptr(), 4}),
        InvalidArgument);
    EXPECT_THROW(core->checkDevice(DeviceItems{dItems.ptr(), 5, 0}, DeviceSpan<std::uint32_t>{0, 5}), InvalidArgument);
    EXPECT_THROW(core->checkDevice(DeviceItems{dItems.ptr(), 5, 1}, dHits.span<std::uint32_t>()), InvalidArgument);

    EXPECT_TRUE(fake->begins.empty()); // nothing reached the backend
    EXPECT_TRUE(fake->calls.empty());
    const auto c = counts(*core, in, 9); // still usable
    for (std::uint32_t i = 0; i < 9; ++i) {
        EXPECT_EQ(c[i], 3 * i + 1);
    }
}

TEST_P(CoreTest, PrepareErrorDrainsStreamsAndRethrowsOnCaller) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.raygen       = [](std::uint32_t b) { return b == 0 ? kRaygenSpin : kRaygenCopy; };
    o.spinCycles   = 400'000'000; // ~0.15-0.2 s on stream 0: still running when batch 2 fails, unless drained
    o.throwAtBatch = 2;
    auto core      = makeCore(o, &fake);
    const std::uint32_t n = 40;
    const auto in         = items(n, times3);
    std::vector<std::uint8_t> flags(n);
    try {
        core->checkHost(HostItems{in, n, 0}, flags, {});
        ADD_FAILURE() << "no exception";
    } catch (const InvalidArgument& e) {
        EXPECT_NE(std::string(e.what()).find("fake prepare failure at batch 2"), std::string::npos) << e.what();
    }
    for (const StreamContext& s : core->streams()) {
        EXPECT_EQ(cudaStreamQuery(s.stream), cudaSuccess) << "stream " << s.index << " not drained";
    }
    if (GetParam() == HostDispatch::SingleThread) {
        EXPECT_EQ(fake->calls.size(), 3u); // batches 0, 1, 2: nothing after the failure
    }
}

TEST_P(CoreTest, ErrorsLeaveTheCheckerUsable) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.throwAtBatch = 3;
    auto core      = makeCore(o, &fake);
    const auto in  = items(kMaxItems, times3);
    std::vector<std::uint8_t> flags(kMaxItems);
    EXPECT_THROW(core->checkHost(HostItems{in, 30, 0}, flags, {}), InvalidArgument);
    // fewer batches: batch 3 never happens, results are right
    const auto c = counts(*core, in, 15);
    for (std::uint32_t i = 0; i < 15; ++i) {
        EXPECT_EQ(c[i], 3 * i + 1);
    }

    FakeBackend* fake2 = nullptr;
    FakeOptions o2;
    o2.throwInBeginCheck = true;
    auto core2           = makeCore(o2, &fake2);
    EXPECT_THROW(core2->checkHost(HostItems{in, 10, 0}, flags, {}), UnsupportedError);
    EXPECT_TRUE(fake2->calls.empty());

    FakeBackend* fake3 = nullptr;
    FakeOptions o3;
    o3.badRaygen = 1; // plan.raygen = 99: a backend bug
    auto core3   = makeCore(o3, &fake3);
    try {
        core3->checkHost(HostItems{in, 20, 0}, flags, {});
        ADD_FAILURE() << "no exception";
    } catch (const Error& e) {
        EXPECT_EQ(e.kind(), ErrorKind::Internal) << e.what();
    }
    for (const StreamContext& s : core3->streams()) {
        EXPECT_EQ(cudaStreamQuery(s.stream), cudaSuccess);
    }
}

TEST_P(CoreTest, ConstructionErrors) {
    {
        DetectorCoreDesc d;
        d.context       = ctx;
        d.config        = cfg;
        d.dof           = kDof;
        d.createBackend = [](const BackendCreateInfo&) -> std::unique_ptr<Backend> { return nullptr; };
        try {
            DetectorCore core(std::move(d));
            ADD_FAILURE() << "no exception";
        } catch (const Error& e) {
            EXPECT_EQ(e.kind(), ErrorKind::Internal);
        }
    }
    {
        DetectorCoreDesc d;
        d.context       = ctx;
        d.config        = cfg;
        d.dof           = kDof;
        d.createBackend = [](const BackendCreateInfo&) -> std::unique_ptr<Backend> {
            throw ConfigError("fake factory failure", "ray.launch_width_cap");
        };
        EXPECT_THROW(DetectorCore core(std::move(d)), ConfigError);
    }
    FakeOptions wrongMode;
    wrongMode.forceReportedMode = true;
    wrongMode.reportedMode      = Mode::Ray;
    EXPECT_THROW(makeCore(wrongMode, nullptr), Error);

    FakeOptions noRaygen;
    noRaygen.editPrograms = [](ProgramSet& ps) { ps.raygens.clear(); };
    EXPECT_THROW(makeCore(noRaygen, nullptr), Error);

    FakeOptions badEntry;
    badEntry.editPrograms = [](ProgramSet& ps) { ps.raygens = {"__raygen__doesNotExist"}; };
    EXPECT_THROW(makeCore(badEntry, nullptr), OptixError);

    FakeOptions sphereWithoutType;
    sphereWithoutType.editPrograms = [](ProgramSet& ps) { ps.builtinIntersection = BuiltinIntersection::Sphere; };
    EXPECT_THROW(makeCore(sphereWithoutType, nullptr), Error);

    // the Context is still fine
    FakeBackend* fake = nullptr;
    auto core         = makeCore({}, &fake);
    const auto in     = items(4, times3);
    EXPECT_EQ(counts(*core, in, 4)[3], 10u);
}

// ---------------------------------------------------------------------------------------------------------------
// Device path, strided input, warmup
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, CheckDeviceAndStridedInput) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.addValue = 4;
    auto core  = makeCore(o, &fake);
    const std::uint32_t n      = 37;
    const std::size_t stride   = fpi + 3;
    const auto packed          = items(n, plus100);
    const auto strided         = items(n, plus100, stride);

    const auto cPacked  = counts(*core, packed, n);
    const auto cStrided = counts(*core, strided, n, stride);
    EXPECT_EQ(cPacked, cStrided);

    for (std::size_t st : {std::size_t(0), stride}) {
        const auto& host = st ? strided : packed;
        DeviceBuffer dItems = DeviceBuffer::fromHost(std::span<const float>(host));
        DeviceBuffer dHits(std::size_t(n + 4) * sizeof(std::uint32_t));
        std::vector<std::uint32_t> init(n + 4, 0xdeadbeef);
        dHits.upload(init.data(), init.size() * sizeof(std::uint32_t));
        core->checkDevice(DeviceItems{dItems.ptr(), n, st}, DeviceSpan<std::uint32_t>{dHits.ptr(), n + 4});
        std::vector<std::uint32_t> out(n + 4);
        dHits.download(out.data(), out.size() * sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(out[i], 104 + i) << "stride " << st << " item " << i;
        }
        for (std::uint32_t i = n; i < n + 4; ++i) {
            EXPECT_EQ(out[i], 0xdeadbeefu) << "written past items.count";
        }
        EXPECT_EQ(core->lastStats().items, n);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// BatchView address rules (relaxed 2026-11-24, rtcd/backend.h): items read in place when the input is packed,
// hit counters counted straight into the result range when a stream's previous batch had the same target,
// single-batch checks on the calling thread. Results must be right in every combination.
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, AddressRulesItemsInPlaceAndDirectHitCounts) {
    for (const bool graph : {false, true}) {
        SCOPED_TRACE(graph ? "graph replay (capturable backend)" : "direct prepare()");
        FakeBackend* fake = nullptr;
        FakeOptions o;
        o.addValue            = 5;
        o.capturableThreshold = graph ? 1u : ~0u;
        auto core             = makeCore(o, &fake);
        const std::size_t fpiBytes = std::size_t(fpi) * sizeof(float);

        // two input buffers (different contents) and two result buffers, all caller-owned device memory
        const auto hostA = items(kMaxItems, plus100);
        const auto hostB = items(kMaxItems, times3);
        DeviceBuffer inA = DeviceBuffer::fromHost(std::span<const float>(hostA));
        DeviceBuffer inB = DeviceBuffer::fromHost(std::span<const float>(hostB));
        DeviceBuffer r0(std::size_t(kMaxItems) * sizeof(std::uint32_t)), r1(std::size_t(kMaxItems) * sizeof(std::uint32_t));
        auto expected = [&](bool a, std::uint32_t i) { return (a ? 100 + i : 3 * i + 1) + 5; };
        auto checkInto = [&](bool a, DeviceBuffer& r, std::uint32_t n, const char* what) {
            const std::vector<std::uint32_t> junk(kMaxItems, 0xdeadbeefu);
            r.upload(junk.data(), junk.size() * sizeof(std::uint32_t)); // the check must zero what it writes
            const std::size_t before = fake->calls.size();
            core->checkDevice(DeviceItems{(a ? inA : inB).ptr(), n, 0}, DeviceSpan<std::uint32_t>{r.ptr(), n});
            std::vector<std::uint32_t> out(kMaxItems);
            r.download(out.data(), out.size() * sizeof(std::uint32_t));
            for (std::uint32_t i = 0; i < n; ++i) {
                EXPECT_EQ(out[i], expected(a, i)) << what << ": n " << n << " item " << i;
            }
            for (std::uint32_t i = n; i < kMaxItems; ++i) {
                EXPECT_EQ(out[i], 0xdeadbeefu) << what << ": written past n at " << i;
            }
            return before; // index of this check's first prepare() call (graph replays add none)
        };

        // (1) single batch, repeated into the SAME result buffer: the first check uses the stream's own counters,
        // later ones count into r0 itself; the check runs on the calling thread in both dispatch modes.
        for (int rep = 0; rep < 3; ++rep) {
            const std::size_t k = checkInto(rep != 1, r0, 3, "single batch, same buffer");
            if (!graph || rep < 2) { // graph: rep 0 captures (own counters), rep 1 captures (r0), rep 2 replays
                ASSERT_EQ(fake->calls.size(), k + 1) << "rep " << rep;
                const auto& c = fake->calls[k];
                EXPECT_EQ(c.thread, std::this_thread::get_id()) << "a single-batch check runs on the caller";
                if (rep == 0) {
                    EXPECT_NE(c.hitCounts, r0.ptr()) << "first check into r0: the stream's own counters";
                } else {
                    EXPECT_EQ(c.hitCounts, r0.ptr()) << "repeated target: counted into the result directly";
                }
                if (graph) {
                    EXPECT_NE(c.items, inA.ptr()) << "graph batches read the staging buffer";
                } else {
                    EXPECT_EQ(c.items, (rep != 1 ? inA : inB).ptr()) << "packed input is read in place";
                }
            } else {
                EXPECT_EQ(fake->calls.size(), k) << "replayed graph (stream 0, 3 items, r0)";
            }
        }
        // (2) alternating result buffers r1 r0 r1 r0 r1 (after r0) and input buffers: the target never repeats ->
        // own counters, still right. Ends on r1, so (3) starts with a new target on stream 0.
        for (int rep = 0; rep < 5; ++rep) {
            const std::size_t k = checkInto(rep % 3 == 0, rep % 2 == 0 ? r1 : r0, 3, "single batch, alternating");
            for (std::size_t j = k; j < fake->calls.size(); ++j) {
                EXPECT_NE(fake->calls[j].hitCounts, r0.ptr());
                EXPECT_NE(fake->calls[j].hitCounts, r1.ptr());
            }
        }
        // (3) multi-batch: 23 items = batches 0..4 on streams 0 1 2 0 1; streams 0 and 1 run two batches each
        // (targets alternate -> own counters), stream 2 one (direct from the second check into the same buffer on)
        for (int rep = 0; rep < 4; ++rep) {
            const bool same = rep >= 1 && rep <= 2; // reps 1, 2 repeat rep 0's buffer; rep 3 switches
            const std::size_t k = checkInto(rep % 2 == 0, rep == 3 ? r1 : r0, 23, "multi batch");
            for (std::size_t j = k; j < fake->calls.size(); ++j) {
                const auto& c = fake->calls[j];
                const DevicePtr target = (rep == 3 ? r1 : r0).ptr() + DevicePtr(c.first) * sizeof(std::uint32_t);
                if (c.streamIndex == 2 && same) {
                    EXPECT_EQ(c.hitCounts, target) << "rep " << rep << " batch " << c.batch;
                } else {
                    EXPECT_NE(c.hitCounts, target) << "rep " << rep << " batch " << c.batch;
                }
                if (!graph) {
                    EXPECT_EQ(c.items, (rep % 2 == 0 ? inA : inB).ptr() + DevicePtr(c.first) * fpiBytes);
                }
            }
        }
        // (4) strided input always goes through the staging buffer
        {
            const std::size_t stride = fpi + 2;
            const auto strided       = items(kMaxItems, plus100, stride);
            DeviceBuffer inS         = DeviceBuffer::fromHost(std::span<const float>(strided));
            for (int rep = 0; rep < 2; ++rep) {
                const std::size_t k = fake->calls.size();
                core->checkDevice(DeviceItems{inS.ptr(), 23, stride}, DeviceSpan<std::uint32_t>{r0.ptr(), 23});
                std::vector<std::uint32_t> out(23);
                r0.download(out.data(), out.size() * sizeof(std::uint32_t));
                for (std::uint32_t i = 0; i < 23; ++i) {
                    EXPECT_EQ(out[i], 105 + i) << "strided item " << i;
                }
                ASSERT_EQ(fake->calls.size(), k + 5); // strided: never replayed
                for (std::size_t j = k; j < fake->calls.size(); ++j) {
                    const DevicePtr p = fake->calls[j].items;
                    EXPECT_TRUE(p < inS.ptr() || p >= inS.ptr() + inS.size()) << "strided items must be staged";
                }
            }
        }
        // (5) host check() interleaved with device checks into the same / other buffers
        for (int rep = 0; rep < 3; ++rep) {
            const auto ch = counts(*core, hostA, 23);
            for (std::uint32_t i = 0; i < 23; ++i) {
                EXPECT_EQ(ch[i], expected(true, i)) << "host check rep " << rep << " item " << i;
            }
            const auto c1 = counts(*core, hostB, 4);
            for (std::uint32_t i = 0; i < 4; ++i) {
                EXPECT_EQ(c1[i], expected(false, i)) << "host single batch rep " << rep << " item " << i;
            }
            checkInto(rep % 2 == 1, r0, 4, "device between host checks");
        }
    }
}

TEST_P(CoreTest, AddressRulesAfterAFailedCheck) {
    for (const bool graph : {false, true}) {
        SCOPED_TRACE(graph ? "graph replay" : "direct prepare()");
        FakeBackend* fake = nullptr;
        FakeOptions o;
        o.addValue            = 1;
        o.throwAtBatch        = 4; // checks with >= 21 items fail at batch 4 (stream 1)
        o.capturableThreshold = graph ? 1u : ~0u;
        auto core             = makeCore(o, &fake);
        const auto host       = items(kMaxItems, times3);
        DeviceBuffer in       = DeviceBuffer::fromHost(std::span<const float>(host));
        DeviceBuffer r(std::size_t(kMaxItems) * sizeof(std::uint32_t));
        auto ok = [&](std::uint32_t n) {
            core->checkDevice(DeviceItems{in.ptr(), n, 0}, DeviceSpan<std::uint32_t>{r.ptr(), n});
            std::vector<std::uint32_t> out(n);
            r.download(out.data(), out.size() * sizeof(std::uint32_t));
            for (std::uint32_t i = 0; i < n; ++i) {
                EXPECT_EQ(out[i], 3 * i + 2) << "n " << n << " item " << i;
            }
        };
        ok(15); // batches 0 1 2 (one per stream)
        ok(15); // all three now count into r directly
        EXPECT_THROW(core->checkDevice(DeviceItems{in.ptr(), 23, 0}, DeviceSpan<std::uint32_t>{r.ptr(), 23}), InvalidArgument);
        for (const StreamContext& s : core->streams()) {
            EXPECT_EQ(cudaStreamQuery(s.stream), cudaSuccess) << "stream " << s.index << " not drained";
        }
        ok(15);
        ok(3);
        ok(15);
        const auto c = counts(*core, host, 20);
        for (std::uint32_t i = 0; i < 20; ++i) {
            EXPECT_EQ(c[i], 3 * i + 2) << "host item " << i;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// CheckerConfig::graphPrepare + Backend::prepareCapturable: zero / staging copy / prepare() replayed as a CUDA graph
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, GraphPrepareCapturesOncePerStreamAndItemCount) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.addValue            = 7;
    o.capturableThreshold = 1;
    ASSERT_TRUE(cfg.graphPrepare); // default on
    auto core = makeCore(o, &fake);
    // batches 5 5 5 5 3 on streams 0 1 2 0 1: graphs (0,5) (1,5) (2,5) (1,3), all with the streams' own hit
    // counters (host checks never count into their pinned result buffer directly; graphs keyed by the hit-count
    // address are covered by AddressRulesItemsInPlaceAndDirectHitCounts)
    const std::uint32_t n = 23;
    for (int rep = 0; rep < 3; ++rep) {
        const auto in = items(n, rep == 1 ? times3 : plus100); // new contents in the same buffers
        const auto c  = counts(*core, in, n);
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(c[i], (rep == 1 ? 3 * i + 1 : 100 + i) + 7) << "item " << i << " rep " << rep;
        }
        EXPECT_EQ(core->lastStats().batches, 5u);
        EXPECT_EQ(core->lastStats().launches, 5u);
    }
    EXPECT_EQ(fake->calls.size(), 4u) << "prepare() must run once per (stream, item count, hits), at capture";
    EXPECT_EQ(fake->begins.size(), 3u); // beginCheck still once per check

    // device items at another address: the staging-copy node is re-pointed
    const auto host     = items(n, times3);
    DeviceBuffer dItems = DeviceBuffer::fromHost(std::span<const float>(host));
    DeviceBuffer dHits(std::size_t(n) * sizeof(std::uint32_t));
    core->checkDevice(DeviceItems{dItems.ptr(), n, 0}, DeviceSpan<std::uint32_t>{dHits.ptr(), n});
    std::vector<std::uint32_t> out(n);
    dHits.download(out.data(), out.size() * sizeof(std::uint32_t));
    for (std::uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(out[i], 3 * i + 1 + 7) << "device item " << i;
    }
    // strided host items are packed by the upload: still replayed
    const std::size_t stride = fpi + 3;
    const auto strided       = items(n, plus100, stride);
    const auto cs            = counts(*core, strided, n, stride);
    for (std::uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(cs[i], 107 + i);
    }
    EXPECT_EQ(fake->calls.size(), 4u); // device and host checks into other targets replay the same graphs
    // strided DEVICE items take the direct path (prepare() called for every batch)
    DeviceBuffer dStrided = DeviceBuffer::fromHost(std::span<const float>(strided));
    core->checkDevice(DeviceItems{dStrided.ptr(), n, stride}, DeviceSpan<std::uint32_t>{dHits.ptr(), n});
    dHits.download(out.data(), out.size() * sizeof(std::uint32_t));
    for (std::uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(out[i], 107 + i) << "strided device item " << i;
    }
    EXPECT_EQ(fake->calls.size(), 4u + 5u);

    // graphPrepare = false: prepare() every batch
    CheckerConfig off = cfg;
    off.graphPrepare  = false;
    FakeBackend* fake2 = nullptr;
    auto core2         = makeCore(o, &fake2, &off);
    counts(*core2, items(n, plus100), n);
    counts(*core2, items(n, plus100), n);
    EXPECT_EQ(fake2->calls.size(), 10u);
}

TEST_P(CoreTest, GraphPrepareManyItemCountsEvictsAndStaysCorrect) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.capturableThreshold = 1;
    CheckerConfig one = cfg;
    one.numStreams    = 1;
    one.maxBatchSize  = 16;
    auto core         = makeCore(o, &fake, &one);
    const auto in     = items(kMaxItems, times3);
    for (int rep = 0; rep < 2; ++rep) {
        for (std::uint32_t n = 1; n <= 16; ++n) { // 16 item counts > the per-stream cache
            const auto c = counts(*core, in, n);
            for (std::uint32_t i = 0; i < n; ++i) {
                ASSERT_EQ(c[i], 3 * i + 1) << "n " << n << " rep " << rep;
            }
        }
    }
}

TEST_P(CoreTest, GraphPrepareCaptureErrorIsCleanedUp) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.capturableThreshold = 1;
    // 23 items: batch 4 (stream 1, 3 items) is the first of its (stream, item count): prepare() throws inside
    // that capture. Batches 0 - 2 captured (s, 5), batch 3 replays (0, 5).
    o.throwAtBatch = 4;
    auto core      = makeCore(o, &fake);
    const auto in  = items(kMaxItems, times3);
    std::vector<std::uint8_t> flags(kMaxItems);
    EXPECT_THROW(core->checkHost(HostItems{in, 23, 0}, flags, {}), InvalidArgument);
    for (const StreamContext& s : core->streams()) {
        cudaStreamCaptureStatus status{};
        EXPECT_EQ(cudaStreamIsCapturing(s.stream, &status), cudaSuccess);
        EXPECT_EQ(status, cudaStreamCaptureStatusNone) << "stream " << s.index << " left capturing";
        EXPECT_EQ(cudaStreamQuery(s.stream), cudaSuccess);
    }
    const auto c = counts(*core, in, 20); // 4 full batches: replays, results right
    for (std::uint32_t i = 0; i < 20; ++i) {
        EXPECT_EQ(c[i], 3 * i + 1);
    }
}

TEST_P(CoreTest, CapturableAndSynchronizingIsAnInternalError) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.capturableThreshold = 1;
    o.syncThreshold       = kBatch;
    auto core             = makeCore(o, &fake);
    const auto in         = items(kMaxItems, times3);
    std::vector<std::uint8_t> flags(kMaxItems);
    try {
        core->checkHost(HostItems{in, 10, 0}, flags, {});
        ADD_FAILURE() << "no exception";
    } catch (const Error& e) {
        EXPECT_EQ(e.kind(), ErrorKind::Internal) << e.what();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Backend::prepareSplit (G3): "zero + staging copy + prepareCompute()" graph-replayed, prepareFinish() direct
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, SplitComputeOncePerKeyFinishEveryBatchNeverCapturing) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.addValue       = 7;
    o.splitThreshold = 1;
    auto core        = makeCore(o, &fake);
    const std::uint32_t n = 23; // batches 5 5 5 5 3 on streams 0 1 2 0 1
    const auto hostA      = items(n, plus100);
    const auto hostB      = items(n, times3);
    DeviceBuffer inA = DeviceBuffer::fromHost(std::span<const float>(hostA));
    DeviceBuffer inB = DeviceBuffer::fromHost(std::span<const float>(hostB));
    DeviceBuffer r0(std::size_t(n) * sizeof(std::uint32_t)), r1(std::size_t(n) * sizeof(std::uint32_t));
    auto check = [&](bool a, DeviceBuffer& r) {
        core->checkDevice(DeviceItems{(a ? inA : inB).ptr(), n, 0}, DeviceSpan<std::uint32_t>{r.ptr(), n});
        std::vector<std::uint32_t> out(n);
        r.download(out.data(), out.size() * sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(out[i], (a ? 100 + i : 3 * i + 1) + 7) << "item " << i;
        }
    };
    std::size_t finishes = 0;
    // check 1: keys (0,5,own) (1,5,own) (2,5,own) (1,3,own); check 2 adds (2,5,r0 + 10 items) (stream 2's only
    // batch repeats its target); check 3 replays everything; check 4 into r1 (new targets: own counters again).
    const std::size_t expectedCaptures[] = {4, 5, 5, 5, 6}; // check 5: stream 2 counts directly into r1: new key
    int k = 0;
    for (const auto& [a, r] : {std::pair<bool, DeviceBuffer*>{true, &r0}, {false, &r0}, {true, &r0}, {false, &r1}, {true, &r1}}) {
        // prepareCompute()'s host-side effect: poison it before every check. On replayed batches it is not run, so
        // the flag stays poisoned; results (checked inside) and prepareFinish's arguments must not care.
        std::fill(fake->hostFlag.begin(), fake->hostFlag.end(), 0xDEADu);
        const std::size_t capturesBefore = fake->computeCalls.size();
        check(a, *r);
        if (fake->computeCalls.size() == capturesBefore) {
            for (std::uint32_t v : fake->hostFlag) {
                EXPECT_EQ(v, 0xDEADu) << "check " << k << ": a fully replayed check ran prepareCompute's host code";
            }
        }
        finishes += 5;
        EXPECT_EQ(fake->computeCalls.size(), expectedCaptures[k]) << "check " << k;
        EXPECT_EQ(fake->finishCalls.size(), finishes) << "check " << k << ": prepareFinish runs for every batch";
        ++k;
    }
    EXPECT_TRUE(fake->calls.empty()) << "split batches with graph_prepare never call prepare()";
    for (const auto& c : fake->computeCalls) {
        EXPECT_TRUE(c.capturing) << "prepareCompute must only be called inside the capture";
    }
    // prepareFinish never inside a capture; its items are the stream's staging buffer (graph batch), the same
    // address prepareCompute was captured with; its hit counters follow the direct-counting rule, also on replays
    // (where prepareCompute's host code -- hostFlag -- did not run).
    for (const auto& f : fake->finishCalls) {
        EXPECT_FALSE(f.capturing) << "batch " << f.batch;
        for (const auto& c : fake->computeCalls) {
            if (c.streamIndex == f.streamIndex) {
                EXPECT_EQ(c.items, f.items) << "staging address";
            }
        }
    }
    // check 2 (finish calls 5..9): stream 2's batch 2 counted directly into r0 at item 10
    for (std::size_t j = 5; j < 10; ++j) {
        const auto& f = fake->finishCalls[j];
        if (f.streamIndex == 2) {
            EXPECT_EQ(f.hitCounts, r0.ptr() + 10 * sizeof(std::uint32_t));
        } else {
            EXPECT_TRUE(f.hitCounts < r0.ptr() || f.hitCounts >= r0.ptr() + r0.size()) << "batch " << f.batch;
        }
    }
    // check 4 (r1, new targets): every batch on the streams' own counters again
    for (std::size_t j = 15; j < 20; ++j) {
        const auto& f = fake->finishCalls[j];
        EXPECT_TRUE(f.hitCounts < r1.ptr() || f.hitCounts >= r1.ptr() + r1.size()) << "batch " << f.batch;
    }
    // host checks and strided device items (not replayed: prepare() every batch) stay right
    const auto ch = counts(*core, hostB, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(ch[i], 3 * i + 1 + 7);
    }
    const std::size_t stride = fpi + 2;
    const auto strided       = items(n, plus100, stride);
    DeviceBuffer inS         = DeviceBuffer::fromHost(std::span<const float>(strided));
    const std::size_t before = fake->calls.size();
    core->checkDevice(DeviceItems{inS.ptr(), n, stride}, DeviceSpan<std::uint32_t>{r0.ptr(), n});
    EXPECT_EQ(fake->calls.size(), before + 5);
    std::vector<std::uint32_t> out(n);
    r0.download(out.data(), out.size() * sizeof(std::uint32_t));
    for (std::uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(out[i], 107 + i);
    }

    // graph_prepare = false: prepare() every batch, never compute / finish
    CheckerConfig off = cfg;
    off.graphPrepare  = false;
    FakeBackend* fake2 = nullptr;
    auto core2         = makeCore(o, &fake2, &off);
    counts(*core2, hostA, n);
    EXPECT_EQ(fake2->calls.size(), 5u);
    EXPECT_TRUE(fake2->computeCalls.empty());
    EXPECT_TRUE(fake2->finishCalls.empty());
}

TEST_P(CoreTest, SplitEvictionAndManyItemCountsStayCorrect) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.splitThreshold = 1;
    CheckerConfig one = cfg;
    one.numStreams    = 1;
    one.maxBatchSize  = 16;
    auto core         = makeCore(o, &fake, &one);
    const auto in     = items(kMaxItems, times3);
    DeviceBuffer dIn  = DeviceBuffer::fromHost(std::span<const float>(in));
    DeviceBuffer r(std::size_t(kMaxItems) * sizeof(std::uint32_t));
    for (int rep = 0; rep < 3; ++rep) {
        for (std::uint32_t n = 1; n <= 16; ++n) { // 16 item counts > the per-stream cache of 8: evictions
            for (int twice = 0; twice < 2; ++twice) { // the second check counts directly into r (another key)
                std::fill(fake->hostFlag.begin(), fake->hostFlag.end(), 0xDEADu); // must not matter (see above)
                core->checkDevice(DeviceItems{dIn.ptr(), n, 0}, DeviceSpan<std::uint32_t>{r.ptr(), n});
                std::vector<std::uint32_t> out(n);
                r.download(out.data(), out.size() * sizeof(std::uint32_t));
                for (std::uint32_t i = 0; i < n; ++i) {
                    ASSERT_EQ(out[i], 3 * i + 1) << "n " << n << " rep " << rep << " twice " << twice;
                }
            }
        }
    }
    for (const auto& f : fake->finishCalls) {
        ASSERT_FALSE(f.capturing);
    }
    EXPECT_EQ(fake->finishCalls.size(), std::size_t(3 * 16 * 2));
    EXPECT_GT(fake->computeCalls.size(), std::size_t(16)) << "evicted keys are captured again";
}

TEST_P(CoreTest, SplitComputeFailureInsideCaptureIsCleanedUp) {
    FakeBackend* fake = nullptr;
    FakeOptions o;
    o.splitThreshold = 1;
    o.throwAtBatch   = 4; // 23 items: batch 4 (stream 1, 3 items) is the first of its key: throws inside the capture
    auto core        = makeCore(o, &fake);
    const auto in    = items(kMaxItems, times3);
    std::vector<std::uint8_t> flags(kMaxItems);
    EXPECT_THROW(core->checkHost(HostItems{in, 23, 0}, flags, {}), InvalidArgument);
    for (const StreamContext& s : core->streams()) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusActive;
        EXPECT_EQ(cudaStreamIsCapturing(s.stream, &st), cudaSuccess);
        EXPECT_EQ(st, cudaStreamCaptureStatusNone) << "stream " << s.index << " left capturing";
        EXPECT_EQ(cudaStreamQuery(s.stream), cudaSuccess) << "stream " << s.index << " not drained";
    }
    for (const auto& f : fake->finishCalls) {
        EXPECT_FALSE(f.capturing);
        EXPECT_NE(f.batch, 4u) << "prepareFinish after a failed prepareCompute";
    }
    for (int rep = 0; rep < 2; ++rep) {
        const auto c = counts(*core, in, 20); // batch 4 never happens
        for (std::uint32_t i = 0; i < 20; ++i) {
            EXPECT_EQ(c[i], 3 * i + 1);
        }
    }
}

TEST_P(CoreTest, SplitAndSynchronizingOrCapturableIsAnInternalError) {
    const auto in = items(kMaxItems, times3);
    std::vector<std::uint8_t> flags(kMaxItems);
    for (int variant = 0; variant < 2; ++variant) {
        FakeBackend* fake = nullptr;
        FakeOptions o;
        o.splitThreshold = 1;
        if (variant == 0) {
            o.syncThreshold = kBatch;
        } else {
            o.capturableThreshold = 1;
        }
        auto core = makeCore(o, &fake);
        try {
            core->checkHost(HostItems{in, 10, 0}, flags, {});
            ADD_FAILURE() << "no exception, variant " << variant;
        } catch (const Error& e) {
            EXPECT_EQ(e.kind(), ErrorKind::Internal) << e.what();
        }
        EXPECT_TRUE(fake->computeCalls.empty());
        EXPECT_TRUE(fake->finishCalls.empty());
    }
}

TEST_P(CoreTest, WarmupRunsOneFullCapacityCheck) {
    FakeBackend* fake = nullptr;
    auto core         = makeCore({}, &fake);
    core->warmup();
    ASSERT_EQ(fake->begins.size(), 1u);
    EXPECT_EQ(fake->begins[0].itemCount, kStreams * kBatch);
    EXPECT_EQ(core->lastStats().batches, kStreams);

    CheckerConfig small = cfg;
    small.maxItemsPerCheck = 7; // capped at maxItemsPerCheck
    FakeBackend* fake2 = nullptr;
    auto core2         = makeCore({}, &fake2, &small);
    core2->warmup();
    ASSERT_EQ(fake2->begins.size(), 1u);
    EXPECT_EQ(fake2->begins[0].itemCount, 7u);
}

// ---------------------------------------------------------------------------------------------------------------
// Cross-check with the single-thread ReferenceBatchRunner (tests/support) on the same fake backend
// ---------------------------------------------------------------------------------------------------------------

TEST_P(CoreTest, SameResultsAsReferenceBatchRunner) {
    TriangleGas gas(*ctx);
    for (int variant = 0; variant < 3; ++variant) {
        FakeOptions o;
        o.dof               = kDof;
        o.addValue          = 3;
        o.traversable       = gas.handle;
        o.includeBatchIndex = variant == 1;
        o.skipBatch         = variant == 2 ? 2u : ~0u;
        o.raygen            = [variant](std::uint32_t b) {
            if (variant == 0) {
                return b % 3 == 0 ? kRaygenTrace : b % 3 == 1 ? kRaygenTraceChMiss : kRaygenCopy;
            }
            return variant == 1 ? kRaygenCopyBatch : kRaygenCopy;
        };

        FakeBackend* fake = nullptr;
        auto core         = makeCore(o, &fake);

        StreamLease lease = ctx->acquireStreams(kStreams);
        std::vector<StreamContext> sc;
        for (std::uint32_t i = 0; i < kStreams; ++i) {
            sc.push_back(StreamContext{i, lease[i]});
        }
        BackendCreateInfo info;
        info.context = ctx;
        info.config  = &cfg;
        info.streams = sc;
        FakeBackend refBackend(info, o);
        internal::ReferenceExecutor exec(ctx, refBackend.programs(), cfg.optix);
        internal::ReferenceBatchRunner runner(ctx, refBackend, exec, cfg, sc, kDof);

        const auto in = items(kMaxItems, [](std::uint32_t i) {
            const float xs[] = {0.25f, 1.5f, 0.75f, -0.5f, 7.0f};
            return xs[i % 5];
        });
        for (std::uint32_t n : {1u, 5u, 14u, 15u, 16u, 37u, 64u}) {
            const std::vector<std::uint32_t> expected = runner.runAll(in, n);
            const std::vector<std::uint32_t> got      = counts(*core, in, n);
            EXPECT_EQ(got, expected) << "variant " << variant << " n " << n;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host-only helpers
// ---------------------------------------------------------------------------------------------------------------

TEST(DetectorWarmupItems, DeterministicWithinLimits) {
    const ItemLayout layout{3, 4};
    const std::vector<float> lo = {-1.0f, 0.0f, 0.0f};
    const std::vector<float> hi = {1.0f, 0.5f, 0.0f}; // joint 2: empty range -> [-pi, pi]
    const auto a = detail::makeWarmupItems(layout, 50, lo, hi);
    const auto b = detail::makeWarmupItems(layout, 50, lo, hi);
    ASSERT_EQ(a.size(), 50u * 12u);
    EXPECT_EQ(a, b);
    bool varied = false;
    for (std::size_t k = 0; k < a.size(); ++k) {
        const std::size_t j = k % 3;
        const float l = j == 2 ? -3.1416f : lo[j], h = j == 2 ? 3.1416f : hi[j];
        EXPECT_GE(a[k], l);
        EXPECT_LE(a[k], h);
        varied = varied || a[k] != a[j];
    }
    EXPECT_TRUE(varied);
}

INSTANTIATE_TEST_SUITE_P(Dispatch, CoreTest, ::testing::Values(HostDispatch::SingleThread, HostDispatch::ThreadPerStream),
    [](const ::testing::TestParamInfo<HostDispatch>& p) {
        return p.param == HostDispatch::SingleThread ? std::string("SingleThread") : std::string("ThreadPerStream");
    });
