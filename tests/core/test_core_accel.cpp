// rtcd_core GPU tests (new_gpu): internal/accel.h. Builds triangle GASes (with / without compaction) and an
// instance AS, then traces rays through them with a small OptiX test program (trace_program.cu, embedded with
// rtcd_add_optix_ir) to prove the structures are complete and correct, not just non-zero handles.
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/accel.h>
#include <rtcd/limits.h>

#include <rtcd_core_test_ir_embedded.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "trace_params.h"

using namespace rtcd;
using namespace rtcd::internal;

namespace {
    std::shared_ptr<Context> ctx() {
        static std::shared_ptr<Context> c = Context::create();
        return c;
    }

    // Unit cube [0,1]^3: 8 vertices, 12 triangles.
    const std::vector<float> kCubeVertices = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::vector<std::uint32_t> kCubeIndices = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7,
        2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3};

    // Minimal pipeline around trace_program.cu: raygen, miss, one triangle hit group.
    class Tracer {
    public:
        explicit Tracer(const Context& c) : optix_(c.optixContext()) {
            OptixModuleCompileOptions mco = {};
            mco.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            mco.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            mco.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            pco_.usesMotionBlur           = 0;
            pco_.traversableGraphFlags    = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
            pco_.numPayloadValues         = 1;
            pco_.numAttributeValues       = 2;
            pco_.exceptionFlags           = OPTIX_EXCEPTION_FLAG_NONE;
            pco_.pipelineLaunchParamsVariableName = kLaunchParamsName;
            pco_.usesPrimitiveTypeFlags   = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            char log[4096];
            size_t logSize = sizeof(log);
            RTCD_OPTIX_CHECK_LOG(optixModuleCreate(optix_, &mco, &pco_,
                                     reinterpret_cast<const char*>(embedded::rtcd_core_test_ir_trace_program),
                                     embedded::rtcd_core_test_ir_trace_program_size, log, &logSize, &module_),
                log);

            OptixProgramGroupOptions pgo = {};
            OptixProgramGroupDesc d[3]   = {};
            d[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            d[0].raygen.module            = module_;
            d[0].raygen.entryFunctionName = "__raygen__trace";
            d[1].kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
            d[1].miss.module              = module_;
            d[1].miss.entryFunctionName   = "__miss__trace";
            d[2].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            d[2].hitgroup.moduleCH            = module_;
            d[2].hitgroup.entryFunctionNameCH = "__closesthit__trace";
            logSize                           = sizeof(log);
            RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(optix_, d, 3, &pgo, log, &logSize, groups_), log);

            OptixPipelineLinkOptions plo = {};
            plo.maxTraceDepth            = 1;
            logSize                      = sizeof(log);
            RTCD_OPTIX_CHECK_LOG(
                optixPipelineCreate(optix_, &pco_, &plo, groups_, 3, log, &logSize, &pipeline_), log);

            SbtRecord rec[3] = {};
            for (int i = 0; i < 3; ++i) {
                RTCD_OPTIX_CHECK(optixSbtRecordPackHeader(groups_[i], &rec[i]));
            }
            records_ = DeviceBuffer::fromHost(std::span<const SbtRecord>(rec, 3));
            sbt_.raygenRecord                = records_.ptr();
            sbt_.missRecordBase              = records_.ptr() + sizeof(SbtRecord);
            sbt_.missRecordStrideInBytes     = sizeof(SbtRecord);
            sbt_.missRecordCount             = 1;
            sbt_.hitgroupRecordBase          = records_.ptr() + 2 * sizeof(SbtRecord);
            sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
            sbt_.hitgroupRecordCount         = 1;
        }
        ~Tracer() {
            optixPipelineDestroy(pipeline_);
            for (auto g : groups_) {
                optixProgramGroupDestroy(g);
            }
            optixModuleDestroy(module_);
        }

        std::vector<std::uint32_t> trace(TraversableHandle handle, const std::vector<TestRay>& rays, StreamHandle s) {
            DeviceBuffer dRays = DeviceBuffer::fromHost(std::span<const TestRay>(rays));
            DeviceBuffer dHits(rays.size() * 4);
            dHits.memsetAsync(0xFF, dHits.size(), s);
            TraceParams p{handle, dRays.ptr(), dHits.ptr(), static_cast<unsigned>(rays.size()), 0};
            DeviceBuffer dParams(sizeof(p));
            dParams.uploadAsync(&p, sizeof(p), s);
            RTCD_OPTIX_CHECK(optixLaunch(
                pipeline_, s, dParams.ptr(), sizeof(p), &sbt_, static_cast<unsigned>(rays.size()), 1, 1));
            RTCD_CUDA_CHECK(cudaStreamSynchronize(s));
            std::vector<std::uint32_t> hits(rays.size());
            dHits.download(hits.data(), hits.size() * 4);
            return hits;
        }

    private:
        OptixDeviceContext optix_;
        OptixPipelineCompileOptions pco_ = {};
        OptixModule module_             = nullptr;
        OptixProgramGroup groups_[3]    = {};
        OptixPipeline pipeline_         = nullptr;
        DeviceBuffer records_;
        OptixShaderBindingTable sbt_ = {};
    };

    TestRay ray(float ox, float oy, float oz, float dx, float dy, float dz, float tmax) {
        return TestRay{{ox, oy, oz}, {dx, dy, dz}, tmax, 0};
    }

    constexpr std::uint32_t kHit = 0x80000000u;
    std::uint32_t instanceOf(std::uint32_t h) { return (h & ~kHit) >> 16; }

    std::size_t uncompactedSize(const Context& c, DevicePtr v, DevicePtr i, std::uint32_t flags) {
        CUdeviceptr vb                      = v;
        unsigned int gflags                 = 0;
        OptixBuildInput in                  = {};
        in.type                             = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        in.triangleArray.vertexFormat       = OPTIX_VERTEX_FORMAT_FLOAT3;
        in.triangleArray.vertexStrideInBytes = 12;
        in.triangleArray.numVertices        = 8;
        in.triangleArray.vertexBuffers      = &vb;
        in.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        in.triangleArray.indexStrideInBytes = 12;
        in.triangleArray.numIndexTriplets   = 12;
        in.triangleArray.indexBuffer        = i;
        in.triangleArray.flags              = &gflags;
        in.triangleArray.numSbtRecords      = 1;
        OptixAccelBuildOptions o            = {};
        o.buildFlags                        = flags;
        o.operation                         = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{};
        RTCD_OPTIX_CHECK(optixAccelComputeMemoryUsage(c.optixContext(), &o, &in, 1, &sizes));
        return sizes.outputSizeInBytes;
    }
} // namespace

TEST(CoreAccel, TriangleGasIsTraceableWithAndWithoutCompaction) {
    auto c     = ctx();
    auto lease = c->acquireStreams(1);
    Tracer tracer(*c);
    DeviceBuffer v = DeviceBuffer::fromHost(std::span<const float>(kCubeVertices));
    DeviceBuffer i = DeviceBuffer::fromHost(std::span<const std::uint32_t>(kCubeIndices));
    const TriangleGasInput in{v.span<const float>(), i.span<const std::uint32_t>(), kGeometryFlagDisableAnyHit};

    const std::vector<TestRay> rays = {
        ray(-5, 0.5f, 0.5f, 1, 0, 0, 10),   // through the cube
        ray(-5, 5, 0.5f, 1, 0, 0, 10),      // passes above
        ray(0.5f, 0.5f, -5, 0, 0, 1, 4),    // stops 1 short of the cube (tmax)
        ray(0.5f, 0.5f, -5, 0, 0, 1, 5.5f), // reaches it
        ray(0.5f, 0.5f, 0.5f, 0, 1, 0, 1),  // starts inside, leaves through y = 1
    };
    const BuildFlags variants[] = {toBuildFlags(BuildPreference::FastTraceCompact),
        toBuildFlags(BuildPreference::FastTrace), toBuildFlags(BuildPreference::FastBuild),
        kBuildFlagPreferFastBuild | kBuildFlagAllowUpdate};
    for (BuildFlags flags : variants) {
        Accel gas = buildTriangleGas(*c, in, flags, lease[0]);
        ASSERT_NE(gas.handle, 0u);
        const std::size_t full = uncompactedSize(*c, v.ptr(), i.ptr(), flags);
        if (flags & kBuildFlagAllowCompaction) {
            EXPECT_LE(gas.buffer.size(), full) << "compacted GAS must not be larger";
        } else {
            EXPECT_EQ(gas.buffer.size(), full);
        }
        auto hits = tracer.trace(gas.handle, rays, lease[0]);
        EXPECT_TRUE(hits[0] & kHit) << flags;
        EXPECT_EQ(hits[1], 0u) << flags;
        EXPECT_EQ(hits[2], 0u) << flags;
        EXPECT_TRUE(hits[3] & kHit) << flags;
        EXPECT_TRUE(hits[4] & kHit) << flags;
    }
    // the compacted variant really is smaller for this mesh (old meshModel::buildGAS kept it only if smaller)
    Accel compact = buildTriangleGas(*c, in, toBuildFlags(BuildPreference::FastTraceCompact), lease[0]);
    EXPECT_LT(compact.buffer.size(), uncompactedSize(*c, v.ptr(), i.ptr(), toBuildFlags(BuildPreference::FastTraceCompact)));
}

TEST(CoreAccel, InstanceAsOverGasesIsTraceable) {
    auto c     = ctx();
    auto lease = c->acquireStreams(1);
    Tracer tracer(*c);
    DeviceBuffer v = DeviceBuffer::fromHost(std::span<const float>(kCubeVertices));
    DeviceBuffer i = DeviceBuffer::fromHost(std::span<const std::uint32_t>(kCubeIndices));
    // scene obstacles: PREFER_FAST_TRACE | ALLOW_COMPACTION, DISABLE_ANYHIT (old meshModel::buildGAS)
    Accel gas = buildTriangleGas(*c, {v.span<const float>(), i.span<const std::uint32_t>(), kGeometryFlagDisableAnyHit},
        kBuildFlagPreferFastTrace | kBuildFlagAllowCompaction, lease[0]);

    // three instances: identity, translated x + 3, scaled 0.5 at y + 3 (all use hit record 0)
    std::vector<OptixInstance> inst(3);
    const float tf[3][12] = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, {1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0},
        {0.5f, 0, 0, 0, 0, 0.5f, 0, 3, 0, 0, 0.5f, 0}};
    for (unsigned k = 0; k < 3; ++k) {
        inst[k]                   = {};
        std::memcpy(inst[k].transform, tf[k], sizeof(tf[k]));
        inst[k].instanceId        = k;
        inst[k].sbtOffset         = 0;
        inst[k].visibilityMask    = 255;
        inst[k].flags             = OPTIX_INSTANCE_FLAG_NONE;
        inst[k].traversableHandle = gas.handle;
    }
    DeviceBuffer dInst = DeviceBuffer::fromHost(std::span<const OptixInstance>(inst));
    for (BuildFlags flags : {BuildFlags{kBuildFlagNone}, BuildFlags{kBuildFlagPreferFastTrace | kBuildFlagAllowCompaction},
             BuildFlags{kBuildFlagPreferFastBuild | kBuildFlagAllowUpdate}}) {
        Accel ias = buildInstanceAs(*c, dInst.ptr(), 3, flags, lease[0]);
        ASSERT_NE(ias.handle, 0u);
        const std::vector<TestRay> rays = {
            ray(-5, 0.5f, 0.5f, 1, 0, 0, 20),    // hits instance 0 first (closest)
            ray(3.5f, 0.5f, -5, 0, 0, 1, 10),    // only instance 1
            ray(0.25f, 3.25f, -5, 0, 0, 1, 10),  // only instance 2 (half-size cube at y in [3, 3.5])
            ray(0.25f, 3.75f, -5, 0, 0, 1, 10),  // above instance 2: miss
            ray(-5, 0.5f, 0.5f, -1, 0, 0, 20),   // away from everything
        };
        auto hits = tracer.trace(ias.handle, rays, lease[0]);
        ASSERT_TRUE(hits[0] & kHit);
        EXPECT_EQ(instanceOf(hits[0]), 0u);
        ASSERT_TRUE(hits[1] & kHit);
        EXPECT_EQ(instanceOf(hits[1]), 1u);
        ASSERT_TRUE(hits[2] & kHit);
        EXPECT_EQ(instanceOf(hits[2]), 2u);
        EXPECT_EQ(hits[3], 0u);
        EXPECT_EQ(hits[4], 0u);
    }
}

TEST(CoreAccel, ArgumentChecks) {
    auto c     = ctx();
    auto lease = c->acquireStreams(1);
    DeviceBuffer v = DeviceBuffer::fromHost(std::span<const float>(kCubeVertices));
    DeviceBuffer i = DeviceBuffer::fromHost(std::span<const std::uint32_t>(kCubeIndices));
    const auto vs = v.span<const float>();
    const auto is = i.span<const std::uint32_t>();
    EXPECT_THROW(buildTriangleGas(*c, {vs.subspan(0, 7), is}, kBuildFlagNone, lease[0]), InvalidArgument);
    EXPECT_THROW(buildTriangleGas(*c, {vs, is.subspan(0, 5)}, kBuildFlagNone, lease[0]), InvalidArgument);
    EXPECT_THROW(buildTriangleGas(*c, {{}, is}, kBuildFlagNone, lease[0]), InvalidArgument);
    EXPECT_THROW(buildTriangleGas(*c, {vs, is, 1u << 5}, kBuildFlagNone, lease[0]), InvalidArgument);
    EXPECT_THROW(buildTriangleGas(*c, {vs, is}, kBuildFlagPreferFastBuild | kBuildFlagPreferFastTrace, lease[0]),
        InvalidArgument);
    EXPECT_THROW(buildTriangleGas(*c, {vs, is}, 1u << 9, lease[0]), InvalidArgument);
    EXPECT_THROW(buildInstanceAs(*c, 0, 3, kBuildFlagNone, lease[0]), InvalidArgument);
    EXPECT_THROW(buildInstanceAs(*c, v.ptr() + 4, 1, kBuildFlagNone, lease[0]), InvalidArgument); // misaligned
}
