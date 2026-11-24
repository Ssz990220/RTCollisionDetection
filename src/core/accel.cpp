// rtcd_core: construction-time acceleration-structure builders of rtcd/internal/accel.h. They reproduce the old
// meshModel::buildGAS (RTCD/Meshes/mesh.h: one triangle build input, one SBT record, compaction when requested
// and smaller) and scene::wrapScene / buildIAS (RTCD/CollisionScenes/scene.h, RTCD/Robot/meshUtils.h).
// Synchronous: they return when the AS (and its compaction) is complete on `stream`.
#include <rtcd/errors.h>
#include <rtcd/internal/accel.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdint>
#include <string>

namespace rtcd::internal {

    // The rtcd flag values are passed to OptiX unchanged.
    static_assert(kBuildFlagAllowUpdate == OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    static_assert(kBuildFlagAllowCompaction == OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
    static_assert(kBuildFlagPreferFastTrace == OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    static_assert(kBuildFlagPreferFastBuild == OPTIX_BUILD_FLAG_PREFER_FAST_BUILD);
    static_assert(kGeometryFlagDisableAnyHit == OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    static_assert(kGeometryFlagRequireSingleAnyHitCall == OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
    static_assert(sizeof(OptixInstance) == 80);

    namespace {
        constexpr BuildFlags kKnownBuildFlags =
            kBuildFlagAllowUpdate | kBuildFlagAllowCompaction | kBuildFlagPreferFastTrace | kBuildFlagPreferFastBuild;

        void checkBuildFlags(const char* fn, BuildFlags flags) {
            if ((flags & ~kKnownBuildFlags) != 0) {
                throw InvalidArgument(std::string(fn) + ": unknown build flag bits " + std::to_string(flags));
            }
            if ((flags & kBuildFlagPreferFastTrace) && (flags & kBuildFlagPreferFastBuild)) {
                throw InvalidArgument(std::string(fn) + ": PREFER_FAST_TRACE and PREFER_FAST_BUILD are exclusive");
            }
        }

        // Runs the build described by (options, input) into a new Accel; compacts it when options ask for
        // compaction and the compacted size is smaller (old buildGAS / buildIAS<COMPACT>). Synchronizes `stream`.
        Accel buildAndMaybeCompact(const Context& context, const OptixAccelBuildOptions& options,
            const OptixBuildInput& input, StreamHandle stream) {
            OptixDeviceContext optix = context.optixContext();
            OptixAccelBufferSizes sizes{};
            RTCD_OPTIX_CHECK(optixAccelComputeMemoryUsage(optix, &options, &input, 1, &sizes));

            DeviceBuffer temp(sizes.tempSizeInBytes);
            Accel out;
            out.buffer.allocate(sizes.outputSizeInBytes);

            const bool compact = (options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
            if (!compact) {
                RTCD_OPTIX_CHECK(optixAccelBuild(optix, stream, &options, &input, 1, temp.ptr(), temp.size(),
                    out.buffer.ptr(), out.buffer.size(), &out.handle, nullptr, 0));
                RTCD_CUDA_CHECK(cudaStreamSynchronize(stream));
                return out;
            }

            DeviceBuffer compactedSize(sizeof(std::uint64_t));
            OptixAccelEmitDesc emit{};
            emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit.result = compactedSize.ptr();
            RTCD_OPTIX_CHECK(optixAccelBuild(optix, stream, &options, &input, 1, temp.ptr(), temp.size(),
                out.buffer.ptr(), out.buffer.size(), &out.handle, &emit, 1));
            std::uint64_t size = 0;
            compactedSize.downloadAsync(&size, sizeof(size), stream);
            RTCD_CUDA_CHECK(cudaStreamSynchronize(stream));

            if (size < sizes.outputSizeInBytes) {
                Accel compacted;
                compacted.buffer.allocate(static_cast<std::size_t>(size));
                RTCD_OPTIX_CHECK(optixAccelCompact(optix, stream, out.handle, compacted.buffer.ptr(),
                    compacted.buffer.size(), &compacted.handle));
                RTCD_CUDA_CHECK(cudaStreamSynchronize(stream));
                return compacted; // the uncompacted buffer is freed with `out`
            }
            return out;
        }
    } // namespace

    Accel buildTriangleGas(const Context& context, const TriangleGasInput& input, BuildFlags flags, StreamHandle stream) {
        checkBuildFlags("buildTriangleGas", flags);
        if (input.vertices.count == 0 || input.vertices.count % 3 != 0) {
            throw InvalidArgument("buildTriangleGas: vertices holds " + std::to_string(input.vertices.count)
                                  + " floats, need a positive multiple of 3");
        }
        if (input.indices.count == 0 || input.indices.count % 3 != 0) {
            throw InvalidArgument("buildTriangleGas: indices holds " + std::to_string(input.indices.count)
                                  + " values, need a positive multiple of 3");
        }
        if (input.vertices.ptr == 0 || input.indices.ptr == 0) {
            throw InvalidArgument("buildTriangleGas: null vertex or index pointer");
        }
        if (input.vertices.count / 3 > 0xFFFFFFFFull || input.indices.count / 3 > 0xFFFFFFFFull) {
            throw InvalidArgument("buildTriangleGas: too many vertices or triangles for one build input");
        }
        if ((input.geometryFlags & ~(kGeometryFlagDisableAnyHit | kGeometryFlagRequireSingleAnyHitCall)) != 0) {
            throw InvalidArgument("buildTriangleGas: unknown geometry flag bits " + std::to_string(input.geometryFlags));
        }
        context.makeCurrent();

        CUdeviceptr vertexBuffer       = input.vertices.ptr;
        const unsigned int inputFlags  = input.geometryFlags;
        OptixBuildInput build          = {};
        build.type                     = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& tri                      = build.triangleArray;
        tri.vertexFormat               = OPTIX_VERTEX_FORMAT_FLOAT3;
        tri.vertexStrideInBytes        = 3 * sizeof(float);
        tri.numVertices                = static_cast<unsigned int>(input.vertices.count / 3);
        tri.vertexBuffers              = &vertexBuffer;
        tri.indexFormat                = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        tri.indexStrideInBytes         = 3 * sizeof(std::uint32_t);
        tri.numIndexTriplets           = static_cast<unsigned int>(input.indices.count / 3);
        tri.indexBuffer                = input.indices.ptr;
        tri.flags                      = &inputFlags;
        tri.numSbtRecords              = 1;
        tri.sbtIndexOffsetBuffer       = 0;
        tri.sbtIndexOffsetSizeInBytes  = 0;
        tri.sbtIndexOffsetStrideInBytes = 0;

        OptixAccelBuildOptions options = {};
        options.buildFlags             = flags;
        options.operation              = OPTIX_BUILD_OPERATION_BUILD;
        return buildAndMaybeCompact(context, options, build, stream);
    }

    Accel buildInstanceAs(
        const Context& context, DevicePtr instances, std::uint32_t count, BuildFlags flags, StreamHandle stream) {
        checkBuildFlags("buildInstanceAs", flags);
        if (count > 0 && instances == 0) {
            throw InvalidArgument("buildInstanceAs: null instance pointer for " + std::to_string(count) + " instances");
        }
        if (instances % OPTIX_INSTANCE_BYTE_ALIGNMENT != 0) {
            throw InvalidArgument("buildInstanceAs: instances must be " + std::to_string(OPTIX_INSTANCE_BYTE_ALIGNMENT)
                                  + "-byte aligned");
        }
        context.makeCurrent();

        OptixBuildInput build            = {};
        build.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        build.instanceArray.instances    = instances;
        build.instanceArray.numInstances = count;

        OptixAccelBuildOptions options = {};
        options.buildFlags             = flags;
        options.operation              = OPTIX_BUILD_OPERATION_BUILD;
        return buildAndMaybeCompact(context, options, build, stream);
    }

} // namespace rtcd::internal
