// TEST-LOCAL STUB of rtcd::internal::buildTriangleGas / buildInstanceAs (include/rtcd/internal/accel.h), owned by
// Wave 1 a (rtcd_core) and not yet implemented there when the Scene was written. tests/scene/CMakeLists.txt adds
// this file only while src/core/CMakeLists.txt does not mention an accel source; DELETE IT AT THE WAVE-1 MERGE.
// Written from the declarations and the old meshModel::buildGAS / scene::wrapScene only (not from 1a's code).

#include <rtcd/errors.h>
#include <rtcd/internal/accel.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

namespace rtcd::internal {

    namespace {
        Accel build(const Context& context, const OptixBuildInput& input, BuildFlags flags, StreamHandle stream) {
            context.makeCurrent();
            OptixAccelBuildOptions options = {};
            options.buildFlags             = flags;
            options.operation              = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes sizes = {};
            RTCD_OPTIX_CHECK(optixAccelComputeMemoryUsage(context.optixContext(), &options, &input, 1, &sizes));

            DeviceBuffer temp(sizes.tempSizeInBytes);
            Accel out;
            out.buffer.allocate(sizes.outputSizeInBytes);

            const bool compact = (flags & kBuildFlagAllowCompaction) != 0;
            DeviceBuffer compactedSize(compact ? sizeof(std::uint64_t) : 0);
            OptixAccelEmitDesc emit = {};
            emit.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit.result             = compactedSize.ptr();

            RTCD_OPTIX_CHECK(optixAccelBuild(context.optixContext(), stream, &options, &input, 1, temp.ptr(), temp.size(),
                out.buffer.ptr(), out.buffer.size(), &out.handle, compact ? &emit : nullptr, compact ? 1u : 0u));
            RTCD_CUDA_CHECK(cudaStreamSynchronize(stream));

            if (compact) {
                std::uint64_t size = 0;
                compactedSize.download(&size, sizeof(size));
                if (size < out.buffer.size()) {
                    DeviceBuffer compacted(size);
                    RTCD_OPTIX_CHECK(optixAccelCompact(
                        context.optixContext(), stream, out.handle, compacted.ptr(), size, &out.handle));
                    RTCD_CUDA_CHECK(cudaStreamSynchronize(stream));
                    out.buffer = std::move(compacted);
                }
            }
            return out;
        }
    } // namespace

    Accel buildTriangleGas(const Context& context, const TriangleGasInput& input, BuildFlags flags, StreamHandle stream) {
        CUdeviceptr vertices       = input.vertices.ptr;
        const unsigned int gFlags  = input.geometryFlags;
        OptixBuildInput bi         = {};
        bi.type                    = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& t                    = bi.triangleArray;
        t.vertexFormat             = OPTIX_VERTEX_FORMAT_FLOAT3;
        t.vertexStrideInBytes      = 3 * sizeof(float);
        t.numVertices              = static_cast<unsigned int>(input.vertices.count / 3);
        t.vertexBuffers            = &vertices;
        t.indexFormat              = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        t.indexStrideInBytes       = 3 * sizeof(std::uint32_t);
        t.numIndexTriplets         = static_cast<unsigned int>(input.indices.count / 3);
        t.indexBuffer              = input.indices.ptr;
        t.flags                    = &gFlags;
        t.numSbtRecords            = 1;
        return build(context, bi, flags, stream);
    }

    Accel buildInstanceAs(const Context& context, DevicePtr instances, std::uint32_t count, BuildFlags flags,
        StreamHandle stream) {
        OptixBuildInput bi            = {};
        bi.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        bi.instanceArray.instances    = instances;
        bi.instanceArray.numInstances = count;
        return build(context, bi, flags, stream);
    }

} // namespace rtcd::internal
