#pragma once
#include <Meshes/meshTypes.h>
#include <Utils/CUDABuffer.h>
#include <Utils/optix7.h>
#include <vector>

using namespace RTCD;

namespace RTCD {
    template <BuildType BUILD>
    void buildIAS(const OptixDeviceContext& context, const OptixAccelBuildOptions& buildOptions,
        const OptixBuildInput& buildInput, CUDABuffer& tempBuffer, CUDABuffer& IASBuffer,
        OptixTraversableHandle& IASHandle, size_t& IASBufferSize, float bufferScale = 1) {

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &buildOptions, &buildInput, 1, &ias_buffer_sizes));

        if (ias_buffer_sizes.tempSizeInBytes * bufferScale > tempBuffer.sizeInBytes) {
            tempBuffer.resize(ias_buffer_sizes.tempSizeInBytes);
        }

        if constexpr (BUILD == BuildType::COMPACT) {

            CUDABuffer d_buffer_temp_output_ias;
            d_buffer_temp_output_ias.alloc(ias_buffer_sizes.outputSizeInBytes);

            OptixAccelEmitDesc emitProperty = {};
            CUDABuffer iasCompactedSizeBuffer;
            iasCompactedSizeBuffer.alloc(sizeof(uint64_t));
            emitProperty.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = iasCompactedSizeBuffer.d_pointer();

            OPTIX_CHECK(optixAccelBuild(context, 0, &buildOptions, &buildInput, 1, tempBuffer.d_pointer(),
                ias_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_ias.d_pointer(),
                ias_buffer_sizes.outputSizeInBytes, &IASHandle, &emitProperty, 1));

            // ==================================================================
            // Compact the IAS
            // ==================================================================
            size_t compacted_ias_size;
            iasCompactedSizeBuffer.download(&compacted_ias_size, 1);

            if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes) {
                IASBuffer.alloc(compacted_ias_size);
                // use handle as input and output
                OPTIX_CHECK(
                    optixAccelCompact(context, 0, IASHandle, IASBuffer.d_pointer(), compacted_ias_size, &IASHandle));
                d_buffer_temp_output_ias.free();
                iasCompactedSizeBuffer.free();

                IASBufferSize = compacted_ias_size;
            } else {
                IASBuffer.move(d_buffer_temp_output_ias);

                IASBufferSize = ias_buffer_sizes.outputSizeInBytes;
            }

            CUDA_CHECK(cudaStreamSynchronize(0));
        } else {
            IASBuffer.alloc(ias_buffer_sizes.outputSizeInBytes * bufferScale);
            OPTIX_CHECK(optixAccelBuild(context, 0, &buildOptions, &buildInput, 1, tempBuffer.d_pointer(),
                ias_buffer_sizes.tempSizeInBytes, IASBuffer.d_pointer(), ias_buffer_sizes.outputSizeInBytes, &IASHandle,
                nullptr, 0));

            IASBufferSize = ias_buffer_sizes.outputSizeInBytes;
            CUDA_SYNC_CHECK();
        }


        if (ias_buffer_sizes.tempUpdateSizeInBytes * bufferScale > tempBuffer.sizeInBytes) {
            tempBuffer.resize(ias_buffer_sizes.tempUpdateSizeInBytes * bufferScale);
        }
    }
} // namespace RTCD
