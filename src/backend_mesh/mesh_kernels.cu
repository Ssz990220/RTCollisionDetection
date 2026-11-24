// rtcd_backend_mesh: instance-buffer kernels (old upInsVisibility in RTCD/Robot/iasRobotKernels.cu and upInsPoses
// in RTCD/Robot/robot.cu). Pure copies of float data, so results are bit-identical to the old kernels.
#include "backend_mesh/mesh_kernels.h"

#include <rtcd/errors.h>

#include <cuda_runtime.h>
#include <optix_types.h>

#include "core/obb_sat.cuh"

namespace rtcd::mesh_detail {

    namespace {
        static_assert(sizeof(OptixInstance) == 80, "OptixInstance layout");

        constexpr unsigned int kThreads = 128; // as the old kernels

        __global__ void transformsAndMasksKernel(
            OptixInstance* instances, const float4* transforms, const unsigned int* counts, const size_t n) {
            const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            const float4* tf = transforms + idx * 3;
            float4* dst      = reinterpret_cast<float4*>(instances[idx].transform); // OptixInstance is 16-byte aligned
            dst[0]           = tf[0];
            dst[1]           = tf[1];
            dst[2]           = tf[2];
            const bool active             = counts == nullptr || counts[idx] > 0;
            instances[idx].visibilityMask = active ? 255u : 0u;
        }

        // Masked path, fused: the OBB test of internal::obbOverlapCounts (same device code, core/obb_sat.cuh, same
        // argument order) decides the visibility mask directly; the count itself is not needed there.
        __global__ void transformsAndOverlapMasksKernel(OptixInstance* instances, const float4* transforms,
            const internal::obb_sat::ObbF3* linkObbs, const internal::obb_sat::ObbF3* sceneObbs, const size_t nScene,
            const size_t n) {
            const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            const float4* tf = transforms + idx * 3;
            float4* dst      = reinterpret_cast<float4*>(instances[idx].transform);
            dst[0]           = tf[0];
            dst[1]           = tf[1];
            dst[2]           = tf[2];
            const internal::obb_sat::ObbF3 obbA = linkObbs[idx];
            bool active                         = false;
            for (size_t s = 0; s < nScene && !active; ++s) {
                active = internal::obb_sat::checkObbCollision(obbA, sceneObbs[s]);
            }
            instances[idx].visibilityMask = active ? 255u : 0u;
        }

        __global__ void transformsParkedKernel(OptixInstance* instances, const float4* transforms, const size_t nActive,
            const size_t total, const ParkTransform park) {
            const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= total) {
                return;
            }
            float* dst = instances[idx].transform;
            if (idx >= nActive) {
                for (int k = 0; k < kTransformFloats; ++k) {
                    dst[k] = park.m[k];
                }
            } else {
                const float4* tf  = transforms + idx * 3;
                float4* dst4      = reinterpret_cast<float4*>(dst);
                dst4[0]           = tf[0];
                dst4[1]           = tf[1];
                dst4[2]           = tf[2];
            }
        }

        unsigned int blocksFor(std::size_t n) { return static_cast<unsigned int>((n + kThreads - 1) / kThreads); }
    } // namespace

    void writeInstanceTransformsAndMasks(
        DevicePtr instances, DevicePtr linkPoses, DevicePtr counts, std::size_t n, StreamHandle stream) {
        if (n == 0) {
            return;
        }
        transformsAndMasksKernel<<<blocksFor(n), kThreads, 0, stream>>>(reinterpret_cast<OptixInstance*>(instances),
            reinterpret_cast<const float4*>(linkPoses), reinterpret_cast<const unsigned int*>(counts), n);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    void writeInstanceTransformsAndOverlapMasks(DevicePtr instances, DevicePtr linkPoses, DeviceSpan<const Obb> linkObbs,
        DeviceSpan<const Obb> sceneObbs, StreamHandle stream) {
        const std::size_t n = linkObbs.count;
        if (n == 0) {
            return;
        }
        using internal::obb_sat::ObbF3;
        transformsAndOverlapMasksKernel<<<blocksFor(n), kThreads, 0, stream>>>(reinterpret_cast<OptixInstance*>(instances),
            reinterpret_cast<const float4*>(linkPoses), reinterpret_cast<const ObbF3*>(linkObbs.ptr),
            reinterpret_cast<const ObbF3*>(sceneObbs.ptr), sceneObbs.count, n);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    void writeInstanceTransformsParked(DevicePtr instances, DevicePtr linkPoses, std::size_t nActive, std::size_t total,
        const ParkTransform& park, StreamHandle stream) {
        if (total == 0) {
            return;
        }
        transformsParkedKernel<<<blocksFor(total), kThreads, 0, stream>>>(reinterpret_cast<OptixInstance*>(instances),
            reinterpret_cast<const float4*>(linkPoses), nActive, total, park);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

} // namespace rtcd::mesh_detail
