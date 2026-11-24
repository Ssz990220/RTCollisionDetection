// src/detector: copyWordsAsync and its graph-node helpers (detector_kernels.h).
#include "detector/detector_kernels.h"

#include <rtcd/errors.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace rtcd::detail {

    namespace {
        constexpr unsigned int kThreads = 128;

        __global__ void copyWordsKernel(std::uint32_t* dst, const std::uint32_t* src, std::size_t words) {
            const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < words) {
                dst[i] = src[i];
            }
        }

        unsigned int blocksFor(std::size_t words) { return static_cast<unsigned int>((words + kThreads - 1) / kThreads); }
        void* kernelFunc() { return reinterpret_cast<void*>(&copyWordsKernel); }
    } // namespace

    void copyWordsAsync(DevicePtr dst, DevicePtr src, std::size_t words, StreamHandle stream) {
        if (words == 0) {
            return;
        }
        copyWordsKernel<<<blocksFor(words), kThreads, 0, stream>>>(
            reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(dst)),
            reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(src)), words);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    bool isCopyWordsNode(CUgraphNode_st* node) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) {
            return false;
        }
        cudaKernelNodeParams p{};
        if (cudaGraphKernelNodeGetParams(node, &p) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        return p.func == kernelFunc();
    }

    void setCopyWordsNode(CUgraphExec_st* exec, CUgraphNode_st* node, DevicePtr dst, DevicePtr src, std::size_t words) {
        std::uint32_t* d       = reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(dst));
        const std::uint32_t* s = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(src));
        void* args[]           = {&d, &s, &words};
        cudaKernelNodeParams p{};
        p.func           = kernelFunc();
        p.gridDim        = dim3(blocksFor(words), 1, 1);
        p.blockDim       = dim3(kThreads, 1, 1);
        p.sharedMemBytes = 0;
        p.kernelParams   = args;
        p.extra          = nullptr;
        RTCD_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec, node, &p));
    }

} // namespace rtcd::detail
