#include "cudaAlgorithms.cuh"
#include <assert.h>
#if defined(_DEBUG) || defined(DEBUG)
#include <iostream>
#include <vector>
#endif
#include <cub/cub.cuh>

namespace RTCD {

    ////////////////////////////////////////////////////////////////////////////////
    // Basic scan codelets
    ////////////////////////////////////////////////////////////////////////////////
    // Naive inclusive scan: O(N * log2(N)) operations
    // Allocate 2 * 'size' local memory, initialize the first half
    // with 'size' zeros avoiding if(pos >= offset) condition evaluation
    // and saving instructions
    inline __device__ uint scan1Inclusive(uint idata, volatile uint* s_Data, uint size, cg::thread_block cta) {
        uint pos    = 2 * threadIdx.x - (threadIdx.x & (size - 1));
        s_Data[pos] = 0;
        pos += size;
        s_Data[pos] = idata;

        for (uint offset = 1; offset < size; offset <<= 1) {
            cg::sync(cta);
            uint t = s_Data[pos] + s_Data[pos - offset];
            cg::sync(cta);
            s_Data[pos] = t;
        }

        return s_Data[pos];
    }

    inline __device__ uint scan1Exclusive(uint idata, volatile uint* s_Data, uint size, cg::thread_block cta) {
        return scan1Inclusive(idata, s_Data, size, cta) - idata;
    }

    inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint* s_Data, uint size, cg::thread_block cta) {
        // Level-0 inclusive scan
        idata4.y += idata4.x;
        idata4.z += idata4.y;
        idata4.w += idata4.z;

        // Level-1 exclusive scan
        uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, cta);

        idata4.x += oval;
        idata4.y += oval;
        idata4.z += oval;
        idata4.w += oval;

        return idata4;
    }

    // Exclusive vector scan: the array to be scanned is stored
    // in local thread memory scope as uint4
    inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint* s_Data, uint size, cg::thread_block cta) {
        uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
        odata4.x -= idata4.x;
        odata4.y -= idata4.y;
        odata4.z -= idata4.z;
        odata4.w -= idata4.w;
        return odata4;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Scan kernels
    ////////////////////////////////////////////////////////////////////////////////
    __global__ void scanExclusiveShared(uint4* d_Dst, uint4* d_Src, uint roundedSize, uint size, uint blockSize) {
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

        uint pos = blockIdx.x * blockDim.x + threadIdx.x;
        if ((pos + 1) * 4 > roundedSize) {
            return;
        }

        // Load data
        uint4 idata4;
        if (pos * 4 < size) {
            if ((pos + 1) * 4 > size) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (pos * 4 + i < size) {
                        ((uint*) &idata4)[i] = ((uint*) d_Src)[pos * 4 + i];
                    } else {
                        ((uint*) &idata4)[i] = 0;
                    }
                }
            } else {
                idata4 = d_Src[pos];
            }
        } else {
            idata4 = make_uint4(0, 0, 0, 0);
        }

        // Calculate exclusive scan
        uint4 odata4 = scan4Exclusive(idata4, s_Data, blockSize, cta);

        // Write back
        d_Dst[pos] = odata4;
    }

    // Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
    __global__ void scanExclusiveShared2(uint* d_Buf, uint* d_Dst, uint* d_Src, uint N) {
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

        // Skip loads and stores for inactive threads of last threadblock (pos >= N)
        uint pos = blockIdx.x * blockDim.x + threadIdx.x;

        // Load top elements
        // Convert results of bottom-level scan back to inclusive
        uint idata = 0;

        if (pos < N) {
            idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos]
                  + d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];
        }

        // Compute
        uint odata = scan1Exclusive(idata, s_Data, N, cta);

        // Avoid out-of-bound access
        if (pos < N) {
            d_Buf[pos] = odata;
        }
    }

    // Final step of large-array scan: combine basic inclusive scan with exclusive
    // scan of top elements of input arrays
    __global__ void uniformUpdate(uint4* d_Data, uint* d_Buffer) {
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        __shared__ uint buf;
        uint pos = blockIdx.x * blockDim.x + threadIdx.x;

        if (threadIdx.x == 0) {
            buf = d_Buffer[blockIdx.x];
        }

        cg::sync(cta);

        uint4 data4 = d_Data[pos];
        data4.x += buf;
        data4.y += buf;
        data4.z += buf;
        data4.w += buf;
        d_Data[pos] = data4;
    }

    static uint iDivUp(uint dividend, uint divisor) {
        return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
    }


    void scanExclusiveShort(
        CUdeviceptr d_Dst, const CUdeviceptr d_Src, const uint arrayLength, const cudaStream_t stream) {
        uint roundedSize = roundUpLog2(arrayLength);

        // Check supported size range
        assert((roundedSize >= MIN_SHORT_ARRAY_SIZE) && (roundedSize <= MAX_SHORT_ARRAY_SIZE));

        scanExclusiveShared<<<(roundedSize + (4 * THREADBLOCK_SIZE) - 1) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE, 0,
            stream>>>(
            reinterpret_cast<uint4*>(d_Dst), reinterpret_cast<uint4*>(d_Src), roundedSize, arrayLength, roundedSize);
    }

    void scanExclusiveLarge(CUdeviceptr d_Dst, const CUdeviceptr d_Src, CUdeviceptr tmp, const uint arrayLength,
        const cudaStream_t stream) {
        // Check power-of-two factorization
        uint roundedSize = roundUpLog2(arrayLength);

        // Check supported size range
        assert(arrayLength > MIN_SHORT_ARRAY_SIZE && arrayLength <= MAX_LARGE_ARRAY_SIZE);

        scanExclusiveShared<<<(roundedSize + (4 * THREADBLOCK_SIZE) - 1) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE, 0,
            stream>>>(reinterpret_cast<uint4*>(d_Dst), reinterpret_cast<uint4*>(d_Src), roundedSize, arrayLength,
            4 * THREADBLOCK_SIZE);
        CUDA_CHECK_LAST("scanExclusiveShared() execution FAILED\n");

        // Not all threadblocks need to be packed with input data:
        // inactive threads of highest threadblock just don't do global reads and
        // writes
        const uint blockCount2 = iDivUp(roundedSize / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
        scanExclusiveShared2<<<blockCount2, THREADBLOCK_SIZE, 0, stream>>>(reinterpret_cast<uint*>(tmp),
            reinterpret_cast<uint*>(d_Dst), reinterpret_cast<uint*>(d_Src), roundedSize / (4 * THREADBLOCK_SIZE));
        CUDA_CHECK_LAST("scanExclusiveShared2() execution FAILED\n");

        uniformUpdate<<<roundedSize / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<uint4*>(d_Dst), reinterpret_cast<uint*>(tmp));
        CUDA_CHECK_LAST("uniformUpdate() execution FAILED\n");
    }

    void exclusiveScan(CUdeviceptr d_Dst, const CUdeviceptr d_Src, CUdeviceptr tmp, const uint arrayLength,
        const cudaStream_t stream) {
        if (arrayLength <= MAX_SHORT_ARRAY_SIZE) {
            scanExclusiveShort(d_Dst, d_Src, arrayLength, stream);
        } else {
            scanExclusiveLarge(d_Dst, d_Src, tmp, arrayLength, stream);
        }
    }

    void exclusiveScan(
        CUdeviceptr input, CUdeviceptr output, RTCD::CUDABuffer& tmp, size_t N, size_t& tmpSize, cudaStream_t stream) {
        if (tmpSize == 0) { // if first time,
            cub::DeviceScan::ExclusiveSum(
                nullptr, tmpSize, reinterpret_cast<uint*>(input), reinterpret_cast<uint*>(input), N, stream);
        }

        if (tmp.sizeInBytes < tmpSize) {
            tmp.freeAsync(stream);
            tmp.allocAsync(tmpSize, stream);
        }

        cub::DeviceScan::ExclusiveSum(reinterpret_cast<void*>(tmp.d_pointer()), tmpSize, reinterpret_cast<uint*>(input),
            reinterpret_cast<uint*>(output), N, stream);
    }
} // namespace RTCD

namespace RTCD { // reduce specialization

    template void reduceSum<uint>(
        const CUdeviceptr d_idata, CUdeviceptr d_odata, const uint size, const cudaStream_t stream);

    template void reduceSum<int>(
        const CUdeviceptr d_idata, CUdeviceptr d_odata, const uint size, const cudaStream_t stream);

    template void reduceSum<float>(
        const CUdeviceptr d_idata, CUdeviceptr d_odata, const uint size, const cudaStream_t stream);

    template void reduceSum<size_t>(
        const CUdeviceptr d_idata, CUdeviceptr d_odata, const uint size, const cudaStream_t stream);

    template void transform_reduce<uint>(const CUdeviceptr d_idata, const CUdeviceptr d_weight, CUdeviceptr d_odata,
        const uint size, const cudaStream_t stream);
} // namespace RTCD

namespace RTCD { // transformArray specialization
    template void transformIsValid<uint, false>(
        const CUdeviceptr src, CUdeviceptr dst, const size_t nEle, const cudaStream_t stream);
    template void transformIsValid<uint, true>(
        const CUdeviceptr src, CUdeviceptr dst, const size_t nEle, const cudaStream_t stream);
} // namespace RTCD

namespace RTCD { // stream compact specialization

    template void compactCpy<true>(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags,
        const CUdeviceptr whereInDst, const uint eleSize, const uint nEle, const cudaStream_t stream);
    template void compactCpy<false>(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags,
        const CUdeviceptr whereInDst, const uint eleSize, const uint nEle, const cudaStream_t stream);

    template void compactCpyCurve<true>(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags,
        const CUdeviceptr whereInDst, const CUdeviceptr whereInSrc, const uint dstEleSize, const uint srcEleSize,
        const uint nEle, const cudaStream_t stream);
    template void compactCpyCurve<false>(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags,
        const CUdeviceptr whereInDst, const CUdeviceptr whereInSrc, const uint dstEleSize, const uint srcEleSize,
        const uint nEle, const cudaStream_t stream);

    template void streamCompact<true>(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer,
        uint* nValid, const uint eleSize, const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst,
        const cudaMemcpyKind kind, const cudaStream_t stream);
    template void streamCompact<false>(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer,
        uint* nValid, const uint eleSize, const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst,
        const cudaMemcpyKind kind, const cudaStream_t stream);

    template void streamCompact<true>(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer,
        const uint eleSize, const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst, const cudaStream_t stream);
    template void streamCompact<false>(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer,
        const uint eleSize, const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst, const cudaStream_t stream);
} // namespace RTCD
