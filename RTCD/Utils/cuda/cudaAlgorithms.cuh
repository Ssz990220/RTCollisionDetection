#pragma once

#include <Utils/CUDABuffer.h>
#include <cooperative_groups.h>
typedef unsigned int uint;

inline constexpr uint THREADBLOCK_SIZE     = 512;
inline constexpr uint MIN_SHORT_ARRAY_SIZE = 4;
inline constexpr uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
inline constexpr uint MIN_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
inline constexpr uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;
inline constexpr uint MAX_BATCH_ELEMENTS   = 64 * 1048576;
namespace RTCD { // template function declaration goes here

    typedef unsigned int uint;

    inline constexpr uint roundUpLog2(uint v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    inline constexpr uint roundToShortArrSize(uint v) {
        return (v + MAX_SHORT_ARRAY_SIZE - 1) / MAX_SHORT_ARRAY_SIZE * MAX_SHORT_ARRAY_SIZE;
    }


    // ReduceSum with buffer. result will be saved at the first element of tempBuffer
    template <class T>
    void reduceSum(const CUdeviceptr d_idata, CUdeviceptr tempBuffer, const uint size, const cudaStream_t stream);


    template <bool flagCondition>
    void compactCpy(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags, const CUdeviceptr whereInDst,
        const uint eleSize, const uint nEle, const cudaStream_t stream);

    template <bool flagCondition>
    void compactCpyCurve(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags, const CUdeviceptr whereInDst,
        const CUdeviceptr whereInSrc, const uint dstEleSize, const uint srcEleSize, const uint nEle,
        const cudaStream_t stream);


    void exclusiveScan(
        CUdeviceptr d_Dst, const CUdeviceptr d_Src, CUdeviceptr tmp, const uint arrayLength, const cudaStream_t stream);

    void weightedExclusiveScan(CUdeviceptr d_Dst, const CUdeviceptr d_Src, const CUdeviceptr d_Weight, CUdeviceptr tmp,
        const uint arrayLength, const cudaStream_t stream);


    // streamCompact
    // @brief StreamCompact a stream of data array. All elements in the array have uniform size, indicated by
    // `eleSize`. Number of valid element is copied to `nValid` asynchornously, copy type specified by `kind`.
    template <bool flagCondition>
    void streamCompact(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer, uint* nValid,
        const uint eleSize, const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst, const cudaMemcpyKind kind,
        const cudaStream_t stream);

    template <bool flagCondition>
    void streamCompact(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer, const uint eleSize,
        const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst, const cudaStream_t stream);

    // template <bool flagCondition>
    // void streamCompact(const CUdeviceptr src, const CUdeviceptr flags, const uint eleSize, const uint nEle,
    //     CUdeviceptr dst, CUdeviceptr whereInDst, const cudaStream_t stream);

    // streamCompact
    // @brief StreamCompact a stream of data array. Elements in the array have varying size, indicated by
    // `streamEleCnt`, its prefix sum version is provided in `whereInSrc`. Number of valid element is copied to
    // `nValid` asynchornously, copy type specified by `kind`.
    template <bool flagCondition>
    void streamCompact(const CUdeviceptr src, const CUdeviceptr whereInSrc, const CUdeviceptr flags,
        CUdeviceptr tempBuffer, uint* nValid, const CUdeviceptr streamEleCnt, const uint eleSize, const uint nEle,
        CUdeviceptr dst, CUdeviceptr whereInDst, const cudaMemcpyKind kind, const cudaStream_t stream);

    template <bool flagCondition>
    void streamCompact(const CUdeviceptr src, const CUdeviceptr whereInSrc, const CUdeviceptr flags,
        CUdeviceptr tempBuffer, const CUdeviceptr streamEleCnt, const uint eleSize, const uint nEle, CUdeviceptr dst,
        CUdeviceptr whereInDst, const cudaStream_t stream);


    template <bool flagCondition>
    struct is_valid {
        __device__ uint operator()(const uint& x) const {
            return (uint) ((flagCondition && x > 0) || (!flagCondition && x == 0));
        }
    };


    template <bool flagCondition>
    struct size_valid {
        __device__ uint operator()(const uint& size, const uint& x) const {
            return size * (uint) ((flagCondition && x > 0) || (!flagCondition && x == 0));
        }
    };

    template <typename T, typename Functor>
    void transformArray(
        const CUdeviceptr input, CUdeviceptr output, Functor functor, const size_t size, const cudaStream_t stream = 0);

    template <typename T, typename Functor>
    void transformArray(const CUdeviceptr input, const CUdeviceptr input2, CUdeviceptr output, Functor functor,
        const size_t size, const cudaStream_t stream = 0);

    template <typename T, bool flagCondition>
    void transformIsValid(const CUdeviceptr input, CUdeviceptr output, const size_t size, const cudaStream_t stream);


    void exclusiveScan(
        CUdeviceptr input, CUdeviceptr output, RTCD::CUDABuffer& tmp, size_t N, size_t& tmpSize, cudaStream_t stream);
} // namespace RTCD

#if defined(__CUDACC__)
namespace RTCD { // template function definition goes here, only in NVCC

    namespace cg = cooperative_groups;

    template <class T>
    struct SharedMemory {
        __device__ inline operator T*() {
            extern __shared__ int __smem[];
            return (T*) __smem;
        }

        __device__ inline operator const T*() const {
            extern __shared__ int __smem[];
            return (T*) __smem;
        }
    };

#include "reduce.inl"
#include "streamCompact.inl"
#include "transform.inl"
} // namespace RTCD


#endif
