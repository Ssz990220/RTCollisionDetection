// rtcd_core: host-callable CUDA algorithms of rtcd/internal/algorithms.h (OBB overlap counts, exclusive scan,
// flag binarization, stable multi-array compaction). Every function only enqueues work on the given stream: no
// allocation, no synchronization. Kernels and device helpers stay inside this translation unit.
#include <rtcd/errors.h>
#include <rtcd/internal/algorithms.h>

#include <cub/device/device_scan.cuh>
#include <cuda/iterator>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <string>

namespace rtcd::internal {

    namespace {

        // ---- OBB separating-axis test ---------------------------------------------------------------------
        // Bit-identical port of RTCD/CollisionDetector/BBCollisionDetection.cu (isSeparatingAxis,
        // checkOBBCollision, BBCDRobotMaskKern) with the float3 helpers of RTCD/Utils/cuUtils.cuh written out in
        // the same expression order (same rounding / FMA contraction as the old translation unit, which is also
        // compiled without fast math).
        struct ObbF3 {
            float3 center;
            float3 halfSize;
            float3 orientation[3];
        };
        static_assert(sizeof(ObbF3) == sizeof(Obb), "rtcd::Obb must have the old RTCD::OBB layout");

        __forceinline__ __device__ float dot3(const float3& a, const float3& b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }
        __forceinline__ __device__ float3 cross3(const float3& a, const float3& b) {
            return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        }
        __forceinline__ __device__ float3 sub3(const float3& a, const float3& b) {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }

        __forceinline__ __device__ bool isSeparatingAxis(const ObbF3& obb1, const ObbF3& obb2, const float3& axis) {
            float3 u[3] = {obb1.orientation[0], obb1.orientation[1], obb1.orientation[2]};
            float3 v[3] = {obb2.orientation[0], obb2.orientation[1], obb2.orientation[2]};

            float ra = obb1.halfSize.x * fabsf(dot3(axis, u[0])) + obb1.halfSize.y * fabsf(dot3(axis, u[1]))
                     + obb1.halfSize.z * fabsf(dot3(axis, u[2]));

            float rb = obb2.halfSize.x * fabsf(dot3(axis, v[0])) + obb2.halfSize.y * fabsf(dot3(axis, v[1]))
                     + obb2.halfSize.z * fabsf(dot3(axis, v[2]));

            float3 t = sub3(obb2.center, obb1.center);
            float d  = fabsf(dot3(t, axis));

            return d > ra + rb;
        }

        __device__ bool checkObbCollision(const ObbF3& obb1, const ObbF3& obb2) {
            float3 u[3] = {obb1.orientation[0], obb1.orientation[1], obb1.orientation[2]};
            float3 v[3] = {obb2.orientation[0], obb2.orientation[1], obb2.orientation[2]};

            bool isColliding = true;
            for (int i = 0; i < 3; ++i) {
                if (isSeparatingAxis(obb1, obb2, u[i]) || isSeparatingAxis(obb1, obb2, v[i])
                    || isSeparatingAxis(obb1, obb2, cross3(u[i], v[0]))
                    || isSeparatingAxis(obb1, obb2, cross3(u[i], v[1]))
                    || isSeparatingAxis(obb1, obb2, cross3(u[i], v[2]))) {
                    isColliding = false;
                    break;
                }
            }
            return isColliding;
        }

        constexpr unsigned int kObbBlockX = 32; // a-OBBs per block (old BBCD_RM_X)
        constexpr unsigned int kObbBlockY = 8;  // b lanes per block (old BBCD_RM_Y)

        __global__ void obbOverlapCountsKernel(
            const ObbF3* a, const ObbF3* b, const size_t nA, const size_t nB, std::uint32_t* counts) {
            __shared__ std::uint32_t partial[kObbBlockY][kObbBlockX];
            const size_t i = size_t(blockIdx.x) * kObbBlockX + threadIdx.x;

            std::uint32_t cnt = 0;
            if (i < nA) {
                const ObbF3 obbA = a[i];
                for (size_t s = threadIdx.y; s < nB; s += kObbBlockY) {
                    cnt += checkObbCollision(obbA, b[s]) ? 1u : 0u;
                }
            }
            partial[threadIdx.y][threadIdx.x] = cnt;
            __syncthreads();
            if (threadIdx.y == 0 && i < nA) {
                std::uint32_t total = 0;
#pragma unroll
                for (unsigned int y = 0; y < kObbBlockY; y++) {
                    total += partial[y][threadIdx.x];
                }
                counts[i] = total;
            }
        }

        // ---- flags ----------------------------------------------------------------------------------------
        struct IsPositive {
            __host__ __device__ std::uint32_t operator()(std::uint32_t x) const { return x > 0 ? 1u : 0u; }
        };

        __global__ void flagsToBinaryKernel(const std::uint32_t* in, std::uint32_t* out, size_t n) {
            const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) {
                out[i] = in[i] > 0 ? 1u : 0u;
            }
        }

        // ---- compaction -----------------------------------------------------------------------------------
        struct CompactArgs {
            std::uint32_t* dst[kMaxCompactArrays];
            const std::uint32_t* src[kMaxCompactArrays];
            std::uint32_t words[kMaxCompactArrays]; // elemBytes / 4
        };

        // blockIdx.y = array; x walks (element, word) pairs so that consecutive threads copy consecutive words.
        // positions = exclusive scan of (flags > 0). Thread 0 of array 0 writes the kept count.
        __global__ void compactCopyKernel(CompactArgs args, const std::uint32_t* flags, const std::uint32_t* positions,
            size_t n, std::uint32_t* count) {
            const unsigned int a = blockIdx.y;
            const size_t t       = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (a == 0 && t == 0) {
                *count = positions[n - 1] + (flags[n - 1] > 0 ? 1u : 0u);
            }
            const std::uint32_t words = args.words[a];
            if (t >= n * words) {
                return;
            }
            const size_t i = t / words;
            if (flags[i] == 0) {
                return;
            }
            const size_t w                                         = t - i * words;
            args.dst[a][size_t(positions[i]) * words + w] = args.src[a][i * words + w];
        }

        using BinaryFlagIterator = cuda::transform_iterator<IsPositive, const std::uint32_t*>;

        constexpr std::size_t kAlign = 256;
        constexpr std::size_t alignUp(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

        int checkedCount(const char* fn, std::size_t n) {
            if (n > std::size_t(INT_MAX)) {
                throw InvalidArgument(std::string(fn) + ": element count " + std::to_string(n)
                                      + " exceeds the supported maximum of " + std::to_string(INT_MAX));
            }
            return static_cast<int>(n);
        }

        std::size_t cubScanBytes(int n) {
            std::size_t bytes = 0;
            RTCD_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<const std::uint32_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr), n));
            return bytes;
        }

        std::size_t cubFlagScanBytes(int n) {
            std::size_t bytes = 0;
            RTCD_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes,
                BinaryFlagIterator(static_cast<const std::uint32_t*>(nullptr), IsPositive{}),
                static_cast<std::uint32_t*>(nullptr), n));
            return bytes;
        }

        template <typename T>
        T* dptr(DevicePtr p) {
            return reinterpret_cast<T*>(static_cast<std::uintptr_t>(p));
        }

    } // namespace

    // ---- OBB culling ----------------------------------------------------------------------------------------

    void obbOverlapCounts(
        DeviceSpan<const Obb> a, DeviceSpan<const Obb> b, DeviceSpan<std::uint32_t> counts, StreamHandle stream) {
        if (counts.count < a.count) {
            throw InvalidArgument("obbOverlapCounts: counts holds " + std::to_string(counts.count)
                                  + " elements, need a.count = " + std::to_string(a.count));
        }
        if (a.count == 0) {
            return;
        }
        if ((a.ptr == 0) || (b.count != 0 && b.ptr == 0) || counts.ptr == 0) {
            throw InvalidArgument("obbOverlapCounts: null device pointer");
        }
        const std::size_t blocks = (a.count + kObbBlockX - 1) / kObbBlockX;
        if (blocks > std::size_t(INT_MAX)) {
            throw InvalidArgument("obbOverlapCounts: a.count = " + std::to_string(a.count) + " is too large");
        }
        const dim3 block(kObbBlockX, kObbBlockY, 1);
        const dim3 grid(static_cast<unsigned int>(blocks), 1, 1);
        obbOverlapCountsKernel<<<grid, block, 0, stream>>>(dptr<const ObbF3>(a.ptr), dptr<const ObbF3>(b.ptr), a.count,
            b.count, dptr<std::uint32_t>(counts.ptr));
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    // ---- scans / compaction ---------------------------------------------------------------------------------

    std::size_t exclusiveScanScratchBytes(std::size_t maxCount) {
        return cubScanBytes(checkedCount("exclusiveScanScratchBytes", maxCount));
    }

    void exclusiveScanU32(DeviceSpan<const std::uint32_t> in, DeviceSpan<std::uint32_t> out,
        DeviceSpan<std::byte> scratch, StreamHandle stream) {
        if (out.count < in.count) {
            throw InvalidArgument("exclusiveScanU32: out holds " + std::to_string(out.count)
                                  + " elements, need " + std::to_string(in.count));
        }
        const int n = checkedCount("exclusiveScanU32", in.count);
        if (n == 0) {
            return;
        }
        std::size_t need = cubScanBytes(n);
        if (scratch.count < need) {
            throw CapacityError("exclusiveScanU32: scratch of " + std::to_string(scratch.count) + " bytes, need "
                                    + std::to_string(need) + " for " + std::to_string(n)
                                    + " elements (size it with exclusiveScanScratchBytes)",
                "scan scratch", need, scratch.count);
        }
        RTCD_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dptr<void>(scratch.ptr), need,
            dptr<const std::uint32_t>(in.ptr), dptr<std::uint32_t>(out.ptr), n, stream));
    }

    void flagsToBinaryU32(DeviceSpan<const std::uint32_t> in, DeviceSpan<std::uint32_t> out, StreamHandle stream) {
        if (out.count < in.count) {
            throw InvalidArgument("flagsToBinaryU32: out holds " + std::to_string(out.count)
                                  + " elements, need " + std::to_string(in.count));
        }
        if (in.count == 0) {
            return;
        }
        constexpr unsigned int threads = 256; // as the old transformIsValid
        const std::size_t blocks       = (in.count + threads - 1) / threads;
        if (blocks > std::size_t(INT_MAX)) {
            throw InvalidArgument("flagsToBinaryU32: count " + std::to_string(in.count) + " is too large");
        }
        flagsToBinaryKernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            dptr<const std::uint32_t>(in.ptr), dptr<std::uint32_t>(out.ptr), in.count);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    std::size_t compactScratchBytes(std::size_t maxCount, std::size_t /*maxElemBytes*/) {
        // [positions: maxCount uint32, 256-aligned][cub scan temp]. Independent of the element size: elements are
        // copied directly from src to dst.
        const int n = checkedCount("compactScratchBytes", maxCount);
        return alignUp(std::size_t(n) * sizeof(std::uint32_t)) + cubFlagScanBytes(n);
    }

    void compactByFlags(std::span<const CompactArray> arrays, DeviceSpan<const std::uint32_t> flags, DevicePtr count,
        DeviceSpan<std::byte> scratch, StreamHandle stream) {
        // ---- validate everything before enqueuing anything
        if (arrays.size() > kMaxCompactArrays) {
            throw InvalidArgument("compactByFlags: " + std::to_string(arrays.size()) + " arrays, at most "
                                  + std::to_string(kMaxCompactArrays) + " are supported");
        }
        if (count == 0) {
            throw InvalidArgument("compactByFlags: null count pointer");
        }
        const int n = checkedCount("compactByFlags", flags.count);
        if (n > 0 && flags.ptr == 0) {
            throw InvalidArgument("compactByFlags: null flags pointer");
        }
        for (std::size_t k = 0; k < arrays.size(); ++k) {
            const CompactArray& a = arrays[k];
            const std::string id  = "compactByFlags: array " + std::to_string(k);
            if (a.elemBytes == 0 || a.elemBytes % 4 != 0) {
                throw InvalidArgument(id + ": elemBytes = " + std::to_string(a.elemBytes)
                                      + " must be a positive multiple of 4");
            }
            if (n == 0) {
                continue;
            }
            if (a.dst == 0 || a.src == 0) {
                throw InvalidArgument(id + ": null dst or src pointer");
            }
            if (a.dst % 4 != 0 || a.src % 4 != 0) {
                throw InvalidArgument(id + ": dst and src must be 4-byte aligned");
            }
            const DevicePtr bytes = DevicePtr(n) * a.elemBytes;
            if (a.dst < a.src + bytes && a.src < a.dst + bytes) {
                throw InvalidArgument(id + ": dst and src overlap");
            }
        }
        std::size_t need = 0;
        std::size_t posBytes = 0;
        std::size_t scanBytes = 0;
        if (n > 0) {
            posBytes  = alignUp(std::size_t(n) * sizeof(std::uint32_t));
            scanBytes = cubFlagScanBytes(n);
            need      = posBytes + scanBytes;
        }
        if (scratch.count < need) {
            throw CapacityError("compactByFlags: scratch of " + std::to_string(scratch.count) + " bytes, need "
                                    + std::to_string(need) + " for " + std::to_string(n)
                                    + " elements (size it with compactScratchBytes)",
                "compaction scratch", need, scratch.count);
        }
        if (n > 0 && scratch.ptr % kAlign != 0) {
            throw InvalidArgument("compactByFlags: scratch must be 256-byte aligned (a DeviceBuffer start is)");
        }

        // ---- enqueue
        if (n == 0) {
            RTCD_CUDA_CHECK(cudaMemsetAsync(dptr<void>(count), 0, sizeof(std::uint32_t), stream));
            return;
        }
        std::uint32_t* positions = dptr<std::uint32_t>(scratch.ptr);
        void* scanTemp           = dptr<void>(scratch.ptr + posBytes);
        RTCD_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scanTemp, scanBytes,
            BinaryFlagIterator(dptr<const std::uint32_t>(flags.ptr), IsPositive{}), positions, n, stream));

        CompactArgs args{};
        std::uint32_t maxWords = 0;
        for (std::size_t k = 0; k < arrays.size(); ++k) {
            args.dst[k]   = dptr<std::uint32_t>(arrays[k].dst);
            args.src[k]   = dptr<const std::uint32_t>(arrays[k].src);
            args.words[k] = arrays[k].elemBytes / 4;
            maxWords      = args.words[k] > maxWords ? args.words[k] : maxWords;
        }
        // With no arrays, launch one "array" of zero words so the count is still written.
        const unsigned int numArrays = arrays.empty() ? 1u : static_cast<unsigned int>(arrays.size());
        if (arrays.empty()) {
            args.words[0] = 0;
        }
        constexpr unsigned int threads = 256;
        const std::size_t work         = std::size_t(n) * (maxWords > 0 ? maxWords : 1);
        const std::size_t blocks       = (work + threads - 1) / threads;
        if (blocks > std::size_t(INT_MAX)) {
            throw InvalidArgument("compactByFlags: too much data for one launch");
        }
        compactCopyKernel<<<dim3(static_cast<unsigned int>(blocks), numArrays, 1), threads, 0, stream>>>(
            args, dptr<const std::uint32_t>(flags.ptr), positions, std::size_t(n), dptr<std::uint32_t>(count));
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

} // namespace rtcd::internal
