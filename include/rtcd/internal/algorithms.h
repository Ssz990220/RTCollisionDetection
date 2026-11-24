#pragma once
// rtcd/internal/algorithms.h -- host-callable CUDA algorithms shared by the backends (library rtcd_core,
// Wave 1 a). INTERNAL: not part of the user API, but a frozen contract between core and the Wave 2 backends.
//
// Every function enqueues work on `stream` and returns immediately (no host synchronization, no allocation).
// Scratch memory is passed in; its required size is a host-side pure function of the capacity, so backends
// allocate scratch once at construction for their maximum element count. Passing too little scratch throws
// CapacityError before anything is enqueued. Launch errors throw CudaError.

#include <rtcd/limits.h>
#include <rtcd/types.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace rtcd::internal {

    // ---- OBB culling ------------------------------------------------------------------------------------------
    // counts[i] = number of j with overlap(a[i], b[j]) for every i < a.count (written, no pre-zeroing needed).
    // Separating-axis test bit-identical to the old BBCDRobotMask / checkOBBCollision
    // (RTCD/CollisionDetector/BBCollisionDetection.cu): 15 axes in the old order, "separated iff d > ra + rb".
    void obbOverlapCounts(DeviceSpan<const Obb> a, DeviceSpan<const Obb> b, DeviceSpan<std::uint32_t> counts,
        StreamHandle stream);

    // ---- scans / compaction ----------------------------------------------------------------------------------
    std::size_t exclusiveScanScratchBytes(std::size_t maxCount);
    // out[i] = sum_{k < i} in[k] (uint32, wrap-around). out may alias in.
    void exclusiveScanU32(DeviceSpan<const std::uint32_t> in, DeviceSpan<std::uint32_t> out,
        DeviceSpan<std::byte> scratch, StreamHandle stream);

    // out[i] = in[i] > 0 ? 1 : 0. out may alias in. (old transformIsValid<uint, true>)
    void flagsToBinaryU32(DeviceSpan<const std::uint32_t> in, DeviceSpan<std::uint32_t> out, StreamHandle stream);

    // Stable compaction of up to 4 parallel arrays by one flag array: element i of every array is kept iff
    // flags[i] > 0, order preserved (equivalent to cub::DeviceSelect::FlaggedIf with flag > 0, which the old IAS
    // path used). *count (device uint32) receives the number of kept elements. dst and src must not overlap.
    struct CompactArray {
        DevicePtr dst            = 0;
        DevicePtr src            = 0;
        std::uint32_t elemBytes  = 0; // multiple of 4
    };
    inline constexpr std::size_t kMaxCompactArrays = 4;
    std::size_t compactScratchBytes(std::size_t maxCount, std::size_t maxElemBytes);
    void compactByFlags(std::span<const CompactArray> arrays, DeviceSpan<const std::uint32_t> flags,
        DevicePtr count, DeviceSpan<std::byte> scratch, StreamHandle stream);

} // namespace rtcd::internal
