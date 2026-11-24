#pragma once
// src/detector (private): the core's one kernel -- a 4-byte-word copy executed by the SMs, used instead of a
// copy-engine memcpy wherever one side may be mapped pinned host memory (checkHost: items read from / counters
// written to pinned buffers, zero-copy). Measured on the RTX 5080 / Windows (tests/experiments/host_breakdown,
// rtcd_exp_transfer_probe): a small H2D cudaMemcpy(Async) followed by GPU work costs 20-60 us until the work can
// start, a D2H copy right after GPU work ~20 us more than in isolation (copy-engine latency); the same transfer
// done by a kernel through the mapped buffer costs about one kernel launch (~6 us incl. the stream sync).
//
// The graph helpers let DetectorCore find and re-point that kernel when it was captured into a CUDA graph (the
// staging copy of CheckerConfig::graphPrepare).

#include <rtcd/types.h>

#include <cstddef>

struct CUgraphExec_st;
struct CUgraphNode_st;

namespace rtcd::detail {

    // dst[i] = src[i] for i < words (uint32 words), enqueued on `stream`. Either pointer may be a device pointer or
    // the device address of mapped pinned host memory. Throws CudaError on a launch failure. words == 0: no-op.
    void copyWordsAsync(DevicePtr dst, DevicePtr src, std::size_t words, StreamHandle stream);

    // True iff `node` (of a captured graph) is a kernel node of copyWordsAsync's kernel.
    bool isCopyWordsNode(CUgraphNode_st* node);

    // Re-points an instantiated copyWordsAsync kernel node (same word count as captured).
    void setCopyWordsNode(CUgraphExec_st* exec, CUgraphNode_st* node, DevicePtr dst, DevicePtr src, std::size_t words);

} // namespace rtcd::detail
