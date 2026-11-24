#pragma once
// src/detector (private): DetectorCore -- the mode-agnostic engine behind CollisionChecker (rtcd/checker.h).
//
// It owns the stream lease, the backend (created through an injected factory), the OptiX pipeline / SBT built
// from the backend's ProgramSet, all per-stream buffers (staging, hit counts, launch params + upload cache), the
// input / result buffers, the worker threads, and runs the batch loop of rtcd/backend.h (design doc section 4).
// It never looks at the mode: the only mode switch of the detector is the default factory
// (backend_factory.h), called by the CollisionChecker facade.
//
// The core does not need a RobotModel / Scene itself: it forwards them to the backend factory (they may be null
// when the factory does not need them, as in the fake-backend tests) and takes the item layout and joint limits
// as plain values. Tests construct a DetectorCore directly with a fake backend; the facade is a thin wrapper.

#include <rtcd/backend.h>
#include <rtcd/checker.h>
#include <rtcd/config.h>
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>

#include "detector/phase_trace.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

// CUDA graph handle types (cudaGraph_t = CUgraph_st*, ...), without including a CUDA header here.
struct CUgraph_st;
struct CUgraphExec_st;
struct CUgraphNode_st;

namespace rtcd::detail {

    class OptixProgramPipeline;
    class StreamWorkers;

    using BackendFactory = std::function<std::unique_ptr<Backend>(const BackendCreateInfo&)>;

    struct DetectorCoreDesc {
        std::shared_ptr<Context> context;
        CheckerConfig config;   // must already be resolved (CheckerConfig::resolved())
        std::uint32_t dof = 0;  // 1 .. kMaxDof
        // Joint limits for warmup() items (dof entries each; non-finite or empty ranges fall back to [-pi, pi]).
        std::vector<float> lowerLimits, upperLimits;
        std::shared_ptr<const RobotModel> robot; // forwarded to the factory only
        std::shared_ptr<const Scene> scene;      // forwarded to the factory only
        BackendFactory createBackend;
    };

    class DetectorCore {
    public:
        // Order: lease numStreams streams, create the backend (factory), check its mode, build the pipeline + SBT,
        // allocate every buffer for the capacities, start the workers (ThreadPerStream). Nothing is allocated later.
        explicit DetectorCore(DetectorCoreDesc desc);
        ~DetectorCore(); // joins workers, drains streams, destroys pipeline then backend; never throws
        DetectorCore(const DetectorCore&)            = delete;
        DetectorCore& operator=(const DetectorCore&) = delete;

        const CheckerConfig& config() const noexcept { return cfg_; }
        ItemLayout itemLayout() const noexcept { return layout_; }
        Backend& backend() noexcept { return *backend_; }
        std::span<const StreamContext> streams() const noexcept { return streamCtx_; }
        // Thread that runs the multi-batch checks' batches of stream s: a worker (ThreadPerStream) or the caller
        // (SingleThread, id()). A single-batch check always runs on the calling thread.
        std::thread::id workerThreadId(std::uint32_t s) const;

        // collides.size() >= items.count; hitCounts empty or >= items.count.
        void checkHost(const HostItems& items, std::span<std::uint8_t> collides, std::span<std::uint32_t> hitCounts);
        void checkDevice(const DeviceItems& items, DeviceSpan<std::uint32_t> hitCounts);
        void warmup();
        CheckStats lastStats() const noexcept { return stats_; }

        // Diagnostic host-time breakdown (phase_trace.h; tests/experiments/host_breakdown): null (default) = off.
        // The trace must have been reset() for this checker's stream count and outlive the checks it records.
        void setPhaseTrace(PhaseTrace* trace) noexcept { trace_ = trace; }

        // Capacity / layout checks of a host item view (count > 0), as checkHost does first. CapacityError,
        // InvalidArgument.
        void validateHostItems(const HostItems& items) const;

    private:
        // A captured "zero hits + staging copy + prepare()" sequence of one stream for one (batch item count,
        // hit-count address): the memset node and the captured params hold that address.
        struct PrepareGraph {
            std::uint32_t itemCount = 0;
            DevicePtr hits          = 0;       // BatchView::hitCounts at capture (memset node, params)
            bool split              = false;   // captured prepareCompute() (Backend::prepareSplit), not prepare()
            CUgraph_st* graph       = nullptr; // kept: owns the staging-copy node handle
            CUgraphExec_st* exec    = nullptr;
            CUgraphNode_st* stagingNode = nullptr;
            DevicePtr stagingSrc    = 0;       // source address the staging-copy node currently reads
            LaunchPlan plan{};                 // prepare()'s plan at capture (params -> `params` below)
            std::array<unsigned char, kMaxLaunchParamsBytes> params{};
        };
        // CheckerConfig::graphPrepare: graphs kept per stream (one per batch item count in use; a check uses at
        // most two: full batches and the last one). Oldest evicted. A miss costs a capture + instantiate, measured
        // ~30 us on the RTX 5080 / Windows (capture ~10 us, cudaGraphInstantiate ~22 us; ~140 us the first time in
        // a process), i.e. about two batches' worth of the ~13 us a hit saves.
        static constexpr std::size_t kMaxPrepareGraphs = 8;

        struct StreamState {
            DeviceBuffer staging; // maxBatchSize * floatsPerItem floats (BatchView::items of graph / strided batches)
            DeviceBuffer hits;    // maxBatchSize uint32 (BatchView::hitCounts unless counted into the result)
            DeviceBuffer params;  // kMaxLaunchParamsBytes
            // Upload cache (old uploadLaunchParams memcmp cache): bytes the params buffer holds.
            std::array<unsigned char, kMaxLaunchParamsBytes> uploaded{};
            std::uint32_t uploadedBytes = 0;
            bool uploadedValid          = false;
            // Counters of the running check.
            std::uint32_t batches = 0, launches = 0, hostSyncs = 0, paramUploads = 0;
            DevicePtr lastTarget = 0; // result address of this stream's previous batch (direct hit counters)
            bool capturing       = false; // true only while graphPrepare() captures on this stream
            std::vector<std::unique_ptr<PrepareGraph>> graphs; // graph-prepare experiment only
        };
        // Where the batches read their items: item i starts at base + i * strideBytes.
        struct ItemSource {
            DevicePtr base          = 0;
            std::size_t strideBytes = 0;
        };

        void validateDeviceItems(const DeviceItems& items, DeviceSpan<std::uint32_t> hitCounts) const;
        void checkCapacity(std::size_t count) const;
        void uploadHostItems(const HostItems& items);
        // Steps 4-8 of the design doc's call sequence; `results` receives `count` uint32 counters.
        void run(std::uint32_t count, const ItemSource& source, DevicePtr results);
        void runBatch(std::uint32_t batch, std::uint32_t count, const ItemSource& source, DevicePtr results);
        // CheckerConfig::graphPrepare: launch (capturing it first if needed) the graph of steps 7.1-7.3 for `view` on
        // stream `s`; returns the plan prepare() returned at capture.
        // split: capture prepareCompute() instead of prepare() and return an empty plan (the caller then calls
        // prepareFinish()); the capture has ended on every path when this returns or throws.
        LaunchPlan graphPrepare(std::uint32_t s, const BatchView& view, DevicePtr src, std::size_t stagingBytes, bool split);
        // Internal error if stream st is inside graphPrepare()'s capture (no OptiX call / prepareFinish there).
        void requireNotCapturing(const StreamState& st, const char* call) const;
        void destroyPrepareGraphs() noexcept;
        void drainStreams() noexcept;
        // Checked; synchronizes streams [0, used) (those that ran batches of the check) even if one fails, throws
        // the first error.
        void synchronizeStreams(std::uint32_t used);
        void mark(std::size_t slot, Phase p) const noexcept {
            if (trace_ != nullptr) {
                trace_->mark(slot, p);
            }
        }

        std::shared_ptr<Context> ctx_;
        CheckerConfig cfg_;
        ItemLayout layout_;
        std::vector<float> lower_, upper_;
        std::shared_ptr<const RobotModel> robot_;
        std::shared_ptr<const Scene> scene_;
        Logger log_;
        StreamLease lease_;
        std::vector<StreamContext> streamCtx_;
        std::unique_ptr<Backend> backend_;
        std::unique_ptr<OptixProgramPipeline> pipeline_;
        std::vector<StreamState> streams_;
        PinnedHostBuffer upload_;   // maxItemsPerCheck packed items (host path, mappedHostTransfer; read in place)
        DeviceBuffer input_, result_; // host path without mappedHostTransfer: device items / counters
        DevicePtr uploadDev_ = 0, downloadDev_ = 0; // device addresses of the two pinned buffers
        PinnedHostBuffer download_; // maxItemsPerCheck uint32 (host path; mappedHostTransfer: written in place)
        bool resultsOnHost_ = false; // the running check writes its counters into download_ (checkHost)
        std::vector<float> warmupItems_;
        CheckStats stats_;
        std::atomic<bool> abort_{false};
        std::mutex errorMutex_;
        std::exception_ptr firstError_;
        PhaseTrace* trace_ = nullptr; // diagnostic, not owned
        std::unique_ptr<StreamWorkers> workers_; // last: destroyed (joined) first
    };

    // Deterministic pseudo-random items for warmup(): `count` packed items of `layout`, joint j uniform in
    // [lower[j], upper[j]] (fixed-seed mt19937, platform independent). Non-finite / empty ranges use [-pi, pi].
    std::vector<float> makeWarmupItems(
        ItemLayout layout, std::size_t count, std::span<const float> lower, std::span<const float> upper);

} // namespace rtcd::detail
