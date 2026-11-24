// src/detector: DetectorCore (detector_core.h) -- the batch loop of rtcd/backend.h, dispatch, buffers, stats.
#include "detector/detector_core.h"

#include "detector/optix_pipeline.h"
#include "detector/stream_workers.h"

#include <rtcd/errors.h>
#include <rtcd/limits.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <utility>

namespace rtcd::detail {

    namespace {
        [[noreturn]] void internalError(const std::string& message) { throw Error(ErrorKind::Internal, message); }
        void* toPtr(DevicePtr p) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(p)); }
        std::string str(std::size_t v) { return std::to_string(v); }
    } // namespace

    // -----------------------------------------------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------------------------------------------

    DetectorCore::DetectorCore(DetectorCoreDesc desc)
        : ctx_(std::move(desc.context)), cfg_(std::move(desc.config)), lower_(std::move(desc.lowerLimits)),
          upper_(std::move(desc.upperLimits)), robot_(std::move(desc.robot)), scene_(std::move(desc.scene)) {
        if (!ctx_) {
            throw InvalidArgument("CollisionChecker: null Context");
        }
        if (!desc.createBackend) {
            internalError("DetectorCore: no backend factory");
        }
        if (desc.dof == 0 || desc.dof > kMaxDof) {
            throw UnsupportedError("CollisionChecker: robot dof = " + str(desc.dof) + " outside [1, " + str(kMaxDof) + "]");
        }
        if (cfg_.numStreams == 0 || cfg_.maxBatchSize == 0 || cfg_.maxItemsPerCheck == 0) {
            internalError("DetectorCore: config is not validated (numStreams, maxBatchSize, maxItemsPerCheck must be >= 1)");
        }
        layout_ = cfg_.itemLayout(desc.dof);
        // The checker's own messages: its verbosity, the Context's sink.
        log_.level = cfg_.verbosity;
        log_.sink  = ctx_->logger().sink;

        ctx_->makeCurrent();
        lease_ = ctx_->acquireStreams(cfg_.numStreams);
        for (std::uint32_t i = 0; i < lease_.size(); ++i) {
            streamCtx_.push_back(StreamContext{i, lease_[i]});
        }

        BackendCreateInfo info;
        info.context = ctx_;
        info.config  = &cfg_;
        info.robot   = robot_;
        info.scene   = scene_;
        info.streams = streamCtx_;
        backend_     = desc.createBackend(info);
        if (!backend_) {
            internalError("CollisionChecker: the backend factory returned null");
        }
        if (backend_->mode() != cfg_.mode) {
            internalError("CollisionChecker: the backend reports mode " + str(static_cast<unsigned>(backend_->mode()))
                          + " but the config has mode " + str(static_cast<unsigned>(cfg_.mode)));
        }

        pipeline_ = std::make_unique<OptixProgramPipeline>(*ctx_, backend_->programs(), cfg_.optix, log_);

        const std::size_t fpiBytes = layout_.floatsPerItem() * sizeof(float);
        streams_.resize(cfg_.numStreams);
        for (StreamState& st : streams_) {
            st.staging.allocate(std::size_t(cfg_.maxBatchSize) * fpiBytes);
            st.hits.allocate(std::size_t(cfg_.maxBatchSize) * sizeof(std::uint32_t));
            st.params.allocate(kMaxLaunchParamsBytes);
        }
        input_.allocate(std::size_t(cfg_.maxItemsPerCheck) * fpiBytes);
        result_.allocate(std::size_t(cfg_.maxItemsPerCheck) * sizeof(std::uint32_t));
        download_.allocate(std::size_t(cfg_.maxItemsPerCheck) * sizeof(std::uint32_t));

        if (cfg_.hostDispatch == HostDispatch::ThreadPerStream) {
            workers_ = std::make_unique<StreamWorkers>(cfg_.numStreams);
        }

    }

    DetectorCore::~DetectorCore() {
        workers_.reset();
        if (ctx_) {
            try {
                ctx_->makeCurrent();
            } catch (...) {
            }
        }
        drainStreams();
        destroyPrepareGraphs();
        pipeline_.reset();
        backend_.reset(); // before the lease returns the streams and before robot / scene / context references go
    }

    void DetectorCore::setTuning(const Tuning& t) {
        tuning_ = t;
        if (workers_) {
            workers_->setSpin(t.joinSpinUs, t.wakeSpinUs);
        }
    }

    std::thread::id DetectorCore::workerThreadId(std::uint32_t s) const {
        if (s >= cfg_.numStreams) {
            throw InvalidArgument("DetectorCore::workerThreadId: stream " + str(s) + " >= " + str(cfg_.numStreams));
        }
        return workers_ ? workers_->threadId(s) : std::thread::id{};
    }

    // -----------------------------------------------------------------------------------------------------------
    // Argument checks (before any GPU work; the checker stays usable)
    // -----------------------------------------------------------------------------------------------------------

    void DetectorCore::checkCapacity(std::size_t count) const {
        if (count > cfg_.maxItemsPerCheck) {
            throw CapacityError("CollisionChecker::check: " + str(count) + " items > max_items_per_check = "
                                    + str(cfg_.maxItemsPerCheck),
                "max_items_per_check", count, cfg_.maxItemsPerCheck);
        }
    }

    void DetectorCore::validateHostItems(const HostItems& items) const {
        checkCapacity(items.count);
        const std::size_t fpi = layout_.floatsPerItem();
        if (items.stride != 0 && items.stride < fpi) {
            throw InvalidArgument("CollisionChecker::check: item stride = " + str(items.stride)
                                  + " floats < floats per item = " + str(fpi));
        }
        if (items.floats.data() == nullptr) {
            throw InvalidArgument("CollisionChecker::check: items.floats is null");
        }
        const std::size_t stride = items.stride == 0 ? fpi : items.stride;
        const std::size_t need   = (items.count - 1) * stride + fpi;
        if (items.floats.size() < need) {
            throw InvalidArgument("CollisionChecker::check: items.floats holds " + str(items.floats.size()) + " floats, "
                                  + str(items.count) + " items of " + str(fpi) + " floats with stride " + str(stride)
                                  + " need " + str(need));
        }
    }

    void DetectorCore::validateDeviceItems(const DeviceItems& items, DeviceSpan<std::uint32_t> hitCounts) const {
        checkCapacity(items.count);
        const std::size_t fpi = layout_.floatsPerItem();
        if (items.stride != 0 && items.stride < fpi) {
            throw InvalidArgument("CollisionChecker::checkDevice: item stride = " + str(items.stride)
                                  + " floats < floats per item = " + str(fpi));
        }
        if (items.floats == 0 || items.floats % alignof(float) != 0) {
            throw InvalidArgument("CollisionChecker::checkDevice: items.floats is null or not 4-byte aligned");
        }
        if (hitCounts.ptr == 0 || hitCounts.ptr % alignof(std::uint32_t) != 0) {
            throw InvalidArgument("CollisionChecker::checkDevice: hitCounts is null or not 4-byte aligned");
        }
        if (hitCounts.count < items.count) {
            throw InvalidArgument("CollisionChecker::checkDevice: hitCounts has " + str(hitCounts.count)
                                  + " entries < " + str(items.count) + " items");
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Public entry points
    // -----------------------------------------------------------------------------------------------------------

    void DetectorCore::checkHost(
        const HostItems& items, std::span<std::uint8_t> collides, std::span<std::uint32_t> hitCounts) {
        if (items.count == 0) {
            stats_ = CheckStats{};
            return;
        }
        validateHostItems(items);
        if (collides.size() < items.count) {
            throw InvalidArgument("CollisionChecker::check: collides has " + str(collides.size()) + " entries < "
                                  + str(items.count) + " items");
        }
        if (!hitCounts.empty() && hitCounts.size() < items.count) {
            throw InvalidArgument("CollisionChecker::check: hitCounts has " + str(hitCounts.size()) + " entries < "
                                  + str(items.count) + " items (pass an empty span to skip the counts)");
        }
        const auto n = static_cast<std::uint32_t>(items.count);
        ctx_->makeCurrent();
        mark(0, Phase::Entry);
        uploadHostItems(items);
        mark(0, Phase::Upload);
        if (tuning_.hostResultsToPinned) {
            // Each batch copies its counters straight into the pinned buffer on its own stream (D2H, async);
            // after the streams are synchronized the counts are there, no separate download.
            resultsOnHost_ = true;
            try {
                run(n, ItemSource{input_.ptr(), layout_.floatsPerItem() * sizeof(float)},
                    static_cast<DevicePtr>(reinterpret_cast<std::uintptr_t>(download_.data())));
            } catch (...) {
                resultsOnHost_ = false;
                throw;
            }
            resultsOnHost_ = false;
        } else {
            run(n, ItemSource{input_.ptr(), layout_.floatsPerItem() * sizeof(float)}, result_.ptr());
            // Step 9: one download of the n counters into the pinned buffer (the streams are synchronized).
            result_.download(download_.data(), std::size_t(n) * sizeof(std::uint32_t));
        }
        mark(0, Phase::Download);
        const std::span<const std::uint32_t> counts = download_.as<std::uint32_t>().first(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            collides[i] = counts[i] > 0 ? 1 : 0;
        }
        if (!hitCounts.empty()) {
            std::copy(counts.begin(), counts.end(), hitCounts.begin());
        }
        mark(0, Phase::Flags);
    }

    void DetectorCore::checkDevice(const DeviceItems& items, DeviceSpan<std::uint32_t> hitCounts) {
        if (items.count == 0) {
            stats_ = CheckStats{};
            return;
        }
        validateDeviceItems(items, hitCounts);
        const std::size_t stride = items.stride == 0 ? layout_.floatsPerItem() : items.stride;
        ctx_->makeCurrent();
        mark(0, Phase::Entry);
        run(static_cast<std::uint32_t>(items.count), ItemSource{items.floats, stride * sizeof(float)}, hitCounts.ptr);
    }

    void DetectorCore::warmup() {
        const std::size_t n = std::min<std::size_t>(std::size_t(cfg_.numStreams) * cfg_.maxBatchSize, cfg_.maxItemsPerCheck);
        if (warmupItems_.empty()) {
            warmupItems_ = makeWarmupItems(layout_, n, lower_, upper_);
        }
        std::vector<std::uint8_t> flags(n);
        checkHost(HostItems{warmupItems_, n, 0}, flags, {});
    }

    // -----------------------------------------------------------------------------------------------------------
    // The check
    // -----------------------------------------------------------------------------------------------------------

    void DetectorCore::uploadHostItems(const HostItems& items) {
        // Step 3: one synchronous copy of the n items into the packed input buffer (strided: cudaMemcpy2D).
        const std::size_t fpiBytes = layout_.floatsPerItem() * sizeof(float);
        if (items.stride == 0 || items.stride == layout_.floatsPerItem()) {
            input_.upload(items.floats.data(), items.count * fpiBytes);
        } else {
            RTCD_CUDA_CHECK(cudaMemcpy2D(toPtr(input_.ptr()), fpiBytes, items.floats.data(), items.stride * sizeof(float),
                fpiBytes, items.count, cudaMemcpyHostToDevice));
        }
    }

    void DetectorCore::run(std::uint32_t count, const ItemSource& source, DevicePtr results) {
        const std::uint32_t B  = cfg_.maxBatchSize;
        const std::uint32_t S  = cfg_.numStreams;
        const std::uint32_t nb = (count + B - 1) / B;
        for (StreamState& st : streams_) {
            st.batches = st.launches = st.hostSyncs = st.paramUploads = 0;
        }

        // Step 5: per-check policy, calling thread, before any prepare(). Nothing is enqueued yet.
        backend_->beginCheck(CheckInfo{count, nb});
        mark(0, Phase::BeginCheck);

        // Step 6 / 7: dispatch.
        mark(0, Phase::Dispatch);
        if (!workers_ || (tuning_.inlineSingleBatch && nb == 1)) {
            try {
                for (std::uint32_t b = 0; b < nb; ++b) {
                    runBatch(b, count, source, results);
                }
            } catch (...) {
                drainStreams();
                throw;
            }
        } else {
            abort_.store(false, std::memory_order_relaxed);
            firstError_ = nullptr;
            const StreamWorkers::Job job = [&](std::uint32_t s) {
                try {
                    ctx_->makeCurrent();
                    for (std::uint32_t b = s; b < nb; b += S) {
                        if (abort_.load(std::memory_order_relaxed)) {
                            return; // another stream failed: issue nothing more
                        }
                        runBatch(b, count, source, results);
                    }
                } catch (...) {
                    abort_.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(errorMutex_);
                    if (!firstError_) {
                        firstError_ = std::current_exception(); // the first failure in time
                    }
                }
            };
            const std::vector<std::exception_ptr> escaped = workers_->run(std::min(S, nb), job);
            std::exception_ptr error = firstError_;
            for (const std::exception_ptr& e : escaped) {
                if (!error && e) {
                    error = e;
                }
            }
            firstError_ = nullptr;
            if (error) {
                // Step 8 on failure: every worker has returned (joined above); drain all streams, rethrow here.
                drainStreams();
                std::rethrow_exception(error);
            }
        }

        mark(0, Phase::Joined);

        // Step 8: synchronize the checker's streams (not the device).
        const std::uint32_t used = std::min(S, nb);
        synchronizeStreams(used);
        if (trace_ != nullptr && trace_->probeIdleSync) {
            synchronizeStreams(used);
            trace_->slots[0].back().phase = Phase::SyncProbe; // the second sync's two marks -> one probe interval
            trace_->slots[0].erase(trace_->slots[0].end() - 2);
        }

        CheckStats stats;
        stats.items = count;
        for (const StreamState& st : streams_) {
            stats.batches += st.batches;
            stats.launches += st.launches;
            stats.hostSyncs += st.hostSyncs;
            stats.paramUploads += st.paramUploads;
        }
        stats_ = stats;
        mark(0, Phase::Stats);
    }

    void DetectorCore::runBatch(std::uint32_t b, std::uint32_t count, const ItemSource& source, DevicePtr results) {
        const std::uint32_t B     = cfg_.maxBatchSize;
        const std::uint32_t s     = b % cfg_.numStreams;
        const std::uint32_t first = b * B;
        const std::uint32_t n     = std::min(B, count - first);
        StreamState& st           = streams_[s];
        const StreamContext& sc   = streamCtx_[s];
        const std::size_t fpi     = layout_.floatsPerItem();
        const std::size_t fpiBytes = fpi * sizeof(float);

        const std::size_t slot = 1 + s;
        mark(slot, Phase::BatchStart);
        const DevicePtr src = source.base + std::uint64_t(first) * source.strideBytes;
        BatchView view;
        view.batchIndex = b;
        view.firstItem  = first;
        view.itemCount  = n;
        view.items      = DeviceSpan<const float>{st.staging.ptr(), std::size_t(n) * fpi};
        view.hitCounts  = DeviceSpan<std::uint32_t>{st.hits.ptr(), n};

        // Direct hit counters: the programs count straight into the check's result range (no result copy). Only
        // when this stream's previous batch had the same target address, so the backend's params (which hold the
        // hit-count pointer) and graphs stay byte-identical between checks into the same result buffer; a batch
        // with a new target uses the stream's own buffer + copy, as before.
        const DevicePtr target = results + std::uint64_t(first) * sizeof(std::uint32_t);
        const bool directHits  = tuning_.directHits && !resultsOnHost_ && target == st.lastTarget;
        st.lastTarget          = target;
        if (directHits) {
            view.hitCounts = DeviceSpan<std::uint32_t>{target, n};
        }
        const bool graph = cfg_.graphPrepare && source.strideBytes == fpiBytes && backend_->prepareCapturable(n);
        // Direct items: a non-graph batch with packed input reads its items where they are (no staging copy).
        const bool directItems = tuning_.directItems && !graph && source.strideBytes == fpiBytes;
        if (directItems) {
            view.items = DeviceSpan<const float>{src, std::size_t(n) * fpi};
        }

        LaunchPlan plan;
        if (graph) {
            // 7.1 - 7.3 as one cudaGraphLaunch (CheckerConfig::graphPrepare, Backend::prepareCapturable).
            if (backend_->synchronizesHost(n)) {
                internalError("backend '" + pipeline_->moduleName() + "': prepareCapturable(" + str(n)
                              + ") and synchronizesHost(" + str(n) + ") are both true");
            }
            plan = graphPrepare(s, view, src, std::size_t(n) * fpiBytes);
            mark(slot, Phase::GraphLaunch);
        } else {
            // 7.1 zero the batch's hit counters.
            RTCD_CUDA_CHECK(cudaMemsetAsync(toPtr(view.hitCounts.ptr), 0, std::size_t(n) * sizeof(std::uint32_t), sc.stream));
            mark(slot, Phase::Zero);
            // 7.2 staging copy of the batch's items (packed in the staging buffer).
            if (directItems) {
                // nothing to copy
            } else if (source.strideBytes == fpiBytes) {
                st.staging.copyFromAsync(src, std::size_t(n) * fpiBytes, sc.stream);
            } else {
                RTCD_CUDA_CHECK(cudaMemcpy2DAsync(toPtr(st.staging.ptr()), fpiBytes, toPtr(src), source.strideBytes, fpiBytes,
                    n, cudaMemcpyDeviceToDevice, sc.stream));
            }
            mark(slot, Phase::Staging);
            // 7.3 mode-specific work.
            if (backend_->synchronizesHost(n)) {
                ++st.hostSyncs;
            }
            plan = backend_->prepare(sc, view);
            mark(slot, Phase::Prepare);
        }

        // 7.4 params upload (skipped when byte-identical to this stream's last upload) + launch.
        if (plan.launch) {
            validateLaunchPlan(plan, pipeline_->raygenCount(), pipeline_->moduleName());
            if (plan.paramsBytes > 0) {
                const bool same = st.uploadedValid && st.uploadedBytes == plan.paramsBytes
                               && std::memcmp(st.uploaded.data(), plan.params, plan.paramsBytes) == 0;
                if (!same) {
                    st.uploadedValid = false; // until the copy is enqueued
                    // Pageable source: CUDA stages it before returning, so the backend may change its host copy
                    // in the next prepare() (old uploadLaunchParams did the same).
                    st.params.uploadAsync(plan.params, plan.paramsBytes, sc.stream);
                    std::memcpy(st.uploaded.data(), plan.params, plan.paramsBytes);
                    st.uploadedBytes = plan.paramsBytes;
                    st.uploadedValid = true;
                    ++st.paramUploads;
                }
            }
            mark(slot, Phase::Params);
            pipeline_->launch(sc.stream, st.params.ptr(), plan.paramsBytes, plan);
            mark(slot, Phase::Launch);
            ++st.launches;
        }

        // 7.5 result copy at firstItem.
        if (!directHits) {
            RTCD_CUDA_CHECK(cudaMemcpyAsync(toPtr(target), toPtr(st.hits.ptr()), std::size_t(n) * sizeof(std::uint32_t),
                resultsOnHost_ ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, sc.stream));
        }
        mark(slot, Phase::ResultCopy);
        ++st.batches;
        mark(slot, Phase::BatchEnd);
    }

    // -----------------------------------------------------------------------------------------------------------
    // CheckerConfig::graphPrepare (CUDA graph replay of steps 7.1 - 7.3)
    // -----------------------------------------------------------------------------------------------------------

    LaunchPlan DetectorCore::graphPrepare(std::uint32_t s, const BatchView& view, DevicePtr src, std::size_t stagingBytes) {
        StreamState& st         = streams_[s];
        const StreamContext& sc = streamCtx_[s];
        const std::uint32_t n   = view.itemCount;

        PrepareGraph* g = nullptr;
        for (const auto& p : st.graphs) {
            if (p->itemCount == n && p->hits == view.hitCounts.ptr) {
                g = p.get();
                break;
            }
        }
        if (g == nullptr) {
            // Capture steps 7.1 - 7.3 once for this (stream, item count). prepareCapturable(n) promises plain CUDA
            // work only and a plan / params that depend only on (stream, n): replaying the graph on new contents of
            // the same staging / hit-count buffers is equivalent to calling prepare() again.
            auto entry       = std::make_unique<PrepareGraph>();
            entry->itemCount = n;
            entry->hits      = view.hitCounts.ptr;
            cudaGraph_t graph = nullptr;
            RTCD_CUDA_CHECK(cudaStreamBeginCapture(sc.stream, cudaStreamCaptureModeThreadLocal));
            LaunchPlan plan;
            try {
                RTCD_CUDA_CHECK(cudaMemsetAsync(toPtr(view.hitCounts.ptr), 0, std::size_t(n) * sizeof(std::uint32_t), sc.stream));
                st.staging.copyFromAsync(src, stagingBytes, sc.stream);
                plan = backend_->prepare(sc, view);
            } catch (...) {
                (void) cudaStreamEndCapture(sc.stream, &graph);
                if (graph != nullptr) {
                    cudaGraphDestroy(graph);
                }
                (void) cudaGetLastError();
                throw;
            }
            RTCD_CUDA_CHECK(cudaStreamEndCapture(sc.stream, &graph));
            entry->graph = graph;
            // The one memcpy node is the staging copy (prepare() of a capturable backend issues no copy itself).
            std::size_t count = 0;
            RTCD_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
            std::vector<cudaGraphNode_t> nodes(count);
            RTCD_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType type{};
                RTCD_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
                if (type == cudaGraphNodeTypeMemcpy) {
                    if (entry->stagingNode != nullptr) {
                        cudaGraphDestroy(graph);
                        internalError("DetectorCore graph prepare: the captured prepare() issued a memcpy of its own");
                    }
                    entry->stagingNode = node;
                }
            }
            const cudaError_t inst = cudaGraphInstantiate(&entry->exec, graph, 0);
            if (inst != cudaSuccess || entry->stagingNode == nullptr) {
                cudaGraphDestroy(graph);
                RTCD_CUDA_CHECK(inst);
                internalError("DetectorCore graph prepare: no staging-copy node in the captured graph");
            }
            entry->stagingSrc = src;
            entry->plan       = plan;
            if (plan.paramsBytes > kMaxLaunchParamsBytes) {
                internalError("DetectorCore graph prepare: paramsBytes > kMaxLaunchParamsBytes");
            }
            if (plan.paramsBytes > 0 && plan.params != nullptr) {
                std::memcpy(entry->params.data(), plan.params, plan.paramsBytes);
            }
            if (st.graphs.size() >= kMaxPrepareGraphs) {
                // Oldest out; its last launch is ordered before anything enqueued later on this stream, and
                // cudaGraphExecDestroy may be called while a launch of it is pending.
                cudaGraphExecDestroy(st.graphs.front()->exec);
                cudaGraphDestroy(st.graphs.front()->graph);
                st.graphs.erase(st.graphs.begin());
            }
            st.graphs.push_back(std::move(entry));
            g = st.graphs.back().get();
        } else if (g->stagingSrc != src) {
            // Same item count, other source address (another batch of the check / other device items).
            RTCD_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(g->exec, g->stagingNode, toPtr(st.staging.ptr()), toPtr(src),
                stagingBytes, cudaMemcpyDeviceToDevice));
            g->stagingSrc = src;
        }
        RTCD_CUDA_CHECK(cudaGraphLaunch(g->exec, sc.stream));
        LaunchPlan plan = g->plan;
        plan.params     = plan.paramsBytes > 0 ? g->params.data() : nullptr;
        return plan;
    }

    void DetectorCore::destroyPrepareGraphs() noexcept {
        for (StreamState& st : streams_) {
            for (const auto& g : st.graphs) {
                if (g->exec != nullptr) {
                    cudaGraphExecDestroy(g->exec);
                }
                if (g->graph != nullptr) {
                    cudaGraphDestroy(g->graph);
                }
            }
            st.graphs.clear();
        }
    }

    void DetectorCore::drainStreams() noexcept {
        for (const StreamContext& sc : streamCtx_) {
            // Errors resurface at the caller's next checked call (or were already thrown).
            (void) cudaStreamSynchronize(sc.stream);
        }
    }

    void DetectorCore::synchronizeStreams(std::uint32_t used) {
        cudaError_t first = cudaSuccess;
        const std::size_t count = tuning_.syncUsedOnly ? std::min<std::size_t>(used, streamCtx_.size()) : streamCtx_.size();
        for (std::uint32_t s = 0; s < count; ++s) {
            if (s == used) {
                mark(0, Phase::SyncUsed);
            }
            const cudaError_t e = cudaStreamSynchronize(streamCtx_[s].stream);
            if (first == cudaSuccess && e != cudaSuccess) {
                first = e;
            }
        }
        if (used >= count) {
            mark(0, Phase::SyncUsed);
        }
        mark(0, Phase::SyncIdle);
        RTCD_CUDA_CHECK(first);
    }

    // -----------------------------------------------------------------------------------------------------------
    // Warmup items
    // -----------------------------------------------------------------------------------------------------------

    std::vector<float> makeWarmupItems(
        ItemLayout layout, std::size_t count, std::span<const float> lower, std::span<const float> upper) {
        constexpr float kPi = 3.14159265358979f;
        std::vector<float> lo(layout.dof, -kPi), hi(layout.dof, kPi);
        for (std::uint32_t j = 0; j < layout.dof; ++j) {
            if (j < lower.size() && j < upper.size() && std::isfinite(lower[j]) && std::isfinite(upper[j])
                && lower[j] < upper[j]) {
                lo[j] = lower[j];
                hi[j] = upper[j];
            }
        }
        std::mt19937 rng(0x52544344u); // "RTCD"; mt19937 output is specified by the standard
        std::vector<float> items(count * layout.floatsPerItem());
        std::size_t k = 0;
        for (std::size_t i = 0; i < count * layout.pointsPerItem; ++i) {
            for (std::uint32_t j = 0; j < layout.dof; ++j) {
                const float u = static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f); // [0, 1)
                items[k++]    = lo[j] + u * (hi[j] - lo[j]);
            }
        }
        return items;
    }

} // namespace rtcd::detail
