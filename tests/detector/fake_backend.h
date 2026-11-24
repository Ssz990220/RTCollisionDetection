#pragma once
// Fake Backend for the detector-core tests: its programs are detector_test_program.cu (embedded as
// rtcd_detector_test_ir), prepare() only fills TestParams (plus a tiny memset on its stream so that it really
// issues CUDA work from the dispatching thread) and records every call.
#include <rtcd/backend.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>

#include <rtcd_detector_test_ir_embedded.h>

#include "detector_test_params.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rtcd_detector_test {

    inline constexpr std::uint32_t kRaygenCopy        = 0;
    inline constexpr std::uint32_t kRaygenCopyBatch   = 1;
    inline constexpr std::uint32_t kRaygenSpin        = 2;
    inline constexpr std::uint32_t kRaygenTrace       = 3;
    inline constexpr std::uint32_t kRaygenTraceChMiss = 4;

    inline rtcd::ProgramSet fakePrograms() {
        rtcd::ProgramSet ps;
        ps.moduleCode = std::span<const unsigned char>(rtcd::embedded::rtcd_detector_test_ir_detector_test_program,
            rtcd::embedded::rtcd_detector_test_ir_detector_test_program_size);
        ps.moduleName          = "detector_test_program";
        ps.raygens             = {"__raygen__copyItems", "__raygen__copyItemsBatch", "__raygen__spin", "__raygen__trace",
                        "__raygen__traceChMiss"};
        ps.miss                = "__miss__mark";
        ps.anyHit              = "__anyhit__count";
        ps.closestHit          = "__closesthit__mark";
        ps.traversableGraph    = rtcd::TraversableGraph::SingleGas;
        ps.maxTraversableDepth = 1;
        ps.numPayloadValues    = 1;
        ps.numAttributeValues  = 2;
        ps.hitGroupRecordCount = 1;
        return ps;
    }

    struct FakeOptions {
        std::uint32_t dof = 2;
        // raygen of batch b (default: copyItems for every batch)
        std::function<std::uint32_t(std::uint32_t batch)> raygen = [](std::uint32_t) { return kRaygenCopy; };
        std::uint32_t addValue          = 0;
        bool includeBatchIndex          = false; // params differ per batch (defeats the upload cache)
        std::uint32_t skipBatch         = ~0u;   // LaunchPlan::launch = false for this batch
        std::uint32_t throwAtBatch      = ~0u;   // prepare() throws InvalidArgument for this batch
        bool throwInBeginCheck          = false; // beginCheck() throws UnsupportedError
        std::uint32_t syncThreshold     = ~0u;   // synchronizesHost(n) = n >= syncThreshold (and prepare syncs)
        std::uint32_t badRaygen         = ~0u;   // batch whose plan has raygen = 99 (invalid plan)
        long long spinCycles            = 0;     // for kRaygenSpin
        rtcd::TraversableHandle traversable = 0;
        std::function<void(rtcd::ProgramSet&)> editPrograms; // applied at construction
        rtcd::Mode reportedMode = rtcd::Mode::LinearCurve;   // set from the config by makeFactory unless forced
        bool forceReportedMode  = false;
    };

    class FakeBackend final : public rtcd::Backend {
    public:
        struct Call {
            std::uint32_t streamIndex = 0;
            rtcd::StreamHandle stream = nullptr;
            std::uint32_t batch = 0, first = 0, count = 0;
            std::size_t itemsCount = 0, hitCountsCount = 0;
            rtcd::DevicePtr items = 0, hitCounts = 0;
            std::thread::id thread;
            bool contextCurrent = false; // the Context's CUDA context was current on the calling thread
        };

        FakeBackend(const rtcd::BackendCreateInfo& info, FakeOptions opts)
            : opts_(std::move(opts)), ctx_(info.context), params_(info.streams.size()) {
            fpi_ = static_cast<std::uint32_t>(info.config->itemLayout(opts_.dof).floatsPerItem());
            mode_ = opts_.forceReportedMode ? opts_.reportedMode : info.config->mode;
            for (const rtcd::StreamContext& s : info.streams) {
                streams.push_back(s);
                scratch_.emplace_back(64);
            }
            programs_ = fakePrograms();
            if (opts_.editPrograms) {
                opts_.editPrograms(programs_);
            }
        }

        rtcd::Mode mode() const noexcept override { return mode_; }
        const rtcd::ProgramSet& programs() const noexcept override { return programs_; }
        void beginCheck(const rtcd::CheckInfo& info) override {
            std::lock_guard<std::mutex> lock(mutex_);
            begins.push_back(info);
            if (opts_.throwInBeginCheck) {
                throw rtcd::UnsupportedError("fake beginCheck failure");
            }
        }
        rtcd::LaunchPlan prepare(const rtcd::StreamContext& stream, const rtcd::BatchView& batch) override {
            Call c;
            c.streamIndex    = stream.index;
            c.stream         = stream.stream;
            c.batch          = batch.batchIndex;
            c.first          = batch.firstItem;
            c.count          = batch.itemCount;
            c.itemsCount     = batch.items.count;
            c.hitCountsCount = batch.hitCounts.count;
            c.items          = batch.items.ptr;
            c.hitCounts      = batch.hitCounts.ptr;
            c.thread         = std::this_thread::get_id();
            CUcontext cur    = nullptr;
            c.contextCurrent = cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur != nullptr && cur == ctx_->cudaContext();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                calls.push_back(c);
            }
            // Real GPU work on the batch's stream from the dispatching thread (needs the current context).
            scratch_.at(stream.index).memsetAsync(0, 64, stream.stream);
            if (batch.batchIndex == opts_.throwAtBatch) {
                throw rtcd::InvalidArgument("fake prepare failure at batch " + std::to_string(batch.batchIndex));
            }
            if (synchronizesHost(batch.itemCount)) {
                RTCD_CUDA_CHECK(cudaStreamSynchronize(stream.stream));
            }
            TestParams& p   = params_.at(stream.index);
            p               = {};
            p.traversable   = opts_.traversable;
            p.items         = batch.items.ptr;
            p.hitCounts     = batch.hitCounts.ptr;
            p.itemCount     = batch.itemCount;
            p.floatsPerItem = fpi_;
            p.addValue      = opts_.addValue;
            p.batchIndex    = opts_.includeBatchIndex ? batch.batchIndex : 0;
            p.spinCycles    = opts_.spinCycles;
            rtcd::LaunchPlan plan;
            plan.launch      = batch.batchIndex != opts_.skipBatch;
            plan.traversable = opts_.traversable;
            plan.width       = batch.itemCount;
            plan.params      = &p;
            plan.paramsBytes = sizeof(TestParams);
            plan.raygen      = batch.batchIndex == opts_.badRaygen ? 99u : opts_.raygen(batch.batchIndex);
            return plan;
        }
        bool synchronizesHost(std::uint32_t itemCount) const noexcept override {
            return itemCount >= opts_.syncThreshold;
        }

        std::vector<rtcd::StreamContext> streams;
        std::vector<rtcd::CheckInfo> begins;
        std::vector<Call> calls;

    private:
        FakeOptions opts_;
        std::shared_ptr<rtcd::Context> ctx_;
        rtcd::Mode mode_ = rtcd::Mode::LinearCurve;
        rtcd::ProgramSet programs_;
        std::vector<TestParams> params_;
        std::vector<rtcd::DeviceBuffer> scratch_;
        std::uint32_t fpi_ = 0;
        std::mutex mutex_;
    };

    // Factory creating a FakeBackend with `opts`; *created receives the instance (owned by the checker).
    inline std::function<std::unique_ptr<rtcd::Backend>(const rtcd::BackendCreateInfo&)> makeFactory(
        FakeOptions opts, FakeBackend** created) {
        return [opts, created](const rtcd::BackendCreateInfo& info) -> std::unique_ptr<rtcd::Backend> {
            auto b = std::make_unique<FakeBackend>(info, opts);
            if (created) {
                *created = b.get();
            }
            return b;
        };
    }

} // namespace rtcd_detector_test
