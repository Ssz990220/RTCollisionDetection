#pragma once
// rtcd/internal/reference_executor.h -- test-support executor (library rtcd_testsupport, tests/support,
// Wave 1 e): builds an OptiX pipeline + SBT from a backend's ProgramSet and runs LaunchPlans, independently of
// the detector core.
// Lets the Wave 2 backend agents test prepare() + programs end to end before (and as a cross-check of) the
// detector core. Not used by the library. Simple on purpose: always uploads params, one pipeline.
//
// Usage in a backend test:
//   auto backend = rtcd::createRayBackend(info);
//   rtcd::internal::ReferenceExecutor exec(ctx, backend->programs(), cfg.optix);
//   rtcd::internal::ReferenceBatchRunner run(ctx, *backend, exec, cfg, info.streams, robot->dof());
//   std::vector<uint32_t> counts = run.runAll(hostItems, itemCount);   // the whole check, single thread

#include <rtcd/backend.h>
#include <rtcd/config.h>
#include <rtcd/context.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rtcd::internal {

    class ReferenceExecutor {
    public:
        ReferenceExecutor(std::shared_ptr<Context> context, const ProgramSet& programs, const OptixOptions& optix);
        ~ReferenceExecutor();
        ReferenceExecutor(const ReferenceExecutor&)            = delete;
        ReferenceExecutor& operator=(const ReferenceExecutor&) = delete;

        // Uploads plan.params to an internal buffer and launches on `stream` (no-op if !plan.launch).
        // Asynchronous; the caller synchronizes.
        void launch(StreamHandle stream, const LaunchPlan& plan);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // Runs the core's batch loop on ONE thread with the backend's streams, using a ReferenceExecutor:
    // batching, staging copy, zeroing, prepare, launch, result copy -- the documented contract of rtcd/backend.h,
    // nothing more. Returns per-item hit counts.
    class ReferenceBatchRunner {
    public:
        ReferenceBatchRunner(std::shared_ptr<Context> context, Backend& backend, ReferenceExecutor& executor,
            const CheckerConfig& resolvedConfig, std::span<const StreamContext> streams, std::uint32_t dof);
        ~ReferenceBatchRunner();
        std::vector<std::uint32_t> runAll(std::span<const float> packedItems, std::uint32_t itemCount);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rtcd::internal
