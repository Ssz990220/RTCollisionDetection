// rtcd_testsupport: ReferenceExecutor / ReferenceBatchRunner (rtcd/internal/reference_executor.h).
//
// Deliberately simple and independent of the detector core (Wave 2 e): one pipeline, one SBT, params uploaded on
// every launch, a single-threaded batch loop. It mirrors the old CollisionDetector's createModule /
// createRaygenPrograms / createMissPrograms / createHitgroupPrograms / createPipeline / buildSBT, driven by a
// ProgramSet instead of `if constexpr (TYPE == ...)`.
#include <rtcd/internal/reference_executor.h>

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/limits.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

namespace rtcd::internal {

    namespace {
        OptixCompileOptimizationLevel toOptix(OptixOptions::OptLevel l) {
            switch (l) {
            case OptixOptions::OptLevel::O0: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            case OptixOptions::OptLevel::O1: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
            case OptixOptions::OptLevel::O2: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
            case OptixOptions::OptLevel::O3: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
            default: return OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            }
        }
        OptixCompileDebugLevel toOptix(OptixOptions::DebugLevel l) {
            switch (l) {
            case OptixOptions::DebugLevel::None: return OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            case OptixOptions::DebugLevel::Minimal: return OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
            case OptixOptions::DebugLevel::Moderate: return OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
            case OptixOptions::DebugLevel::Full: return OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            default: return OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
            }
        }
        OptixPrimitiveType toOptix(BuiltinIntersection b) {
            switch (b) {
            case BuiltinIntersection::Sphere: return OPTIX_PRIMITIVE_TYPE_SPHERE;
            case BuiltinIntersection::RoundLinear: return OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
            case BuiltinIntersection::RoundQuadratic: return OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
            case BuiltinIntersection::RoundCubic: return OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
            default: return OPTIX_PRIMITIVE_TYPE_TRIANGLE;
            }
        }
        const char* nameOrNull(const std::string& s) { return s.empty() ? nullptr : s.c_str(); }

        static_assert(sizeof(SbtRecord) % OPTIX_SBT_RECORD_ALIGNMENT == 0, "SbtRecord stride");
        static_assert(kSbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE, "SBT header size");
    } // namespace

    // -----------------------------------------------------------------------------------------------------------
    // ReferenceExecutor
    // -----------------------------------------------------------------------------------------------------------

    struct ReferenceExecutor::Impl {
        std::shared_ptr<Context> ctx;
        std::string moduleName;
        std::vector<std::string> raygenNames;
        OptixModule module   = nullptr;
        OptixModule isModule = nullptr; // built-in IS module: owned by the OptiX device context, never destroyed
                                        // here (the old detector did not destroy it either)
        std::vector<OptixProgramGroup> raygenPGs;
        OptixProgramGroup missPG = nullptr;
        OptixProgramGroup hitPG  = nullptr;
        OptixPipeline pipeline   = nullptr;
        DeviceBuffer raygenRecords, missRecords, hitRecords;
        OptixShaderBindingTable sbt{};
        // One launch-param block per stream, created on the stream's first launch. A deque, not a vector:
        // emplace_back never moves existing elements, so the reference paramsFor() returns stays valid while
        // another thread's first launch on a different stream appends (a vector could reallocate under it).
        std::mutex paramsMutex;
        std::deque<std::pair<StreamHandle, DeviceBuffer>> params;

        DeviceBuffer& paramsFor(StreamHandle s) {
            std::lock_guard<std::mutex> lock(paramsMutex);
            for (auto& [stream, buf] : params) {
                if (stream == s) {
                    return buf;
                }
            }
            params.emplace_back(s, DeviceBuffer(kMaxLaunchParamsBytes));
            return params.back().second;
        }

        void destroy() noexcept {
            if (pipeline) {
                optixPipelineDestroy(pipeline);
            }
            if (hitPG) {
                optixProgramGroupDestroy(hitPG);
            }
            if (missPG) {
                optixProgramGroupDestroy(missPG);
            }
            for (auto pg : raygenPGs) {
                if (pg) {
                    optixProgramGroupDestroy(pg);
                }
            }
            if (module) {
                optixModuleDestroy(module);
            }
            pipeline = nullptr;
            hitPG = missPG = nullptr;
            raygenPGs.clear();
            module = isModule = nullptr;
        }
    };

    ReferenceExecutor::ReferenceExecutor(
        std::shared_ptr<Context> context, const ProgramSet& programs, const OptixOptions& optix)
        : impl_(std::make_unique<Impl>()) {
        if (!context) {
            throw InvalidArgument("ReferenceExecutor: null context");
        }
        if (programs.moduleCode.empty()) {
            throw InvalidArgument("ReferenceExecutor: ProgramSet '" + programs.moduleName + "' has no module code");
        }
        if (programs.raygens.empty()) {
            throw InvalidArgument("ReferenceExecutor: ProgramSet '" + programs.moduleName + "' has no raygen");
        }
        if (programs.hitGroupRecordCount == 0) {
            throw InvalidArgument("ReferenceExecutor: ProgramSet '" + programs.moduleName
                                  + "' has hitGroupRecordCount == 0");
        }
        Impl& d       = *impl_;
        d.ctx         = std::move(context);
        d.moduleName  = programs.moduleName;
        d.raygenNames = programs.raygens;
        d.ctx->makeCurrent();
        const OptixDeviceContext octx = d.ctx->optixContext();

        try {
            OptixModuleCompileOptions mco{};
            mco.maxRegisterCount = static_cast<int>(optix.maxRegisterCount);
            mco.optLevel         = toOptix(optix.optLevel);
            mco.debugLevel       = toOptix(optix.debugLevel);

            OptixPipelineCompileOptions pco{};
            pco.usesMotionBlur                   = 0;
            pco.traversableGraphFlags            = static_cast<unsigned int>(programs.traversableGraph);
            pco.numPayloadValues                 = static_cast<int>(programs.numPayloadValues);
            pco.numAttributeValues               = static_cast<int>(programs.numAttributeValues);
            pco.exceptionFlags                   = optix.exceptionFlags;
            pco.pipelineLaunchParamsVariableName = kLaunchParamsName;
            pco.usesPrimitiveTypeFlags           = programs.primitiveTypes;

            char log[8192];
            size_t logSize = sizeof(log);
            log[0]         = 0;
            RTCD_OPTIX_CHECK_LOG(optixModuleCreate(octx, &mco, &pco, reinterpret_cast<const char*>(programs.moduleCode.data()),
                                     programs.moduleCode.size(), log, &logSize, &d.module),
                log);

            if (programs.builtinIntersection != BuiltinIntersection::None) {
                OptixBuiltinISOptions iso{};
                iso.builtinISModuleType = toOptix(programs.builtinIntersection);
                iso.buildFlags          = programs.builtinIsBuildFlags;
                iso.usesMotionBlur      = 0;
                iso.curveEndcapFlags    = programs.builtinIntersection == BuiltinIntersection::Sphere
                                            ? 0u
                                            : static_cast<unsigned int>(OPTIX_CURVE_ENDCAP_ON);
                RTCD_OPTIX_CHECK(optixBuiltinISModuleGet(octx, &mco, &pco, &iso, &d.isModule));
            }

            OptixProgramGroupOptions pgo{};
            for (const std::string& rg : programs.raygens) {
                OptixProgramGroupDesc desc{};
                desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                desc.raygen.module            = d.module;
                desc.raygen.entryFunctionName = rg.c_str();
                OptixProgramGroup pg          = nullptr;
                logSize                       = sizeof(log);
                log[0]                        = 0;
                RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(octx, &desc, 1, &pgo, log, &logSize, &pg), log);
                d.raygenPGs.push_back(pg);
            }
            {
                OptixProgramGroupDesc desc{};
                desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                desc.miss.module            = programs.miss.empty() ? nullptr : d.module;
                desc.miss.entryFunctionName = nameOrNull(programs.miss);
                logSize                     = sizeof(log);
                log[0]                      = 0;
                RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(octx, &desc, 1, &pgo, log, &logSize, &d.missPG), log);
            }
            {
                OptixProgramGroupDesc desc{};
                desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                desc.hitgroup.moduleAH            = programs.anyHit.empty() ? nullptr : d.module;
                desc.hitgroup.entryFunctionNameAH = nameOrNull(programs.anyHit);
                desc.hitgroup.moduleCH            = programs.closestHit.empty() ? nullptr : d.module;
                desc.hitgroup.entryFunctionNameCH = nameOrNull(programs.closestHit);
                desc.hitgroup.moduleIS            = d.isModule; // nullptr = built-in triangles
                desc.hitgroup.entryFunctionNameIS = nullptr;
                logSize                           = sizeof(log);
                log[0]                            = 0;
                RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(octx, &desc, 1, &pgo, log, &logSize, &d.hitPG), log);
            }

            std::vector<OptixProgramGroup> groups = d.raygenPGs;
            groups.push_back(d.missPG);
            groups.push_back(d.hitPG);
            OptixPipelineLinkOptions plo{};
            plo.maxTraceDepth = optix.maxTraceDepth;
            logSize           = sizeof(log);
            log[0]            = 0;
            RTCD_OPTIX_CHECK_LOG(optixPipelineCreate(octx, &pco, &plo, groups.data(), static_cast<unsigned int>(groups.size()),
                                     log, &logSize, &d.pipeline),
                log);
            if (optix.printPipelineLog && logSize > 1) {
                d.ctx->logger().log(LogLevel::Info, std::string("ReferenceExecutor pipeline log: ") + log);
            }

            OptixStackSizes ss{};
            for (OptixProgramGroup pg : groups) {
                RTCD_OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &ss, d.pipeline));
            }
            unsigned int dcTrav = 0, dcState = 0, cont = 0;
            // No program traces recursively: stack sizes for trace depth 1, as the old detector.
            RTCD_OPTIX_CHECK(optixUtilComputeStackSizes(&ss, 1, 0, 0, &dcTrav, &dcState, &cont));
            RTCD_OPTIX_CHECK(optixPipelineSetStackSize(d.pipeline, dcTrav, dcState, cont, programs.maxTraversableDepth));

            // SBT: one raygen record per entry (a launch points sbt.raygenRecord at record plan.raygen), one miss
            // record, hitGroupRecordCount identical hit records. data = 0 everywhere.
            auto pack = [](OptixProgramGroup pg, std::size_t n) {
                std::vector<SbtRecord> recs(n);
                for (auto& r : recs) {
                    std::memset(&r, 0, sizeof(r));
                    RTCD_OPTIX_CHECK(optixSbtRecordPackHeader(pg, &r));
                }
                return recs;
            };
            std::vector<SbtRecord> rg;
            for (OptixProgramGroup pg : d.raygenPGs) {
                rg.push_back(pack(pg, 1)[0]);
            }
            const std::vector<SbtRecord> ms = pack(d.missPG, 1);
            const std::vector<SbtRecord> hg = pack(d.hitPG, programs.hitGroupRecordCount);
            d.raygenRecords = DeviceBuffer::fromHost(std::span<const SbtRecord>(rg));
            d.missRecords   = DeviceBuffer::fromHost(std::span<const SbtRecord>(ms));
            d.hitRecords    = DeviceBuffer::fromHost(std::span<const SbtRecord>(hg));
            d.sbt.raygenRecord                = d.raygenRecords.ptr();
            d.sbt.missRecordBase              = d.missRecords.ptr();
            d.sbt.missRecordStrideInBytes     = sizeof(SbtRecord);
            d.sbt.missRecordCount             = 1;
            d.sbt.hitgroupRecordBase          = d.hitRecords.ptr();
            d.sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
            d.sbt.hitgroupRecordCount         = programs.hitGroupRecordCount;
        } catch (...) {
            d.destroy();
            throw;
        }
    }

    ReferenceExecutor::~ReferenceExecutor() {
        if (impl_) {
            impl_->destroy();
        }
    }

    void ReferenceExecutor::launch(StreamHandle stream, const LaunchPlan& plan) {
        if (!plan.launch) {
            return;
        }
        Impl& d = *impl_;
        if (plan.raygen >= d.raygenPGs.size()) {
            throw InvalidArgument("ReferenceExecutor(" + d.moduleName + "): plan.raygen = " + std::to_string(plan.raygen)
                                  + " but the ProgramSet has " + std::to_string(d.raygenPGs.size()) + " raygens");
        }
        if (plan.width == 0 || plan.height == 0 || plan.depth == 0) {
            throw InvalidArgument("ReferenceExecutor(" + d.moduleName + "): launch dimensions must be >= 1 (got "
                                  + std::to_string(plan.width) + " x " + std::to_string(plan.height) + " x "
                                  + std::to_string(plan.depth) + ")");
        }
        const std::uint64_t size = std::uint64_t(plan.width) * plan.height * plan.depth;
        if (size > kMaxLaunchSize) {
            throw InvalidArgument("ReferenceExecutor(" + d.moduleName + "): launch size " + std::to_string(size)
                                  + " exceeds 2^30");
        }
        if (plan.paramsBytes > kMaxLaunchParamsBytes || (plan.paramsBytes > 0 && plan.params == nullptr)) {
            throw InvalidArgument("ReferenceExecutor(" + d.moduleName + "): invalid launch params ("
                                  + std::to_string(plan.paramsBytes) + " bytes)");
        }
        d.ctx->makeCurrent();
        DeviceBuffer& pbuf = d.paramsFor(stream);
        if (plan.paramsBytes > 0) {
            // pageable source: CUDA stages it before returning, so the backend may reuse its host memory
            pbuf.uploadAsync(plan.params, plan.paramsBytes, stream);
        }
        OptixShaderBindingTable sbt = d.sbt;
        sbt.raygenRecord            = d.raygenRecords.ptr() + std::uint64_t(plan.raygen) * sizeof(SbtRecord);
        RTCD_OPTIX_CHECK(optixLaunch(d.pipeline, stream, pbuf.ptr(), plan.paramsBytes, &sbt, plan.width, plan.height,
            plan.depth));
    }

    // -----------------------------------------------------------------------------------------------------------
    // ReferenceBatchRunner
    // -----------------------------------------------------------------------------------------------------------

    struct ReferenceBatchRunner::Impl {
        std::shared_ptr<Context> ctx;
        Backend* backend           = nullptr;
        ReferenceExecutor* exec    = nullptr;
        CheckerConfig cfg;
        std::vector<StreamContext> streams;
        std::size_t floatsPerItem = 0;
        DeviceBuffer input;  // maxItemsPerCheck items
        DeviceBuffer result; // maxItemsPerCheck uint32
        std::vector<DeviceBuffer> staging; // per stream: maxBatchSize items
        std::vector<DeviceBuffer> hits;    // per stream: maxBatchSize uint32

        void drain() noexcept {
            for (const auto& s : streams) {
                cudaStreamSynchronize(s.stream); // errors resurface at the caller's next checked call
            }
        }
    };

    ReferenceBatchRunner::ReferenceBatchRunner(std::shared_ptr<Context> context, Backend& backend,
        ReferenceExecutor& executor, const CheckerConfig& resolvedConfig, std::span<const StreamContext> streams,
        std::uint32_t dof)
        : impl_(std::make_unique<Impl>()) {
        if (!context) {
            throw InvalidArgument("ReferenceBatchRunner: null context");
        }
        if (streams.empty()) {
            throw InvalidArgument("ReferenceBatchRunner: no streams");
        }
        for (std::size_t i = 0; i < streams.size(); ++i) {
            if (streams[i].index != i) {
                throw InvalidArgument("ReferenceBatchRunner: streams[" + std::to_string(i) + "].index = "
                                      + std::to_string(streams[i].index) + " (must equal its position)");
            }
        }
        if (resolvedConfig.maxBatchSize == 0 || resolvedConfig.maxItemsPerCheck == 0 || dof == 0) {
            throw InvalidArgument("ReferenceBatchRunner: maxBatchSize, maxItemsPerCheck and dof must be >= 1");
        }
        Impl& d        = *impl_;
        d.ctx          = std::move(context);
        d.backend      = &backend;
        d.exec         = &executor;
        d.cfg          = resolvedConfig;
        d.streams.assign(streams.begin(), streams.end());
        d.floatsPerItem = resolvedConfig.itemLayout(dof).floatsPerItem();
        d.ctx->makeCurrent();
        d.input.allocate(std::size_t(d.cfg.maxItemsPerCheck) * d.floatsPerItem * sizeof(float));
        d.result.allocate(std::size_t(d.cfg.maxItemsPerCheck) * sizeof(std::uint32_t));
        for (std::size_t i = 0; i < d.streams.size(); ++i) {
            d.staging.emplace_back(std::size_t(d.cfg.maxBatchSize) * d.floatsPerItem * sizeof(float));
            d.hits.emplace_back(std::size_t(d.cfg.maxBatchSize) * sizeof(std::uint32_t));
        }
    }

    ReferenceBatchRunner::~ReferenceBatchRunner() {
        if (impl_) {
            impl_->drain();
        }
    }

    std::vector<std::uint32_t> ReferenceBatchRunner::runAll(std::span<const float> packedItems, std::uint32_t itemCount) {
        Impl& d = *impl_;
        if (itemCount == 0) {
            return {};
        }
        if (itemCount > d.cfg.maxItemsPerCheck) {
            throw CapacityError("ReferenceBatchRunner: " + std::to_string(itemCount) + " items > max_items_per_check = "
                                    + std::to_string(d.cfg.maxItemsPerCheck),
                "max_items_per_check", itemCount, d.cfg.maxItemsPerCheck);
        }
        const std::size_t floats = std::size_t(itemCount) * d.floatsPerItem;
        if (packedItems.size() < floats) {
            throw InvalidArgument("ReferenceBatchRunner: " + std::to_string(packedItems.size()) + " floats for "
                                  + std::to_string(itemCount) + " items of " + std::to_string(d.floatsPerItem) + " floats");
        }
        d.ctx->makeCurrent();
        d.input.upload(packedItems.data(), floats * sizeof(float));

        const std::uint32_t B  = d.cfg.maxBatchSize;
        const std::uint32_t nb = (itemCount + B - 1) / B;
        const auto S           = static_cast<std::uint32_t>(d.streams.size());
        d.backend->beginCheck(CheckInfo{itemCount, nb});
        try {
            for (std::uint32_t b = 0; b < nb; ++b) {
                const std::uint32_t s     = b % S;
                const StreamContext& sc   = d.streams[s];
                const std::uint32_t first = b * B;
                const std::uint32_t n     = (std::min)(B, itemCount - first); // windows.h (optix_stubs.h) min macro
                DeviceBuffer& stage       = d.staging[s];
                DeviceBuffer& hit         = d.hits[s];
                // 7.1 zero the stream's hit counters, 7.2 staging copy (rtcd/backend.h, design doc section 4). The
                // reference keeps the most conservative form the contract allows (staging buffer and per-stream
                // counters for every batch); the core may read items in place and count into the result directly,
                // so backends must not rely on either address being stable.
                hit.memsetAsync(0, std::size_t(n) * sizeof(std::uint32_t), sc.stream);
                stage.copyFromAsync(d.input.ptr() + std::uint64_t(first) * d.floatsPerItem * sizeof(float),
                    std::size_t(n) * d.floatsPerItem * sizeof(float), sc.stream);
                BatchView view;
                view.batchIndex = b;
                view.firstItem  = first;
                view.itemCount  = n;
                view.items      = DeviceSpan<const float>{stage.ptr(), std::size_t(n) * d.floatsPerItem};
                view.hitCounts  = DeviceSpan<std::uint32_t>{hit.ptr(), n};
                // 7.3 prepare, 7.4 launch
                const LaunchPlan plan = d.backend->prepare(sc, view);
                d.exec->launch(sc.stream, plan);
                // 7.5 result copy
                d.result.copyFromAsync(hit.ptr(), std::size_t(n) * sizeof(std::uint32_t), sc.stream,
                    std::size_t(first) * sizeof(std::uint32_t));
            }
            for (const auto& sc : d.streams) {
                RTCD_CUDA_CHECK(cudaStreamSynchronize(sc.stream));
            }
        } catch (...) {
            d.drain();
            throw;
        }
        std::vector<std::uint32_t> counts(itemCount);
        d.result.download(counts.data(), counts.size() * sizeof(std::uint32_t));
        return counts;
    }

} // namespace rtcd::internal
