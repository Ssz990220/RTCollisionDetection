// src/detector: OptixProgramPipeline (optix_pipeline.h).
#include "detector/optix_pipeline.h"

#include <rtcd/errors.h>
#include <rtcd/limits.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace rtcd::detail {

    // The rtcd mirrors of OptiX enums used here must keep the OptiX numeric values (backend.h, config.h).
    static_assert(kPrimitiveDefault == 0);
    static_assert(kPrimitiveRoundQuadratic == OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE);
    static_assert(kPrimitiveRoundCubic == OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE);
    static_assert(kPrimitiveRoundLinear == OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR);
    static_assert(kPrimitiveSphere == OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE);
    static_assert(kPrimitiveTriangle == static_cast<std::uint32_t>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE));
    static_assert(static_cast<std::uint32_t>(TraversableGraph::Any) == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY);
    static_assert(static_cast<std::uint32_t>(TraversableGraph::SingleGas) == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS);
    static_assert(static_cast<std::uint32_t>(TraversableGraph::SingleLevelInstancing)
                  == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING);
    static_assert(OptixOptions::kExceptionNone == OPTIX_EXCEPTION_FLAG_NONE);
    static_assert(OptixOptions::kExceptionStackOverflow == OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW);
    static_assert(OptixOptions::kExceptionTraceDepth == OPTIX_EXCEPTION_FLAG_TRACE_DEPTH);
    static_assert(OptixOptions::kExceptionUser == OPTIX_EXCEPTION_FLAG_USER);
    static_assert(kBuildFlagAllowUpdate == OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    static_assert(kBuildFlagAllowCompaction == OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
    static_assert(kBuildFlagPreferFastTrace == OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    static_assert(kBuildFlagPreferFastBuild == OPTIX_BUILD_FLAG_PREFER_FAST_BUILD);
    static_assert(sizeof(SbtRecord) % OPTIX_SBT_RECORD_ALIGNMENT == 0, "SbtRecord stride");
    static_assert(alignof(SbtRecord) >= OPTIX_SBT_RECORD_ALIGNMENT, "SbtRecord alignment");
    static_assert(kSbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE, "SBT header size");

    namespace {
        [[noreturn]] void internalError(const std::string& message) { throw Error(ErrorKind::Internal, message); }

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
        std::uint32_t primitiveBit(BuiltinIntersection b) {
            switch (b) {
            case BuiltinIntersection::Sphere: return kPrimitiveSphere;
            case BuiltinIntersection::RoundLinear: return kPrimitiveRoundLinear;
            case BuiltinIntersection::RoundQuadratic: return kPrimitiveRoundQuadratic;
            case BuiltinIntersection::RoundCubic: return kPrimitiveRoundCubic;
            default: return 0;
            }
        }
        const char* nameOrNull(const std::string& s) { return s.empty() ? nullptr : s.c_str(); }
        std::string programSetName(const ProgramSet& p) { return "ProgramSet '" + p.moduleName + "'"; }

        // Compile logs: OptiX NUL-terminates and reports the size written (including the NUL).
        constexpr std::size_t kLogSize = 8192;
        struct CompileLog {
            char text[kLogSize];
            std::size_t size = kLogSize;
            void reset() {
                text[0] = 0;
                size    = kLogSize;
            }
            bool nonEmpty() const { return size > 1 && text[0] != 0; }
        };
    } // namespace

    void validateProgramSet(const ProgramSet& p) {
        if (p.moduleCode.empty()) {
            internalError(programSetName(p) + " has no module code");
        }
        if (p.raygens.empty()) {
            internalError(programSetName(p) + " has no raygen entry point");
        }
        for (std::size_t i = 0; i < p.raygens.size(); ++i) {
            if (p.raygens[i].empty()) {
                internalError(programSetName(p) + ": raygens[" + std::to_string(i) + "] is empty");
            }
        }
        if (p.hitGroupRecordCount == 0) {
            internalError(programSetName(p) + " has hitGroupRecordCount == 0");
        }
        if (p.maxTraversableDepth == 0) {
            internalError(programSetName(p) + " has maxTraversableDepth == 0");
        }
        if (p.builtinIntersection != BuiltinIntersection::None && p.primitiveTypes != kPrimitiveDefault
            && (p.primitiveTypes & primitiveBit(p.builtinIntersection)) == 0) {
            internalError(programSetName(p) + ": primitiveTypes = " + std::to_string(p.primitiveTypes)
                          + " does not include the primitive of its built-in intersection program");
        }
        if (p.builtinIntersection != BuiltinIntersection::None && p.primitiveTypes == kPrimitiveDefault) {
            internalError(programSetName(p)
                          + ": a built-in intersection program needs its primitive type in primitiveTypes (0 = "
                            "triangles + custom only)");
        }
    }

    void validateLaunchPlan(const LaunchPlan& plan, std::uint32_t raygenCount, const std::string& moduleName) {
        if (plan.raygen >= raygenCount) {
            internalError("backend '" + moduleName + "': LaunchPlan::raygen = " + std::to_string(plan.raygen)
                          + " but the ProgramSet has " + std::to_string(raygenCount) + " raygens");
        }
        if (plan.width == 0 || plan.height == 0 || plan.depth == 0) {
            internalError("backend '" + moduleName + "': LaunchPlan dimensions must be >= 1 (got "
                          + std::to_string(plan.width) + " x " + std::to_string(plan.height) + " x "
                          + std::to_string(plan.depth) + ")");
        }
        const std::uint64_t size = std::uint64_t(plan.width) * plan.height * plan.depth;
        if (size > kMaxLaunchSize) {
            internalError("backend '" + moduleName + "': launch size " + std::to_string(size) + " exceeds "
                          + std::to_string(kMaxLaunchSize));
        }
        if (plan.paramsBytes > kMaxLaunchParamsBytes) {
            internalError("backend '" + moduleName + "': LaunchPlan::paramsBytes = " + std::to_string(plan.paramsBytes)
                          + " > kMaxLaunchParamsBytes = " + std::to_string(kMaxLaunchParamsBytes));
        }
        if (plan.paramsBytes > 0 && plan.params == nullptr) {
            internalError("backend '" + moduleName + "': LaunchPlan::params is null but paramsBytes = "
                          + std::to_string(plan.paramsBytes));
        }
    }

    struct OptixProgramPipeline::Impl {
        std::string moduleName;
        OptixModule module   = nullptr;
        OptixModule isModule = nullptr; // built-in IS module: owned by the OptiX device context (never destroyed,
                                        // as in the old detector)
        std::vector<OptixProgramGroup> raygenPGs;
        OptixProgramGroup missPG = nullptr;
        OptixProgramGroup hitPG  = nullptr;
        OptixPipeline pipeline   = nullptr;
        DeviceBuffer raygenRecords, missRecords, hitRecords;
        std::vector<OptixShaderBindingTable> sbts; // one per raygen (raygenRecord points at that raygen's record)

        void destroy() noexcept {
            // OptiX destroy calls only fail on invalid handles; nothing to report.
            if (pipeline) {
                optixPipelineDestroy(pipeline);
            }
            if (hitPG) {
                optixProgramGroupDestroy(hitPG);
            }
            if (missPG) {
                optixProgramGroupDestroy(missPG);
            }
            for (OptixProgramGroup pg : raygenPGs) {
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

    OptixProgramPipeline::OptixProgramPipeline(
        const Context& context, const ProgramSet& programs, const OptixOptions& optix, const Logger& logger)
        : impl_(std::make_unique<Impl>()) {
        validateProgramSet(programs);
        Impl& d      = *impl_;
        d.moduleName = programs.moduleName;
        context.makeCurrent();
        const OptixDeviceContext octx = context.optixContext();
        const std::string who         = "checker pipeline (" + programs.moduleName + ")";
        auto logCompile               = [&](const char* step, const CompileLog& log) {
            if (log.nonEmpty() && logger.enabled(LogLevel::Debug)) {
                logger.log(LogLevel::Debug, who + " " + step + " log: " + log.text);
            }
        };

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

            CompileLog log;
            log.reset();
            RTCD_OPTIX_CHECK_LOG(optixModuleCreate(octx, &mco, &pco, reinterpret_cast<const char*>(programs.moduleCode.data()),
                                     programs.moduleCode.size(), log.text, &log.size, &d.module),
                log.text);
            logCompile("module", log);

            if (programs.builtinIntersection != BuiltinIntersection::None) {
                // Build flags must be exactly those of the GAS it intersects (ProgramSet contract); curves get end
                // caps on (old createModule: curveEndcapFlags = 1).
                OptixBuiltinISOptions iso{};
                iso.builtinISModuleType = toOptix(programs.builtinIntersection);
                iso.buildFlags          = programs.builtinIsBuildFlags;
                iso.usesMotionBlur      = 0;
                iso.curveEndcapFlags    = programs.builtinIntersection == BuiltinIntersection::Sphere
                                            ? 0u
                                            : static_cast<unsigned int>(OPTIX_CURVE_ENDCAP_ON);
                RTCD_OPTIX_CHECK(optixBuiltinISModuleGet(octx, &mco, &pco, &iso, &d.isModule));
            }

            const OptixProgramGroupOptions pgo{};
            for (const std::string& rg : programs.raygens) {
                OptixProgramGroupDesc desc{};
                desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                desc.raygen.module            = d.module;
                desc.raygen.entryFunctionName = rg.c_str();
                OptixProgramGroup pg          = nullptr;
                log.reset();
                RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(octx, &desc, 1, &pgo, log.text, &log.size, &pg), log.text);
                d.raygenPGs.push_back(pg);
                logCompile("raygen program group", log);
            }
            {
                // Empty miss name = null miss program (module nullptr), as the old createMissPrograms for every
                // mode but RayHitMode::FirstHitMiss. One miss record either way.
                OptixProgramGroupDesc desc{};
                desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                desc.miss.module            = programs.miss.empty() ? nullptr : d.module;
                desc.miss.entryFunctionName = nameOrNull(programs.miss);
                log.reset();
                RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(octx, &desc, 1, &pgo, log.text, &log.size, &d.missPG), log.text);
                logCompile("miss program group", log);
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
                log.reset();
                RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(octx, &desc, 1, &pgo, log.text, &log.size, &d.hitPG), log.text);
                logCompile("hit group", log);
            }

            std::vector<OptixProgramGroup> groups = d.raygenPGs;
            groups.push_back(d.missPG);
            groups.push_back(d.hitPG);
            OptixPipelineLinkOptions plo{};
            plo.maxTraceDepth = optix.maxTraceDepth;
            log.reset();
            RTCD_OPTIX_CHECK_LOG(optixPipelineCreate(octx, &pco, &plo, groups.data(),
                                     static_cast<unsigned int>(groups.size()), log.text, &log.size, &d.pipeline),
                log.text);
            if (optix.printPipelineLog && log.nonEmpty()) {
                // An explicit request: bypasses the checker's verbosity filter (old PRINTIT / printPipelineLog
                // printed unconditionally).
                Logger direct = logger;
                direct.level  = LogLevel::Debug;
                direct.log(LogLevel::Info, who + " pipeline log: " + log.text);
            }

            // Stack sizes as the old createPipeline: no recursive traces (trace depth 1), no callables; the
            // traversable depth comes from the ProgramSet (2 with instancing, 1 for a single GAS).
            OptixStackSizes ss{};
            for (OptixProgramGroup pg : groups) {
                RTCD_OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &ss, d.pipeline));
            }
            unsigned int dcTrav = 0, dcState = 0, cont = 0;
            RTCD_OPTIX_CHECK(optixUtilComputeStackSizes(&ss, 1, 0, 0, &dcTrav, &dcState, &cont));
            RTCD_OPTIX_CHECK(optixPipelineSetStackSize(d.pipeline, dcTrav, dcState, cont, programs.maxTraversableDepth));

            // SBT: one raygen record per entry, one miss record, hitGroupRecordCount identical hit records; data 0
            // (old buildSBT: data = nullptr / 0 everywhere).
            auto pack = [](OptixProgramGroup pg, std::size_t n) {
                std::vector<SbtRecord> recs(n);
                for (SbtRecord& r : recs) {
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
            d.raygenRecords                 = DeviceBuffer::fromHost(std::span<const SbtRecord>(rg));
            d.missRecords                   = DeviceBuffer::fromHost(std::span<const SbtRecord>(ms));
            d.hitRecords                    = DeviceBuffer::fromHost(std::span<const SbtRecord>(hg));

            OptixShaderBindingTable sbt{};
            sbt.missRecordBase              = d.missRecords.ptr();
            sbt.missRecordStrideInBytes     = sizeof(SbtRecord);
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = d.hitRecords.ptr();
            sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
            sbt.hitgroupRecordCount         = programs.hitGroupRecordCount;
            for (std::size_t i = 0; i < d.raygenPGs.size(); ++i) {
                sbt.raygenRecord = d.raygenRecords.ptr() + i * sizeof(SbtRecord);
                d.sbts.push_back(sbt);
            }
        } catch (...) {
            d.destroy();
            throw;
        }
    }

    OptixProgramPipeline::~OptixProgramPipeline() {
        if (impl_) {
            impl_->destroy();
        }
    }

    std::uint32_t OptixProgramPipeline::raygenCount() const noexcept {
        return static_cast<std::uint32_t>(impl_->sbts.size());
    }

    const std::string& OptixProgramPipeline::moduleName() const noexcept { return impl_->moduleName; }

    void OptixProgramPipeline::launch(
        StreamHandle stream, DevicePtr params, std::uint32_t paramsBytes, const LaunchPlan& plan) const {
        const Impl& d = *impl_;
        RTCD_OPTIX_CHECK(optixLaunch(
            d.pipeline, stream, params, paramsBytes, &d.sbts[plan.raygen], plan.width, plan.height, plan.depth));
    }

} // namespace rtcd::detail
