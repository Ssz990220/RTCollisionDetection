#pragma once
// src/detector (private): the OptiX objects of one checker, built from a backend's ProgramSet -- module (+ built-in
// IS module), program groups, ONE pipeline, the SBT -- and the launch of a LaunchPlan on it.
//
// Old stack: CollisionDetector::createModule / createRaygenPrograms / createMissPrograms / createHitgroupPrograms /
// createPipeline / buildSBT / destroyOptix (RTCD/CollisionDetector/CollisionDetector.h), which decided all of this
// with `if constexpr (TYPE == ...)`. Differences: one pipeline per checker instead of one identical pipeline per
// stream (design doc Q2: OptiX allows concurrent launches of one pipeline on several streams), one raygen record
// per ProgramSet::raygens entry (a plan selects one), errors by exception.
//
// Threading: construction / destruction on one thread; launch() is const and may be called concurrently from
// several threads for different streams (it only reads the precomputed SBTs).

#include <rtcd/backend.h>
#include <rtcd/config.h>
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>

#include <cstdint>
#include <memory>
#include <string>

namespace rtcd::detail {

    // Throws rtcd::Error(ErrorKind::Internal) naming the first problem of a ProgramSet that would otherwise
    // surface as an obscure OptiX error (no module code, no raygen, empty entry name, hitGroupRecordCount == 0,
    // maxTraversableDepth == 0, built-in IS without its primitive-type bit).
    void validateProgramSet(const ProgramSet& programs);

    // Throws rtcd::Error(ErrorKind::Internal) if a LaunchPlan with launch == true breaks the backend.h rules
    // (raygen index, dimensions >= 1 and product <= kMaxLaunchSize, paramsBytes <= kMaxLaunchParamsBytes, params
    // non-null when paramsBytes > 0). A broken plan is a backend bug, not a caller error.
    void validateLaunchPlan(const LaunchPlan& plan, std::uint32_t raygenCount, const std::string& moduleName);

    class OptixProgramPipeline {
    public:
        // `logger` receives the pipeline log (Info, only with optix.printPipelineLog) and non-empty module /
        // program-group compile logs (Debug). Throws OptixError (with the compile log), CudaError, Error(Internal).
        OptixProgramPipeline(const Context& context, const ProgramSet& programs, const OptixOptions& optix,
            const Logger& logger);
        ~OptixProgramPipeline(); // never throws
        OptixProgramPipeline(const OptixProgramPipeline&)            = delete;
        OptixProgramPipeline& operator=(const OptixProgramPipeline&) = delete;

        std::uint32_t raygenCount() const noexcept;
        const std::string& moduleName() const noexcept;

        // optixLaunch(pipeline, stream, params, paramsBytes, sbt(raygen = plan.raygen), w, h, d). The plan must have
        // passed validateLaunchPlan. Asynchronous.
        void launch(StreamHandle stream, DevicePtr params, std::uint32_t paramsBytes, const LaunchPlan& plan) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rtcd::detail
