#pragma once
// New-stack CheckerConfig equivalent to an old-stack profile (old_runner.h), so both stacks run the same setup by
// construction. Pure struct code (no library calls), so it is unit-tested without the new stack.

#include "old_runner.h"

#include <rtcd/config.h>
#include <rtcd/context.h>

#include <filesystem>

namespace rtcd::sbs {

    // Mode, streams, batch size, item capacity, OBB culling (explicit On / Off as the old useOBB), build
    // preferences (COMPACT -> FastTraceCompact, FAST_TRACE -> FastTrace, FAST_BUILD -> FastBuild), trajectory /
    // control points, host dispatch ThreadPerStream (old USE_MULTI_THREADING), OptiX options buildDefault() (old
    // OptixCompileConfig::buildDefault()), splineWeightsDir as given. Everything else stays at the new defaults,
    // which equal the old ones (compaction min poses 128, ray width cap 262144, skip-hit on, closest-hit, refit
    // auto, geometry flags auto, collapse root link OBB, scene edges auto).
    CheckerConfig newConfigForProfile(const OldProfile& profile, const std::filesystem::path& splineWeightsDir);

    // Applies the old-syntax settings (old_runner.h OldSettings) to the new structs, the same way the old runner
    // applies them to the old stack. Throws ConfigError.
    void applyOldSettings(const OldSettings& settings, CheckerConfig& config, ContextOptions& context);

} // namespace rtcd::sbs
