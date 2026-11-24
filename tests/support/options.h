#pragma once
// support/options.h -- environment / command-line knobs shared by the apps (regression, benchmarks). The library
// never reads the environment (design doc 3.12); the apps do, through these helpers.
//
// Legacy environment variables of the old stack's benchmarks (removed; tag old-stack-final) are accepted unchanged:
//   RTCD_OPTIX_CFG="key=value,..."   old OptixCompileConfig::setFields keys: opt, debug, maxreg, exceptions,
//                                    tracedepth, rayhit, pipelinelog (-> CheckerConfig), validation, loglevel
//                                    (-> ContextOptions)
//   RTCD_CURVE_GEOM_FLAG=none|single_anyhit      -> curve.geometryFlags
//   RTCD_BM_REFIT_EVERY=<K>                      -> curve.refit (0 always, 1 never, K >= 2 every K)
// New-stack variable:
//   RTCD_OVERRIDES="path=value,..."  rtcd::applyOverrides syntax (rtcd/json_config.h), applied last by the apps.

#include <rtcd/config.h>
#include <rtcd/context.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rtcd::support {

    // Applies an old RTCD_OPTIX_CFG list ("key=value,key=value", same grammar and value sets as
    // RTCD::OptixCompileConfig::setFields) to the new structs. Throws ConfigError naming the bad entry. Either
    // pointer may be null (keys that target it are then validated but ignored).
    void applyLegacyOptixConfig(std::string_view list, CheckerConfig* config, ContextOptions* context);

    // RTCD_CURVE_GEOM_FLAG value -> CurveOptions::geometryFlags. Throws ConfigError for other values.
    GeometryFlags parseLegacyCurveGeometryFlag(std::string_view value);
    // RTCD_BM_REFIT_EVERY value (old setRefitPolicy(rebuildEvery)) -> RefitPolicy. Throws ConfigError.
    RefitPolicy parseLegacyRefitEvery(std::string_view value);

    // What the environment asks for (all optional).
    struct EnvSettings {
        std::optional<std::string> legacyOptixConfig; // RTCD_OPTIX_CFG
        std::optional<std::string> curveGeometryFlag; // RTCD_CURVE_GEOM_FLAG
        std::optional<std::string> refitEvery;        // RTCD_BM_REFIT_EVERY
        std::optional<std::string> overrides;         // RTCD_OVERRIDES
    };
    EnvSettings readEnvSettings();

    // Applies the legacy part of `env` (RTCD_OPTIX_CFG, RTCD_CURVE_GEOM_FLAG, RTCD_BM_REFIT_EVERY) to the structs.
    // RTCD_OVERRIDES is NOT applied here (it needs rtcd_json); apps call rtcd::applyOverrides with it.
    void applyLegacyEnv(const EnvSettings& env, CheckerConfig* config, ContextOptions* context);

    // Splits "1,2,4" / "1-4096" (powers of two from a to b) / "1:4096:x2" into item counts. Throws
    // InvalidArgument.
    std::vector<std::size_t> parseCountList(std::string_view text);
    // Powers of two from 1 to maxN inclusive (the old benchmarks' Range(1, 1 << 12) with multiplier 2).
    std::vector<std::size_t> powersOfTwo(std::size_t maxN);

    // Short human-readable description of the fields that matter for timing comparisons.
    std::string describeConfig(const CheckerConfig& config);
    std::string modeName(Mode mode); // JSON spelling ("mesh", "ray", "linear_curve", ...)

} // namespace rtcd::support
