#pragma once
// Benchmark-only hook: lets one benchmark executable sweep OptiX settings through environment variables
// (the library itself never reads the environment).
//
//   RTCD_OPTIX_CFG="key=value,key=value"   applied to OptixCompileConfig::buildDefault(), keys see
//                                          OptixCompileConfig::setField (opt, debug, maxreg, exceptions, tracedepth,
//                                          rayhit, pipelinelog, validation, loglevel)
//   RTCD_CURVE_GEOM_FLAG=none|single_anyhit curve build-input geometry flag (batchOptixCurveRobot::setGeometryFlags)
#include <CollisionDetector/OptixCompileConfig.h>
#include <optix_types.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace RTCD {
    inline OptixCompileConfig benchmarkOptixConfig() {
        OptixCompileConfig cfg = OptixCompileConfig::buildDefault();
        if (const char* v = std::getenv("RTCD_OPTIX_CFG")) {
            if (!cfg.setFields(v)) {
                std::fprintf(stderr, "RTCD_OPTIX_CFG: invalid entry in \"%s\"\n", v);
                std::exit(1);
            }
        }
        std::fprintf(stderr, "[optix cfg] %s\n", cfg.describe().c_str());
        return cfg;
    }

    // Returns the curve geometry flag requested by RTCD_CURVE_GEOM_FLAG (OPTIX_GEOMETRY_FLAG_NONE if unset).
    inline unsigned int benchmarkCurveGeometryFlags() {
        unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
        if (const char* v = std::getenv("RTCD_CURVE_GEOM_FLAG")) {
            const std::string s(v);
            if (s == "single_anyhit") {
                flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
            } else if (s != "none") {
                std::fprintf(stderr, "RTCD_CURVE_GEOM_FLAG: unknown value \"%s\"\n", v);
                std::exit(1);
            }
        }
        return flags;
    }
} // namespace RTCD
