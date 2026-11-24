#pragma once
// Runtime OptiX program compile / link / context settings of a CollisionDetector.
//
// Set with CollisionDetector::setCompileConfig() (before setupOptix(), or afterwards: a change re-creates the
// modules, program groups and pipelines and rebuilds the SBT). Defaults reproduce the behaviour of the former
// compile-time macro selection: release() in Release builds, debug() when _DEBUG / DEBUG is defined.
//
// The library never reads environment variables or command lines; setField() lets benchmarks and tests map
// "key=value" strings (from env vars or argv) onto the struct so one executable can sweep settings.
#include <optix_types.h>

#include <cstdlib>
#include <string>

namespace RTCD {

    // How RAY_ONLINE (RobotToObs) rays find out whether they hit the scene. All modes give the same yes/no per ray
    // (the scene triangles have any-hit disabled and nothing else is read from the hit).
    enum class RayHitMode : int {
        // optixTrace, DISABLE_ANYHIT; full closest-hit search, closest-hit program sets payload = 1.
        CLOSEST_HIT = 0,
        // as CLOSEST_HIT plus TERMINATE_ON_FIRST_HIT: traversal stops at the first hit, closest-hit sets payload = 1.
        FIRST_HIT_CH = 1,
        // TERMINATE_ON_FIRST_HIT | DISABLE_CLOSESTHIT, payload starts at 1 and the miss program clears it.
        FIRST_HIT_MISS = 2,
        // optixTraverse with TERMINATE_ON_FIRST_HIT, then optixHitObjectIsHit(): no closest-hit / miss program runs.
        FIRST_HIT_TRAVERSE = 3,
    };

    struct OptixCompileConfig {
        // OptixModuleCompileOptions
        OptixCompileOptimizationLevel optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        // DEFAULT is documented as "currently minimal" (line info kept), not NONE.
        OptixCompileDebugLevel debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
        int maxRegisterCount              = 50; // 0 = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT (no limit)
        // OptixPipelineCompileOptions
        unsigned int exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        // OptixPipelineLinkOptions. No program traces recursively; the stack sizes are always computed for depth 1.
        unsigned int maxTraceDepth = 2;
        // RAY_ONLINE only: raygen entry / ray flags (see RayHitMode).
        RayHitMode rayHitMode = RayHitMode::CLOSEST_HIT;
        // Print the pipeline link log (formerly only with PRINTIT).
        bool printPipelineLog = false;

        // Device-context options, used by createContext*() in Utils/Test/cudaCtxUtils.h (not by the detector).
        bool contextValidation = false; // OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
        int contextLogLevel    = 0; // 0 = no log callback, otherwise optixDeviceContextSetLogCallback level (1-4)

        static OptixCompileConfig release() { return {}; }

        static OptixCompileConfig debug() {
            OptixCompileConfig c;
            c.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            c.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            c.exceptionFlags    = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            c.contextValidation = true;
            return c;
        }

        // Former OPTIX_PROFILE branch: optimized code with moderate debug info for Nsight Compute.
        static OptixCompileConfig profile() {
            OptixCompileConfig c;
            c.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
            return c;
        }

        // Defaults for the build type the including translation unit is compiled as.
        static OptixCompileConfig buildDefault() {
#if defined(_DEBUG) || defined(DEBUG)
            OptixCompileConfig c = debug();
#else
            OptixCompileConfig c = release();
#endif
#ifdef PRINTIT
            c.printPipelineLog = true;
            c.contextLogLevel  = 4;
#endif
            return c;
        }

        // Fields that affect modules / program groups / pipelines (the context fields do not).
        bool samePipeline(const OptixCompileConfig& o) const {
            return optLevel == o.optLevel && debugLevel == o.debugLevel && maxRegisterCount == o.maxRegisterCount
                && exceptionFlags == o.exceptionFlags && maxTraceDepth == o.maxTraceDepth
                && rayHitMode == o.rayHitMode && printPipelineLog == o.printPipelineLog;
        }

        // Sets one field from strings. Keys: opt (default|0|1|2|3), debug (default|none|minimal|moderate|full),
        // maxreg (int), exceptions (none|debug|<int>), tracedepth (int), rayhit (0-3 or
        // closest|firsthit_ch|firsthit_miss|traverse), pipelinelog (0|1), validation (0|1), loglevel (0-4).
        // Returns false for an unknown key or value (the struct is then unchanged).
        bool setField(const std::string& key, const std::string& value) {
            auto toInt = [&](int& out) {
                char* end = nullptr;
                const long v = std::strtol(value.c_str(), &end, 10);
                if (value.empty() || *end != '\0') {
                    return false;
                }
                out = static_cast<int>(v);
                return true;
            };
            int i = 0;
            if (key == "opt") {
                if (value == "default") {
                    optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
                } else if (toInt(i) && i >= 0 && i <= 3) {
                    optLevel = static_cast<OptixCompileOptimizationLevel>(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 + i);
                } else {
                    return false;
                }
            } else if (key == "debug") {
                if (value == "default") {
                    debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
                } else if (value == "none") {
                    debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
                } else if (value == "minimal") {
                    debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
                } else if (value == "moderate") {
                    debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
                } else if (value == "full") {
                    debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
                } else {
                    return false;
                }
            } else if (key == "maxreg") {
                if (!toInt(i) || i < 0) {
                    return false;
                }
                maxRegisterCount = i;
            } else if (key == "exceptions") {
                if (value == "none") {
                    exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
                } else if (value == "debug") {
                    exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
                } else if (toInt(i) && i >= 0) {
                    exceptionFlags = static_cast<unsigned int>(i);
                } else {
                    return false;
                }
            } else if (key == "tracedepth") {
                if (!toInt(i) || i < 1) {
                    return false;
                }
                maxTraceDepth = static_cast<unsigned int>(i);
            } else if (key == "rayhit") {
                if (value == "closest") {
                    i = 0;
                } else if (value == "firsthit_ch") {
                    i = 1;
                } else if (value == "firsthit_miss") {
                    i = 2;
                } else if (value == "traverse") {
                    i = 3;
                } else if (!toInt(i) || i < 0 || i > 3) {
                    return false;
                }
                rayHitMode = static_cast<RayHitMode>(i);
            } else if (key == "pipelinelog") {
                if (!toInt(i)) {
                    return false;
                }
                printPipelineLog = i != 0;
            } else if (key == "validation") {
                if (!toInt(i)) {
                    return false;
                }
                contextValidation = i != 0;
            } else if (key == "loglevel") {
                if (!toInt(i) || i < 0 || i > 4) {
                    return false;
                }
                contextLogLevel = i;
            } else {
                return false;
            }
            return true;
        }

        // Applies a comma separated "key=value,key=value" list; false if any entry is malformed or unknown.
        bool setFields(const std::string& list) {
            size_t pos = 0;
            bool ok    = true;
            while (pos <= list.size()) {
                size_t end = list.find(',', pos);
                if (end == std::string::npos) {
                    end = list.size();
                }
                const std::string item = list.substr(pos, end - pos);
                if (!item.empty()) {
                    const size_t eq = item.find('=');
                    ok &= eq != std::string::npos && setField(item.substr(0, eq), item.substr(eq + 1));
                }
                pos = end + 1;
            }
            return ok;
        }

        std::string describe() const {
            return "opt=" + std::to_string(static_cast<int>(optLevel)) + " debug=0x"
                 + toHex(static_cast<unsigned int>(debugLevel)) + " maxreg=" + std::to_string(maxRegisterCount)
                 + " exceptions=" + std::to_string(exceptionFlags) + " tracedepth=" + std::to_string(maxTraceDepth)
                 + " rayhit=" + std::to_string(static_cast<int>(rayHitMode));
        }

    private:
        static std::string toHex(unsigned int v) {
            const char* digits = "0123456789abcdef";
            std::string s;
            do {
                s.insert(s.begin(), digits[v & 0xf]);
                v >>= 4;
            } while (v);
            return s;
        }
    };

} // namespace RTCD
