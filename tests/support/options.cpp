#include <support/options.h>
#include <support/paths.h>

#include <rtcd/errors.h>

#include <charconv>
#include <sstream>

namespace rtcd::support {

    namespace {
        std::string trim(std::string_view s) {
            std::size_t a = 0, b = s.size();
            while (a < b && (s[a] == ' ' || s[a] == '\t')) {
                ++a;
            }
            while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) {
                --b;
            }
            return std::string(s.substr(a, b - a));
        }

        // Same acceptance as the old strtol-based toInt: optional sign, digits, nothing else.
        bool toInt(const std::string& v, long long& out) {
            if (v.empty()) {
                return false;
            }
            const char* first = v.data();
            if (*first == '+') {
                ++first;
            }
            const auto [ptr, ec] = std::from_chars(first, v.data() + v.size(), out);
            return ec == std::errc() && ptr == v.data() + v.size();
        }

        [[noreturn]] void bad(std::string_view what, const std::string& item) {
            throw ConfigError("RTCD_OPTIX_CFG: invalid entry '" + item + "' (" + std::string(what) + ")", "optix");
        }
    } // namespace

    void applyLegacyOptixConfig(std::string_view list, CheckerConfig* config, ContextOptions* context) {
        // Work on copies so a bad entry leaves the targets unchanged (old setField semantics per entry; here the
        // whole list is all-or-nothing, which is stricter and simpler for the apps).
        CheckerConfig cfg   = config ? *config : CheckerConfig{};
        ContextOptions ctx  = context ? *context : ContextOptions{};
        std::size_t pos = 0;
        while (pos <= list.size()) {
            std::size_t end = list.find(',', pos);
            if (end == std::string_view::npos) {
                end = list.size();
            }
            const std::string item = trim(list.substr(pos, end - pos));
            pos                    = end + 1;
            if (item.empty()) {
                continue;
            }
            const std::size_t eq = item.find('=');
            if (eq == std::string::npos) {
                bad("expected key=value", item);
            }
            const std::string key   = trim(std::string_view(item).substr(0, eq));
            const std::string value = trim(std::string_view(item).substr(eq + 1));
            long long i             = 0;
            using OL = OptixOptions::OptLevel;
            using DL = OptixOptions::DebugLevel;
            if (key == "opt") {
                if (value == "default") {
                    cfg.optix.optLevel = OL::Default;
                } else if (toInt(value, i) && i >= 0 && i <= 3) {
                    cfg.optix.optLevel = static_cast<OL>(static_cast<int>(OL::O0) + static_cast<int>(i));
                } else {
                    bad("opt: default|0|1|2|3", item);
                }
            } else if (key == "debug") {
                if (value == "default") {
                    cfg.optix.debugLevel = DL::Default;
                } else if (value == "none") {
                    cfg.optix.debugLevel = DL::None;
                } else if (value == "minimal") {
                    cfg.optix.debugLevel = DL::Minimal;
                } else if (value == "moderate") {
                    cfg.optix.debugLevel = DL::Moderate;
                } else if (value == "full") {
                    cfg.optix.debugLevel = DL::Full;
                } else {
                    bad("debug: default|none|minimal|moderate|full", item);
                }
            } else if (key == "maxreg") {
                if (!toInt(value, i) || i < 0) {
                    bad("maxreg: integer >= 0", item);
                }
                cfg.optix.maxRegisterCount = static_cast<std::uint32_t>(i);
            } else if (key == "exceptions") {
                if (value == "none") {
                    cfg.optix.exceptionFlags = OptixOptions::kExceptionNone;
                } else if (value == "debug") {
                    cfg.optix.exceptionFlags = OptixOptions::kExceptionTraceDepth | OptixOptions::kExceptionStackOverflow;
                } else if (toInt(value, i) && i >= 0) {
                    const std::uint32_t known = OptixOptions::kExceptionStackOverflow | OptixOptions::kExceptionTraceDepth
                                              | OptixOptions::kExceptionUser;
                    if ((static_cast<std::uint64_t>(i) & ~std::uint64_t(known)) != 0) {
                        bad("exceptions: only bits 1 (stack overflow), 2 (trace depth), 4 (user) exist in OptiX 9", item);
                    }
                    cfg.optix.exceptionFlags = static_cast<std::uint32_t>(i);
                } else {
                    bad("exceptions: none|debug|<bits>", item);
                }
            } else if (key == "tracedepth") {
                if (!toInt(value, i) || i < 1) {
                    bad("tracedepth: integer >= 1", item);
                }
                cfg.optix.maxTraceDepth = static_cast<std::uint32_t>(i);
            } else if (key == "rayhit") {
                if (value == "closest") {
                    i = 0;
                } else if (value == "firsthit_ch") {
                    i = 1;
                } else if (value == "firsthit_miss") {
                    i = 2;
                } else if (value == "traverse") {
                    i = 3;
                } else if (!toInt(value, i) || i < 0 || i > 3) {
                    bad("rayhit: closest|firsthit_ch|firsthit_miss|traverse|0-3", item);
                }
                cfg.ray.hitMode = static_cast<RayHitMode>(i); // same numeric values as the old RayHitMode
            } else if (key == "pipelinelog") {
                if (!toInt(value, i)) {
                    bad("pipelinelog: integer", item);
                }
                cfg.optix.printPipelineLog = i != 0;
            } else if (key == "validation") {
                if (!toInt(value, i)) {
                    bad("validation: integer", item);
                }
                ctx.optixValidation = i != 0;
            } else if (key == "loglevel") {
                if (!toInt(value, i) || i < 0 || i > 4) {
                    bad("loglevel: 0-4", item);
                }
                ctx.optixLogLevel = static_cast<int>(i);
            } else {
                bad("unknown key", item);
            }
        }
        if (config) {
            *config = cfg;
        }
        if (context) {
            *context = ctx;
        }
    }

    GeometryFlags parseLegacyCurveGeometryFlag(std::string_view value) {
        const std::string v = trim(value);
        if (v == "none") {
            return kGeometryFlagNone;
        }
        if (v == "single_anyhit") {
            return kGeometryFlagRequireSingleAnyHitCall;
        }
        throw ConfigError("RTCD_CURVE_GEOM_FLAG: unknown value '" + v + "' (none|single_anyhit)", "curve.geometry_flags");
    }

    RefitPolicy parseLegacyRefitEvery(std::string_view value) {
        const std::string v = trim(value);
        long long k         = 0;
        if (!toInt(v, k) || k < 0) {
            throw ConfigError("RTCD_BM_REFIT_EVERY: '" + v + "' is not an integer >= 0", "curve.refit");
        }
        RefitPolicy p;
        if (k == 0) {
            p.kind = RefitPolicy::Kind::Always;
        } else if (k == 1) {
            p.kind = RefitPolicy::Kind::Never;
        } else {
            p.kind  = RefitPolicy::Kind::Every;
            p.every = static_cast<std::uint32_t>(k);
        }
        return p;
    }

    EnvSettings readEnvSettings() {
        EnvSettings e;
        e.legacyOptixConfig = getEnv("RTCD_OPTIX_CFG");
        e.curveGeometryFlag = getEnv("RTCD_CURVE_GEOM_FLAG");
        e.refitEvery        = getEnv("RTCD_BM_REFIT_EVERY");
        e.overrides         = getEnv("RTCD_OVERRIDES");
        return e;
    }

    void applyLegacyEnv(const EnvSettings& env, CheckerConfig* config, ContextOptions* context) {
        if (env.legacyOptixConfig) {
            applyLegacyOptixConfig(*env.legacyOptixConfig, config, context);
        }
        if (env.curveGeometryFlag) {
            const GeometryFlags f = parseLegacyCurveGeometryFlag(*env.curveGeometryFlag);
            if (config) {
                config->curve.geometryFlags = f;
            }
        }
        if (env.refitEvery) {
            const RefitPolicy p = parseLegacyRefitEvery(*env.refitEvery);
            if (config) {
                config->curve.refit = p;
            }
        }
    }

    std::vector<std::size_t> powersOfTwo(std::size_t maxN) {
        std::vector<std::size_t> v;
        for (std::size_t n = 1; n <= maxN; n *= 2) {
            v.push_back(n);
        }
        return v;
    }

    std::vector<std::size_t> parseCountList(std::string_view text) {
        std::vector<std::size_t> out;
        std::size_t pos = 0;
        auto num        = [&](const std::string& s) {
            long long v = 0;
            if (!toInt(s, v) || v < 1) {
                throw InvalidArgument("count list '" + std::string(text) + "': '" + s + "' is not an integer >= 1");
            }
            return static_cast<std::size_t>(v);
        };
        while (pos <= text.size()) {
            std::size_t end = text.find(',', pos);
            if (end == std::string_view::npos) {
                end = text.size();
            }
            const std::string item = trim(text.substr(pos, end - pos));
            pos                    = end + 1;
            if (item.empty()) {
                continue;
            }
            const std::size_t dash = item.find('-');
            if (dash != std::string::npos) { // a-b: powers of two times a, up to b
                const std::size_t a = num(trim(std::string_view(item).substr(0, dash)));
                const std::size_t b = num(trim(std::string_view(item).substr(dash + 1)));
                if (b < a) {
                    throw InvalidArgument("count list '" + std::string(text) + "': empty range '" + item + "'");
                }
                for (std::size_t n = a; n <= b; n *= 2) {
                    out.push_back(n);
                }
            } else {
                out.push_back(num(item));
            }
        }
        if (out.empty()) {
            throw InvalidArgument("count list is empty");
        }
        return out;
    }

    std::string modeName(Mode mode) {
        switch (mode) {
        case Mode::Mesh: return "mesh";
        case Mode::Ray: return "ray";
        case Mode::LinearCurve: return "linear_curve";
        case Mode::QuadraticCurve: return "quadratic_curve";
        case Mode::CubicCurve: return "cubic_curve";
        }
        return "?";
    }

    std::string describeConfig(const CheckerConfig& c) {
        auto build = [](BuildPreference p) {
            return p == BuildPreference::FastBuild ? "fast_build" : p == BuildPreference::FastTrace ? "fast_trace" : "fast_trace_compact";
        };
        auto toggle = [](Toggle t) { return t == Toggle::Auto ? "auto" : t == Toggle::On ? "on" : "off"; };
        std::ostringstream s;
        s << "mode=" << modeName(c.mode) << " streams=" << c.numStreams << " batch=" << c.maxBatchSize
          << " max_items=" << c.maxItemsPerCheck << " obb=" << toggle(c.obbCulling)
          << " dispatch=" << (c.hostDispatch == HostDispatch::SingleThread ? "single_thread" : "thread_per_stream")
          << " optix(opt=" << int(c.optix.optLevel) << " debug=" << int(c.optix.debugLevel)
          << " maxreg=" << c.optix.maxRegisterCount << " exc=" << c.optix.exceptionFlags
          << " depth=" << c.optix.maxTraceDepth << ")";
        if (c.mode == Mode::Mesh) {
            s << " link_gas=" << build(c.mesh.linkGasBuild) << " compaction_min=" << c.mesh.compactionMinPoses;
        } else if (c.mode == Mode::Ray) {
            s << " width_cap=" << c.ray.launchWidthCap << " skip_hit=" << c.ray.skipHitItems
              << " hit_mode=" << int(c.ray.hitMode);
        } else {
            s << " traj=" << c.curve.trajectoryPoints << " ctrl=" << c.curve.controlPoints
              << " gas=" << build(c.curve.gasBuild) << " refit=" << int(c.curve.refit.kind);
            if (c.curve.refit.kind == RefitPolicy::Kind::Every) {
                s << "/" << c.curve.refit.every;
            }
            s << " geom_flags=" << (c.curve.geometryFlags ? std::to_string(*c.curve.geometryFlags) : std::string("auto"));
        }
        return s.str();
    }

} // namespace rtcd::support
