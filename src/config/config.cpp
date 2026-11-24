// rtcd_config: CheckerConfig::validate() and CheckerConfig::resolved() (include/rtcd/config.h).
//
// validate() collects every violated rule before throwing one ConfigError whose what() has one line per problem,
// "<dotted.json.path>: <message>", and whose field() is the first offending path. The JSON module prefixes each
// line with the source file. Rules follow docs/design/interfaces.md section 3.7; in addition every enum field must
// hold one of its declared values (only reachable from C++, the JSON loader cannot produce other values).
#include <rtcd/config.h>
#include <rtcd/errors.h>
#include <rtcd/limits.h>

#include <string>
#include <vector>

namespace rtcd {

    namespace {

        struct Problems {
            std::vector<std::pair<std::string, std::string>> items; // (field, message)

            void add(std::string field, std::string message) { items.emplace_back(std::move(field), std::move(message)); }

            void throwIfAny() const {
                if (items.empty()) {
                    return;
                }
                std::string text;
                for (const auto& [field, message] : items) {
                    if (!text.empty()) {
                        text += '\n';
                    }
                    text += field + ": " + message;
                }
                throw ConfigError(text, items.front().first);
            }
        };

        std::string str(std::uint64_t v) { return std::to_string(v); }

        const char* modeName(Mode m) {
            switch (m) {
            case Mode::Mesh: return "mesh";
            case Mode::Ray: return "ray";
            case Mode::LinearCurve: return "linear_curve";
            case Mode::QuadraticCurve: return "quadratic_curve";
            case Mode::CubicCurve: return "cubic_curve";
            }
            return "?";
        }

        bool validMode(Mode m) { return static_cast<unsigned>(m) <= static_cast<unsigned>(Mode::CubicCurve); }
        bool validBuild(BuildPreference b) {
            return static_cast<unsigned>(b) <= static_cast<unsigned>(BuildPreference::FastTraceCompact);
        }

        // Effective control points per curve (Phantom adds one phantom point at each end for degree > 1).
        std::uint32_t controlPointsPerCurve(const CheckerConfig& c) {
            return c.curve.controlPoints + (c.curve.endPoints == CurveEndPoints::Phantom ? 2u : 0u);
        }

        void checkEnums(const CheckerConfig& c, Problems& p) {
            if (!validMode(c.mode)) {
                p.add("mode", "invalid value " + str(static_cast<unsigned>(c.mode)));
            }
            if (static_cast<unsigned>(c.hostDispatch) > static_cast<unsigned>(HostDispatch::ThreadPerStream)) {
                p.add("host_dispatch", "invalid value " + str(static_cast<unsigned>(c.hostDispatch)));
            }
            if (static_cast<unsigned>(c.obbCulling) > static_cast<unsigned>(Toggle::On)) {
                p.add("obb_culling", "invalid value " + str(static_cast<unsigned>(c.obbCulling)));
            }
            if (c.sceneEdges && static_cast<unsigned>(*c.sceneEdges) > static_cast<unsigned>(EdgeSet::Loop)) {
                p.add("scene_edges", "invalid value " + str(static_cast<unsigned>(*c.sceneEdges)));
            }
            if (static_cast<unsigned>(c.verbosity) > static_cast<unsigned>(LogLevel::Debug)) {
                p.add("verbosity", "invalid value " + str(static_cast<unsigned>(c.verbosity)));
            }
            if (static_cast<unsigned>(c.optix.optLevel) > static_cast<unsigned>(OptixOptions::OptLevel::O3)) {
                p.add("optix.opt_level", "invalid value " + str(static_cast<unsigned>(c.optix.optLevel)));
            }
            if (static_cast<unsigned>(c.optix.debugLevel) > static_cast<unsigned>(OptixOptions::DebugLevel::Full)) {
                p.add("optix.debug_level", "invalid value " + str(static_cast<unsigned>(c.optix.debugLevel)));
            }
            constexpr std::uint32_t knownExceptions = OptixOptions::kExceptionStackOverflow
                                                    | OptixOptions::kExceptionTraceDepth | OptixOptions::kExceptionUser;
            if ((c.optix.exceptionFlags & ~knownExceptions) != 0) {
                p.add("optix.exceptions", "unknown exception bits " + str(c.optix.exceptionFlags & ~knownExceptions));
            }
            if (!validBuild(c.mesh.linkGasBuild)) {
                p.add("mesh.link_gas_build", "invalid value " + str(static_cast<unsigned>(c.mesh.linkGasBuild)));
            }
            if (static_cast<unsigned>(c.ray.hitMode) > static_cast<unsigned>(RayHitMode::FirstHitTraverse)) {
                p.add("ray.hit_mode", "invalid value " + str(static_cast<unsigned>(c.ray.hitMode)));
            }
            if (static_cast<unsigned>(c.curve.endPoints) > static_cast<unsigned>(CurveEndPoints::Phantom)) {
                p.add("curve.end_points", "invalid value " + str(static_cast<unsigned>(c.curve.endPoints)));
            }
            if (!validBuild(c.curve.gasBuild)) {
                p.add("curve.gas_build", "invalid value " + str(static_cast<unsigned>(c.curve.gasBuild)));
            }
            if (static_cast<unsigned>(c.curve.refit.kind) > static_cast<unsigned>(RefitPolicy::Kind::Every)) {
                p.add("curve.refit", "invalid kind " + str(static_cast<unsigned>(c.curve.refit.kind)));
            }
        }

        void checkCurve(const CheckerConfig& c, Problems& p) {
            const std::uint32_t degree = curveDegree(c.mode);
            const CurveOptions& cv     = c.curve;
            if (cv.trajectoryPoints < 2 || cv.trajectoryPoints > kMaxTrajectoryPoints) {
                p.add("curve.trajectory_points", str(cv.trajectoryPoints) + " must be in [2, " + str(kMaxTrajectoryPoints)
                                                     + "] (kMaxTrajectoryPoints)");
            }
            if (degree >= 2) {
                if (cv.controlPoints <= degree || cv.controlPoints >= cv.trajectoryPoints) {
                    p.add("curve.control_points", str(cv.controlPoints) + " must be > " + str(degree)
                                                      + " (the curve degree of " + modeName(c.mode)
                                                      + ") and < curve.trajectory_points = " + str(cv.trajectoryPoints));
                }
                const std::uint64_t weights = std::uint64_t(controlPointsPerCurve(c)) * cv.trajectoryPoints;
                if (weights > kMaxSplineWeights) {
                    p.add("curve.control_points",
                        str(controlPointsPerCurve(c)) + " control points per curve"
                            + (cv.endPoints == CurveEndPoints::Phantom ? " (control_points + 2 phantom points)" : "")
                            + " * curve.trajectory_points = " + str(cv.trajectoryPoints) + " = " + str(weights)
                            + " spline weights exceed kMaxSplineWeights = " + str(kMaxSplineWeights));
                }
                if (cv.splineWeightsDir.empty()) {
                    p.add("curve.spline_weights_dir", std::string("must be set for ") + modeName(c.mode));
                }
            }
            if (cv.refit.kind == RefitPolicy::Kind::Every && cv.refit.every < 2) {
                p.add("curve.refit", "{\"every\": " + str(cv.refit.every)
                                         + "} must have every >= 2 (use \"never\" to rebuild every batch)");
            }
            if (c.obbCulling == Toggle::On && c.mode == Mode::CubicCurve) {
                p.add("obb_culling", "true is not supported with cubic_curve (as in the old stack)");
            }
            if (cv.geometryFlags && *cv.geometryFlags != kGeometryFlagNone
                && *cv.geometryFlags != kGeometryFlagRequireSingleAnyHitCall) {
                p.add("curve.geometry_flags", "value " + str(*cv.geometryFlags)
                                                  + " is not allowed (only \"auto\", \"none\", \"single_anyhit\")");
            }
        }

    } // namespace

    void CheckerConfig::validate() const {
        Problems p;
        checkEnums(*this, p);
        if (numStreams < 1) {
            p.add("streams", "0 must be >= 1");
        }
        if (maxBatchSize < 1) {
            p.add("max_batch_size", "0 must be >= 1");
        }
        if (maxItemsPerCheck < 1) {
            p.add("max_items_per_check", "0 must be >= 1");
        }
        if (optix.maxTraceDepth < 1) {
            p.add("optix.max_trace_depth", "0 must be >= 1");
        }
        // mesh.compaction_min_poses: any value (0 = always compact, > max_batch_size = never).
        // scene_edges: ignored by Ray, any value accepted.
        if (validMode(mode) && isContinuous(mode)) {
            checkCurve(*this, p);
        }
        p.throwIfAny();
    }

    CheckerConfig CheckerConfig::resolved() const {
        validate();
        CheckerConfig r = *this;
        const bool continuous = isContinuous(mode);

        if (r.obbCulling == Toggle::Auto) {
            r.obbCulling = continuous ? Toggle::Off : Toggle::On;
        }
        if (!r.sceneEdges) {
            // Unique for Mesh (and Ray, which ignores it), Loop for the curve modes.
            r.sceneEdges = continuous ? EdgeSet::Loop : EdgeSet::Unique;
        }
        if (r.curve.autoRefitMaxBatches < 0) {
            r.curve.autoRefitMaxBatches = mode == Mode::LinearCurve ? 2 : 1;
        }
        if (!r.curve.geometryFlags) {
            r.curve.geometryFlags = mode == Mode::QuadraticCurve ? GeometryFlags{kGeometryFlagRequireSingleAnyHitCall}
                                                                 : GeometryFlags{kGeometryFlagNone};
        }
        // OBB culling compacts the curve topology every batch: the GAS is never refit (old curveUseRefit = !useOBB),
        // so the effective policy is Never (and the backend does not set ALLOW_UPDATE).
        if (continuous && r.obbCulling == Toggle::On) {
            r.curve.refit = RefitPolicy{RefitPolicy::Kind::Never, 0};
        }
        return r;
    }

} // namespace rtcd
