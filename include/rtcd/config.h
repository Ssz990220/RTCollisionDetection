#pragma once
// rtcd/config.h -- CheckerConfig: every option of a CollisionChecker, as plain data (no JSON dependency; the JSON
// module rtcd/json_config.h maps files onto these structs).
//
// Three kinds of parameters (agreed design):
//   [runtime]   runtime configuration, fixed at construction of the checker (changing it = new checker);
//   [capacity]  a maximum given at construction; buffers are allocated once for it; per-call values must be
//               <= the capacity (checked, CapacityError);
//   [fixed]     sizes fixed at construction that every call must use exactly (trajectory / control point counts:
//               they determine curve topology and the spline weight table).
// True compile-time constants are only in rtcd/limits.h.
//
// Every field below states its default, its kind and its JSON key (snake_case, nested objects as shown).
//
// validate() / resolved():
//   * validate() checks the configuration on its own (no files, no robot, no GPU) and throws ConfigError whose
//     what() lists EVERY violated rule, one per line, each prefixed with the field's dotted JSON path; field()
//     is the first offending path.
//   * resolved() = validate() + a copy with every Auto / "per-mode default" value replaced by its effective
//     value (Toggle::Auto, EdgeSet via sceneEdges, autoRefitMaxBatches < 0, ...). The checker stores and reports
//     the resolved copy (CollisionChecker::config(), dumpCheckerConfig()).
//   * Checks that need the robot, the scene or files (e.g. RAY launch size, spline weight file present) run in
//     the CollisionChecker constructor and throw ConfigError / IoError / CapacityError there.
// Implemented in src/config (Wave 1 b), no JSON dependency.

#include <rtcd/geometry.h>
#include <rtcd/types.h>

#include <cstdint>
#include <filesystem>
#include <optional>

namespace rtcd {

    // OptiX compile / link options (old RTCD::OptixCompileConfig minus the context fields, which moved to
    // ContextOptions, and rayHitMode, which moved to RayOptions). Enum values are independent of OptiX; src/core
    // maps them. JSON object "optix".
    struct OptixOptions {
        enum class OptLevel : std::uint8_t { Default, O0, O1, O2, O3 };              // "opt_level": "default"|0..3
        enum class DebugLevel : std::uint8_t { Default, None, Minimal, Moderate, Full }; // "debug_level"
        enum ExceptionBits : std::uint32_t {                                            // "exceptions": list of names
            kExceptionNone          = 0,      // "none"
            kExceptionStackOverflow = 1u << 0, // "stack_overflow"
            kExceptionTraceDepth    = 1u << 1, // "trace_depth"
            kExceptionUser          = 1u << 2, // "user"
        };

        OptLevel optLevel           = OptLevel::Default;   // [runtime] "opt_level"
        DebugLevel debugLevel       = DebugLevel::Default; // [runtime] "debug_level" (OptiX DEFAULT = minimal)
        std::uint32_t maxRegisterCount = 50;               // [runtime] "max_register_count"; 0 = OptiX default
        std::uint32_t exceptionFlags   = kExceptionNone;   // [runtime] "exceptions"
        std::uint32_t maxTraceDepth    = 2;                // [runtime] "max_trace_depth" (>= 1); stack sizes are
                                                           //           always computed for trace depth 1
        bool printPipelineLog          = false;            // [runtime] "print_pipeline_log" (to the Logger, Info)

        static OptixOptions release() { return {}; }
        static OptixOptions debug() {
            OptixOptions o;
            o.optLevel       = OptLevel::O0;
            o.debugLevel     = DebugLevel::Full;
            o.exceptionFlags = kExceptionTraceDepth | kExceptionStackOverflow;
            return o;
        }
        static OptixOptions profile() { // optimized code with moderate debug info (Nsight Compute)
            OptixOptions o;
            o.debugLevel = DebugLevel::Moderate;
            return o;
        }
        // Defaults for the build type of the INCLUDING translation unit (old OptixCompileConfig::buildDefault).
        static OptixOptions buildDefault() {
#if defined(_DEBUG) || defined(DEBUG)
            return debug();
#else
            return release();
#endif
        }
    };

    // Mesh backend (ObsToRobot, old batchIASRobot + CollisionDetector<MESH>). JSON object "mesh".
    struct MeshOptions {
        // [runtime] "link_gas_build": build preference of the per-link triangle GAS (built once at construction).
        // Old regression / benchmarks: COMPACT.
        BuildPreference linkGasBuild = BuildPreference::FastTraceCompact;
        // [runtime] "compaction_min_poses": with OBB culling, batches with itemCount >= this stream-compact the
        // surviving instances (one device->host count read + stream sync per batch, smaller IAS build); smaller
        // batches build the IAS over all instances with culled ones at visibility mask 0 (no host round trip).
        // 0 = always compact; > max_batch_size = never compact. Default 128 = measured crossover (BM_IAS, RTX 5080).
        std::uint32_t compactionMinPoses = 128;
        // [runtime] "park_transform": transform given to instances that must not collide (unused pose slots of a
        // partial batch on the refit path). Default = old safeTransform: scale 0.001, 0.1 m below the origin.
        Transform parkTransform{{0.001f, 0, 0, 0, 0, 0.001f, 0, 0, 0, 0, 0.001f, -0.1f}};
    };

    // Ray backend (RobotToObs, old batchRayRobot + CollisionDetector<RAY_ONLINE>). JSON object "ray".
    struct RayOptions {
        // [runtime] "launch_width_cap": optixLaunch width = min(itemCount * raysPerRobot, cap); the raygen
        // grid-strides over all ray ids. 0 = no cap. Default 262144 = best or near-best BM_RAY at every N.
        std::uint32_t launchWidthCap = 262144;
        // [runtime] "skip_hit_items": skip rays of items already known to collide. Flags unchanged, hit counts
        // become lower bounds. Default true.
        bool skipHitItems = true;
        // [runtime] "hit_mode". Default ClosestHit.
        RayHitMode hitMode = RayHitMode::ClosestHit;
    };

    // Refit policy of the curve GAS (old batchOptixCurveRobot::setRefitPolicy). Collision flags never depend on
    // it; only build vs. trace cost does. A refit (OPTIX_BUILD_OPERATION_UPDATE) is only ever done when the batch
    // has the same item count as the last full build on that stream; otherwise the GAS is rebuilt.
    struct RefitPolicy {
        enum class Kind : std::uint8_t {
            Auto,   // "auto": refit while the check has <= CurveOptions::autoRefitMaxBatches batches, else rebuild
            Always, // "always": refit whenever possible (old rebuildEvery = 0)
            Never,  // "never": full rebuild every batch (old rebuildEvery = 1)
            Every,  // {"every": K}: full rebuild every K-th build of a stream (K >= 2), refits in between
        };
        Kind kind           = Kind::Auto;
        std::uint32_t every = 0; // only for Kind::Every
    };

    // Curve backends (old batchOptixCurveRobot). JSON object "curve".
    struct CurveOptions {
        // [fixed] "trajectory_points": poses per trajectory item (>= 2, <= kMaxTrajectoryPoints). Default 16.
        std::uint32_t trajectoryPoints = 16;
        // [fixed] "control_points": B-spline control points fitted per sphere trajectory (quadratic / cubic only;
        // ignored for linear curves, whose vertices are the trajectory points). Must satisfy
        // degree < control_points < trajectory_points and control_points * trajectory_points <= kMaxSplineWeights.
        // Default 8.
        std::uint32_t controlPoints = 8;
        CurveEndPoints endPoints = CurveEndPoints::Normal; // [runtime] "end_points"
        // [runtime] "gas_build": build preference of the per-batch curve GAS. Default FastBuild (old regression).
        // ALLOW_UPDATE is added automatically when refits are possible (OBB culling off).
        BuildPreference gasBuild = BuildPreference::FastBuild;
        RefitPolicy refit;                                  // [runtime] "refit"
        // [runtime] "auto_refit_max_batches": threshold of RefitPolicy::Auto. < 0 = per-degree default
        // (resolved): linear 2, quadratic 1, cubic 1 (measured BM_CURVE_Shelf, batch 16).
        std::int32_t autoRefitMaxBatches = -1;
        // [runtime] "geometry_flags": curve build-input geometry flag: kGeometryFlagNone (default) or
        // kGeometryFlagRequireSingleAnyHitCall (disables curve segment splitting; changes cost and hit counts,
        // never flags). JSON: "none" | "single_anyhit".
        GeometryFlags geometryFlags = kGeometryFlagNone;
        // [runtime] "spline_weights_dir": directory holding <degree name>/<end points>/<control>_<trajectory>.txt
        // (degree name "quad" / "cubic", end points "normal" / "repeated" / "phantom"). Required for quadratic and
        // cubic curves; the JSON loader resolves it relative to the config file. Empty = not set.
        std::filesystem::path splineWeightsDir;
    };

    // How check() drives the streams from the host.
    enum class HostDispatch : std::uint8_t {
        SingleThread,    // "single_thread": the calling thread issues every batch in order (old check() without
                         //                  USE_MULTI_THREADING)
        ThreadPerStream, // "thread_per_stream": one persistent worker thread per stream issues that stream's
                         //                  batches in order (old USE_MULTI_THREADING, used by the regression and
                         //                  the benchmarks). Lets a host sync on one stream (IAS compaction, curve
                         //                  OBB) overlap work issued on the others.
    };

    struct CheckerConfig {
        Mode mode = Mode::Mesh;                                // [runtime] "mode"
        std::uint32_t numStreams      = 4;                     // [runtime] "streams" (>= 1); batch b runs on stream
                                                               //           b % numStreams
        HostDispatch hostDispatch     = HostDispatch::ThreadPerStream; // [runtime] "host_dispatch"
        std::uint32_t maxBatchSize    = 1024;                  // [capacity] "max_batch_size": items per batch (old
                                                               //           BATCHSIZE); every batch but the last of a
                                                               //           check is full
        std::uint32_t maxItemsPerCheck = 8192;                 // [capacity] "max_items_per_check": items of one
                                                               //           check() call (input staging + results)
        Toggle obbCulling = Toggle::Auto;                      // [runtime] "obb_culling": link / segment OBB vs.
                                                               //           obstacle OBB pre-test. Auto: On for Mesh
                                                               //           and Ray, Off for curves (old configs)
        // [runtime] "collapse_root_link_obb": shrink link 0's OBB to its center point before OBB culling
        // (reproduces the old SKIP_BASE define of RTCD/Robot/robot.h, which the goldens were recorded with).
        bool collapseRootLinkObb = true;
        // [runtime] "scene_edges": edge set used as rays by Mesh and curve modes. nullopt ("auto") = Unique for
        // Mesh, Loop for curves (old regression / benchmarks). Ignored by Ray.
        std::optional<EdgeSet> sceneEdges;
        LogLevel verbosity = LogLevel::Warning;                // [runtime] "verbosity": level of this checker's
                                                               //           messages (sink = the Context's Logger)
        OptixOptions optix = OptixOptions::buildDefault();     // "optix"
        MeshOptions mesh;                                      // "mesh"   (read only when mode == Mesh)
        RayOptions ray;                                        // "ray"    (read only when mode == Ray)
        CurveOptions curve;                                    // "curve"  (read only for curve modes)

        // See the header comment. Implemented in src/config.
        void validate() const;
        CheckerConfig resolved() const;

        // Convenience accessors valid on a resolved() config.
        bool obbCullingEnabled() const { return obbCulling == Toggle::On; }
        ItemLayout itemLayout(std::uint32_t dof) const {
            return ItemLayout{dof, isContinuous(mode) ? curve.trajectoryPoints : 1u};
        }
    };

} // namespace rtcd
