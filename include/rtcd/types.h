#pragma once
// rtcd/types.h -- small shared value types of the public API: handle aliases (so user code compiles without
// CUDA / OptiX headers), enums, item (pose / trajectory) views, logging.
//
// Plain C++20, no CUDA / OptiX includes. Implemented inline (nothing to link) except Logger::defaultSink().

#include <rtcd/limits.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

// Forward declarations of the opaque CUDA / OptiX handle structs. These are the exact tags the CUDA and OptiX
// headers use (cudaStream_t == CUstream_st*, CUcontext == CUctx_st*, OptixDeviceContext ==
// OptixDeviceContext_t*), so the aliases below convert to the real handle types without casts.
struct CUstream_st;
struct CUctx_st;
struct OptixDeviceContext_t;

namespace rtcd {

    // ---------------------------------------------------------------------------------------------------------
    // Handle aliases
    // ---------------------------------------------------------------------------------------------------------

    using StreamHandle        = CUstream_st*;            // == cudaStream_t / CUstream
    using CudaContextHandle   = CUctx_st*;               // == CUcontext
    using OptixContextHandle  = OptixDeviceContext_t*;   // == OptixDeviceContext
    using DevicePtr           = unsigned long long;      // == CUdeviceptr (64-bit), a device address
    using TraversableHandle   = unsigned long long;      // == OptixTraversableHandle

    // Typed, non-owning view of device memory: `count` elements of T starting at device address `ptr`.
    // Never dereferenced on the host.
    template <typename T>
    struct DeviceSpan {
        DevicePtr ptr     = 0;
        std::size_t count = 0;

        constexpr std::size_t bytes() const { return count * sizeof(T); }
        constexpr bool empty() const { return count == 0; }
        constexpr DevicePtr at(std::size_t i) const { return ptr + i * sizeof(T); } // address of element i
        constexpr DeviceSpan subspan(std::size_t offset, std::size_t n) const {
            return DeviceSpan{ptr + offset * sizeof(T), n};
        }
        constexpr operator DeviceSpan<const T>() const
            requires(!std::is_const_v<T>)
        {
            return DeviceSpan<const T>{ptr, count};
        }
    };

    // ---------------------------------------------------------------------------------------------------------
    // Enums (runtime configuration values; JSON spellings in brackets)
    // ---------------------------------------------------------------------------------------------------------

    // Collision-checking mode = backend. One checker has exactly one mode for its lifetime.
    enum class Mode : std::uint8_t {
        Mesh,            // ["mesh"]   ObsToRobot, discrete poses: scene edges traced against a per-batch robot IAS
                         //            (one instance per (pose, link), link mesh GAS). Old LinkType::MESH.
        Ray,             // ["ray"]    RobotToObs, discrete poses: robot link edges traced against the scene IAS.
                         //            Old LinkType::RAY_ONLINE.
        LinearCurve,     // ["linear_curve"]    continuous: swept spheres as round linear curves (degree 1).
        QuadraticCurve,  // ["quadratic_curve"] continuous: round quadratic B-splines (degree 2).
        CubicCurve,      // ["cubic_curve"]     continuous: round cubic B-splines (degree 3).
    };

    // true for the curve modes: an item is a trajectory of CurveOptions::trajectoryPoints poses.
    constexpr bool isContinuous(Mode m) { return m == Mode::LinearCurve || m == Mode::QuadraticCurve || m == Mode::CubicCurve; }
    // B-spline degree of a curve mode, 0 for the discrete modes.
    constexpr std::uint32_t curveDegree(Mode m) {
        return m == Mode::LinearCurve ? 1u : m == Mode::QuadraticCurve ? 2u : m == Mode::CubicCurve ? 3u : 0u;
    }

    // Acceleration-structure build preference (old BuildType). Maps to OptiX build flags:
    //   FastBuild        -> PREFER_FAST_BUILD                       (old FAST_BUILD)
    //   FastTrace        -> PREFER_FAST_TRACE                       (old FAST_TRACE)
    //   FastTraceCompact -> PREFER_FAST_TRACE | ALLOW_COMPACTION, and the AS is compacted after the build (old COMPACT)
    enum class BuildPreference : std::uint8_t { FastBuild, FastTrace, FastTraceCompact }; // ["fast_build", "fast_trace", "fast_trace_compact"]

    // OptiX build-flag bits, numerically equal to OptixBuildFlags (static_asserted in src/core).
    enum BuildFlagBits : std::uint32_t {
        kBuildFlagNone            = 0,
        kBuildFlagAllowUpdate     = 1u << 0,
        kBuildFlagAllowCompaction = 1u << 1,
        kBuildFlagPreferFastTrace = 1u << 2,
        kBuildFlagPreferFastBuild = 1u << 3,
    };
    using BuildFlags = std::uint32_t;

    // OptiX geometry-flag bits, numerically equal to OptixGeometryFlags (static_asserted in src/core).
    enum GeometryFlagBits : std::uint32_t {
        kGeometryFlagNone                    = 0,
        kGeometryFlagDisableAnyHit           = 1u << 0,
        kGeometryFlagRequireSingleAnyHitCall = 1u << 1,
    };
    using GeometryFlags = std::uint32_t;

    constexpr BuildFlags toBuildFlags(BuildPreference p) {
        return p == BuildPreference::FastBuild   ? BuildFlags{kBuildFlagPreferFastBuild}
             : p == BuildPreference::FastTrace   ? BuildFlags{kBuildFlagPreferFastTrace}
                                                 : BuildFlags{kBuildFlagPreferFastTrace | kBuildFlagAllowCompaction};
    }

    // End-point treatment of the B-spline fit of a trajectory (old curveEndPointType). Selects the spline weight
    // file directory and the control-point count per curve:
    //   Normal   -> controlPoints per curve                         ["normal"]
    //   Repeated -> controlPoints per curve                         ["repeated"]
    //   Phantom  -> controlPoints + 2 per curve (degree > 1)        ["phantom"]  (no weight files exist yet)
    enum class CurveEndPoints : std::uint8_t { Normal, Repeated, Phantom };

    // Which scene edges become rays in the edge-ray modes (Mesh, curves).
    enum class EdgeSet : std::uint8_t {
        Unique, // ["unique"] every unique triangle edge once (old uniqueEdgeIndices; old default for discrete modes)
        Loop,   // ["loop"]   consistently oriented edge loops of a watertight mesh (old loopEdgeIndices; old curve modes)
    };

    // How RAY-mode rays decide hit / no hit (old RTCD::RayHitMode, same numeric values). All four give the same
    // yes/no per ray; they differ only in cost.
    enum class RayHitMode : std::uint8_t {
        ClosestHit       = 0, // ["closest_hit"]    optixTrace, closest-hit program sets the payload
        FirstHitCH       = 1, // ["first_hit_ch"]   + TERMINATE_ON_FIRST_HIT
        FirstHitMiss     = 2, // ["first_hit_miss"] TERMINATE_ON_FIRST_HIT | DISABLE_CLOSESTHIT, miss clears payload
        FirstHitTraverse = 3, // ["traverse"]       optixTraverse + optixHitObjectIsHit(), no CH / miss program
    };

    // Three-valued switch whose Auto value is resolved per mode by CheckerConfig::resolved().
    enum class Toggle : std::uint8_t { Auto, Off, On }; // ["auto", false, true]

    // ---------------------------------------------------------------------------------------------------------
    // Item views (input of CollisionChecker::check)
    // ---------------------------------------------------------------------------------------------------------
    //
    // An ITEM is what gets one collision flag:
    //   discrete modes (Mesh, Ray): one pose = dof floats;
    //   curve modes:                one trajectory = trajectoryPoints poses = trajectoryPoints * dof floats.
    // Layout of the input (both host and device), all float32, joint values in radians:
    //   item i starts at float index i * stride (stride in floats; 0 means packed, i.e. stride = floatsPerItem);
    //   inside an item, waypoint p (0 <= p < pointsPerItem) starts at p * dof;
    //   inside a pose, joint j (0 <= j < dof) is at offset j, joints in chain order root -> tip
    //   (RobotModel::jointNames()).
    // So a packed batch of trajectories is row-major [item][waypoint][joint].
    struct ItemLayout {
        std::uint32_t dof           = 0; // RobotModel::dof()
        std::uint32_t pointsPerItem = 1; // 1 for discrete modes, CurveOptions::trajectoryPoints for curves
        constexpr std::size_t floatsPerItem() const { return std::size_t(dof) * pointsPerItem; }
    };

    // Host-resident items. `floats` must hold at least (count - 1) * stride + floatsPerItem floats
    // (checked: InvalidArgument). The memory is only read during the call.
    struct HostItems {
        std::span<const float> floats;
        std::size_t count  = 0;
        std::size_t stride = 0; // floats between consecutive items; 0 = packed; otherwise >= floatsPerItem
    };

    // Device-resident items (memory on the checker's device, owned by the caller, at least 4-byte aligned).
    // Same layout rules as HostItems; the size cannot be checked, the caller guarantees it.
    struct DeviceItems {
        DevicePtr floats  = 0;
        std::size_t count = 0;
        std::size_t stride = 0; // floats; 0 = packed
    };

    // ---------------------------------------------------------------------------------------------------------
    // Logging
    // ---------------------------------------------------------------------------------------------------------

    enum class LogLevel : std::uint8_t { Error, Warning, Info, Debug }; // ["error", "warning", "info", "debug"]

    // Library diagnostics go through a Logger, never to std::cout directly. Messages with a level above `level`
    // are dropped. `sink` may be called from any thread the library uses (stream worker threads included); it
    // must be thread-safe. An empty sink means defaultSink() (stderr, one line per message, serialized).
    struct Logger {
        LogLevel level = LogLevel::Warning;
        std::function<void(LogLevel, std::string_view)> sink;

        bool enabled(LogLevel l) const { return l <= level; }
        void log(LogLevel l, std::string_view message) const {
            if (!enabled(l)) {
                return;
            }
            if (sink) {
                sink(l, message);
            } else {
                defaultSink(l, message);
            }
        }
        static void defaultSink(LogLevel l, std::string_view message); // implemented in src/core
    };

} // namespace rtcd
