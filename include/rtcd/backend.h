#pragma once
// rtcd/backend.h -- the seam between the mode-agnostic detector core and the per-mode backends.
//
// One backend per Mode (Mesh, Ray, curves). The detector core (CollisionChecker implementation, Wave 2 e) builds
// ONE OptiX pipeline from the backend's ProgramSet, then for every check runs one loop over batches:
//
//     backend.beginCheck(info)                                  // once per check, host only
//     for batch b (on stream s = b % numStreams, in batch order per stream):
//         core: zero stream s's hit-count buffer                           (BatchView::hitCounts)
//         core: copy the batch's items into stream s's staging buffer      (BatchView::items)
//         plan = backend.prepare(streams[s], batch)                        // per batch
//         core: upload plan.params (skipped if byte-identical to stream s's last upload), optixLaunch(...)
//               unless plan.launch == false
//         core: copy hitCounts into the check result at batch.firstItem
//     core: synchronize the checker's streams
//
// There are no mode branches in that loop; everything mode-specific is inside prepare() and the programs.
// Every virtual call here is once per checker, once per check or once per batch, never per item.
// Current device: the core makes the Context's device current (Context::makeCurrent()) on every thread that
// calls beginCheck() or prepare() -- the checker's calling thread and, with HostDispatch::ThreadPerStream, each
// worker thread -- before the first such call. Backends therefore do not call makeCurrent() in beginCheck() /
// prepare(). (The core also makes it current before createBackend(); the backends make it current again in
// their constructors, so they can be constructed directly, e.g. by tests.)
//
// Headers: no OptiX / CUDA includes. Program descriptions use rtcd enums numerically equal to the OptiX ones
// (static_asserted in src/detector).

#include <rtcd/config.h>
#include <rtcd/context.h>
#include <rtcd/limits.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>
#include <rtcd/types.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rtcd {

    // ---------------------------------------------------------------------------------------------------------
    // Program declaration (once per checker): what the core needs to build modules, program groups, the
    // pipeline and the SBT without knowing the mode.
    // ---------------------------------------------------------------------------------------------------------

    // Numerically equal to OptixPrimitiveTypeFlags.
    enum PrimitiveTypeBits : std::uint32_t {
        kPrimitiveDefault          = 0,          // 0 = OptiX default (triangles + custom)
        kPrimitiveRoundQuadratic   = 1u << 1,    // OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE
        kPrimitiveRoundCubic       = 1u << 2,    // OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE
        kPrimitiveRoundLinear      = 1u << 3,    // OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR
        kPrimitiveSphere           = 1u << 6,    // OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE
        kPrimitiveTriangle         = 1u << 31,   // OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE
    };

    // Numerically equal to OptixTraversableGraphFlags.
    enum class TraversableGraph : std::uint32_t {
        Any                   = 0,
        SingleGas             = 1u << 0, // curve modes: the robot curve GAS is the root
        SingleLevelInstancing = 1u << 1, // Mesh (robot IAS -> link GAS) and Ray (scene IAS -> obstacle GAS)
    };

    // Built-in intersection program of the hit group (OptiX built-in IS module). For curves the core creates the
    // module with ProgramSet::builtinIsBuildFlags and curveEndcapFlags = 1 (OPTIX_CURVE_ENDCAP_ON).
    enum class BuiltinIntersection : std::uint8_t { None, Sphere, RoundLinear, RoundQuadratic, RoundCubic };

    struct ProgramSet {
        // OptiX-IR (or PTX) of the backend's program translation unit, embedded in the backend library. Must stay
        // valid for the backend's lifetime. moduleName is used in logs and error messages only.
        std::span<const unsigned char> moduleCode;
        std::string moduleName;

        // Raygen entry points ("__raygen__..."). A LaunchPlan selects one by index. The core creates one program
        // group and one SBT raygen record per entry (all in the one pipeline).
        std::vector<std::string> raygens;
        // Miss entry point; empty = null miss program (module nullptr, as the old stack did for all but
        // RayHitMode::FirstHitMiss). One miss record.
        std::string miss;
        // The single hit group. Empty name = no program of that kind.
        std::string anyHit;
        std::string closestHit;
        BuiltinIntersection builtinIntersection = BuiltinIntersection::None;
        // Build flags of the GAS the built-in IS module intersects (must be EXACTLY the flags the backend builds
        // that GAS with, or OptiX faults at trace time). Ignored when builtinIntersection == None.
        BuildFlags builtinIsBuildFlags = kBuildFlagNone;

        // Pipeline compile options.
        TraversableGraph traversableGraph = TraversableGraph::SingleLevelInstancing;
        std::uint32_t primitiveTypes      = kPrimitiveDefault; // PrimitiveTypeBits
        std::uint32_t numPayloadValues    = 0;                 // Ray: 1, others 0
        std::uint32_t numAttributeValues  = 2;
        // optixPipelineSetStackSize maxTraversableDepth: 2 with instancing, 1 for a single GAS.
        std::uint32_t maxTraversableDepth = 2;

        // Number of hit-group SBT records, all packed from the one hit group, data = 0. Mesh:
        // maxBatchSize * numLinks (instance sbtOffset = pose * numLinks + link, old layout); Ray: numObstacles
        // (scene instance o uses record o); curves: 1.
        std::uint32_t hitGroupRecordCount = 1;
    };

    // ---------------------------------------------------------------------------------------------------------
    // Per-check and per-batch data
    // ---------------------------------------------------------------------------------------------------------

    // One stream of the checker, as seen by a backend. `index` selects the backend's per-stream state
    // (0 .. numStreams - 1); the backend allocated that state at construction for every index.
    struct StreamContext {
        std::uint32_t index = 0;
        StreamHandle stream = nullptr;
    };

    struct CheckInfo {
        std::uint32_t itemCount  = 0; // items of this check, 1 .. maxItemsPerCheck
        std::uint32_t batchCount = 0; // ceil(itemCount / maxBatchSize)
    };

    struct BatchView {
        std::uint32_t batchIndex = 0; // b
        std::uint32_t firstItem  = 0; // b * maxBatchSize: index of the batch's first item in the check
        std::uint32_t itemCount  = 0; // 1 .. maxBatchSize (only the last batch may be partial)
        // The batch's items, packed (stride = ItemLayout::floatsPerItem), in this stream's staging buffer:
        // itemCount * floatsPerItem floats. The address is the same for every batch of a stream (stable).
        DeviceSpan<const float> items;
        // Per-item hit counters of this batch, uint32, itemCount entries, ZEROED by the core before prepare() on
        // the same stream. This is the ONLY result location: the backend's programs must add the hits of item i
        // (0 <= i < itemCount, batch-local) to hitCounts[i] with atomicAdd; item i collides iff hitCounts[i] > 0
        // after the launch. Stable address per stream; capacity maxBatchSize entries.
        DeviceSpan<std::uint32_t> hitCounts;
    };

    // What prepare() asks the core to launch.
    struct LaunchPlan {
        // false: nothing can hit (e.g. every instance OBB-culled on the compaction path); the core skips the
        // launch and the batch's hit counts stay 0.
        bool launch = true;
        // Root traversable of this launch (also inside params; the core uses it only for checks / diagnostics).
        TraversableHandle traversable = 0;
        // optixLaunch dimensions, each >= 1, width * height * depth <= kMaxLaunchSize.
        std::uint32_t width  = 0;
        std::uint32_t height = 1;
        std::uint32_t depth  = 1;
        // Launch parameters (e.g. EdgeRayParams / RobotRayParams from rtcd/limits.h): host memory owned by the
        // backend, paramsBytes <= kMaxLaunchParamsBytes, valid and unchanged until the next prepare() for the
        // same stream index. The core copies it into the stream's parameter buffer (cudaMemcpyAsync) only if
        // the bytes differ from that buffer's last upload.
        const void* params       = nullptr;
        std::uint32_t paramsBytes = 0;
        // Index into ProgramSet::raygens.
        std::uint32_t raygen = 0;
    };

    // ---------------------------------------------------------------------------------------------------------
    // Backend
    // ---------------------------------------------------------------------------------------------------------

    struct BackendCreateInfo {
        std::shared_ptr<Context> context;
        const CheckerConfig* config = nullptr;           // RESOLVED config; outlives the backend
        std::shared_ptr<const RobotModel> robot;
        std::shared_ptr<const Scene> scene;              // built on `context`
        std::span<const StreamContext> streams;          // config->numStreams entries; index i == streams[i].index
    };

    // Contract of every backend:
    //
    // Construction (createBackend): allocates ALL device memory the backend will ever use, for every stream index,
    //   sized by config->maxBatchSize and the robot / scene (scratch buffers included); builds static acceleration
    //   structures (link GASes); uploads static data (kinematics table, link OBB templates with
    //   collapseRootLinkObb applied, rays, sphere arrays, spline weights); computes programs(). May synchronize.
    //   Throws ConfigError / IoError / CapacityError / UnsupportedError for combinations that need the robot,
    //   scene or files (e.g. Ray: maxBatchSize * raysPerRobot must be < kMaxLaunchSize).
    //
    // programs(): constant after construction (the core reads it once). Build flags of every GAS a built-in IS
    //   module intersects are fixed at construction, so module flags and GAS flags can never diverge.
    //
    // beginCheck(info): host-only (no CUDA calls), called on the checker's calling thread before any prepare()
    //   of the check. Used for per-check policies (curve refit decision from batchCount).
    //
    // prepare(stream, batch):
    //   * enqueues GPU work ONLY on stream.stream (kinematics, OBB culling, AS build / refit, compaction) and
    //     returns the LaunchPlan; must not allocate or free device memory; must not call cudaDeviceSynchronize;
    //   * may block the host on stream.stream (cudaStreamSynchronize or a blocking D2H copy) ONLY if
    //     synchronizesHost(batch.itemCount) is true (Mesh compaction path, curve OBB path);
    //   * touches only the per-stream state of stream.index plus immutable shared state, so the core may call
    //     prepare() for DIFFERENT stream indices concurrently from different host threads
    //     (HostDispatch::ThreadPerStream); calls for the same index are always sequential and in batch order;
    //   * reads batch.items and writes batch.hitCounts only through work enqueued on stream.stream;
    //   * per-stream state (instance buffers, GAS, params) stays in use by the GPU until the stream passes the
    //     launch of that batch; reusing it in the next prepare() of the same index is safe by stream order;
    //   * throws rtcd::Error on failure; the core then drains all streams and rethrows on the calling thread.
    //
    // synchronizesHost(itemCount): exact, host-only predicate: true iff prepare() of a batch with that item
    //   count blocks the host. The core uses it for statistics and, with prepareCapturable(), to decide whether a
    //   batch can be captured into a CUDA graph.
    //
    // prepareCapturable(itemCount) (added 2026-11-24, CheckerConfig::graphPrepare): exact, host-only predicate:
    //   true iff prepare() for a batch of that item count
    //     (a) enqueues only plain CUDA work on its stream: kernels, memset / memcpy (no OptiX API call of any
    //         kind, no host synchronization, no device allocation), and
    //     (b) returns a LaunchPlan and params that depend only on (stream index, itemCount),
    //   so that replaying the captured work (stream capture -> CUDA graph) is equivalent to calling prepare() again
    //   on new contents of the same buffers. true implies !synchronizesHost(itemCount) (the core checks it).
    //   With CheckerConfig::graphPrepare the core then captures "zero hits, staging copy, prepare()" once per
    //   (stream, itemCount) and replays it with one cudaGraphLaunch per batch instead of calling prepare().
    //   HARD RULE: no OptiX call inside a stream capture. Measured on OptiX 9.1 / driver 610.88
    //   (tests/experiments/graph_capture): optixLaunch and optixAccelBuild inside a capture are silently dropped
    //   from the graph (only a 64-byte param memcpy, resp. nothing or part of the build, is recorded; the replay
    //   traces nothing; a partially captured quadratic-curve build crashes with an illegal address), or fail with
    //   OPTIX_ERROR_CUDA_ERROR (cuEventQuery not permitted while capturing), and in every case they leave the
    //   OptiX device context with an invalid internal event (later launches fail with OPTIX_ERROR_INVALID_VALUE,
    //   context destruction fails). A backend whose prepare() builds or refits an acceleration structure must
    //   return false. Default: false.
    class Backend {
    public:
        virtual ~Backend() = default;
        virtual Mode mode() const noexcept                                          = 0;
        virtual const ProgramSet& programs() const noexcept                        = 0;
        virtual void beginCheck(const CheckInfo& info)                              = 0;
        virtual LaunchPlan prepare(const StreamContext& stream, const BatchView& batch) = 0;
        virtual bool synchronizesHost(std::uint32_t itemCount) const noexcept      = 0;
        virtual bool prepareCapturable(std::uint32_t itemCount) const noexcept {
            (void) itemCount;
            return false;
        }
    };

    // Factories, one per backend library (Wave 2 f, g, h). The detector core calls the one matching
    // config->mode (the only mode switch of the core, at construction).
    std::unique_ptr<Backend> createMeshBackend(const BackendCreateInfo& info);  // Mode::Mesh            (f)
    std::unique_ptr<Backend> createRayBackend(const BackendCreateInfo& info);   // Mode::Ray             (g)
    std::unique_ptr<Backend> createCurveBackend(const BackendCreateInfo& info); // Mode::*Curve          (h)

} // namespace rtcd
