// rtcd_backend_mesh: createMeshBackend (rtcd/backend.h), Mode::Mesh = ObsToRobot.
//
// Robot links are instanced triangle GASes in one IAS per stream; the scene's edge rays are traced against it and
// the any-hit counts hits per item through an instance -> item map. Reproduces the old stack exactly, including
// its Level-2 optimizations at their defaults (docs/design/interfaces.md, section 5 "Mesh (f)"):
//
//   old (RTCD/...)                                         here
//   Robot/batchIASRobot.h  optixMeshLink / buildGAS        construction: one triangle GAS per link
//                          (Meshes/mesh.h buildGAS)        (internal::buildTriangleGas, REQUIRE_SINGLE_ANYHIT_CALL)
//   streamUniqueRobot::buildAS (template instances,        construction, per stream: instance template, initial
//                          buildIAS<FAST_TRACE> x1.5)      IAS build over all instances, 1.5x buffers
//   batchIASRobot ctor mapIndex                            itemMapTemplate (instance -> pose, shared)
//   robot::loadURDF SKIP_BASE                              CheckerConfig::collapseRootLinkObb
//   detectWithOBB (non-RAY): fkine (batchFkineAdjOBB),     prepare(), OBB on: linkPosesAndObbs, obbOverlapCounts,
//     detectOBB (BBCDRobotMask), updateWithMask (V3)       then masked or compaction path (below)
//   detect: update = fkineOBB + updateIAS (upInsPoses with prepare(), OBB off: linkPosesAndObbs, parked
//     safeTransform, IAS UPDATE over all instances)        transforms, IAS UPDATE over all instances
//   getMapIndex (compacted ? indexMap : template)          EdgeRayParams::primToItem
//   CCCuda.cu __raygen__oneside / __anyhit__IAS            mesh_programs.cu __raygen__edgeRays / __anyhit__instanceItem
//
// updateWithMask, OBB on (old V3):
//   * itemCount >= mesh.compactionMinPoses ("compaction path"): instance transforms written with visibility 255,
//     the OBB-surviving instances and their item indices stream-compacted (internal::compactByFlags, replacing the
//     old two cub::DeviceSelect::FlaggedIf calls), the survivor count read back and the stream synchronized (the
//     ONLY host sync of this backend: synchronizesHost() is true exactly here); no survivor -> launch = false;
//     else IAS BUILD over the survivors, primToItem = compacted map.
//   * itemCount < mesh.compactionMinPoses ("masked path", no host round trip): transforms written with
//     visibility mask 0 for culled instances, IAS BUILD over all itemCount * numLinks instances, primToItem =
//     template map. Always launches.
#include <rtcd/backend.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/accel.h>
#include <rtcd/internal/algorithms.h>
#include <rtcd/internal/kinematics.h>

#include "backend_mesh/mesh_kernels.h"

#include <rtcd_backend_mesh_ir_embedded.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace rtcd {

    namespace {
        static_assert(sizeof(OptixInstance) == 80, "OptixInstance layout");
        static_assert(sizeof(EdgeRayParams) <= kMaxLaunchParamsBytes);

        constexpr const char* kRaygen = "__raygen__edgeRays";
        constexpr const char* kAnyHit = "__anyhit__instanceItem";

        // Per-batch robot IAS build options (old streamUniqueRobot buildOptions / updateOptions).
        constexpr unsigned int kIasBuildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        // Old buildIAS(..., bufferScale = 1.5): slack on the IAS output and temp buffers.
        constexpr double kIasBufferScale = 1.5;

        std::size_t scaled(std::size_t bytes) {
            return static_cast<std::size_t>(std::ceil(static_cast<double>(bytes) * kIasBufferScale));
        }

        // All per-stream state (old streamUniqueRobot + the detector's robot-mask buffer). Only prepare() of this
        // stream index touches it.
        struct StreamState {
            StreamHandle stream = nullptr;
            DeviceBuffer linkPoses;        // maxBatchSize * numLinks * 12 floats (pose-major)
            DeviceBuffer linkObbs;         // maxBatchSize * numLinks Obb
            DeviceBuffer obbCounts;        // maxBatchSize * numLinks uint32 (old robot mask)
            DeviceBuffer instances;        // maxBatchSize * numLinks OptixInstance, template order
            DeviceBuffer compactInstances; // compaction path: surviving instances
            DeviceBuffer compactMap;       // compaction path: surviving instance -> item
            DeviceBuffer compactCount;     // compaction path: device uint32
            DeviceBuffer compactScratch;   // compactByFlags scratch
            PinnedHostBuffer hostCount;    // compaction path: survivor count read back
            DeviceBuffer iasTemp;
            DeviceBuffer iasOutput;
            TraversableHandle ias = 0;
            EdgeRayParams params{};        // host copy handed to the core (valid until the next prepare)
        };

        class MeshBackend final : public Backend {
        public:
            explicit MeshBackend(const BackendCreateInfo& info);
            ~MeshBackend() override;

            Mode mode() const noexcept override { return Mode::Mesh; }
            const ProgramSet& programs() const noexcept override { return programs_; }
            void beginCheck(const CheckInfo&) override {}
            LaunchPlan prepare(const StreamContext& stream, const BatchView& batch) override;
            bool synchronizesHost(std::uint32_t itemCount) const noexcept override {
                return obbCulling_ && itemCount >= compactionMinPoses_;
            }
            // Split (G3): every path without a host sync, i.e. the masked path (OBB culling on, itemCount <
            // compactionMinPoses) and the OBB-off path. prepareCompute = FK + link OBBs + instance transforms /
            // masks (kernels only); prepareFinish = IAS build / refit + params + plan.
            bool prepareSplit(std::uint32_t itemCount) const noexcept override {
                return itemCount >= 1 && itemCount <= maxBatchSize_ && !synchronizesHost(itemCount);
            }
            void prepareCompute(const StreamContext& stream, const BatchView& batch) override;
            LaunchPlan prepareFinish(const StreamContext& stream, const BatchView& batch) override;

        private:
            StreamState& checkedState(const StreamContext& stream, const BatchView& batch, const char* fn);
            // The kernels of a split batch (no host-side state is touched: replays do not run this host code).
            void computeKernels(StreamState& s, const BatchView& batch);
            // The OptiX part of a split batch and the params / plan, from the arguments alone.
            LaunchPlan finishSplit(StreamState& s, const BatchView& batch);
            LaunchPlan finalPlan(StreamState& s, const BatchView& batch, DevicePtr primToItem);
            void buildIas(StreamState& s, DevicePtr instances, std::uint32_t count, OptixBuildOperation op);
            void drain() noexcept;

            std::shared_ptr<Context> context_;
            CheckerConfig config_; // resolved copy (the pointer in BackendCreateInfo outlives us anyway)
            std::shared_ptr<const RobotModel> robot_;
            std::shared_ptr<const Scene> scene_;

            std::uint32_t numLinks_      = 0;
            std::uint32_t maxBatchSize_  = 0;
            std::uint32_t maxInstances_  = 0; // maxBatchSize * numLinks
            bool obbCulling_             = false;
            std::uint32_t compactionMinPoses_ = 0;
            mesh_detail::ParkTransform park_{};
            const EdgeRays* edges_ = nullptr;

            DeviceBuffer kinematics_;
            internal::KinematicsRef kinRef_{};
            DeviceBuffer obbTemplates_;               // numLinks Obb (root collapsed if configured)
            std::vector<internal::Accel> linkGas_;    // one per link
            DeviceBuffer itemMapTemplate_;            // instance -> item, maxInstances uint32 (shared, immutable)
            std::vector<StreamState> streams_;

            ProgramSet programs_;
        };

        MeshBackend::MeshBackend(const BackendCreateInfo& info) {
            if (!info.context) {
                throw InvalidArgument("createMeshBackend: null context");
            }
            if (info.config == nullptr) {
                throw InvalidArgument("createMeshBackend: null config");
            }
            if (!info.robot) {
                throw InvalidArgument("createMeshBackend: null robot");
            }
            if (!info.scene) {
                throw InvalidArgument("createMeshBackend: null scene");
            }
            if (info.config->mode != Mode::Mesh) {
                throw InvalidArgument("createMeshBackend: config mode is not Mode::Mesh");
            }
            if (info.scene->context() != info.context) {
                throw InvalidArgument("createMeshBackend: the scene was built on a different Context");
            }
            context_ = info.context;
            config_  = *info.config;
            robot_   = info.robot;
            scene_   = info.scene;

            if (!config_.sceneEdges.has_value() || config_.obbCulling == Toggle::Auto) {
                throw InvalidArgument("createMeshBackend: config is not resolved (CheckerConfig::resolved())");
            }
            if (info.streams.size() != config_.numStreams || info.streams.empty()) {
                throw InvalidArgument("createMeshBackend: " + std::to_string(info.streams.size()) + " streams given, "
                                      + std::to_string(config_.numStreams) + " configured");
            }
            for (std::size_t i = 0; i < info.streams.size(); ++i) {
                if (info.streams[i].index != i) {
                    throw InvalidArgument("createMeshBackend: streams[" + std::to_string(i) + "].index = "
                                          + std::to_string(info.streams[i].index) + " (must equal its position)");
                }
            }

            const RobotModel& robot = *robot_;
            if (robot.dof() < 1 || robot.dof() > kMaxDof || robot.numLinks() != robot.dof() + 1
                || robot.links().size() != robot.numLinks()) {
                throw InvalidArgument("createMeshBackend: inconsistent robot model (dof " + std::to_string(robot.dof())
                                      + ", " + std::to_string(robot.links().size()) + " links)");
            }
            numLinks_           = robot.numLinks();
            maxBatchSize_       = config_.maxBatchSize;
            obbCulling_         = config_.obbCullingEnabled();
            compactionMinPoses_ = config_.mesh.compactionMinPoses;
            std::memcpy(park_.m, config_.mesh.parkTransform.m.data(), sizeof(park_.m));

            // Instance count, SBT record count and the device-side int compaction count are 32-bit; OptiX limits
            // the instances of one IAS further (checked against the device property below).
            const std::uint64_t maxInstances = std::uint64_t(maxBatchSize_) * numLinks_;
            unsigned int optixMaxInstances   = 0;
            {
                context_->makeCurrent();
                RTCD_OPTIX_CHECK(optixDeviceContextGetProperty(context_->optixContext(),
                    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS, &optixMaxInstances, sizeof(optixMaxInstances)));
            }
            constexpr std::uint64_t kInt32Max = 0x7FFFFFFFull;
            if (maxInstances > kInt32Max || maxInstances > optixMaxInstances) {
                throw CapacityError("createMeshBackend: max_batch_size * numLinks = " + std::to_string(maxInstances)
                                        + " robot link instances exceed the per-IAS limit",
                    "max_batch_size * numLinks", maxInstances,
                    (std::min)(std::uint64_t(optixMaxInstances), kInt32Max));
            }
            maxInstances_ = static_cast<std::uint32_t>(maxInstances);

            edges_ = &scene_->edgeRays(*config_.sceneEdges); // throws UnsupportedError for Loop on a non-watertight scene
            if (edges_->count == 0) {
                throw InvalidArgument("createMeshBackend: the scene has no edge rays of the selected edge set");
            }
            if (edges_->count > kMaxLaunchSize) {
                throw CapacityError("createMeshBackend: " + std::to_string(edges_->count)
                                        + " scene edge rays exceed the OptiX launch size limit",
                    "scene edge rays", edges_->count, kMaxLaunchSize);
            }

            context_->makeCurrent();

            // ---- static data: kinematics table, link OBB templates (old robot::uploadOBBs with SKIP_BASE) ----
            kinematics_ = internal::uploadKinematics(robot.kinematics());
            kinRef_     = internal::KinematicsRef{&robot.kinematics(), kinematics_.ptr()};
            {
                std::vector<Obb> obbs;
                obbs.reserve(numLinks_);
                for (const LinkGeometry& l : robot.links()) {
                    obbs.push_back(l.obb);
                }
                if (config_.collapseRootLinkObb) {
                    obbs[0].halfSize[0] = obbs[0].halfSize[1] = obbs[0].halfSize[2] = 0.0f;
                }
                obbTemplates_ = DeviceBuffer::fromHost(std::span<const Obb>(obbs));
            }

            // ---- link GASes (old optixMeshLink -> meshModel::buildGAS<BUILD>, geometry REQUIRE_SINGLE_ANYHIT_CALL) ----
            const BuildFlags linkFlags = toBuildFlags(config_.mesh.linkGasBuild);
            linkGas_.reserve(numLinks_);
            for (std::uint32_t l = 0; l < numLinks_; ++l) {
                const TriangleMesh& m = robot.links()[l].mesh;
                if (m.triangles.empty() || m.vertices.empty()) {
                    throw UnsupportedError("createMeshBackend: link " + std::to_string(l) + " ('"
                                           + robot.links()[l].name + "') has no triangle mesh");
                }
                static_assert(sizeof(Vec3) == 3 * sizeof(float));
                static_assert(sizeof(m.triangles[0]) == 3 * sizeof(std::uint32_t));
                const DeviceBuffer vertices = DeviceBuffer::fromHost(std::span<const Vec3>(m.vertices));
                const DeviceBuffer indices =
                    DeviceBuffer::fromHost(std::span<const std::array<std::uint32_t, 3>>(m.triangles));
                internal::TriangleGasInput input;
                input.vertices      = DeviceSpan<const float>{vertices.ptr(), m.vertices.size() * 3};
                input.indices       = DeviceSpan<const std::uint32_t>{indices.ptr(), m.triangles.size() * 3};
                input.geometryFlags = kGeometryFlagRequireSingleAnyHitCall;
                linkGas_.push_back(internal::buildTriangleGas(*context_, input, linkFlags, info.streams[0].stream));
            }

            // ---- instance -> item template map (old batchIASRobot mapIndex) and the instance template ----
            std::vector<std::uint32_t> map(maxInstances_);
            std::vector<OptixInstance> instances(maxInstances_);
            const Transform identity;
            for (std::uint32_t p = 0; p < maxBatchSize_; ++p) {
                for (std::uint32_t l = 0; l < numLinks_; ++l) {
                    const std::uint32_t i = p * numLinks_ + l;
                    map[i]                = p;
                    OptixInstance& inst   = instances[i];
                    std::memset(&inst, 0, sizeof(inst));
                    inst.visibilityMask    = 255;
                    inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
                    inst.traversableHandle = linkGas_[l].handle;
                    inst.sbtOffset         = i;
                    std::memcpy(inst.transform, identity.m.data(), sizeof(inst.transform));
                }
            }
            itemMapTemplate_ = DeviceBuffer::fromHost(std::span<const std::uint32_t>(map));

            // ---- IAS sizes for the full instance count (old buildIAS: x1.5 output, temp >= max(build, update)) ----
            OptixAccelBufferSizes sizes{};
            {
                OptixBuildInput in{};
                in.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                in.instanceArray.instances    = 0;
                in.instanceArray.numInstances = maxInstances_;
                OptixAccelBuildOptions opt{};
                opt.buildFlags            = kIasBuildFlags;
                opt.motionOptions.numKeys = 1;
                opt.operation             = OPTIX_BUILD_OPERATION_BUILD;
                RTCD_OPTIX_CHECK(optixAccelComputeMemoryUsage(context_->optixContext(), &opt, &in, 1, &sizes));
            }
            const std::size_t tempBytes =
                scaled((std::max)(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes));
            const std::size_t outputBytes  = scaled(sizes.outputSizeInBytes);
            const std::size_t scratchBytes = internal::compactScratchBytes(maxInstances_, sizeof(OptixInstance));

            // ---- per-stream state ----
            streams_.resize(info.streams.size());
            for (std::size_t si = 0; si < info.streams.size(); ++si) {
                StreamState& s = streams_[si];
                s.stream       = info.streams[si].stream;
                s.linkPoses.allocate(std::size_t(maxInstances_) * kTransformFloats * sizeof(float));
                s.linkObbs.allocate(std::size_t(maxInstances_) * sizeof(Obb));
                s.obbCounts.allocate(std::size_t(maxInstances_) * sizeof(std::uint32_t));
                s.instances        = DeviceBuffer::fromHost(std::span<const OptixInstance>(instances));
                s.compactInstances = DeviceBuffer::fromHost(std::span<const OptixInstance>(instances));
                s.compactMap.allocate(std::size_t(maxInstances_) * sizeof(std::uint32_t));
                s.compactCount.allocate(sizeof(std::uint32_t));
                s.compactScratch.allocate(scratchBytes);
                s.hostCount.allocate(sizeof(std::uint32_t));
                s.iasTemp.allocate(tempBytes);
                s.iasOutput.allocate(outputBytes);
                // Initial build over all instances (identity transforms), old streamUniqueRobot::buildAS. The OBB-off
                // path refits this IAS every batch.
                buildIas(s, s.instances.ptr(), maxInstances_, OPTIX_BUILD_OPERATION_BUILD);
                s.params.edgeVertices = edges_->vertices.ptr;
                s.params.edgeCount    = edges_->count;
                s.params.primToItem   = itemMapTemplate_.ptr();
            }
            for (const StreamState& s : streams_) {
                RTCD_CUDA_CHECK(cudaStreamSynchronize(s.stream));
            }

            // ---- programs (old createModule / createRaygenPrograms / createHitgroupPrograms / buildSBT, MESH) ----
            programs_.moduleCode = std::span<const unsigned char>(
                embedded::rtcd_backend_mesh_ir_mesh_programs, embedded::rtcd_backend_mesh_ir_mesh_programs_size);
            programs_.moduleName          = "rtcd_backend_mesh/mesh_programs";
            programs_.raygens             = {kRaygen};
            programs_.miss                = {};
            programs_.anyHit              = kAnyHit;
            programs_.closestHit          = {};
            programs_.builtinIntersection = BuiltinIntersection::None;
            programs_.builtinIsBuildFlags = kBuildFlagNone;
            programs_.traversableGraph    = TraversableGraph::SingleLevelInstancing;
            programs_.primitiveTypes      = kPrimitiveDefault;
            programs_.numPayloadValues    = 0;
            programs_.numAttributeValues  = 2;
            programs_.maxTraversableDepth = 2;
            programs_.hitGroupRecordCount = maxInstances_;
        }

        MeshBackend::~MeshBackend() { drain(); }

        void MeshBackend::drain() noexcept {
            for (const StreamState& s : streams_) {
                if (s.stream != nullptr) {
                    cudaStreamSynchronize(s.stream); // buffers below are freed after the GPU is done with them
                }
            }
        }

        void MeshBackend::buildIas(StreamState& s, DevicePtr instances, std::uint32_t count, OptixBuildOperation op) {
            OptixBuildInput in{};
            in.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            in.instanceArray.instances    = instances;
            in.instanceArray.numInstances = count;
            OptixAccelBuildOptions opt{};
            opt.buildFlags            = kIasBuildFlags;
            opt.motionOptions.numKeys = 1;
            opt.operation             = op;
            OptixTraversableHandle handle = s.ias;
            RTCD_OPTIX_CHECK(optixAccelBuild(context_->optixContext(), s.stream, &opt, &in, 1, s.iasTemp.ptr(),
                s.iasTemp.size(), s.iasOutput.ptr(), s.iasOutput.size(), &handle, nullptr, 0));
            s.ias = handle;
        }

        StreamState& MeshBackend::checkedState(
            const StreamContext& stream, const BatchView& batch, const char* fn) {
            // ---- caller errors first (nothing enqueued yet) ----
            if (stream.index >= streams_.size()) {
                throw InvalidArgument(std::string("MeshBackend::") + fn + ": stream index " + std::to_string(stream.index)
                                      + " >= " + std::to_string(streams_.size()));
            }
            StreamState& s = streams_[stream.index];
            if (stream.stream != s.stream) {
                throw InvalidArgument(std::string("MeshBackend::") + fn + ": stream " + std::to_string(stream.index)
                                      + " is not the stream given at construction");
            }
            const std::uint32_t n = batch.itemCount;
            if (n == 0 || n > maxBatchSize_) {
                throw CapacityError(std::string("MeshBackend::") + fn + ": batch of " + std::to_string(n)
                                        + " items (1 .. max_batch_size " + std::to_string(maxBatchSize_) + ")",
                    "max_batch_size", n, maxBatchSize_);
            }
            const std::size_t dof = robot_->dof();
            if (batch.items.count < std::size_t(n) * dof || batch.hitCounts.count < n) {
                throw InvalidArgument(std::string("MeshBackend::") + fn + ": batch views too small for "
                                      + std::to_string(n) + " items");
            }
            return s;
        }

        void MeshBackend::computeKernels(StreamState& s, const BatchView& batch) {
            const std::uint32_t n    = batch.itemCount;
            const std::uint32_t nIns = n * numLinks_;
            const DeviceSpan<float> linkPoses{s.linkPoses.ptr(), std::size_t(nIns) * kTransformFloats};
            const DeviceSpan<Obb> linkObbs{s.linkObbs.ptr(), nIns};
            // Fused FK + link OBBs (old fkineOBB = batchFkineAdjOBB; the old OBB-off update() used it too).
            internal::linkPosesAndObbs(kinRef_, batch.items, n, obbTemplates_.span<const Obb>(), linkPoses, linkObbs, s.stream);
            if (!obbCulling_) {
                // Old detect(): update = fkineOBB + updateIAS: transforms of the batch, park transform for the unused
                // slots of a partial batch (the IAS refit in finishSplit covers all instances). Masks stay 255.
                mesh_detail::writeInstanceTransformsParked(s.instances.ptr(), s.linkPoses.ptr(), nIns, maxInstances_, park_, s.stream);
            } else {
                // Masked path (old detectOBB + updateWithMask V3, fused): culled instances get visibility mask 0. The
                // overlap decision is internal::obbOverlapCounts' (shared SAT code), without the count buffer.
                mesh_detail::writeInstanceTransformsAndOverlapMasks(s.instances.ptr(), s.linkPoses.ptr(),
                    DeviceSpan<const Obb>{linkObbs.ptr, nIns}, scene_->obbs(), s.stream);
            }
        }

        LaunchPlan MeshBackend::finishSplit(StreamState& s, const BatchView& batch) {
            if (!obbCulling_) {
                buildIas(s, s.instances.ptr(), maxInstances_, OPTIX_BUILD_OPERATION_UPDATE);
            } else {
                buildIas(s, s.instances.ptr(), batch.itemCount * numLinks_, OPTIX_BUILD_OPERATION_BUILD);
            }
            return finalPlan(s, batch, itemMapTemplate_.ptr());
        }

        LaunchPlan MeshBackend::finalPlan(StreamState& s, const BatchView& batch, DevicePtr primToItem) {
            // Every params field is set here, from the arguments and the stream's IAS (prepareCompute sets none).
            s.params.hitCounts    = batch.hitCounts.ptr;
            s.params.primToItem   = primToItem;
            s.params.traversable  = s.ias;
            s.params.edgeVertices = edges_->vertices.ptr;
            s.params.edgeCount    = edges_->count;
            LaunchPlan plan;
            plan.launch      = true;
            plan.traversable = s.ias;
            plan.width       = edges_->count;
            plan.height      = 1;
            plan.depth       = 1;
            plan.params      = &s.params;
            plan.paramsBytes = sizeof(EdgeRayParams);
            plan.raygen      = 0;
            return plan;
        }

        void MeshBackend::prepareCompute(const StreamContext& stream, const BatchView& batch) {
            StreamState& s = checkedState(stream, batch, "prepareCompute");
            if (!prepareSplit(batch.itemCount)) {
                throw InvalidArgument("MeshBackend::prepareCompute: batch of " + std::to_string(batch.itemCount)
                                      + " items takes the compaction path (not split)");
            }
            computeKernels(s, batch);
        }

        LaunchPlan MeshBackend::prepareFinish(const StreamContext& stream, const BatchView& batch) {
            StreamState& s = checkedState(stream, batch, "prepareFinish");
            if (!prepareSplit(batch.itemCount)) {
                throw InvalidArgument("MeshBackend::prepareFinish: batch of " + std::to_string(batch.itemCount)
                                      + " items takes the compaction path (not split)");
            }
            return finishSplit(s, batch);
        }

        LaunchPlan MeshBackend::prepare(const StreamContext& stream, const BatchView& batch) {
            StreamState& s        = checkedState(stream, batch, "prepare");
            const std::uint32_t n = batch.itemCount;
            if (prepareSplit(n)) {
                computeKernels(s, batch);
                return finishSplit(s, batch);
            }

            // Compaction path (OBB culling on, n >= compactionMinPoses): one host sync.
            const std::uint32_t nIns = n * numLinks_;
            const DeviceSpan<float> linkPoses{s.linkPoses.ptr(), std::size_t(nIns) * kTransformFloats};
            const DeviceSpan<Obb> linkObbs{s.linkObbs.ptr(), nIns};
            internal::linkPosesAndObbs(kinRef_, batch.items, n, obbTemplates_.span<const Obb>(), linkPoses, linkObbs, s.stream);
            // Old detectOBB (BBCDRobotMask): per link-instance overlap count with the obstacle OBBs.
            const DeviceSpan<std::uint32_t> counts{s.obbCounts.ptr(), nIns};
            internal::obbOverlapCounts(linkObbs, scene_->obbs(), counts, s.stream);
            mesh_detail::writeInstanceTransformsAndMasks(s.instances.ptr(), s.linkPoses.ptr(), 0, nIns, s.stream);
            const std::array<internal::CompactArray, 2> arrays{
                internal::CompactArray{s.compactInstances.ptr(), s.instances.ptr(),
                    static_cast<std::uint32_t>(sizeof(OptixInstance))},
                internal::CompactArray{s.compactMap.ptr(), itemMapTemplate_.ptr(),
                    static_cast<std::uint32_t>(sizeof(std::uint32_t))},
            };
            internal::compactByFlags(arrays, DeviceSpan<const std::uint32_t>(counts), s.compactCount.ptr(),
                s.compactScratch.span<std::byte>(), s.stream);
            s.compactCount.downloadAsync(s.hostCount.data(), sizeof(std::uint32_t), s.stream);
            RTCD_CUDA_CHECK(cudaStreamSynchronize(s.stream));
            const std::uint32_t survivors = s.hostCount.as<std::uint32_t>()[0];
            if (survivors == 0) {
                LaunchPlan plan  = finalPlan(s, batch, itemMapTemplate_.ptr());
                plan.launch      = false; // nothing can hit; hit counts stay 0
                return plan;
            }
            buildIas(s, s.compactInstances.ptr(), survivors, OPTIX_BUILD_OPERATION_BUILD);
            return finalPlan(s, batch, s.compactMap.ptr());
        }
    } // namespace

    std::unique_ptr<Backend> createMeshBackend(const BackendCreateInfo& info) {
        return std::make_unique<MeshBackend>(info);
    }

} // namespace rtcd
