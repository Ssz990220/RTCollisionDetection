// rtcd_backend_ray: createRayBackend (rtcd/backend.h), Mode::Ray = old LinkType::RAY_ONLINE (RobotToObs).
//
// Old-stack reference (frozen, the specification of this file):
//   RTCD/Robot/batchRayRobot.h            streamRayRobot / batchRayRobot: ray templates (every link's unique edges
//                                         in link order), ray -> link map, per-stream link transforms and link
//                                         OBBs, updateWithMask storing only the mask, launch width cap, skip-hit;
//   RTCD/CollisionDetector/CollisionDetector.h
//                                         detectWithOBB RAY branch (fkine = batchFkineAdj + batchOBBTransform,
//                                         detectOBB, launch over all rays with a capped width, never a host sync),
//                                         createRaygenPrograms / createMissPrograms / createHitgroupPrograms
//                                         (program choice by rayHitMode), buildSBT (numObstacles hit records);
//   RTCD/CollisionDetector/CCRay.cu       the programs (ported to ray_programs.cu).
//
// Construction (all device memory, sized by maxBatchSize):
//   shared: robot rays (6 floats per ray, robot-local, link order), ray -> link map, link OBB template (root link
//           collapsed if collapseRootLinkObb), kinematics table, and with OBB culling off an all-ones link mask;
//   per stream: link poses [maxBatchSize * numLinks * 12], link OBBs and link mask [maxBatchSize * numLinks] (OBB
//           culling on only), the host RobotRayParams handed out by prepare().
// prepare(): linkPoses (old batchFkineAdj); with OBB culling transformObbs (old batchOBBTransform) +
//   obbOverlapCounts against the scene OBBs into the link mask (old detectOBB: BBCD robot mask; only "> 0" is read,
//   and the overlap count is > 0 exactly when BBCD's atomically accumulated robot mask is); params; plan with
//   width = min(itemCount * raysPerRobot, launchWidthCap) (cap 0 = no cap). Never synchronizes the host.
#include <rtcd/backend.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/algorithms.h>
#include <rtcd/internal/kinematics.h>
#include <rtcd/limits.h>

#include <rtcd_backend_ray_ir_embedded.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rtcd {

    namespace {

        // Raygen entry of each RayHitMode (old createRaygenPrograms).
        const char* raygenName(RayHitMode mode) {
            switch (mode) {
            case RayHitMode::FirstHitCH: return "__raygen__robotRayFirstHitCH";
            case RayHitMode::FirstHitMiss: return "__raygen__robotRayFirstHitMiss";
            case RayHitMode::FirstHitTraverse: return "__raygen__robotRayFirstHitTraverse";
            case RayHitMode::ClosestHit: break;
            }
            return "__raygen__robotRayCompact";
        }

        class RayBackend final : public Backend {
        public:
            explicit RayBackend(const BackendCreateInfo& info);

            Mode mode() const noexcept override { return Mode::Ray; }
            const ProgramSet& programs() const noexcept override { return programs_; }
            void beginCheck(const CheckInfo&) override {} // no per-check policy in RAY mode
            LaunchPlan prepare(const StreamContext& stream, const BatchView& batch) override;
            bool synchronizesHost(std::uint32_t) const noexcept override { return false; }

        private:
            struct PerStream {
                DeviceBuffer linkPoses; // float [maxBatchSize * numLinks * 12], pose-major
                DeviceBuffer linkObbs;  // Obb [maxBatchSize * numLinks]       (OBB culling only)
                DeviceBuffer linkMask;  // uint32 [maxBatchSize * numLinks]    (OBB culling only)
                RobotRayParams params{};
            };

            std::shared_ptr<Context> ctx_;
            std::shared_ptr<const RobotModel> robot_;
            std::shared_ptr<const Scene> scene_;
            std::uint32_t maxBatchSize_  = 0;
            std::uint32_t dof_           = 0;
            std::uint32_t numLinks_      = 0;
            std::uint32_t raysPerRobot_  = 0;
            std::uint32_t launchWidthCap_ = 0;
            bool skipHitItems_           = true;
            bool obbCulling_             = true;

            KinematicsTable table_{};      // host copy (KinematicsRef::host)
            DeviceBuffer kinematics_;      // device copy (KinematicsRef::device)
            DeviceBuffer rays_;            // float [raysPerRobot * 6]
            DeviceBuffer rayLink_;         // int32 [raysPerRobot]
            DeviceBuffer obbTemplate_;     // Obb [numLinks]
            DeviceBuffer constantMask_;    // uint32 [maxBatchSize * numLinks], all ones (OBB culling off only)
            std::vector<PerStream> streams_;
            ProgramSet programs_;
        };

        RayBackend::RayBackend(const BackendCreateInfo& info) {
            if (!info.context || info.config == nullptr || !info.robot || !info.scene) {
                throw InvalidArgument("createRayBackend: context, config, robot and scene are required");
            }
            const CheckerConfig& cfg = *info.config;
            if (cfg.mode != Mode::Ray) {
                throw InvalidArgument("createRayBackend: config.mode is not Mode::Ray");
            }
            if (cfg.obbCulling == Toggle::Auto) {
                throw InvalidArgument("createRayBackend: the config must be resolved (obb_culling is still auto)");
            }
            if (cfg.maxBatchSize == 0) {
                throw InvalidArgument("createRayBackend: max_batch_size must be >= 1");
            }
            if (info.streams.size() != cfg.numStreams || info.streams.empty()) {
                throw InvalidArgument("createRayBackend: " + std::to_string(info.streams.size())
                                      + " streams given, config.streams = " + std::to_string(cfg.numStreams));
            }
            for (std::size_t i = 0; i < info.streams.size(); ++i) {
                if (info.streams[i].index != i) {
                    throw InvalidArgument("createRayBackend: streams[" + std::to_string(i) + "].index = "
                                          + std::to_string(info.streams[i].index) + " (must equal its position)");
                }
            }
            if (info.scene->context() != info.context) {
                throw InvalidArgument("createRayBackend: the scene was built on a different Context");
            }

            ctx_            = info.context;
            robot_          = info.robot;
            scene_          = info.scene;
            maxBatchSize_   = cfg.maxBatchSize;
            dof_            = robot_->dof();
            numLinks_       = robot_->numLinks();
            launchWidthCap_ = cfg.ray.launchWidthCap;
            skipHitItems_   = cfg.ray.skipHitItems;
            obbCulling_     = cfg.obbCullingEnabled();
            table_          = robot_->kinematics();

            if (dof_ < 1 || dof_ > kMaxDof || numLinks_ != dof_ + 1) {
                throw UnsupportedError("createRayBackend: robot '" + robot_->name() + "' has dof "
                                       + std::to_string(dof_) + " (supported: 1 .. " + std::to_string(kMaxDof) + ")");
            }
            const auto links = robot_->links();
            if (links.size() != numLinks_) {
                throw InvalidArgument("createRayBackend: robot '" + robot_->name() + "' has "
                                      + std::to_string(links.size()) + " link geometries, numLinks = "
                                      + std::to_string(numLinks_));
            }

            // Robot rays: every link's unique edges, links in chain order (old batchRayRobot::prepareRobot).
            std::vector<float> rays;
            std::vector<std::int32_t> rayLink;
            for (std::uint32_t l = 0; l < numLinks_; ++l) {
                const TriangleMesh& mesh = links[l].mesh;
                for (const Edge& e : mesh.uniqueEdges) {
                    const Vec3& a = mesh.vertices.at(e[0]);
                    const Vec3& b = mesh.vertices.at(e[1]);
                    rays.insert(rays.end(), {a[0], a[1], a[2], b[0], b[1], b[2]});
                    rayLink.push_back(static_cast<std::int32_t>(l));
                }
            }
            raysPerRobot_ = static_cast<std::uint32_t>(rayLink.size());
            if (raysPerRobot_ == 0) {
                throw UnsupportedError("createRayBackend: robot '" + robot_->name() + "' has no mesh edges (no rays)");
            }
            // Launch ids are uint32 and an OptiX launch is limited to 2^30 (old assert in the batchRayRobot ctor).
            const std::uint64_t maxRays = std::uint64_t(maxBatchSize_) * raysPerRobot_;
            if (maxRays >= kMaxLaunchSize) {
                throw CapacityError("createRayBackend: max_batch_size (" + std::to_string(maxBatchSize_)
                                        + ") * rays per robot (" + std::to_string(raysPerRobot_) + ") = "
                                        + std::to_string(maxRays) + " must be < 2^30 (OptiX launch size limit)",
                    "max_batch_size * rays_per_robot", static_cast<std::size_t>(maxRays),
                    static_cast<std::size_t>(kMaxLaunchSize - 1));
            }

            // Link OBB template (old robot<DOF>::loadURDF + SKIP_BASE).
            std::vector<Obb> obbs;
            obbs.reserve(numLinks_);
            for (const LinkGeometry& link : links) {
                obbs.push_back(link.obb);
            }
            if (cfg.collapseRootLinkObb) {
                obbs[0].halfSize[0] = obbs[0].halfSize[1] = obbs[0].halfSize[2] = 0.0f;
            }

            ctx_->makeCurrent();
            kinematics_  = internal::uploadKinematics(table_);
            rays_        = DeviceBuffer::fromHost(std::span<const float>(rays));
            rayLink_     = DeviceBuffer::fromHost(std::span<const std::int32_t>(rayLink));
            obbTemplate_ = DeviceBuffer::fromHost(std::span<const Obb>(obbs));

            const std::size_t linkSlots = std::size_t(maxBatchSize_) * numLinks_;
            if (!obbCulling_) {
                // Q10: OBB culling off traces every ray (the old stack crashed here).
                const std::vector<std::uint32_t> ones(linkSlots, 1u);
                constantMask_ = DeviceBuffer::fromHost(std::span<const std::uint32_t>(ones));
            }

            streams_.resize(info.streams.size());
            for (PerStream& s : streams_) {
                s.linkPoses.allocate(linkSlots * kTransformFloats * sizeof(float));
                if (obbCulling_) {
                    s.linkObbs.allocate(linkSlots * sizeof(Obb));
                    s.linkMask.allocate(linkSlots * sizeof(std::uint32_t));
                }
                // Constant per stream; only hitCounts and totalRayCount change between batches.
                RobotRayParams& p = s.params;
                p.traversable     = static_cast<unsigned long long>(scene_->traversable());
                p.rays            = rays_.ptr();
                p.rayLink         = rayLink_.ptr();
                p.linkMask        = obbCulling_ ? s.linkMask.ptr() : constantMask_.ptr();
                p.linkPoses       = s.linkPoses.ptr();
                p.hitCounts       = 0;
                p.totalRayCount   = 0;
                p.raysPerRobot    = raysPerRobot_;
                p.linksPerRobot   = numLinks_;
                p.skipHitItems    = skipHitItems_ ? 1u : 0u;
            }

            // Programs (old createModule / createRaygenPrograms / createMissPrograms / createHitgroupPrograms /
            // createPipeline / buildSBT for RAY_ONLINE).
            programs_.moduleCode = std::span<const unsigned char>(
                embedded::rtcd_backend_ray_ir_ray_programs, embedded::rtcd_backend_ray_ir_ray_programs_size);
            programs_.moduleName = "rtcd_backend_ray/ray_programs.cu";
            programs_.raygens    = {raygenName(cfg.ray.hitMode)};
            programs_.miss       = cfg.ray.hitMode == RayHitMode::FirstHitMiss ? "__miss__radiance" : "";
            programs_.anyHit     = "";
            programs_.closestHit =
                (cfg.ray.hitMode == RayHitMode::ClosestHit || cfg.ray.hitMode == RayHitMode::FirstHitCH)
                    ? "__closesthit__radiance"
                    : "";
            programs_.builtinIntersection = BuiltinIntersection::None;
            programs_.builtinIsBuildFlags = kBuildFlagNone;
            programs_.traversableGraph    = TraversableGraph::SingleLevelInstancing;
            programs_.primitiveTypes      = kPrimitiveDefault;
            programs_.numPayloadValues    = 1;
            programs_.numAttributeValues  = 2;
            programs_.maxTraversableDepth = 2;
            programs_.hitGroupRecordCount = scene_->numObstacles(); // scene instance o -> record o
        }

        LaunchPlan RayBackend::prepare(const StreamContext& stream, const BatchView& batch) {
            if (stream.index >= streams_.size()) {
                throw InvalidArgument("RayBackend::prepare: stream index " + std::to_string(stream.index)
                                      + " >= " + std::to_string(streams_.size()) + " streams");
            }
            const std::uint32_t n = batch.itemCount;
            if (n == 0) {
                throw InvalidArgument("RayBackend::prepare: batch.itemCount must be >= 1");
            }
            if (n > maxBatchSize_) {
                throw CapacityError("RayBackend::prepare: batch of " + std::to_string(n) + " items > max_batch_size = "
                                        + std::to_string(maxBatchSize_),
                    "max_batch_size", n, maxBatchSize_);
            }
            if (batch.items.count < std::size_t(n) * dof_ || batch.items.ptr == 0) {
                throw InvalidArgument("RayBackend::prepare: batch.items holds " + std::to_string(batch.items.count)
                                      + " floats, " + std::to_string(std::size_t(n) * dof_) + " needed");
            }
            if (batch.hitCounts.count < n || batch.hitCounts.ptr == 0) {
                throw InvalidArgument("RayBackend::prepare: batch.hitCounts holds "
                                      + std::to_string(batch.hitCounts.count) + " entries, " + std::to_string(n)
                                      + " needed");
            }

            PerStream& s                  = streams_[stream.index];
            const std::size_t linkSlots   = std::size_t(n) * numLinks_;
            const DeviceSpan<float> poses = s.linkPoses.span<float>();
            const internal::KinematicsRef kin{&table_, kinematics_.ptr()};

            // Old detectWithOBB RAY branch: rayRobot->fkine (batchFkineAdj + batchOBBTransform), detectOBB.
            internal::linkPoses(kin, batch.items, n, poses, stream.stream);
            if (obbCulling_) {
                const DeviceSpan<Obb> obbs = s.linkObbs.span<Obb>().subspan(0, linkSlots);
                internal::transformObbs(poses, numLinks_, n, obbTemplate_.span<const Obb>(), obbs, stream.stream);
                internal::obbOverlapCounts(obbs, scene_->obbs(), s.linkMask.span<std::uint32_t>().subspan(0, linkSlots),
                    stream.stream);
            }

            // Old updateWithMask + getRayInfo: every ray of the batch, launch width capped, no host round trip.
            const std::uint32_t total = n * raysPerRobot_; // < 2^30 (checked at construction)
            s.params.hitCounts        = batch.hitCounts.ptr;
            s.params.totalRayCount    = total;

            LaunchPlan plan;
            plan.launch      = true;
            plan.traversable = scene_->traversable();
            plan.width       = (launchWidthCap_ != 0 && total > launchWidthCap_) ? launchWidthCap_ : total;
            plan.height      = 1;
            plan.depth       = 1;
            plan.params      = &s.params;
            plan.paramsBytes = sizeof(RobotRayParams);
            plan.raygen      = 0;
            return plan;
        }

    } // namespace

    std::unique_ptr<Backend> createRayBackend(const BackendCreateInfo& info) {
        return std::make_unique<RayBackend>(info);
    }

} // namespace rtcd
