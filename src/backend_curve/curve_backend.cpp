// rtcd_backend_curve: createCurveBackend (include/rtcd/backend.h; design doc section 5 "Curves").
//
// Swept robot spheres as OptiX round curves (linear, quadratic B-spline, cubic B-spline), one curve per
// (trajectory item, robot sphere), radius = sphere radius, all curves of a batch in ONE curve GAS per stream; the
// scene's edge rays (loop edges by default) are traced against it and the any-hit program counts hits per item
// through the segment -> item map. Port of RTCD/Robot/batchCurveRobot.h (streamCurveRobot / batchOptixCurveRobot)
// and the curve paths of RTCD/CollisionDetector/CollisionDetector.h (detect, detectWithOBB / detectOBB).
//
// Level-2 optimizations kept (old defaults):
//   * fused point generation (FK + sphere centers + control points in one kernel) whenever the configuration fits
//     the fused kernel; otherwise trajectoryLinkPoses + sphere centers (+ control points), same values;
//   * curve GAS built with ALLOW_UPDATE when OBB culling is off, refit (OPTIX_BUILD_OPERATION_UPDATE) instead of a
//     rebuild per the refit policy (beginCheck decides from the batch count; a refit only when the batch has the
//     item count of the stream's last full build);
//   * geometry flags from the resolved config (auto: single any-hit for quadratic), and the built-in IS module
//     flags equal to the GAS build flags (ProgramSet::builtinIsBuildFlags).
//
// Host synchronization: prepare() blocks the host (one 4-byte D2H read of the surviving segment count + stream
// sync) exactly when OBB culling is on; synchronizesHost() returns that. Without OBB culling prepare() only
// enqueues work.
#include <rtcd/backend.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/algorithms.h>
#include <rtcd/internal/kinematics.h>

#include "curve_kernels.h"

#include <rtcd_backend_curve_ir_embedded.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace rtcd {

    namespace {

        namespace fs = std::filesystem;

        static_assert(kBuildFlagAllowUpdate == OPTIX_BUILD_FLAG_ALLOW_UPDATE, "build flag values");
        static_assert(kBuildFlagPreferFastBuild == OPTIX_BUILD_FLAG_PREFER_FAST_BUILD, "build flag values");
        static_assert(kGeometryFlagRequireSingleAnyHitCall == OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
            "geometry flag values");
        static_assert(sizeof(DevicePtr) == sizeof(CUdeviceptr), "device pointer size");

        const char* endPointsName(CurveEndPoints e) {
            switch (e) {
            case CurveEndPoints::Repeated: return "repeated";
            case CurveEndPoints::Phantom: return "phantom";
            default: return "normal";
            }
        }

        std::string str(std::uint64_t v) { return std::to_string(v); }

        std::uint32_t checkedU32(std::uint64_t v, const char* what, std::uint64_t capacityItems) {
            if (v > std::numeric_limits<std::uint32_t>::max()) {
                throw CapacityError(std::string("curve backend: ") + what + " = " + str(v) + " exceeds 2^32 - 1",
                    "max_batch_size", capacityItems, capacityItems);
            }
            return static_cast<std::uint32_t>(v);
        }

        // Spline weight file (one float per line, row-major [controlPoint][trajectoryPoint]), read like the old
        // loadWeight (operator>> on float); unlike the old code, a wrong count or a malformed token is an error.
        std::vector<float> loadSplineWeights(const fs::path& file, std::size_t expected) {
            if (!fs::exists(file)) {
                throw IoError("spline weight file " + file.string() + " does not exist", file);
            }
            std::ifstream in(file);
            if (!in) {
                throw IoError("cannot open spline weight file " + file.string(), file);
            }
            std::vector<float> w;
            w.reserve(expected);
            float v = 0;
            while (in >> v) {
                w.push_back(v);
            }
            if (!in.eof()) {
                throw IoError("spline weight file " + file.string() + ": token " + str(w.size() + 1)
                                  + " is not a number",
                    file);
            }
            if (w.size() != expected) {
                throw IoError("spline weight file " + file.string() + " holds " + str(w.size()) + " weights, "
                                  + str(expected) + " expected (control points per curve x trajectory points)",
                    file);
            }
            return w;
        }

        class CurveBackend final : public Backend {
        public:
            explicit CurveBackend(const BackendCreateInfo& info);
            ~CurveBackend() override = default;

            Mode mode() const noexcept override { return mode_; }
            const ProgramSet& programs() const noexcept override { return programs_; }
            void beginCheck(const CheckInfo& info) override;
            LaunchPlan prepare(const StreamContext& stream, const BatchView& batch) override;
            bool synchronizesHost(std::uint32_t itemCount) const noexcept override { return obb_ && itemCount > 0; }

        private:
            struct StreamState {
                DeviceBuffer points;    // curve vertices of the batch: [item][sphere][vertex][3]
                DeviceBuffer linkPoses; // unfused path only: trajectory link poses
                DeviceBuffer trajPts;   // unfused path, degree >= 2: sphere centers before the control-point fit
                DeviceBuffer gas, temp;
                // OBB culling only
                DeviceBuffer segObbs, counts, where, scanScratch;
                DeviceBuffer cmpVertices, cmpRadii, cmpIndex, cmpMap, cmpCount;
                PinnedHostBuffer hostCount;

                CUdeviceptr vertexPtr = 0; // the build input stores the address of these
                CUdeviceptr widthPtr  = 0;
                OptixBuildInput input{};
                OptixAccelBuildOptions options{};
                OptixTraversableHandle handle = 0;
                std::size_t builtItems = 0; // item count of the last full build (0 = no valid GAS)
                std::size_t refits     = 0; // refits since the last full build
                EdgeRayParams params{};
            };

            void generatePoints(StreamState& st, StreamHandle stream, DeviceSpan<const float> q, std::uint32_t n);
            void build(StreamState& st, StreamHandle stream, std::uint32_t vertices, std::uint32_t primitives,
                bool refit);

            std::shared_ptr<Context> ctx_;
            const CheckerConfig* cfg_ = nullptr;
            std::shared_ptr<const RobotModel> robot_;
            std::shared_ptr<const Scene> scene_;
            Mode mode_            = Mode::LinearCurve;
            std::uint32_t degree_ = 1;
            bool obb_             = false;
            bool allowUpdate_     = false;
            bool fused_           = false;

            std::uint32_t dof_ = 0, spheres_ = 0, trajPoints_ = 0, maxBatch_ = 0;
            std::uint32_t verticesPerCurve_ = 0, segmentsPerCurve_ = 0, verticesPerSegment_ = 0;
            std::uint32_t verticesPerItem_ = 0, segmentsPerItem_ = 0;
            std::uint32_t ctrlPointsForKernel_ = 0; // 0 for linear curves
            BuildFlags buildFlags_             = 0;
            GeometryFlags geometryFlags_       = 0;
            OptixPrimitiveType curveType_      = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;

            // rebuild policy of the current check (old streamCurveRobot::rebuildEvery): 0 = refit whenever
            // possible, K >= 1 = full rebuild at least every K-th build. Written by beginCheck (calling thread,
            // before any prepare of the check), read by prepare.
            std::size_t rebuildEvery_ = 0;

            const EdgeRays* edges_ = nullptr;

            // immutable device data shared by all streams
            DeviceBuffer kinematics_, sphereCenters_, sphereLink_, weights_;
            DeviceBuffer radiiTemplate_;   // [maxBatch * verticesPerItem] per-vertex radius
            DeviceBuffer indexTemplate_;   // [maxBatch * segmentsPerItem] first vertex of each segment
            DeviceBuffer segToItem_;       // [maxBatch * segmentsPerItem] segment -> batch-local item

            std::vector<StreamState> streams_;
            ProgramSet programs_;
        };

        CurveBackend::CurveBackend(const BackendCreateInfo& info)
            : ctx_(info.context), cfg_(info.config), robot_(info.robot), scene_(info.scene) {
            if (!ctx_ || !cfg_ || !robot_ || !scene_) {
                throw InvalidArgument("createCurveBackend: context, config, robot and scene are required");
            }
            const CheckerConfig& cfg = *cfg_;
            if (!isContinuous(cfg.mode)) {
                throw InvalidArgument("createCurveBackend: mode is not a curve mode");
            }
            if (info.streams.size() != cfg.numStreams || info.streams.empty()) {
                throw InvalidArgument("createCurveBackend: " + str(info.streams.size()) + " streams given, config has "
                                      + str(cfg.numStreams));
            }
            for (std::size_t i = 0; i < info.streams.size(); ++i) {
                if (info.streams[i].index != i) {
                    throw InvalidArgument("createCurveBackend: streams[" + str(i) + "].index = "
                                          + str(info.streams[i].index) + " (must equal its position)");
                }
            }
            mode_   = cfg.mode;
            degree_ = curveDegree(mode_);
            obb_    = cfg.obbCulling == Toggle::On;
            if (obb_ && degree_ == 3) {
                throw UnsupportedError("OBB culling is not implemented for cubic curves (as in the old stack)");
            }
            if (!robot_->hasSpheres() || robot_->totalSpheres() == 0) {
                throw ConfigError("curve modes need robot spheres (robot.geometry.spheres)", "robot.geometry.spheres");
            }
            const CurveOptions& cv = cfg.curve;
            dof_                   = robot_->dof();
            spheres_               = robot_->totalSpheres();
            trajPoints_            = cv.trajectoryPoints;
            maxBatch_              = cfg.maxBatchSize;
            if (dof_ < 1 || dof_ > kMaxDof) {
                throw UnsupportedError("curve backend: robot DOF " + str(dof_) + " outside [1, kMaxDof]");
            }
            if (trajPoints_ < 2 || trajPoints_ > kMaxTrajectoryPoints) {
                throw ConfigError("curve.trajectory_points " + str(trajPoints_) + " outside [2, kMaxTrajectoryPoints]",
                    "curve.trajectory_points");
            }

            // ---- topology (old curveParam) ----
            if (degree_ == 1) {
                verticesPerCurve_ = trajPoints_;
                segmentsPerCurve_ = trajPoints_ - 1;
            } else {
                verticesPerCurve_ = cv.controlPoints + (cv.endPoints == CurveEndPoints::Phantom ? 2u : 0u);
                if (verticesPerCurve_ <= degree_) {
                    throw ConfigError("curve.control_points " + str(cv.controlPoints) + " must be > degree "
                                          + str(degree_),
                        "curve.control_points");
                }
                segmentsPerCurve_ = verticesPerCurve_ - degree_;
                if (std::uint64_t(verticesPerCurve_) * trajPoints_ > kMaxSplineWeights) {
                    throw CapacityError("curve backend: " + str(verticesPerCurve_) + " x " + str(trajPoints_)
                                            + " spline weights exceed kMaxSplineWeights",
                        "kMaxSplineWeights", std::uint64_t(verticesPerCurve_) * trajPoints_, kMaxSplineWeights);
                }
            }
            verticesPerSegment_  = degree_ + 1;
            ctrlPointsForKernel_ = degree_ == 1 ? 0u : verticesPerCurve_;
            verticesPerItem_     = checkedU32(std::uint64_t(verticesPerCurve_) * spheres_, "vertices per item", maxBatch_);
            segmentsPerItem_     = checkedU32(std::uint64_t(segmentsPerCurve_) * spheres_, "segments per item", maxBatch_);
            const std::uint32_t maxVertices =
                checkedU32(std::uint64_t(verticesPerItem_) * maxBatch_, "vertices per batch", maxBatch_);
            const std::uint32_t maxSegments =
                checkedU32(std::uint64_t(segmentsPerItem_) * maxBatch_, "segments per batch", maxBatch_);
            const std::uint32_t maxCmpVertices =
                checkedU32(std::uint64_t(maxSegments) * verticesPerSegment_, "compacted vertices per batch", maxBatch_);
            checkedU32(std::uint64_t(maxVertices) * 3, "vertex floats per batch", maxBatch_);
            checkedU32(std::uint64_t(maxCmpVertices) * 3, "compacted vertex floats per batch", maxBatch_);

            // ---- flags (old curveGASBuildFlags / curveDefaultGeometryFlags, resolved by the config) ----
            allowUpdate_   = !obb_; // old curveUseRefit = !useOBB: refittable whenever OBB culling is off
            buildFlags_    = toBuildFlags(cv.gasBuild) | (allowUpdate_ ? BuildFlags{kBuildFlagAllowUpdate} : 0u);
            geometryFlags_ = cv.geometryFlags.value_or(degree_ == 2 ? GeometryFlags{kGeometryFlagRequireSingleAnyHitCall}
                                                                    : GeometryFlags{kGeometryFlagNone});
            curveType_     = degree_ == 1   ? OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR
                           : degree_ == 2 ? OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE
                                          : OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
            fused_ = curve::fusedCurvePointsSupported(dof_, trajPoints_, spheres_, ctrlPointsForKernel_);

            // ---- scene rays ----
            edges_ = &scene_->edgeRays(cfg.sceneEdges.value_or(EdgeSet::Loop));
            if (edges_->count > kMaxLaunchSize) {
                throw CapacityError("curve backend: " + str(edges_->count) + " scene edge rays exceed the launch limit",
                    "launch_size", edges_->count, kMaxLaunchSize);
            }

            ctx_->makeCurrent();

            // ---- static device data ----
            kinematics_ = internal::uploadKinematics(robot_->kinematics());
            sphereCenters_ = DeviceBuffer::fromHost(robot_->sphereCenters());
            sphereLink_    = DeviceBuffer::fromHost(robot_->sphereLink());
            if (degree_ >= 2) {
                const fs::path file = cv.splineWeightsDir / (degree_ == 2 ? "quad" : "cubic") / endPointsName(cv.endPoints)
                                    / (str(cv.controlPoints) + "_" + str(trajPoints_) + ".txt");
                const std::vector<float> w = loadSplineWeights(file, std::size_t(verticesPerCurve_) * trajPoints_);
                weights_                   = DeviceBuffer::fromHost(std::span<const float>(w));
            }
            {
                // old curveParam: index buffer, per-vertex radii, segment -> item map (template for a full batch)
                const std::span<const float> sphereRadii = robot_->sphereRadii();
                std::vector<float> radii(maxVertices);
                std::vector<std::uint32_t> index(maxSegments), segItem(maxSegments);
                const std::uint64_t curves = std::uint64_t(maxBatch_) * spheres_;
                for (std::uint64_t c = 0; c < curves; ++c) {
                    const float r = sphereRadii[c % spheres_];
                    for (std::uint32_t v = 0; v < verticesPerCurve_; ++v) {
                        radii[c * verticesPerCurve_ + v] = r;
                    }
                    for (std::uint32_t s = 0; s < segmentsPerCurve_; ++s) {
                        index[c * segmentsPerCurve_ + s] = static_cast<std::uint32_t>(c * verticesPerCurve_ + s);
                    }
                }
                for (std::uint32_t i = 0; i < maxBatch_; ++i) {
                    std::fill(segItem.begin() + std::size_t(i) * segmentsPerItem_,
                        segItem.begin() + std::size_t(i + 1) * segmentsPerItem_, i);
                }
                radiiTemplate_ = DeviceBuffer::fromHost(std::span<const float>(radii));
                indexTemplate_ = DeviceBuffer::fromHost(std::span<const std::uint32_t>(index));
                segToItem_     = DeviceBuffer::fromHost(std::span<const std::uint32_t>(segItem));
            }

            // ---- per stream (old streamCurveRobot ctor + preAllocCurveBuffer) ----
            const OptixDeviceContext octx = ctx_->optixContext();
            streams_.resize(info.streams.size());
            for (StreamState& st : streams_) {
                st.points.allocate(std::size_t(maxVertices) * 3 * sizeof(float));
                if (!fused_) {
                    st.linkPoses.allocate(std::size_t(maxBatch_) * (dof_ + 1) * trajPoints_ * 12 * sizeof(float));
                    if (degree_ >= 2) {
                        st.trajPts.allocate(std::size_t(maxBatch_) * spheres_ * trajPoints_ * 3 * sizeof(float));
                    }
                }
                st.vertexPtr = st.points.ptr();
                st.widthPtr  = radiiTemplate_.ptr();

                OptixBuildInputCurveArray& ca = st.input.curveArray;
                st.input.type                 = OPTIX_BUILD_INPUT_TYPE_CURVES;
                ca.curveType                  = curveType_;
                ca.vertexBuffers              = &st.vertexPtr;
                ca.numVertices                = maxVertices;
                ca.vertexStrideInBytes        = sizeof(float) * 3;
                ca.widthBuffers               = &st.widthPtr;
                ca.widthStrideInBytes         = sizeof(float);
                ca.normalBuffers              = nullptr;
                ca.normalStrideInBytes        = 0;
                ca.indexBuffer                = indexTemplate_.ptr();
                ca.indexStrideInBytes         = sizeof(std::uint32_t);
                ca.numPrimitives              = maxSegments;
                ca.flag                       = geometryFlags_;
                ca.primitiveIndexOffset       = 0;
                ca.endcapFlags                = OPTIX_CURVE_ENDCAP_ON;

                st.options.buildFlags = buildFlags_;
                st.options.operation  = OPTIX_BUILD_OPERATION_BUILD;

                if (obb_) {
                    const std::size_t segs = maxSegments;
                    st.segObbs.allocate(segs * sizeof(Obb));
                    st.counts.allocate(segs * sizeof(std::uint32_t));
                    st.where.allocate(segs * sizeof(std::uint32_t));
                    st.scanScratch.allocate((std::max)(internal::exclusiveScanScratchBytes(segs), std::size_t(16)));
                    st.cmpVertices.allocate(std::size_t(maxCmpVertices) * 3 * sizeof(float));
                    st.cmpRadii.allocate(std::size_t(maxCmpVertices) * sizeof(float));
                    st.cmpIndex.allocate(segs * sizeof(std::uint32_t));
                    st.cmpMap.allocate(segs * sizeof(std::uint32_t));
                    st.cmpCount.allocate(sizeof(std::uint32_t));
                    st.hostCount.allocate(sizeof(std::uint32_t));
                    // the GAS is always built from the compacted arrays (old: build input overwritten when useOBB)
                    st.vertexPtr   = st.cmpVertices.ptr();
                    st.widthPtr    = st.cmpRadii.ptr();
                    ca.indexBuffer = st.cmpIndex.ptr();
                    ca.numVertices = maxCmpVertices; // largest compacted input: every segment survives
                }

                OptixAccelBufferSizes sizes{};
                RTCD_OPTIX_CHECK(optixAccelComputeMemoryUsage(octx, &st.options, &st.input, 1, &sizes));
                st.gas.allocate(sizes.outputSizeInBytes);
                // sizes are for the full batch; a refit needs tempUpdateSizeInBytes
                st.temp.allocate((std::max)(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes));

                st.params.edgeVertices = edges_->vertices.ptr;
                st.params.edgeCount    = edges_->count;
            }

            // ---- programs (old createModule / createRaygenPrograms / createHitgroupPrograms for the curve types) ----
            programs_.moduleCode = std::span<const unsigned char>(embedded::rtcd_backend_curve_ir_curve_programs,
                embedded::rtcd_backend_curve_ir_curve_programs_size);
            programs_.moduleName = "rtcd_backend_curve/curve_programs";
            programs_.raygens    = {"__raygen__oneside"};
            programs_.miss.clear();
            programs_.anyHit              = "__anyhit__GAS";
            programs_.closestHit          = "";
            programs_.builtinIntersection = degree_ == 1   ? BuiltinIntersection::RoundLinear
                                          : degree_ == 2 ? BuiltinIntersection::RoundQuadratic
                                                         : BuiltinIntersection::RoundCubic;
            programs_.builtinIsBuildFlags = buildFlags_;
            programs_.traversableGraph    = TraversableGraph::SingleGas;
            programs_.primitiveTypes      = degree_ == 1   ? kPrimitiveRoundLinear
                                          : degree_ == 2 ? kPrimitiveRoundQuadratic
                                                         : kPrimitiveRoundCubic;
            programs_.numPayloadValues    = 0;
            programs_.numAttributeValues  = 2;
            programs_.maxTraversableDepth = 1;
            programs_.hitGroupRecordCount = 1;

            if (cfg.verbosity >= LogLevel::Info) {
                std::ostringstream m;
                m << "curve backend: degree " << degree_ << ", " << spheres_ << " spheres, " << verticesPerItem_
                  << " vertices / " << segmentsPerItem_ << " segments per item, point generation "
                  << (fused_ ? "fused" : "unfused") << ", build flags " << buildFlags_ << ", geometry flags "
                  << geometryFlags_ << (obb_ ? ", OBB culling" : "") << ", GAS " << streams_[0].gas.size()
                  << " B, temp " << streams_[0].temp.size() << " B per stream";
                ctx_->logger().log(LogLevel::Info, m.str());
            }
        }

        void CurveBackend::beginCheck(const CheckInfo& info) {
            // old batchOptixCurveRobot::applyRefitPolicy (setBatchCountHint)
            const RefitPolicy& p = cfg_->curve.refit;
            switch (p.kind) {
            case RefitPolicy::Kind::Always: rebuildEvery_ = 0; break;
            case RefitPolicy::Kind::Never: rebuildEvery_ = 1; break;
            case RefitPolicy::Kind::Every: rebuildEvery_ = p.every; break;
            default: {
                const std::int64_t maxBatches = cfg_->curve.autoRefitMaxBatches < 0
                                                  ? (mode_ == Mode::LinearCurve ? 2 : 1)
                                                  : cfg_->curve.autoRefitMaxBatches;
                rebuildEvery_ = std::int64_t(info.batchCount) <= maxBatches ? 0 : 1;
                break;
            }
            }
        }

        void CurveBackend::generatePoints(StreamState& st, StreamHandle stream, DeviceSpan<const float> q, std::uint32_t n) {
            // old streamCurveRobot::genCurvePoints
            if (fused_) {
                curve::fusedCurvePoints(kinematics_.ptr(), dof_, q.ptr, sphereCenters_.ptr(), sphereLink_.ptr(),
                    weights_.ptr(), st.points.ptr(), trajPoints_, n, spheres_, ctrlPointsForKernel_, stream);
                return;
            }
            const internal::KinematicsRef kin{&robot_->kinematics(), kinematics_.ptr()};
            internal::trajectoryLinkPoses(kin, q, trajPoints_, n, st.linkPoses.span<float>(), stream);
            if (degree_ == 1) {
                curve::sphereCentersTraj(st.linkPoses.ptr(), sphereCenters_.ptr(), sphereLink_.ptr(), st.points.ptr(),
                    trajPoints_, n, spheres_, dof_, stream);
            } else {
                curve::sphereCentersTraj(st.linkPoses.ptr(), sphereCenters_.ptr(), sphereLink_.ptr(), st.trajPts.ptr(),
                    trajPoints_, n, spheres_, dof_, stream);
                curve::controlPoints(st.trajPts.ptr(), st.points.ptr(), weights_.ptr(), spheres_ * n,
                    ctrlPointsForKernel_, trajPoints_, stream);
            }
        }

        void CurveBackend::build(
            StreamState& st, StreamHandle stream, std::uint32_t vertices, std::uint32_t primitives, bool refit) {
            st.input.curveArray.numVertices   = vertices;
            st.input.curveArray.numPrimitives = primitives;
            st.options.operation              = refit ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
            RTCD_OPTIX_CHECK(optixAccelBuild(ctx_->optixContext(), stream, &st.options, &st.input, 1, st.temp.ptr(),
                st.temp.size(), st.gas.ptr(), st.gas.size(), &st.handle, nullptr, 0));
        }

        LaunchPlan CurveBackend::prepare(const StreamContext& stream, const BatchView& batch) {
            if (stream.index >= streams_.size()) {
                throw InvalidArgument("curve backend: stream index " + str(stream.index) + " >= " + str(streams_.size()));
            }
            const std::uint32_t n = batch.itemCount;
            if (n == 0 || n > maxBatch_) {
                throw CapacityError("curve backend: batch of " + str(n) + " items (capacity max_batch_size = "
                                        + str(maxBatch_) + ", at least 1)",
                    "max_batch_size", n, maxBatch_);
            }
            const std::size_t floats = std::size_t(n) * trajPoints_ * dof_;
            if (batch.items.count < floats || batch.hitCounts.count < n) {
                throw InvalidArgument("curve backend: batch views too small (" + str(batch.items.count) + " floats for "
                                      + str(floats) + ", " + str(batch.hitCounts.count) + " counters for " + str(n) + ")");
            }
            StreamState& st         = streams_[stream.index];
            const StreamHandle s    = stream.stream;
            LaunchPlan plan;

            generatePoints(st, s, batch.items, n);

            if (!obb_) {
                // old streamCurveRobot::buildCurveGAS: a refit keeps the BVH topology of the last full build, so it is
                // only valid for the same item count; rebuildEvery_ 0 = refit whenever possible, K = full rebuild at
                // least every K-th build.
                const bool refit = allowUpdate_ && st.builtItems == n && (rebuildEvery_ == 0 || st.refits + 1 < rebuildEvery_);
                build(st, s, verticesPerItem_ * n, segmentsPerItem_ * n, refit);
                if (refit) {
                    st.refits++;
                } else {
                    st.builtItems = n;
                    st.refits     = 0;
                }
                st.params.primToItem = segToItem_.ptr();
            } else {
                // old fkine (useOBB) -> updateOBB, detectOBB (BBCDRobotMask), updateWithMask; plus the compacted
                // segment -> item map (design doc Q5: the old path mapped compacted primitive indices through the
                // uncompacted template).
                const std::uint32_t nSeg = segmentsPerItem_ * n;
                curve::curveSegmentObbs(degree_, st.points.ptr(), indexTemplate_.ptr(), radiiTemplate_.ptr(),
                    st.segObbs.ptr(), nSeg, s);
                const DeviceSpan<std::uint32_t> counts{st.counts.ptr(), nSeg};
                internal::obbOverlapCounts(DeviceSpan<const Obb>{st.segObbs.ptr(), nSeg}, scene_->obbs(), counts, s);
                internal::flagsToBinaryU32(counts, counts, s);
                internal::exclusiveScanU32(counts, DeviceSpan<std::uint32_t>{st.where.ptr(), nSeg},
                    st.scanScratch.span<std::byte>(), s);
                curve::compactSegments(st.points.ptr(), radiiTemplate_.ptr(), indexTemplate_.ptr(), segToItem_.ptr(),
                    st.counts.ptr(), st.where.ptr(), nSeg, verticesPerSegment_, st.cmpVertices.ptr(), st.cmpRadii.ptr(),
                    st.cmpIndex.ptr(), st.cmpMap.ptr(), st.cmpCount.ptr(), s);
                st.cmpCount.downloadAsync(st.hostCount.data(), sizeof(std::uint32_t), s);
                RTCD_CUDA_CHECK(cudaStreamSynchronize(s)); // synchronizesHost() == true on this path
                const std::uint32_t kept = st.hostCount.as<std::uint32_t>()[0];
                if (kept == 0) {
                    plan.launch = false; // nothing near an obstacle: hit counts stay 0
                    return plan;
                }
                build(st, s, kept * verticesPerSegment_, kept, false);
                st.params.primToItem = st.cmpMap.ptr();
            }

            st.params.traversable = st.handle;
            st.params.hitCounts   = batch.hitCounts.ptr;
            plan.launch           = edges_->count > 0;
            plan.traversable      = st.handle;
            plan.width            = (std::max)(edges_->count, 1u);
            plan.height           = 1;
            plan.depth            = 1;
            plan.params           = &st.params;
            plan.paramsBytes      = sizeof(EdgeRayParams);
            plan.raygen           = 0;
            return plan;
        }

    } // namespace

    std::unique_ptr<Backend> createCurveBackend(const BackendCreateInfo& info) {
        return std::make_unique<CurveBackend>(info);
    }

} // namespace rtcd
