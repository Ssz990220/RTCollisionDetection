// rtcd::Scene: reproduces RTCD/CollisionScenes/{obstacle.h, scene.h} (and the scene part of batchScene.h that the
// backends use): obstacle meshes in the world frame, one compacted triangle GAS per obstacle, one IAS over them,
// the obstacle edge rays of both edge sets and the obstacle OBBs on the device.
//
// Old -> new:
//   obstacle(path, Affine3f pose, scale, loop)   loadObjMesh(path, pose, scale) (rtcd_geometry)
//   obstacle::singleVertices (2 float3 per ray)  EdgeRays::vertices (6 floats per ray), one buffer per edge set
//   scene::obsVertCnt / whereInEdge (vertices)   EdgeRays::rayCount / firstRay (rays = vertices / 2)
//   scene::obbs / dOBBs                          ObstacleInfo::obb / obbs()
//   meshModel::buildGAS + scene::wrapScene       internal::buildTriangleGas / buildInstanceAs (rtcd_core)
// The old scene built only the edge set it was asked for (buildSharedScene(loop)); this one builds Unique always
// and Loop when every obstacle is watertight (interfaces.md Q12).

#include <rtcd/errors.h>
#include <rtcd/internal/accel.h>
#include <rtcd/scene.h>

#include <optix_types.h>

#include <cstring>
#include <string>

namespace rtcd {

    namespace {
        static_assert(sizeof(OptixInstance) == 80, "OptixInstance layout");

        struct EdgeSetData {
            DeviceBuffer buffer;
            EdgeRays rays;
        };

        void buildEdgeSet(EdgeSetData& out, const std::vector<TriangleMesh>& meshes, bool loop) {
            std::vector<float> host;
            std::size_t total = 0;
            for (const TriangleMesh& m : meshes) {
                total += (loop ? m.loopEdges : m.uniqueEdges).size();
            }
            host.reserve(total * 6);
            out.rays.firstRay.clear();
            out.rays.rayCount.clear();
            std::uint32_t first = 0;
            for (const TriangleMesh& m : meshes) {
                const std::vector<Edge>& edges = loop ? m.loopEdges : m.uniqueEdges;
                for (const Edge& e : edges) {
                    const Vec3& a = m.vertices[e[0]];
                    const Vec3& b = m.vertices[e[1]];
                    host.insert(host.end(), {a[0], a[1], a[2], b[0], b[1], b[2]});
                }
                out.rays.firstRay.push_back(first);
                out.rays.rayCount.push_back(static_cast<std::uint32_t>(edges.size()));
                first += static_cast<std::uint32_t>(edges.size());
            }
            out.buffer        = DeviceBuffer::fromHost(std::span<const float>(host));
            out.rays.vertices = out.buffer.span<float>();
            out.rays.count    = first;
        }
    } // namespace

    struct Scene::Impl {
        std::shared_ptr<Context> context;
        SceneDesc desc;
        std::vector<TriangleMesh> meshes;
        std::vector<ObstacleInfo> infos;

        std::vector<internal::Accel> gases;
        DeviceBuffer instances;
        internal::Accel ias;

        EdgeSetData unique;
        EdgeSetData loop;
        bool hasLoop = false;
        std::string firstNonWatertight;

        DeviceBuffer obbBuffer;
    };

    Scene::Scene(std::shared_ptr<Context> context, const SceneDesc& desc) : impl_(std::make_unique<Impl>()) {
        if (!context) {
            throw InvalidArgument("Scene: context is null");
        }
        if (desc.obstacles.empty()) {
            throw InvalidArgument("Scene '" + desc.name + "': the scene has no obstacles");
        }
        Impl& s   = *impl_;
        s.context = std::move(context);
        s.desc    = desc;

        // ---- host: meshes, OBBs (old obstacle(path, pose, scale) + scene::addObstacle) ----
        s.meshes.reserve(desc.obstacles.size());
        s.infos.reserve(desc.obstacles.size());
        for (std::size_t o = 0; o < desc.obstacles.size(); ++o) {
            const ObstacleDesc& ob = desc.obstacles[o];
            const std::string name = ob.name.empty() ? ob.mesh.stem().string() : ob.name;
            if (!(ob.scale > 0.0f)) {
                throw InvalidArgument("Scene '" + desc.name + "': obstacle " + std::to_string(o) + " ('" + name
                                      + "'): scale = " + std::to_string(ob.scale) + " must be > 0");
            }
            s.meshes.push_back(loadObjMesh(ob.mesh, ob.pose, ob.scale));
            const TriangleMesh& m = s.meshes.back();
            ObstacleInfo info;
            info.name         = name;
            info.obb          = fitObb(m.vertices);
            info.numVertices  = static_cast<std::uint32_t>(m.vertices.size());
            info.numTriangles = static_cast<std::uint32_t>(m.triangles.size());
            info.watertight   = m.watertight;
            if (!m.watertight && s.firstNonWatertight.empty()) {
                s.firstNonWatertight = name + " (" + ob.mesh.string() + ")";
            }
            s.infos.push_back(std::move(info));
        }
        s.hasLoop = s.firstNonWatertight.empty();

        // ---- device ----
        s.context->makeCurrent();

        // Edge rays and OBBs (old scene::uploadToDevice).
        buildEdgeSet(s.unique, s.meshes, false);
        if (s.hasLoop) {
            buildEdgeSet(s.loop, s.meshes, true);
        }
        std::vector<Obb> obbs;
        obbs.reserve(s.infos.size());
        for (const ObstacleInfo& info : s.infos) {
            obbs.push_back(info.obb);
        }
        s.obbBuffer = DeviceBuffer::fromHost(std::span<const Obb>(obbs));

        // One compacted triangle GAS per obstacle (old meshModel::buildGAS: PREFER_FAST_TRACE | ALLOW_COMPACTION,
        // geometry DISABLE_ANYHIT, one SBT record). The vertex / index buffers are only needed during the build.
        s.gases.reserve(s.meshes.size());
        for (const TriangleMesh& m : s.meshes) {
            static_assert(sizeof(Vec3) == 3 * sizeof(float));
            static_assert(sizeof(m.triangles[0]) == 3 * sizeof(std::uint32_t));
            const DeviceBuffer vertices = DeviceBuffer::fromHost(std::span<const Vec3>(m.vertices));
            const DeviceBuffer indices  = DeviceBuffer::fromHost(std::span<const std::array<std::uint32_t, 3>>(m.triangles));
            internal::TriangleGasInput input;
            input.vertices      = DeviceSpan<const float>{vertices.ptr(), m.vertices.size() * 3};
            input.indices       = DeviceSpan<const std::uint32_t>{indices.ptr(), m.triangles.size() * 3};
            input.geometryFlags = kGeometryFlagDisableAnyHit;
            s.gases.push_back(internal::buildTriangleGas(
                *s.context, input, kBuildFlagPreferFastTrace | kBuildFlagAllowCompaction, nullptr));
        }

        // IAS (old scene::wrapScene): identity transforms, visibility 255, instanceId 0, flags NONE, build NONE.
        // Instance o uses SBT offset o (the old code left every offset at 0; its numObstacles hit records were
        // identical, so both give the same hits; scene.h specifies offset o).
        std::vector<OptixInstance> inst(s.gases.size());
        for (std::size_t o = 0; o < s.gases.size(); ++o) {
            OptixInstance& i = inst[o];
            std::memset(&i, 0, sizeof(i));
            const Transform identity;
            std::memcpy(i.transform, identity.m.data(), sizeof(i.transform));
            i.instanceId        = 0;
            i.sbtOffset         = static_cast<unsigned int>(o);
            i.visibilityMask    = 255;
            i.flags             = OPTIX_INSTANCE_FLAG_NONE;
            i.traversableHandle = s.gases[o].handle;
        }
        s.instances = DeviceBuffer::fromHost(std::span<const OptixInstance>(inst));
        s.ias       = internal::buildInstanceAs(
            *s.context, s.instances.ptr(), static_cast<std::uint32_t>(inst.size()), kBuildFlagNone, nullptr);
    }

    Scene::~Scene() = default;

    const std::shared_ptr<Context>& Scene::context() const noexcept {
        return impl_->context;
    }
    const SceneDesc& Scene::desc() const noexcept {
        return impl_->desc;
    }
    std::uint32_t Scene::numObstacles() const noexcept {
        return static_cast<std::uint32_t>(impl_->infos.size());
    }
    std::span<const ObstacleInfo> Scene::obstacles() const noexcept {
        return impl_->infos;
    }
    TraversableHandle Scene::traversable() const noexcept {
        return impl_->ias.handle;
    }
    bool Scene::hasEdgeSet(EdgeSet set) const noexcept {
        return set == EdgeSet::Unique || impl_->hasLoop;
    }
    const EdgeRays& Scene::edgeRays(EdgeSet set) const {
        if (set == EdgeSet::Unique) {
            return impl_->unique.rays;
        }
        if (!impl_->hasLoop) {
            throw UnsupportedError("Scene '" + impl_->desc.name + "': loop edges are not available because obstacle "
                                   + impl_->firstNonWatertight + " is not watertight");
        }
        return impl_->loop.rays;
    }
    DeviceSpan<const Obb> Scene::obbs() const noexcept {
        return impl_->obbBuffer.span<const Obb>();
    }
    std::span<const TriangleMesh> Scene::meshes() const noexcept {
        return impl_->meshes;
    }

} // namespace rtcd
