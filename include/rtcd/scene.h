#pragma once
// rtcd/scene.h -- the static obstacle scene.
//
//   SceneDesc  plain description (what the JSON scene file holds): obstacle meshes with pose and scale.
//   Scene      the scene built on the GPU from a SceneDesc: per-obstacle triangle GAS + one IAS over them, the
//              obstacle edge rays of both edge sets, and the obstacle OBBs -- exactly what the backends read.
//
// Ownership / lifetime: a Scene is immutable after construction and is shared read-only (shared_ptr<const Scene>)
// by any number of CollisionCheckers built on the SAME Context (checked by the checker). It keeps its Context
// alive. All device memory it exposes stays valid and unchanged for the Scene's lifetime.
//
// Threading: all const member functions are thread-safe; they only return stored values (no CUDA calls).
//
// Errors: construction throws IoError (mesh file missing / unreadable / not OBJ), InvalidArgument (empty scene,
// scale <= 0), CudaError / OptixError. edgeRays(EdgeSet::Loop) throws UnsupportedError naming the first
// non-watertight obstacle if loop edges are not available.
//
// Library rtcd_scene (src/scene, Wave 1 c), reproducing RTCD/CollisionScenes/{obstacle.h, scene.h}; meshes,
// edges and OBBs through the rtcd_geometry functions declared in rtcd/geometry.h.

#include <rtcd/context.h>
#include <rtcd/geometry.h>
#include <rtcd/limits.h>
#include <rtcd/types.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rtcd {

    struct ObstacleDesc {
        std::string name;                // free text, used in messages and dumps (default: mesh file stem)
        std::filesystem::path mesh;      // OBJ file; absolute after JSON loading (resolved against the scene file)
        Transform pose;                  // obstacle -> world
        float scale = 1.0f;              // uniform, applied before the pose: world = pose * (scale * v)
    };

    struct SceneDesc {
        std::string name;                   // free text
        std::vector<ObstacleDesc> obstacles; // order = obstacle index everywhere (SBT offset, OBB index, edge order)
    };

    // A set of scene edge rays: rays[r] = (origin, target), 6 floats per ray (two float3), all obstacles
    // concatenated in obstacle order. Per obstacle o, its rays are [firstRay[o], firstRay[o] + rayCount[o]).
    struct EdgeRays {
        DeviceSpan<const float> vertices;  // 6 * count floats
        std::uint32_t count = 0;           // number of rays
        std::vector<std::uint32_t> firstRay;
        std::vector<std::uint32_t> rayCount;
    };

    struct ObstacleInfo {
        std::string name;
        Obb obb;                       // world frame, best fit over the transformed vertices (old meshModel::obb)
        std::uint32_t numVertices  = 0;
        std::uint32_t numTriangles = 0;
        bool watertight            = false;
    };

    class Scene {
    public:
        // Loads every obstacle mesh, applies scale and pose to the vertices, builds per obstacle a triangle GAS
        // (PREFER_FAST_TRACE | ALLOW_COMPACTION, compacted, geometry flag DISABLE_ANYHIT, identity instance
        // transform) and one IAS over them (build flags NONE; instance o: sbtOffset o, visibilityMask 255,
        // instanceId 0), uploads both edge sets (Unique always, Loop if every obstacle is watertight) and the
        // OBBs. Synchronous; the Scene is ready when the constructor returns.
        Scene(std::shared_ptr<Context> context, const SceneDesc& desc);
        ~Scene();
        Scene(const Scene&)            = delete;
        Scene& operator=(const Scene&) = delete;

        const std::shared_ptr<Context>& context() const noexcept;
        const SceneDesc& desc() const noexcept;
        std::uint32_t numObstacles() const noexcept;
        std::span<const ObstacleInfo> obstacles() const noexcept;

        // IAS over all obstacle GASes (RAY mode traces against it). Hit-group SBT records needed: numObstacles()
        // (instance o uses record o).
        TraversableHandle traversable() const noexcept;

        bool hasEdgeSet(EdgeSet set) const noexcept;
        const EdgeRays& edgeRays(EdgeSet set) const; // throws UnsupportedError if !hasEdgeSet(set)

        // Obstacle OBBs on the device, numObstacles() entries in obstacle order (old scene::getdOBBs()).
        DeviceSpan<const Obb> obbs() const noexcept;

        // Host copy of the world-space meshes (for tests, reference checks and visualization).
        std::span<const TriangleMesh> meshes() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rtcd
