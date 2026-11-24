// Parity of rtcd_geometry with the old stack's loader (RTCD/Meshes/mesh.h, Obb.h). Host only (label new_unit),
// but it includes old-stack headers: DELETE THIS TEST TOGETHER WITH THE OLD STACK.
//
// Compared bit for bit (float bit patterns, index values, order):
//   * every Panda link mesh, loaded the way the old robot loads it (meshModel(urdfLink, meshDir, sphereDir) on
//     models/franka_description/panda.urdf), against loadObjMesh(file): vertices, triangles, unique edges, loop
//     edges, link OBB (fitObb of the vertices), and the link sphere files against loadSpheres;
//   * every obstacle of the three shelf scenes (RTCD/CollisionScenes/Scene/shelf.h, denseShelf.h, shelfSimple.h),
//     loaded as meshModel(path, Eigen::Affine3f(translate), 1.0) like the old obstacle class, against
//     loadObjMesh(path, Transform::translation(...), 1.0f): the same five items.
//   * computeEdges on the loaded triangles reproduces the edges again (idempotent), as rtcd_robot uses it.

#ifndef NOMINMAX
#define NOMINMAX // optix_stubs.h (via mesh.h) includes <windows.h>; its min / max macros break RTCD/Utils/cuUtils.cuh
#endif
#include <rtcd/geometry.h>

// old stack (reference implementation); mesh.h first, as the old tests do (the OptiX table definition pulls in
// <windows.h> and its min / max macros)
#include <Meshes/mesh.h>
#include <optix_function_table_definition.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    const std::string kModels = "models";

    std::uint32_t bits(float f) {
        return std::bit_cast<std::uint32_t>(f);
    }

    ::testing::AssertionResult sameFloat3(const float3& o, const rtcd::Vec3& n) {
        if (bits(o.x) == bits(n[0]) && bits(o.y) == bits(n[1]) && bits(o.z) == bits(n[2])) {
            return ::testing::AssertionSuccess();
        }
        return ::testing::AssertionFailure() << "old (" << o.x << ", " << o.y << ", " << o.z << ") new (" << n[0] << ", "
                                             << n[1] << ", " << n[2] << ")";
    }

    ::testing::AssertionResult sameObb(const RTCD::OBB& o, const rtcd::Obb& n) {
        static_assert(sizeof(RTCD::OBB) == sizeof(rtcd::Obb));
        if (std::memcmp(&o, &n, sizeof(n)) == 0) {
            return ::testing::AssertionSuccess();
        }
        auto f = ::testing::AssertionFailure() << "OBB differs:";
        const float* a = reinterpret_cast<const float*>(&o);
        const float* b = reinterpret_cast<const float*>(&n);
        for (int i = 0; i < 15; ++i) {
            if (bits(a[i]) != bits(b[i])) {
                f << " [" << i << "] old " << a[i] << " new " << b[i];
            }
        }
        return f;
    }

    void expectEdgesEqual(const std::vector<int2>& old, const std::vector<rtcd::Edge>& now, const char* what) {
        ASSERT_EQ(old.size(), now.size()) << what;
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < old.size(); ++i) {
            if (static_cast<std::uint32_t>(old[i].x) != now[i][0] || static_cast<std::uint32_t>(old[i].y) != now[i][1]) {
                if (mismatches++ < 5) {
                    ADD_FAILURE() << what << " " << i << ": old (" << old[i].x << ", " << old[i].y << ") new (" << now[i][0]
                                  << ", " << now[i][1] << ")";
                }
            }
        }
        EXPECT_EQ(mismatches, 0u) << what;
    }

    // Everything the old meshModel holds that the new TriangleMesh / fitObb reproduce.
    void expectSameMesh(const RTCD::meshModel& old, const rtcd::TriangleMesh& now, const std::string& label) {
        SCOPED_TRACE(label);
        ASSERT_EQ(old.mesh.vertices.size(), now.vertices.size());
        std::size_t vertexMismatches = 0;
        for (std::size_t i = 0; i < now.vertices.size(); ++i) {
            const auto r = sameFloat3(old.mesh.vertices[i], now.vertices[i]);
            if (!r && vertexMismatches++ < 5) {
                ADD_FAILURE() << "vertex " << i << ": " << r.message();
            }
        }
        EXPECT_EQ(vertexMismatches, 0u);

        ASSERT_EQ(old.mesh.indices.size(), now.triangles.size());
        std::size_t triMismatches = 0;
        for (std::size_t i = 0; i < now.triangles.size(); ++i) {
            const int3 o = old.mesh.indices[i];
            const auto& n = now.triangles[i];
            if (static_cast<std::uint32_t>(o.x) != n[0] || static_cast<std::uint32_t>(o.y) != n[1]
                || static_cast<std::uint32_t>(o.z) != n[2]) {
                ++triMismatches;
            }
        }
        EXPECT_EQ(triMismatches, 0u);

        expectEdgesEqual(old.mesh.uniqueEdgeIndices, now.uniqueEdges, "unique edge");
        // The old loader throws for non-watertight meshes, so a constructed old mesh is watertight.
        EXPECT_TRUE(now.watertight);
        expectEdgesEqual(old.mesh.loopEdgeIndices, now.loopEdges, "loop edge");

        EXPECT_TRUE(sameObb(old.getOBB(), rtcd::fitObb(now.vertices)));

        // computeEdges is idempotent on the loaded triangles (rtcd_robot recomputes edges after merging).
        rtcd::TriangleMesh again;
        again.vertices  = now.vertices;
        again.triangles = now.triangles;
        rtcd::computeEdges(again);
        EXPECT_EQ(again.uniqueEdges, now.uniqueEdges);
        EXPECT_EQ(again.loopEdges, now.loopEdges);
        EXPECT_EQ(again.watertight, now.watertight);
    }

    struct ObstacleRef {
        const char* file;
        float x, y, z;
    };

    // RTCD/CollisionScenes/Scene/shelf.h ("medium shelf", the goldens' scene)
    const std::vector<ObstacleRef> kShelf = {{"curobo/shelves.obj", 0.45f, 0, 0.4f}, {"curobo/bin.obj", 0, -0.6f, 0},
        {"curobo/bin.obj", 0, 0.6f, 0}, {"curobo/Bob.obj", -0.4f, 0, 0}, {"curobo/Cow.obj", -0.4f, 0, 0},
        {"curobo/Fish.obj", -0.4f, 0, 0}, {"curobo/Sheep.obj", -0.4f, 0, 0}, {"curobo/Snakeboard.obj", -0.4f, 0, 0}};
    // RTCD/CollisionScenes/Scene/denseShelf.h
    const std::vector<ObstacleRef> kDenseShelf = {{"curobo/shelves.obj", 0.45f, 0, 0.4f},
        {"curobo/bin_dense.obj", 0, -0.6f, 0}, {"curobo/bin_dense.obj", 0, 0.6f, 0}, {"curobo/Bob.obj", -0.4f, 0, 0},
        {"curobo/Face1.obj", -0.4f, 0, 0}, {"curobo/Face2.obj", -0.4f, 0, 0}, {"curobo/Face3.obj", -0.4f, 0, 0},
        {"curobo/Cow.obj", -0.4f, 0, 0}, {"curobo/Fish.obj", -0.4f, 0, 0}, {"curobo/Sheep.obj", -0.4f, 0, 0},
        {"curobo/Snakeboard.obj", -0.4f, 0, 0}};
    // RTCD/CollisionScenes/Scene/shelfSimple.h
    const std::vector<ObstacleRef> kShelfSimple = {
        {"curobo/shelves.obj", 0.45f, 0, 0.4f}, {"curobo/bin.obj", 0, -0.6f, 0}, {"curobo/bin.obj", 0, 0.6f, 0}};

    void checkObstacles(const std::vector<ObstacleRef>& obstacles) {
        for (const ObstacleRef& ob : obstacles) {
            const std::string path = kModels + "/" + ob.file;
            Eigen::Affine3f pose   = Eigen::Affine3f::Identity();
            pose.translate(Eigen::Vector3f(ob.x, ob.y, ob.z));
            const RTCD::meshModel old(path, pose, 1.0f);
            const rtcd::TriangleMesh now = rtcd::loadObjMesh(path, rtcd::Transform::translation(ob.x, ob.y, ob.z), 1.0f);
            expectSameMesh(old, now, path + " @ (" + std::to_string(ob.x) + ", " + std::to_string(ob.y) + ", "
                                         + std::to_string(ob.z) + ")");
        }
    }

} // namespace

TEST(GeometryParity, PandaLinkMeshesAndSpheres) {
    const std::string urdfPath  = kModels + "/franka_description/panda.urdf";
    const std::string meshDir   = kModels + "/franka_description/meshes/Simplified/watertight/";
    const std::string sphereDir = kModels + "/franka_description/spheres/";
    const urdf::ModelInterfaceSharedPtr model = urdf::parseURDFFile(urdfPath);
    ASSERT_TRUE(model) << urdfPath;

    int linksChecked = 0;
    for (const auto& [name, link] : model->links_) {
        if (!link->visual || !link->visual->geometry || link->visual->geometry->type != urdf::Geometry::MESH) {
            continue;
        }
        const std::string urdfFile = static_cast<const urdf::Mesh*>(link->visual->geometry.get())->filename;
        const std::string meshFile = meshDir + "/" + urdfFile.substr(urdfFile.rfind('/') + 1);
        const std::string sphFile  = sphereDir + "/" + name + ".txt";
        if (!fs::exists(meshFile) || !fs::exists(sphFile)) {
            continue; // the old robot only loads links whose files exist (hand / fingers are not part of it)
        }
        const RTCD::meshModel old(link, meshDir, sphereDir); // old robot<DOF>::loadURDF path
        const rtcd::TriangleMesh now = rtcd::loadObjMesh(meshFile);
        expectSameMesh(old, now, name);

        const std::vector<rtcd::Sphere> spheres = rtcd::loadSpheres(sphFile);
        ASSERT_EQ(spheres.size(), old.sphereOrigins.size()) << name;
        ASSERT_EQ(spheres.size(), old.sphereRadii.size()) << name;
        for (std::size_t i = 0; i < spheres.size(); ++i) {
            EXPECT_TRUE(sameFloat3(old.sphereOrigins[i], spheres[i].center)) << name << " sphere " << i;
            EXPECT_EQ(bits(old.sphereRadii[i]), bits(spheres[i].radius)) << name << " sphere " << i;
        }
        ++linksChecked;
    }
    EXPECT_EQ(linksChecked, 8); // panda_link0 .. panda_link7
}

TEST(GeometryParity, ShelfObstacles) {
    checkObstacles(kShelf);
}

TEST(GeometryParity, DenseShelfObstacles) {
    checkObstacles(kDenseShelf);
}

TEST(GeometryParity, ShelfSimpleObstacles) {
    checkObstacles(kShelfSimple);
}

// Scaled and rotated poses are not used by the old scenes but are part of the loadObjMesh contract
// (`vertex *= scale; transformPoint(vertex, Affine3f)`): compare with the old meshModel on a rotated pose.
TEST(GeometryParity, RotatedScaledPose) {
    const std::string path = kModels + "/curobo/Sheep.obj";
    const rtcd::Transform pose = rtcd::Transform::fromXyzRpy(0.1, -0.2, 0.3, 0.3, -0.7, 1.1);
    Eigen::Affine3f affine     = Eigen::Affine3f::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            affine.matrix()(r, c) = pose.m[static_cast<std::size_t>(r * 4 + c)];
        }
    }
    const RTCD::meshModel old(path, affine, 1.7f);
    expectSameMesh(old, rtcd::loadObjMesh(path, pose, 1.7f), "Sheep rotated, scale 1.7");
}
