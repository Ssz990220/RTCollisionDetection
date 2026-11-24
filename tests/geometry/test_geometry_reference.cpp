// rtcd_geometry vs the stored mesh summaries (tests/data/reference/meshes.txt, written from the OLD loader
// RTCD::meshModel before the old stack was removed). Host only (label new_unit). Compared exactly (counts, FNV-1a
// hashes of the vertex / triangle / edge arrays, OBB float bits):
//   * every Panda link mesh, resolved as the robot loader does (mesh directory + file name of the URDF visual mesh),
//     loaded with loadObjMesh(file), and its sphere file with loadSpheres;
//   * every obstacle of the three shelf scenes (configs/scenes/shelf*.json), loadObjMesh(path, translation, 1);
//   * one rotated / scaled pose (part of the loadObjMesh contract, unused by the scenes);
//   * computeEdges on the loaded triangles reproduces the edges (idempotent), as rtcd_robot uses it.
#include <rtcd/geometry.h>
#include <rtcd/json_config.h>

#include <support/mesh_reference.h>
#include <support/paths.h>

#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace sup = rtcd::support;

namespace {

    const std::string kModels = "models";

    void expectSameEdgesAfterRecompute(const rtcd::TriangleMesh& now) {
        rtcd::TriangleMesh again;
        again.vertices  = now.vertices;
        again.triangles = now.triangles;
        rtcd::computeEdges(again);
        EXPECT_EQ(again.uniqueEdges, now.uniqueEdges);
        EXPECT_EQ(again.loopEdges, now.loopEdges);
        EXPECT_EQ(again.watertight, now.watertight);
    }

    void checkScene(const char* sceneFile, std::size_t expectedObstacles) {
        const rtcd::SceneDesc scene = rtcd::loadSceneDesc(sup::configsDir() / "scenes" / sceneFile);
        ASSERT_EQ(scene.obstacles.size(), expectedObstacles);
        const fs::path models = fs::weakly_canonical(fs::path(kModels));
        for (const rtcd::ObstacleDesc& ob : scene.obstacles) {
            const std::string rel = fs::relative(fs::weakly_canonical(ob.mesh), models).generic_string();
            const auto& m         = ob.pose.m;
            // the shelf scenes place obstacles by translation only, scale 1
            ASSERT_EQ(m[0], 1.0f);
            ASSERT_EQ(m[5], 1.0f);
            ASSERT_EQ(m[10], 1.0f);
            ASSERT_EQ(ob.scale, 1.0f);
            const rtcd::TriangleMesh now = rtcd::loadObjMesh(ob.mesh, ob.pose, ob.scale);
            sup::expectMeshMatchesReference(sup::obstacleLabel(rel, m[3], m[7], m[11]), now, rtcd::fitObb(now.vertices));
            expectSameEdgesAfterRecompute(now);
        }
    }

} // namespace

TEST(GeometryReference, PandaLinkMeshesAndSpheres) {
    const std::string urdfPath  = kModels + "/franka_description/panda.urdf";
    const std::string meshDir   = kModels + "/franka_description/meshes/Simplified/watertight";
    const std::string sphereDir = kModels + "/franka_description/spheres";
    const urdf::ModelInterfaceSharedPtr model = urdf::parseURDFFile(urdfPath);
    ASSERT_TRUE(model) << urdfPath;
    int linksChecked = 0;
    for (const auto& [name, link] : model->links_) {
        if (!sup::meshReference().has("link." + name + ".vertices")) {
            continue; // hand / fingers: not part of the robot model
        }
        ASSERT_TRUE(link->visual && link->visual->geometry && link->visual->geometry->type == urdf::Geometry::MESH) << name;
        const std::string urdfFile = static_cast<const urdf::Mesh*>(link->visual->geometry.get())->filename;
        const fs::path meshFile    = fs::path(meshDir) / fs::path(urdfFile).filename();
        const rtcd::TriangleMesh now = rtcd::loadObjMesh(meshFile);
        sup::expectMeshMatchesReference("link." + name, now, rtcd::fitObb(now.vertices));
        expectSameEdgesAfterRecompute(now);
        sup::expectSpheresMatchReference("link." + name, rtcd::loadSpheres(fs::path(sphereDir) / (name + ".txt")));
        ++linksChecked;
    }
    EXPECT_EQ(linksChecked, 8); // panda_link0 .. panda_link7
}

TEST(GeometryReference, ShelfObstacles) {
    checkScene("shelf.json", 8);
}

TEST(GeometryReference, DenseShelfObstacles) {
    checkScene("shelf_dense.json", 11);
}

TEST(GeometryReference, ShelfSimpleObstacles) {
    checkScene("shelf_simple.json", 3);
}

TEST(GeometryReference, RotatedScaledPose) {
    const rtcd::Transform pose   = rtcd::Transform::fromXyzRpy(0.1, -0.2, 0.3, 0.3, -0.7, 1.1);
    const rtcd::TriangleMesh now = rtcd::loadObjMesh(kModels + "/curobo/Sheep.obj", pose, 1.7f);
    sup::expectMeshMatchesReference("rotated.curobo/Sheep.obj", now, rtcd::fitObb(now.vertices));
}
