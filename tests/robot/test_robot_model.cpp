// tests/robot: RobotModel::load (new_unit, no GPU).
//
// The loaded Panda is compared with stored references written from the OLD stack before it was removed
// (tests/data/reference): panda_kinematics.txt (the hand-written pandas.h table) and meshes.txt (link.<name>: the
// old per-link loader RTCD::meshModel(urdfLink, meshDir, sphereDir) that robot<DOF>::loadURDF used, in chain
// order). Format: tests/support/mesh_reference.h.
#include <rtcd/errors.h>
#include <rtcd/robot_model.h>

#include <support/mesh_reference.h>
#include <support/reference.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace rtcd;

namespace {

    const fs::path kUrdf    = "models/franka_description/panda.urdf";
    const fs::path kMeshDir = "models/franka_description/meshes/Simplified/watertight";
    const fs::path kSphDir  = "models/franka_description/spheres";

    RobotDesc pandaDesc(bool spheres = true) {
        RobotDesc d;
        d.description = kUrdf;
        d.mesh.dir    = kMeshDir;
        if (spheres) {
            d.spheres = GeometrySource{kSphDir, "xyzr", {}};
        }
        return d;
    }
    // A temporary directory for synthetic URDF / sphere files.
    struct TempDir {
        fs::path dir;
        TempDir() {
            dir = fs::temp_directory_path() / ("rtcd_robot_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed())
                                               + "_" + ::testing::UnitTest::GetInstance()->current_test_info()->name());
            fs::create_directories(dir);
        }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
        fs::path write(const std::string& name, const std::string& text) const {
            std::ofstream(dir / name) << text;
            return dir / name;
        }
    };

} // namespace

TEST(RobotModel, PandaMatchesTheStoredOldRobotLoader) {
    const auto m = RobotModel::load(pandaDesc());
    ASSERT_EQ(m->dof(), 7u);
    ASSERT_EQ(m->numLinks(), 8u);
    EXPECT_EQ(m->name(), "panda");
    EXPECT_TRUE(m->hasSpheres());
    const auto kin = support::ReferenceFile::load(support::referenceDir() / "panda_kinematics.txt");
    for (std::uint32_t j = 0; j < m->dof(); ++j) {
        const std::vector<float> want = kin.floats("cos_mask." + std::to_string(j));
        ASSERT_EQ(want.size(), 12u);
        EXPECT_EQ(std::memcmp(&m->kinematics().cosMask[j][0], want.data(), 12 * sizeof(float)), 0) << "joint " << j;
    }

    std::vector<float> centers, radii;
    std::vector<std::int32_t> link;
    for (std::uint32_t l = 0; l < m->numLinks(); ++l) {
        const LinkGeometry& g = m->links()[l];
        SCOPED_TRACE(g.name);
        EXPECT_EQ(g.name, "panda_link" + std::to_string(l));
        EXPECT_TRUE(g.mergedLinks.empty());
        // mesh, edges and the link OBB (not collapsed: the old SKIP_BASE collapse is the backends' job)
        support::expectMeshMatchesReference("link." + g.name, g.mesh, g.obb);
        support::expectSpheresMatchReference("link." + g.name, g.spheres);
        EXPECT_EQ(g.meshFile, kMeshDir / (l == 7 ? "link7_nohand.obj" : "link" + std::to_string(l) + ".obj"));
        EXPECT_EQ(g.sphereFile, kSphDir / (g.name + ".txt"));
        for (const Sphere& s : g.spheres) {
            centers.insert(centers.end(), s.center.begin(), s.center.end());
            radii.push_back(s.radius);
            link.push_back(static_cast<std::int32_t>(l));
        }
    }
    // Flat arrays = old sphereCentersTemplate / sphereRadiusTemplate / sphereIdxToLinkIdx: link spheres in chain order.
    EXPECT_EQ(m->totalSpheres(), 62u);
    ASSERT_EQ(m->sphereCenters().size(), centers.size());
    EXPECT_EQ(std::memcmp(m->sphereCenters().data(), centers.data(), centers.size() * sizeof(float)), 0);
    ASSERT_EQ(m->sphereRadii().size(), radii.size());
    EXPECT_EQ(std::memcmp(m->sphereRadii().data(), radii.data(), radii.size() * sizeof(float)), 0);
    EXPECT_TRUE(std::equal(link.begin(), link.end(), m->sphereLink().begin(), m->sphereLink().end()));
}
TEST(RobotModel, JointNamesLimitsAndHostFk) {
    const auto m = RobotModel::load(pandaDesc(false));
    EXPECT_FALSE(m->hasSpheres());
    EXPECT_EQ(m->totalSpheres(), 0u);
    ASSERT_EQ(m->jointNames().size(), 7u);
    EXPECT_EQ(m->jointNames()[3], "panda_joint4");
    EXPECT_EQ(m->lowerLimits()[3], -3.1416f);
    EXPECT_EQ(m->upperLimits()[3], 0.0f);
    EXPECT_EQ(m->upperLimits()[5], 3.8223f);
    for (const auto& l : m->links()) {
        EXPECT_TRUE(l.spheres.empty());
        EXPECT_TRUE(l.sphereFile.empty());
    }

    // q = 0: the well-known Panda zero pose; flange (link 7) at x = 0.088, z = 0.333 + 0.316 + 0.384 = 1.033.
    std::vector<float> q(7, 0.0f), out(8 * 12);
    m->linkPoses(q, out);
    EXPECT_NEAR(out[7 * 12 + 3], 0.088f, 1e-6f);
    EXPECT_NEAR(out[7 * 12 + 7], 0.0f, 1e-6f);
    EXPECT_NEAR(out[7 * 12 + 11], 1.033f, 1e-6f);
    EXPECT_NEAR(out[1 * 12 + 11], 0.333f, 1e-6f);
    // rotation of the flange at q = 0: diag(1, -1, -1)
    EXPECT_NEAR(out[7 * 12 + 0], 1.0f, 1e-6f);
    EXPECT_NEAR(out[7 * 12 + 5], -1.0f, 1e-6f);
    EXPECT_NEAR(out[7 * 12 + 10], -1.0f, 1e-6f);

    std::vector<float> small(12);
    EXPECT_THROW(m->linkPoses(q, small), InvalidArgument);
    EXPECT_THROW(m->linkPoses(std::span<const float>(q).first(6), out), InvalidArgument);
}

TEST(RobotModel, BasePoseGoesIntoTheTable) {
    RobotDesc d = pandaDesc(false);
    d.basePose  = Transform::translation(0.5f, -0.25f, 0.125f);
    const auto m = RobotModel::load(d);
    EXPECT_EQ(m->kinematics().base[3], 0.5f);
    EXPECT_EQ(m->kinematics().base[7], -0.25f);
    EXPECT_EQ(m->kinematics().base[11], 0.125f);
    EXPECT_EQ(m->desc().basePose.m, d.basePose.m);
}

TEST(RobotModel, OverridesAndErrors) {
    {
        // per-link override (relative to dir): link 7 gets link6's mesh
        RobotDesc d = pandaDesc(false);
        d.mesh.files["panda_link7"] = "link6.obj";
        const auto m = RobotModel::load(d);
        EXPECT_EQ(m->links()[7].meshFile, kMeshDir / "link6.obj");
        EXPECT_EQ(m->links()[7].mesh.vertices, m->links()[6].mesh.vertices);
    }
    {
        RobotDesc d = pandaDesc(false);
        d.mesh.format = "stl";
        EXPECT_THROW(RobotModel::load(d), UnsupportedError);
    }
    {
        RobotDesc d   = pandaDesc(true);
        d.spheres->format = "npy";
        EXPECT_THROW(RobotModel::load(d), UnsupportedError);
    }
    {
        RobotDesc d = pandaDesc(true);
        d.spheres->files["panda_link3"] = "no_such_file.txt";
        EXPECT_THROW(RobotModel::load(d), IoError);
    }
    {
        RobotDesc d = pandaDesc(false);
        d.mesh.dir  = "models/no_such_dir";
        EXPECT_THROW(RobotModel::load(d), IoError);
    }
    {
        RobotDesc d   = pandaDesc(false);
        d.description = "models/no_such.urdf";
        EXPECT_THROW(RobotModel::load(d), IoError);
    }
}

// Fixed-joint merging: a "tool" link with geometry attached to link 1 by a fixed joint is merged into link 1
// (vertices transformed, triangles offset, edges recomputed, spheres appended in the link-1 frame).
TEST(RobotModel, FixedLinksAreMerged) {
    TempDir tmp;
    const std::string meshDir = fs::absolute(kMeshDir).generic_string();
    const std::string urdf    = R"(<robot name="m">
  <link name="a"><visual><geometry><mesh filename="x/link0.obj"/></geometry></visual></link>
  <link name="b"><visual><geometry><mesh filename="x/link1.obj"/></geometry></visual></link>
  <link name="tool"><visual><origin xyz="0 0 0.5"/><geometry><mesh filename="x/link2.obj"/></geometry></visual></link>
  <link name="frame"/>
  <joint name="j" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0 0 1"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="f" type="fixed"><parent link="b"/><child link="tool"/><origin xyz="0.25 0 0"/></joint>
  <joint name="g" type="fixed"><parent link="b"/><child link="frame"/><origin xyz="9 9 9"/></joint>
</robot>)";
    RobotDesc d;
    d.description = tmp.write("m.urdf", urdf);
    d.mesh.dir    = meshDir;
    tmp.write("a.txt", "0 0 0 0.1\n");
    tmp.write("b.txt", "# b\n0 0 0.1 0.05\n\n");
    tmp.write("tool.txt", "0.01 0 0 0.02\n");
    d.spheres = GeometrySource{tmp.dir, "xyzr", {}};

    const auto m = RobotModel::load(d);
    ASSERT_EQ(m->numLinks(), 2u);
    const LinkGeometry& b = m->links()[1];
    ASSERT_EQ(b.mergedLinks, std::vector<std::string>{"tool"});
    const TriangleMesh l1 = loadObjMesh(kMeshDir / "link1.obj");
    const TriangleMesh l2 = loadObjMesh(kMeshDir / "link2.obj");
    ASSERT_EQ(b.mesh.vertices.size(), l1.vertices.size() + l2.vertices.size());
    ASSERT_EQ(b.mesh.triangles.size(), l1.triangles.size() + l2.triangles.size());
    EXPECT_EQ(b.mesh.vertices[0], l1.vertices[0]);
    const Vec3 v = b.mesh.vertices[l1.vertices.size()];
    EXPECT_NEAR(v[0], l2.vertices[0][0] + 0.25f, 1e-6f);
    EXPECT_NEAR(v[2], l2.vertices[0][2] + 0.5f, 1e-6f);
    const auto& t = b.mesh.triangles[l1.triangles.size()];
    EXPECT_EQ(t[0], l2.triangles[0][0] + l1.vertices.size());
    EXPECT_EQ(b.mesh.uniqueEdges.size(), l1.uniqueEdges.size() + l2.uniqueEdges.size());
    EXPECT_EQ(b.meshFile, fs::path(meshDir) / "link1.obj");
    EXPECT_EQ(b.obb.center[0], fitObb(b.mesh.vertices).center[0]);

    ASSERT_EQ(b.spheres.size(), 2u);
    EXPECT_NEAR(b.spheres[1].center[0], 0.26f, 1e-6f); // tool sphere moved into the b frame
    EXPECT_EQ(b.spheres[1].radius, 0.02f);
    EXPECT_EQ(m->totalSpheres(), 3u);
    EXPECT_EQ(m->sphereLink()[0], 0);
    EXPECT_EQ(m->sphereLink()[1], 1);
    EXPECT_EQ(m->sphereLink()[2], 1);
    EXPECT_EQ(m->kinematics().oneMask[0][11], 1.0f);
}

TEST(RobotModel, ChainLinkWithoutMeshIsUnsupported) {
    TempDir tmp;
    const std::string urdf = R"(<robot name="m">
  <link name="a"><visual><geometry><mesh filename="link0.obj"/></geometry></visual></link>
  <link name="b"/>
  <joint name="j" type="continuous"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/></joint>
</robot>)";
    RobotDesc d;
    d.description = tmp.write("m.urdf", urdf);
    d.mesh.dir    = kMeshDir;
    EXPECT_THROW(RobotModel::load(d), UnsupportedError);
}

TEST(RobotModel, PrimitiveGeometryIsUnsupported) {
    TempDir tmp;
    const std::string urdf = R"(<robot name="m">
  <link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
  <link name="b"><visual><geometry><mesh filename="link1.obj"/></geometry></visual></link>
  <joint name="j" type="continuous"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/></joint>
</robot>)";
    RobotDesc d;
    d.description = tmp.write("m.urdf", urdf);
    d.mesh.dir    = kMeshDir;
    EXPECT_THROW(RobotModel::load(d), UnsupportedError);
}
