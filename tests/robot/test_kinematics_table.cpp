// tests/robot: URDF -> chain -> KinematicsTable (new_unit, no GPU).
//
// OLD-STACK PARITY TEST: includes RTCD/Robot/models/pandas.h (the hand-written Panda masks) as the reference.
// Delete the pandas.h comparison together with the old stack.
#include "robot/urdf_chain.h"

#include <rtcd/errors.h>

#include <Robot/models/pandas.h> // old stack: Dim, baseT, cosMask, sinMask, oneMask, lowerBound, upperBound

#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

using namespace rtcd;
using namespace rtcd::robot_detail;

namespace {

    const std::filesystem::path kPandaUrdf = "models/franka_description/panda.urdf";

    std::string hexf(float v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g (0x%08x)", v, std::bit_cast<std::uint32_t>(v));
        return buf;
    }

    // Number of entries whose bits differ; prints each.
    int compareBits(const char* what, const float* got, const float* want, std::size_t n, std::size_t rowLen = 12) {
        int diffs = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i])) {
                ADD_FAILURE() << what << "[joint " << i / rowLen << "][" << i % rowLen << "]: derived " << hexf(got[i])
                              << ", pandas.h " << hexf(want[i]);
                ++diffs;
            }
        }
        return diffs;
    }

    Chain chainFromString(const std::string& xml) {
        std::shared_ptr<const urdf::ModelInterface> m = urdf::parseURDF(xml);
        if (!m) {
            throw std::runtime_error("test URDF does not parse");
        }
        return buildChain(m, MeshSource::Visual, "test.urdf");
    }

    // Rodrigues rotation in double, for checking the masks of arbitrary axes.
    Affine rotAxis(const std::array<double, 3>& k, double q) {
        const double c = std::cos(q), s = std::sin(q), v = 1 - c;
        const double x = k[0], y = k[1], z = k[2];
        return Affine{c + x * x * v, x * y * v - z * s, x * z * v + y * s, 0, y * x * v + z * s, c + y * y * v,
            y * z * v - x * s, 0, z * x * v - y * s, z * y * v + x * s, c + z * z * v, 0};
    }

} // namespace

// THE make-or-break result of Wave 1 d: the table derived at run time equals pandas.h bit for bit.
TEST(KinematicsTable, PandaUrdfEqualsPandasHBitForBit) {
    const Chain chain         = loadChain(kPandaUrdf);
    const KinematicsTable tab = kinematicsFromChain(chain);
    ASSERT_EQ(tab.dof, static_cast<std::uint32_t>(Dim));
    ASSERT_EQ(tab.numLinks, static_cast<std::uint32_t>(Dim + 1));

    int diffs = 0;
    diffs += compareBits("cosMask", &tab.cosMask[0][0], cosMask.data(), cosMask.size());
    diffs += compareBits("sinMask", &tab.sinMask[0][0], sinMask.data(), sinMask.size());
    diffs += compareBits("oneMask", &tab.oneMask[0][0], oneMask.data(), oneMask.size());
    diffs += compareBits("base", tab.base, baseT.data(), baseT.size());
    // Rows j >= dof must be +0.0 exactly.
    for (std::size_t j = Dim; j < kMaxDof; ++j) {
        for (std::size_t e = 0; e < kTransformFloats; ++e) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(tab.cosMask[j][e]), 0u);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(tab.sinMask[j][e]), 0u);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(tab.oneMask[j][e]), 0u);
        }
    }
    EXPECT_EQ(tab.reserved[0], 0u);
    EXPECT_EQ(tab.reserved[1], 0u);
    EXPECT_EQ(diffs, 0) << "derived Panda table differs from RTCD/Robot/models/pandas.h";
}

// Without the 1e-9 snap the cos(+-pi/2) residues of the URDF's 1.57079632679 would survive as tiny non-zeros:
// documents why the rule exists (open question Q21).
TEST(KinematicsTable, SnapIsNeededForPanda) {
    const Chain chain = loadChain(kPandaUrdf);
    bool anyTiny      = false;
    for (const auto& j : chain.joints) {
        for (double v : j.origin) {
            if (v != 0.0 && std::fabs(v) < kSnapTolerance) {
                anyTiny = true;
            }
        }
    }
    EXPECT_TRUE(anyTiny);
}

TEST(KinematicsTable, PandaChainNamesAndLimits) {
    const Chain chain = loadChain(kPandaUrdf);
    EXPECT_EQ(chain.robotName, "panda");
    ASSERT_EQ(chain.links.size(), 8u);
    ASSERT_EQ(chain.joints.size(), 7u);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(chain.links[i].name, "panda_link" + std::to_string(i));
        EXPECT_EQ(chain.links[i].parts.size(), 1u);
        EXPECT_TRUE(chain.links[i].mergedLinks.empty());
    }
    const double lower[7] = {-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671};
    const double upper[7] = {2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671};
    for (int j = 0; j < 7; ++j) {
        EXPECT_EQ(chain.joints[j].name, "panda_joint" + std::to_string(j + 1));
        EXPECT_FALSE(chain.joints[j].continuous);
        EXPECT_EQ(chain.joints[j].lower, lower[j]);
        EXPECT_EQ(chain.joints[j].upper, upper[j]);
    }
    EXPECT_EQ(chain.rootFixed, kIdentityAffine);
}

TEST(KinematicsTable, MasksReproduceRotationForArbitraryAxis) {
    const Affine A = compose(Affine{1, 0, 0, 0.1, 0, 1, 0, -0.2, 0, 0, 1, 0.3},
        Affine{0.36, 0.48, -0.8, 0, -0.8, 0.6, 0, 0, 0.48, 0.64, 0.6, 0});
    const double n = std::sqrt(1.0 + 4.0 + 9.0);
    const std::array<double, 3> k{1 / n, -2 / n, 3 / n};
    const JointMasks m = jointMasks(A, k);
    for (double q : {-2.5, -1.0, 0.0, 0.3, 1.2, 3.0}) {
        const Affine want = compose(A, rotAxis(k, q));
        for (int e = 0; e < 12; ++e) {
            const double got = std::cos(q) * m.cosMask[e] + std::sin(q) * m.sinMask[e] + m.oneMask[e];
            EXPECT_NEAR(got, want[e], 1e-12) << "q " << q << " entry " << e;
        }
    }
}

namespace {
    const char* kLinkMesh = "<visual><geometry><mesh filename=\"a/b/m.obj\"/></geometry></visual>";
    std::string link(const std::string& name, bool geom = true) {
        return "<link name=\"" + name + "\">" + (geom ? kLinkMesh : "") + "</link>";
    }
    std::string joint(const std::string& name, const std::string& type, const std::string& parent,
        const std::string& child, const std::string& xyz = "0 0 0", const std::string& rpy = "0 0 0",
        const std::string& axis = "0 0 1") {
        std::string s = "<joint name=\"" + name + "\" type=\"" + type + "\"><parent link=\"" + parent
                      + "\"/><child link=\"" + child + "\"/><origin xyz=\"" + xyz + "\" rpy=\"" + rpy + "\"/>";
        if (type != "fixed") {
            s += "<axis xyz=\"" + axis + "\"/>";
        }
        if (type == "revolute" || type == "prismatic") {
            s += "<limit lower=\"-1\" upper=\"2\" effort=\"1\" velocity=\"1\"/>";
        }
        return s + "</joint>";
    }
    std::string robot(const std::string& body) { return "<robot name=\"r\">" + body + "</robot>"; }
} // namespace

TEST(KinematicsTable, FixedLinksAreMergedOrDropped) {
    // world (no geometry) -fixed-> base -rev-> l1 -fixed-> tool (geometry) ; l1 -fixed-> frame (no geometry)
    //   frame -rev-> l2
    const std::string xml = robot(link("world", false) + link("base") + link("l1") + link("tool") + link("frame", false)
                                  + link("l2") + joint("w", "fixed", "world", "base", "0 0 1")
                                  + joint("j1", "revolute", "base", "l1", "0 0 0.5")
                                  + joint("t", "fixed", "l1", "tool", "0.1 0 0")
                                  + joint("f", "fixed", "l1", "frame", "0 0.2 0")
                                  + joint("j2", "continuous", "frame", "l2", "0 0 0.3", "0 0 0", "1 0 0"));
    const Chain c = chainFromString(xml);
    ASSERT_EQ(c.links.size(), 3u);
    ASSERT_EQ(c.joints.size(), 2u);
    EXPECT_EQ(c.links[0].name, "base");
    EXPECT_EQ(c.rootFixed[11], 1.0); // world -> base folded into the base transform
    EXPECT_EQ(c.links[1].name, "l1");
    ASSERT_EQ(c.links[1].mergedLinks.size(), 1u);
    EXPECT_EQ(c.links[1].mergedLinks[0], "tool");
    ASSERT_EQ(c.links[1].parts.size(), 2u);
    EXPECT_EQ(c.links[1].parts[1].inChainLink[3], 0.1);
    EXPECT_EQ(c.links[2].name, "l2");
    // j2's origin includes the fixed frame offset.
    EXPECT_EQ(c.joints[1].origin[7], 0.2);
    EXPECT_EQ(c.joints[1].origin[11], 0.3);
    EXPECT_TRUE(c.joints[1].continuous);
    EXPECT_EQ(c.joints[1].lower, -std::numbers::pi);
    EXPECT_EQ(c.joints[1].upper, std::numbers::pi);
    EXPECT_EQ(c.joints[0].lower, -1.0);
    EXPECT_EQ(c.joints[0].upper, 2.0);

    const KinematicsTable t = kinematicsFromChain(c, Transform::translation(1, 2, 3));
    EXPECT_EQ(t.dof, 2u);
    EXPECT_EQ(t.base[3], 1.0f);
    EXPECT_EQ(t.base[7], 2.0f);
    EXPECT_EQ(t.base[11], 4.0f); // 3 + world->base 1
}

TEST(KinematicsTable, RejectsUnsupportedJointsAndBranches) {
    EXPECT_THROW(chainFromString(robot(link("a") + link("b") + joint("p", "prismatic", "a", "b"))), UnsupportedError);
    EXPECT_THROW(chainFromString(robot(link("a") + link("b") + joint("p", "floating", "a", "b"))), UnsupportedError);
    EXPECT_THROW(chainFromString(robot(link("a") + link("b") + link("c") + joint("j1", "revolute", "a", "b")
                                       + joint("j2", "revolute", "a", "c"))),
        UnsupportedError);
    // a branch hidden behind a fixed joint is still a branch
    EXPECT_THROW(chainFromString(robot(link("a") + link("f", false) + link("b") + link("c")
                                       + joint("j1", "revolute", "a", "b") + joint("x", "fixed", "a", "f")
                                       + joint("j2", "revolute", "f", "c"))),
        UnsupportedError);
    // no actuated joint at all
    EXPECT_THROW(chainFromString(robot(link("a") + link("b") + joint("x", "fixed", "a", "b"))), UnsupportedError);
}

TEST(KinematicsTable, RejectsTooManyJoints) {
    std::string body = link("l0");
    for (int i = 0; i < static_cast<int>(kMaxDof) + 1; ++i) {
        body += link("l" + std::to_string(i + 1))
              + joint("j" + std::to_string(i), "revolute", "l" + std::to_string(i), "l" + std::to_string(i + 1));
    }
    EXPECT_THROW(chainFromString(robot(body)), UnsupportedError);
}

TEST(KinematicsTable, MissingOrBadFileIsIoError) {
    EXPECT_THROW(loadChain("models/does_not_exist.urdf"), IoError);
    EXPECT_THROW(loadChain("CMakeLists.txt"), IoError);
}
