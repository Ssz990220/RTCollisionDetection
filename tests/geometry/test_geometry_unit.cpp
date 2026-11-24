// Unit tests of rtcd_geometry without stored references: sphere-file rules, edge construction on small
// hand-made meshes, loader errors, OBB basics. Label new_unit (no GPU).

#include <rtcd/errors.h>
#include <rtcd/geometry.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

namespace fs = std::filesystem;
using namespace rtcd;

namespace {

    // A per-test temporary directory under the system temp folder, removed at the end.
    class TempDir {
    public:
        TempDir() {
            static std::atomic<int> counter{0};
            dir_ = fs::temp_directory_path()
                 / ("rtcd_geometry_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_"
                     + std::to_string(counter++));
            fs::create_directories(dir_);
        }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(dir_, ec);
        }
        fs::path write(const std::string& name, const std::string& content) const {
            const fs::path p = dir_ / name;
            std::ofstream out(p, std::ios::binary);
            out << content;
            return p;
        }
        const fs::path& path() const { return dir_; }

    private:
        fs::path dir_;
    };

    TriangleMesh makeMesh(std::vector<std::array<std::uint32_t, 3>> triangles) {
        TriangleMesh m;
        m.triangles = std::move(triangles);
        computeEdges(m);
        return m;
    }

    // Tetrahedron with consistently oriented faces.
    const std::vector<std::array<std::uint32_t, 3>> kTetra = {{0, 1, 2}, {0, 3, 1}, {1, 3, 2}, {2, 3, 0}};

    std::set<std::pair<std::uint32_t, std::uint32_t>> undirected(const std::vector<Edge>& edges) {
        std::set<std::pair<std::uint32_t, std::uint32_t>> s;
        for (const Edge& e : edges) {
            s.insert({std::min(e[0], e[1]), std::max(e[0], e[1])});
        }
        return s;
    }

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// loadSpheres
// ---------------------------------------------------------------------------------------------------------------

TEST(LoadSpheres, ParsesLinesCommentsBlankAndCrlf) {
    TempDir t;
    const fs::path f = t.write("s.txt", "# header\r\n0.1 0.2 0.3 0.05\r\n\r\n   \n-1 2e-1 3 1.5\n# tail");
    const auto s     = loadSpheres(f);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[0].center[0], 0.1f);
    EXPECT_EQ(s[0].center[1], 0.2f);
    EXPECT_EQ(s[0].center[2], 0.3f);
    EXPECT_EQ(s[0].radius, 0.05f);
    EXPECT_EQ(s[1].center[0], -1.0f);
    EXPECT_EQ(s[1].center[1], 0.2f);
    EXPECT_EQ(s[1].radius, 1.5f);
}

TEST(LoadSpheres, EmptyFileGivesEmptyVector) {
    TempDir t;
    EXPECT_TRUE(loadSpheres(t.write("e.txt", "")).empty());
    EXPECT_TRUE(loadSpheres(t.write("c.txt", "# only a comment\n\n")).empty());
}

TEST(LoadSpheres, MalformedLineIsIoErrorWithLine) {
    TempDir t;
    const fs::path f = t.write("bad.txt", "0 0 0 1\n# c\n0 0 x 1\n");
    try {
        loadSpheres(f);
        FAIL() << "expected IoError";
    } catch (const IoError& e) {
        EXPECT_EQ(e.line(), 3u);
        EXPECT_EQ(e.path(), f);
    }
    EXPECT_THROW(loadSpheres(t.write("short.txt", "0 0 0\n")), IoError);
    EXPECT_THROW(loadSpheres(t.write("long.txt", "0 0 0 1 2\n")), IoError);
}

TEST(LoadSpheres, NonPositiveRadiusIsIoError) {
    TempDir t;
    try {
        loadSpheres(t.write("r.txt", "0 0 0 1\n0 0 0 0\n"));
        FAIL() << "expected IoError";
    } catch (const IoError& e) {
        EXPECT_EQ(e.line(), 2u);
    }
    EXPECT_THROW(loadSpheres(t.write("neg.txt", "0 0 0 -1\n")), IoError);
}

TEST(LoadSpheres, MissingFileAndUnsupportedFormat) {
    TempDir t;
    EXPECT_THROW(loadSpheres(t.path() / "missing.txt"), IoError);
    const fs::path f = t.write("s.txt", "0 0 0 1\n");
    EXPECT_THROW(loadSpheres(f, "xyz"), UnsupportedError);
}

TEST(LoadSpheres, PandaFilesHold62Spheres) {
    std::size_t total = 0;
    for (int i = 0; i <= 7; ++i) {
        const auto s = loadSpheres(fs::path("models/franka_description/spheres") / ("panda_link" + std::to_string(i) + ".txt"));
        EXPECT_FALSE(s.empty()) << i;
        total += s.size();
    }
    EXPECT_EQ(total, 62u);
}

// ---------------------------------------------------------------------------------------------------------------
// computeEdges
// ---------------------------------------------------------------------------------------------------------------

TEST(ComputeEdges, TetrahedronIsWatertight) {
    const TriangleMesh m = makeMesh(kTetra);
    EXPECT_TRUE(m.watertight);
    EXPECT_EQ(m.uniqueEdges.size(), 6u);
    EXPECT_EQ(undirected(m.uniqueEdges).size(), 6u);
    // Loop edges cover every undirected edge; the old BFS may also add an edge in both directions (here 8 directed
    // edges for 6 undirected ones; the exact sets are proven against the old loader in test_geometry_parity).
    EXPECT_EQ(undirected(m.loopEdges), undirected(m.uniqueEdges));
    EXPECT_EQ(m.loopEdges.size(), 8u);
}

TEST(ComputeEdges, UniqueEdgeOrderAndDirectionFirstOccurrence) {
    // Triangles (0,1,2) then (2,1,3): edges (0,1),(1,2),(2,0), then (2,1) is a duplicate of (1,2), (1,3),(3,2).
    const TriangleMesh m = makeMesh({{0, 1, 2}, {2, 1, 3}});
    const std::vector<Edge> expected = {{0, 1}, {1, 2}, {2, 0}, {1, 3}, {3, 2}};
    EXPECT_EQ(m.uniqueEdges, expected);
}

// The old "watertight" check only verifies that the BFS gave every triangle a consistently oriented edge loop.
// The BFS always achieves that (an edge, once ordered, is never removed, and every case of the BFS completes the
// loop of the triangle it visits), so the old loader accepts open meshes too. Reproduced, not "fixed".
TEST(ComputeEdges, OldCheckAcceptsOpenMeshes) {
    const TriangleMesh m = makeMesh({{0, 1, 2}, {2, 1, 3}});
    EXPECT_TRUE(m.watertight);
    // T0 (0,1,2) all forward; T1 (2,1,3) sees 2->1 as backward, so 1->3 and 3->2 are added reversed (3->1, 2->3).
    const std::vector<Edge> loop = {{0, 1}, {1, 2}, {2, 0}, {3, 1}, {2, 3}};
    EXPECT_EQ(m.loopEdges, loop);
    const TriangleMesh one = makeMesh({{0, 1, 2}});
    EXPECT_TRUE(one.watertight);
    const std::vector<Edge> loop1 = {{0, 1}, {1, 2}, {2, 0}};
    EXPECT_EQ(one.loopEdges, loop1);
}

TEST(ComputeEdges, NeighbourFollowsSharedEdgeDirection) {
    // T1 = (0,1,3) meets the ordered edge 0->1 as FORWARD, so its two new edges are added forward (1->3, 3->0).
    const TriangleMesh m = makeMesh({{0, 1, 2}, {0, 1, 3}});
    EXPECT_TRUE(m.watertight);
    const std::vector<Edge> loop = {{0, 1}, {1, 2}, {2, 0}, {1, 3}, {3, 0}};
    EXPECT_EQ(m.loopEdges, loop);
}

TEST(ComputeEdges, RecomputeReplacesPreviousResults) {
    TriangleMesh m = makeMesh(kTetra);
    m.triangles.pop_back();
    computeEdges(m);
    EXPECT_EQ(m.uniqueEdges.size(), 6u);
    EXPECT_EQ(undirected(m.loopEdges), undirected(m.uniqueEdges));
    m.triangles.clear();
    computeEdges(m);
    EXPECT_TRUE(m.uniqueEdges.empty());
    EXPECT_FALSE(m.watertight);
}

// ---------------------------------------------------------------------------------------------------------------
// loadObjMesh
// ---------------------------------------------------------------------------------------------------------------

namespace {
    const char* kTetraObj = "# tetra\n"
                            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                            "v 0 0 0\n" // duplicate position: must be welded with vertex 1
                            "f 1 2 3\nf 5 4 2\nf 2 4 3\nf 3 4 1\n";
}

TEST(LoadObjMesh, WeldsAndOrdersVertices) {
    TempDir t;
    const TriangleMesh m = loadObjMesh(t.write("tetra.obj", kTetraObj));
    ASSERT_EQ(m.vertices.size(), 4u);
    EXPECT_EQ(m.vertices[0], (Vec3{0, 0, 0}));
    EXPECT_EQ(m.vertices[1], (Vec3{1, 0, 0}));
    EXPECT_EQ(m.vertices[2], (Vec3{0, 1, 0}));
    EXPECT_EQ(m.vertices[3], (Vec3{0, 0, 1}));
    const std::vector<std::array<std::uint32_t, 3>> tris = {{0, 1, 2}, {0, 3, 1}, {1, 3, 2}, {2, 3, 0}};
    EXPECT_EQ(m.triangles, tris);
    EXPECT_TRUE(m.watertight);
    EXPECT_EQ(m.uniqueEdges.size(), 6u);
}

TEST(LoadObjMesh, AppliesScaleThenPose) {
    TempDir t;
    const fs::path f     = t.write("tetra.obj", kTetraObj);
    const Transform pose = Transform::fromXyzRpy(1, 2, 3, 0, 0, 1.5707963267948966);
    const TriangleMesh m = loadObjMesh(f, pose, 2.0f);
    ASSERT_EQ(m.vertices.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        const Vec3 raw = loadObjMesh(f).vertices[i];
        const Vec3 exp = pose.apply({raw[0] * 2.0f, raw[1] * 2.0f, raw[2] * 2.0f});
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(m.vertices[i][k], exp[k], 1e-6f);
        }
    }
    EXPECT_NEAR(m.vertices[1][0], 1.0f, 1e-6f); // (2,0,0) rotated by +90 deg about z = (0,2,0), + (1,2,3)
    EXPECT_NEAR(m.vertices[1][1], 4.0f, 1e-6f);
}

TEST(LoadObjMesh, Errors) {
    TempDir t;
    EXPECT_THROW(loadObjMesh(t.path() / "missing.obj"), IoError);
    EXPECT_THROW(loadObjMesh(t.path()), IoError); // a directory
    const fs::path f = t.write("tetra.obj", kTetraObj);
    EXPECT_THROW(loadObjMesh(f, Transform{}, 0.0f), InvalidArgument);
    EXPECT_THROW(loadObjMesh(f, Transform{}, -1.0f), InvalidArgument);
    EXPECT_THROW(loadObjMesh(t.write("empty.obj", "# nothing\n")), IoError);
    EXPECT_THROW(loadObjMesh(t.write("bad.obj", "v 0 0 0\nf 1 2 3\n")), IoError);
}

// ---------------------------------------------------------------------------------------------------------------
// fitObb
// ---------------------------------------------------------------------------------------------------------------

TEST(FitObb, AxisAlignedBox) {
    std::vector<Vec3> pts;
    for (float x : {-2.0f, 2.0f}) {
        for (float y : {-1.0f, 1.0f}) {
            for (float z : {-0.5f, 0.5f}) {
                pts.push_back({x + 10, y, z});
            }
        }
    }
    const Obb o = fitObb(pts);
    EXPECT_NEAR(o.center[0], 10.0f, 1e-5f);
    EXPECT_NEAR(o.center[1], 0.0f, 1e-5f);
    EXPECT_NEAR(o.center[2], 0.0f, 1e-5f);
    EXPECT_NEAR(o.halfSize[0], 2.0f, 1e-5f); // descending variance: x, y, z
    EXPECT_NEAR(o.halfSize[1], 1.0f, 1e-5f);
    EXPECT_NEAR(o.halfSize[2], 0.5f, 1e-5f);
    EXPECT_NEAR(std::abs(o.axes[0][0]), 1.0f, 1e-5f);
    EXPECT_NEAR(std::abs(o.axes[1][1]), 1.0f, 1e-5f);
    EXPECT_NEAR(std::abs(o.axes[2][2]), 1.0f, 1e-5f);
    // right-handed
    const float* a = o.axes[0];
    const float* b = o.axes[1];
    const float* c = o.axes[2];
    const float det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    EXPECT_GT(det, 0.0f);
}

TEST(FitObb, EmptyThrows) {
    EXPECT_THROW(fitObb({}), InvalidArgument);
}
