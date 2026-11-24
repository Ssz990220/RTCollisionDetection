#pragma once
// support/mesh_reference.h -- header-only GoogleTest helpers comparing loaded meshes / spheres with the stored mesh
// summaries of tests/data/reference/meshes.txt (written from the OLD loader RTCD::meshModel before the old stack
// was removed; see tests/data/README.md). Used by tests/geometry and tests/robot.
//
// Per mesh label L (link.<urdf link name>, obstacle.<file>@<x>,<y>,<z>, rotated.<file>):
//   L.vertices     <count> <FNV-1a of the float xyz triples>
//   L.triangles    <count> <FNV-1a of the uint32 index triples>
//   L.unique_edges <count> <FNV-1a of the uint32 pairs>
//   L.loop_edges   <count> <FNV-1a of the uint32 pairs>
//   L.obb          15 hex floats (center, half sizes, axes; the old OBB fit of the vertices)
//   L.spheres      <count> <FNV-1a of the center xyz floats> <FNV-1a of the radii>   (links only)
#include <support/reference.h>

#include <rtcd/geometry.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdio>
#include <string>
#include <vector>

namespace rtcd::support {

    // "obstacle.<file>@<x>,<y>,<z>" (%g): the label of an obstacle mesh loaded at a translation, scale 1.
    inline std::string obstacleLabel(const std::string& file, float x, float y, float z) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "@%g,%g,%g", x, y, z);
        return "obstacle." + file + buf;
    }

    inline const ReferenceFile& meshReference() {
        static const ReferenceFile f = ReferenceFile::load(referenceDir() / "meshes.txt");
        return f;
    }

    template <typename T>
    void expectCountAndHash(const ReferenceFile& ref, const std::string& key, const std::vector<T>& v) {
        ASSERT_TRUE(ref.has(key)) << key << " missing in " << ref.path();
        EXPECT_EQ(v.size(), ref.integer(key, 0)) << key;
        EXPECT_EQ(hex64(fnv1a(v)), hex64(ref.hash(key, 1))) << key;
    }

    inline void expectMeshMatchesReference(const std::string& label, const TriangleMesh& mesh, const Obb& obb) {
        SCOPED_TRACE(label);
        const ReferenceFile& ref = meshReference();
        expectCountAndHash(ref, label + ".vertices", mesh.vertices);
        expectCountAndHash(ref, label + ".triangles", mesh.triangles);
        expectCountAndHash(ref, label + ".unique_edges", mesh.uniqueEdges);
        expectCountAndHash(ref, label + ".loop_edges", mesh.loopEdges);
        // The old loader threw for non-watertight meshes, so every stored mesh is watertight.
        EXPECT_TRUE(mesh.watertight);
        const std::vector<float> want = ref.floats(label + ".obb");
        ASSERT_EQ(want.size(), 15u);
        const float* got = reinterpret_cast<const float*>(&obb);
        for (int i = 0; i < 15; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(want[i]))
                << "obb[" << i << "]: " << got[i] << " vs stored " << want[i];
        }
    }

    inline void expectSpheresMatchReference(const std::string& label, const std::vector<Sphere>& spheres) {
        SCOPED_TRACE(label);
        const ReferenceFile& ref = meshReference();
        const std::string key    = label + ".spheres";
        ASSERT_TRUE(ref.has(key)) << key;
        std::vector<float> centers, radii;
        for (const Sphere& s : spheres) {
            centers.insert(centers.end(), s.center.begin(), s.center.end());
            radii.push_back(s.radius);
        }
        EXPECT_EQ(spheres.size(), ref.integer(key, 0));
        EXPECT_EQ(hex64(fnv1a(centers)), hex64(ref.hash(key, 1))) << "sphere centers";
        EXPECT_EQ(hex64(fnv1a(radii)), hex64(ref.hash(key, 2))) << "sphere radii";
    }

} // namespace rtcd::support
