#pragma once
// rtcd/geometry.h -- host-side geometry value types shared by the robot model, the scene and the backends:
// rigid transforms, triangle meshes with their edge sets, spheres (plain data, trivial inline helpers), plus the
// mesh / sphere loaders and the OBB fit of library rtcd_geometry (declared at the end).
//
// Units: meters and radians. Frames: a link's geometry is expressed in that link's frame; obstacles in the world.

#include <rtcd/limits.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace rtcd {

    // Rigid transform, row-major 3x4 [R | t] (same layout as KinematicsTable entries and OptixInstance::transform).
    struct Transform {
        std::array<float, kTransformFloats> m{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

        static constexpr Transform identity() { return Transform{}; }
        static constexpr Transform translation(float x, float y, float z) {
            return Transform{{1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z}};
        }
        // URDF convention: fixed-axis roll (x), pitch (y), yaw (z): R = Rz(yaw) * Ry(pitch) * Rx(roll).
        // Computed in double, rounded to float.
        static Transform fromXyzRpy(double x, double y, double z, double roll, double pitch, double yaw) {
            const double cr = std::cos(roll), sr = std::sin(roll);
            const double cp = std::cos(pitch), sp = std::sin(pitch);
            const double cy = std::cos(yaw), sy = std::sin(yaw);
            return Transform{{static_cast<float>(cy * cp), static_cast<float>(cy * sp * sr - sy * cr),
                static_cast<float>(cy * sp * cr + sy * sr), static_cast<float>(x), static_cast<float>(sy * cp),
                static_cast<float>(sy * sp * sr + cy * cr), static_cast<float>(sy * sp * cr - cy * sr),
                static_cast<float>(y), static_cast<float>(-sp), static_cast<float>(cp * sr),
                static_cast<float>(cp * cr), static_cast<float>(z)}};
        }
        // Unit quaternion (x, y, z, w); normalized here. Computed in double, rounded to float.
        static Transform fromXyzQuat(double x, double y, double z, double qx, double qy, double qz, double qw) {
            const double n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            qx /= n, qy /= n, qz /= n, qw /= n;
            return Transform{{static_cast<float>(1 - 2 * (qy * qy + qz * qz)), static_cast<float>(2 * (qx * qy - qz * qw)),
                static_cast<float>(2 * (qx * qz + qy * qw)), static_cast<float>(x),
                static_cast<float>(2 * (qx * qy + qz * qw)), static_cast<float>(1 - 2 * (qx * qx + qz * qz)),
                static_cast<float>(2 * (qy * qz - qx * qw)), static_cast<float>(y),
                static_cast<float>(2 * (qx * qz - qy * qw)), static_cast<float>(2 * (qy * qz + qx * qw)),
                static_cast<float>(1 - 2 * (qx * qx + qy * qy)), static_cast<float>(z)}};
        }

        // this * other (apply `other` first). Float arithmetic, row by row.
        Transform operator*(const Transform& o) const {
            Transform r;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float v = m[i * 4 + 0] * o.m[0 + j] + m[i * 4 + 1] * o.m[4 + j] + m[i * 4 + 2] * o.m[8 + j];
                    if (j == 3) {
                        v += m[i * 4 + 3];
                    }
                    r.m[i * 4 + j] = v;
                }
            }
            return r;
        }
        std::array<float, 3> apply(const std::array<float, 3>& p) const {
            return {m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3], m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
                m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]};
        }
    };

    using Vec3 = std::array<float, 3>;
    using Edge = std::array<std::uint32_t, 2>; // vertex indices (origin, target): one ray origin -> target

    // Triangle mesh as loaded by the geometry module (OBJ via tinyobjloader, triangulated). The loader must
    // reproduce the old RTCD::meshModel::loadOBJ exactly, because ray sets, ray directions and OBBs feed the
    // goldens:
    //   * vertices deduplicated by exact float equality, numbered in first-use order over the faces
    //     (shapes in file order; within a shape, faces grouped by material id in ascending id order);
    //   * triangles in that same order;
    //   * uniqueEdges: for each triangle (a, b, c) the edges (a,b), (b,c), (c,a) in that order, each undirected
    //     edge kept once (first occurrence and its direction win);
    //   * loopEdges: the consistently oriented edge cover built by the old BFS over triangle adjacency
    //     (loopEdgeIndices); empty and watertight == false if the mesh is not watertight.
    // Vertex positions already include any scale / pose applied at load (obstacles: world = pose * (scale * v);
    // robot links: link frame, URDF visual origin and scale applied).
    struct TriangleMesh {
        std::vector<Vec3> vertices;
        std::vector<std::array<std::uint32_t, 3>> triangles;
        std::vector<Edge> uniqueEdges;
        std::vector<Edge> loopEdges;
        bool watertight = false;

        bool empty() const { return triangles.empty(); }
    };

    struct Sphere {
        Vec3 center{0, 0, 0}; // in the link frame
        float radius = 0;
    };

    // -----------------------------------------------------------------------------------------------------------
    // Geometry I/O and fitting: library rtcd_geometry (src/geometry, Wave 1 c). Host only, no GPU, no Context.
    // These are the seam between rtcd_geometry and its users rtcd_scene (obstacles) and rtcd_robot (link meshes,
    // spheres, link OBBs). Reentrant.
    // -----------------------------------------------------------------------------------------------------------

    // Loads an OBJ file (tinyobjloader, triangulate) exactly as the old RTCD::meshModel::loadOBJ (rules at
    // TriangleMesh): vertices are deduplicated on the positions AS READ from the file; afterwards every vertex
    // becomes pose * (scale * v) (old meshModel(path, transform, scale); the implementation must reproduce the
    // old float results of `vertex *= scale; transformPoint(vertex, Eigen::Affine3f)`); edges and loop edges by
    // computeEdges(). Throws IoError (missing / unreadable / not OBJ) and InvalidArgument (scale <= 0).
    // With identity pose and scale 1 the vertices are returned exactly as read, with no arithmetic at all (this
    // preserves -0.0f coordinates, which the Panda link0 / link5 / link6 meshes contain and the goldens depend on);
    // any other pose / scale applies `vertex *= scale`, then the affine transform, in float, in that order.
    TriangleMesh loadObjMesh(const std::filesystem::path& file, const Transform& pose = Transform{}, float scale = 1.0f);

    // Recomputes mesh.uniqueEdges, mesh.loopEdges and mesh.watertight from mesh.triangles (old loadOBJ edge and
    // loop-edge construction; edges are topology only, vertex positions are not read). Used by loadObjMesh and by
    // rtcd_robot after merging link meshes.
    void computeEdges(TriangleMesh& mesh);

    // Best-fitting OBB of a point set, exactly the old computeBestFittingOBB (RTCD/Meshes/Obb.h). Throws
    // InvalidArgument for an empty set.
    Obb fitObb(std::span<const Vec3> points);

    // Reads a sphere file. format "xyzr" only (UnsupportedError otherwise): UTF-8 text, one "x y z r" per line,
    // empty lines and lines starting with '#' skipped, anything else IoError(file, line); r > 0 required
    // (IoError(file, line)); missing file IoError. File order is kept. An empty file gives an empty vector.
    std::vector<Sphere> loadSpheres(const std::filesystem::path& file, std::string_view format = "xyzr");

} // namespace rtcd
