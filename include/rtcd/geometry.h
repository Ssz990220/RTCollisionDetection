#pragma once
// rtcd/geometry.h -- host-side geometry value types shared by the robot model, the scene and the backends:
// rigid transforms, triangle meshes with their edge sets, spheres. Plain data; trivial inline helpers only.
//
// Units: meters and radians. Frames: a link's geometry is expressed in that link's frame; obstacles in the world.

#include <rtcd/limits.h>

#include <array>
#include <cmath>
#include <cstdint>
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

} // namespace rtcd
