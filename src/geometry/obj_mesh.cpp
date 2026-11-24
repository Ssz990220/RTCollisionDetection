// loadObjMesh: OBJ loading exactly as the old RTCD::meshModel::loadOBJ + meshModel(path, transform, scale)
// (RTCD/Meshes/mesh.h). See rtcd/geometry.h (TriangleMesh) for the rules reproduced here.

#include <rtcd/errors.h>
#include <rtcd/geometry.h>

#include <Eigen/Geometry>
#include <tiny_obj_loader.h>

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <system_error>

namespace rtcd {

    namespace {

        // Old float3less: lexicographic (x, y, z) with operator<. Note: +0.0f and -0.0f compare equal and are merged
        // (the first one read is kept), exactly as in the old loader.
        struct Vec3Less {
            bool operator()(const Vec3& l, const Vec3& r) const {
                if (l[0] < r[0]) {
                    return true;
                }
                if (l[0] > r[0]) {
                    return false;
                }
                if (l[1] < r[1]) {
                    return true;
                }
                if (l[1] > r[1]) {
                    return false;
                }
                return l[2] < r[2];
            }
        };

        std::uint32_t addVertex(TriangleMesh& mesh, const Vec3& v, std::map<Vec3, std::uint32_t, Vec3Less>& known) {
            const auto it = known.find(v);
            if (it != known.end()) {
                return it->second;
            }
            const auto id = static_cast<std::uint32_t>(mesh.vertices.size());
            known.emplace(v, id);
            mesh.vertices.push_back(v);
            return id;
        }

        // The old loader passed objFile.substr(0, objFile.rfind('/') + 1) as the material base directory and
        // tinyobj's LoadObj(const char* filename, ...) appends the platform separator if it is missing. Reproduced
        // here for the stream overload (which lets us open the file through std::filesystem, wide-path safe).
        std::string materialBaseDir(const std::filesystem::path& file) {
            const std::string generic = file.generic_string();
            std::string baseDir       = generic.substr(0, generic.rfind('/') + 1);
            if (!baseDir.empty()) {
#ifndef _WIN32
                const char dirsep = '/';
#else
                const char dirsep = '\\';
#endif
                if (baseDir.back() != dirsep) {
                    baseDir += dirsep;
                }
            }
            return baseDir;
        }

    } // namespace

    TriangleMesh loadObjMesh(const std::filesystem::path& file, const Transform& pose, float scale) {
        if (!(scale > 0.0f)) {
            throw InvalidArgument("loadObjMesh: scale = " + std::to_string(scale) + " must be > 0 (file "
                                  + file.string() + ")");
        }
        std::error_code ec;
        if (!std::filesystem::is_regular_file(file, ec)) {
            throw IoError("OBJ mesh file does not exist or is not a regular file", file);
        }
        std::ifstream in(file);
        if (!in) {
            throw IoError("cannot open OBJ mesh file", file);
        }

        tinyobj::attrib_t attributes;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        tinyobj::MaterialFileReader materialReader(materialBaseDir(file));
        const bool readOK = tinyobj::LoadObj(&attributes, &shapes, &materials, &warn, &err, &in, &materialReader,
            /* triangulate */ true);
        if (!readOK) {
            throw IoError("cannot parse OBJ mesh: " + err, file);
        }

        TriangleMesh mesh;
        std::map<Vec3, std::uint32_t, Vec3Less> knownVertices;
        const std::size_t numPositions = attributes.vertices.size() / 3;
        auto position                  = [&](const tinyobj::index_t& idx) -> Vec3 {
            if (idx.vertex_index < 0 || static_cast<std::size_t>(idx.vertex_index) >= numPositions) {
                throw IoError("face references vertex index " + std::to_string(idx.vertex_index) + " out of range [0, "
                                  + std::to_string(numPositions) + ")",
                    file);
            }
            const float* p = attributes.vertices.data() + 3 * static_cast<std::size_t>(idx.vertex_index);
            return Vec3{p[0], p[1], p[2]};
        };

        // The old loader kept ONE material-id set across all shapes (declared outside the shape loop) and, per
        // shape, visited the ids in ascending order, taking that shape's faces of each id. Ids not used by the
        // shape select no face, so this equals "per shape, faces grouped by ascending material id".
        std::set<int> materialIDs;
        for (const tinyobj::shape_t& shape : shapes) {
            for (const int faceMatID : shape.mesh.material_ids) {
                materialIDs.insert(faceMatID);
            }
            if (shape.mesh.indices.size() < 3 * shape.mesh.material_ids.size()) {
                throw IoError("shape '" + shape.name + "' has fewer indices than 3 per face", file);
            }
            for (const int materialID : materialIDs) {
                for (std::size_t faceID = 0; faceID < shape.mesh.material_ids.size(); ++faceID) {
                    if (shape.mesh.material_ids[faceID] != materialID) {
                        continue;
                    }
                    const Vec3 v0 = position(shape.mesh.indices[3 * faceID + 0]);
                    const Vec3 v1 = position(shape.mesh.indices[3 * faceID + 1]);
                    const Vec3 v2 = position(shape.mesh.indices[3 * faceID + 2]);

                    const std::uint32_t g0 = addVertex(mesh, v0, knownVertices);
                    const std::uint32_t g1 = addVertex(mesh, v1, knownVertices);
                    const std::uint32_t g2 = addVertex(mesh, v2, knownVertices);
                    mesh.triangles.push_back({g0, g1, g2});
                }
            }
        }
        if (mesh.triangles.empty()) {
            throw IoError("OBJ mesh contains no triangles", file);
        }

        // Identity pose and unit scale: the vertices stay exactly as read. This is the old robot-link path
        // (meshModel(urdfLink, meshDir, sphereDir) applied no transform at all). Running the affine product instead
        // would give equal values but turn -0.0f coordinates into +0.0f (-0 + 0*y + 0*z + 0 = +0), and the Panda
        // link meshes link0 / link5 / link6 contain -0 coordinates.
        if (pose.m == Transform::identity().m && scale == 1.0f) {
            computeEdges(mesh);
            return mesh;
        }

        // Old meshModel(path, transform, scale): `vertex *= scale; transformPoint(vertex, transform)` with an
        // Eigen::Affine3f, reproduced with the same Eigen expression so the float results are identical.
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                transform.matrix()(r, c) = pose.m[static_cast<std::size_t>(r * 4 + c)];
            }
        }
        for (Vec3& v : mesh.vertices) {
            v[0] *= scale;
            v[1] *= scale;
            v[2] *= scale;
            const Eigen::Vector3f p(v[0], v[1], v[2]);
            const Eigen::Vector3f q = transform * p;
            v                       = Vec3{q[0], q[1], q[2]};
        }

        computeEdges(mesh);
        return mesh;
    }

} // namespace rtcd
