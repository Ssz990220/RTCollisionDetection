#pragma once
#if defined(_WIN32)
#include <Eigen/Geometry>
#include <Eigen/core>
#elif defined(__linux__)
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#endif
#include <iostream>
#include <optix_stubs.h>
#include <queue>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#undef near
#undef far
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>
#include <vector>
#define TINYOBJLOADER_IMPLEMENTATION
#include <Meshes/Obb.h>
#include <Meshes/meshTypes.h>
#include <Utils/CUDABuffer.h>
#include <Utils/cuUtils.cuh>
#include <Utils/optix7.h>
#include <filesystem>
#include <sys/stat.h>
#include <tiny_obj_loader.h>


namespace std {
    template <>
    struct hash<int2> {
        std::size_t operator()(const int2& k) const { return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1); }
    };
} // namespace std

namespace RTCD {

    enum class edgeType : int { BACKWARD = -1, UNDEF = 0, FORWARD = 1, BIDIRECT = 2 };

    // Function to check if all edges are of the same type (forward or backward)
    static inline bool doesLoop(int& forwardCount, int& backwardCount) {
        return (forwardCount == 3 || backwardCount == 3);
    }

    // Function to check if two edges are of the same type (forward or backward)
    static inline bool doesAgree(int& forwardCount, int& backwardCount) {
        return (forwardCount == 2 || backwardCount == 2);
    }

    // Function to find the edge that breaks the rule
    static inline int findRuleBreaker(
        const std::array<edgeType, 3>& edges, const int forwardCount, const int backwardCount) {
        if (forwardCount == backwardCount) { // choose the first direction that is not BIDIRECT
            for (int i = 0; i < 3; i++) {
                if (edges[i] != edgeType::BIDIRECT) {
                    return i;
                }
            }
        }
        if (forwardCount < backwardCount) { // the forward one is the rule breaker
            for (int i = 0; i < 3; i++) {
                if (edges[i] == edgeType::FORWARD) {
                    return i;
                }
            }
        } else { // the backward one is the rule breaker
            for (int i = 0; i < 3; i++) {
                if (edges[i] == edgeType::BACKWARD) {
                    return i;
                }
            }
        }
    }

    struct TriangleMesh {
        ~TriangleMesh() = default;

        // Vertices for the mesh
        std::vector<float3> vertices;

        // we don't need this for collision detection
        // std::vector<float3> normals;

        // Indices for faces, group by int3, users should be careful not using duplicated triangle
        std::vector<int3> indices;

        // Indices for unique edges, group by int2
        std::vector<int2> uniqueEdgeIndices;

        std::unordered_map<int2, std::set<int>> edge_to_triangles;

        std::vector<int2> loopEdgeIndices;
    };

    struct float3less {
        bool operator()(const float3& lhs, const float3& rhs) const {
            if (lhs.x < rhs.x) {
                return true;
            }
            if (lhs.x > rhs.x) {
                return false;
            }
            if (lhs.y < rhs.y) {
                return true;
            }
            if (lhs.y > rhs.y) {
                return false;
            }
            if (lhs.z < rhs.z) {
                return true;
            }
            if (lhs.z > rhs.z) {
                return false;
            }
            return false;
        }
    };

    static inline void transformPoint(float3& point, const Eigen::Affine3f& transform) {
        Eigen::Vector3f homogeneousPoint(point.x, point.y, point.z);
        Eigen::Vector3f transformedPoint = transform * homogeneousPoint;
        point.x                          = transformedPoint[0];
        point.y                          = transformedPoint[1];
        point.z                          = transformedPoint[2];
    }

    class meshModel {
    public:
        meshModel() { bodyTransform.setIdentity(); }
        meshModel(const std::vector<float3>& vertices, const std::vector<int3>& indices,
            Eigen::Affine3f baseTransform = Eigen::Affine3f::Identity());
        meshModel(const std::string_view meshPath, float scale = 1);
        meshModel(const std::string_view meshPath, Eigen::Affine3f baseTransform, float scale = 1);
        meshModel(urdf::LinkConstSharedPtr urdfLink, const std::string_view meshDir, const std::string_view sphereDir);

        // Transforms
        Eigen::Matrix<float, 3, 4, Eigen::RowMajor> bodyTransform;
        // sutil::Aabb bounds;

        // Optix GAS
        OptixTraversableHandle traversableHandle;
        CUDABuffer asBuffer;
        bool wrapped        = false;
        bool sphereWrapped  = false;
        bool uploaded       = false;
        bool sphereUploaded = false;
        CUDABuffer vertexDeviceBuffer;
        CUDABuffer indexDeviceBuffer;
        CUDABuffer edgeDeviceBuffer;
        OptixBuildInput triangleInput = {};
        CUDABuffer localTempBuffer;
        // Geometry data
        TriangleMesh mesh;
        OBB obb;
        OBB sphereOBB;
        std::vector<float3> sphereOrigins; // Origins for spheres wrap the link
        CUDABuffer sphereDeviceBuffer;
        std::vector<float> sphereRadii; // Radii for spheres wrap the link
        CUDABuffer sphereRadiiDeviceBuffer;
        size_t sphereCount;

        // Functions
        // Wrap meshes to Optix GAS
        void uploadToDevice();
        void uploadSphereToDevice();
        // template <bool compact>
        void buildGAS(OptixDeviceContext optixContext, CUDABuffer& tempBuffer);

        template <BuildType BUILD>
        OptixTraversableHandle buildGAS(const OptixDeviceContext optixContext, CUDABuffer& tempBuffer,
            const unsigned int buildFlags, const unsigned int inputFlags);

        template <BuildType BUILD>
        OptixTraversableHandle buildSphereGAS(const OptixDeviceContext optixContext, CUDABuffer& tempBuffer,
            const unsigned int buildFlags, const unsigned int inputFlags);
        // template <bool compact>
        // void buildGAS(OptixDeviceContext optixContext);
        void loadOBJ(const std::string_view objFile);
        // Aabb getAABB() const { return aabb; }
        OBB getOBB() const { return obb; }
        OBB getSphereOBB() const { return sphereOBB; }

    private:
        void loadSphers(const std::string sphereFiles);
        // void loadDummySpheres();
    };
} // namespace RTCD

namespace RTCD {
    // TODO: Make this a hash function
    static inline const int hashIndexPair(const int x, const int y) noexcept {
        int cantorHash = ((x + y) * (x + y + 1) / 2) + y;
        return cantorHash;
    }

    static inline void addEdge(
        TriangleMesh& mesh, const int idx0, const int idx1, std::unordered_set<int>& knownEdges) {
        if (knownEdges.contains(hashIndexPair(idx1, idx0)) || knownEdges.contains(hashIndexPair(idx0, idx1))) {
            return;
        }
        knownEdges.insert(hashIndexPair(idx1, idx0));

        mesh.uniqueEdgeIndices.push_back(make_int2(idx0, idx1));
    }

    // Add unique edges to the mesh by traversing the triangles
    static void addTriangleEdges(
        TriangleMesh& mesh, const int idx0, const int idx1, const int idx2, std::unordered_set<int>& knownEdges) {
        addEdge(mesh, idx0, idx1, knownEdges);
        addEdge(mesh, idx1, idx2, knownEdges);
        addEdge(mesh, idx2, idx0, knownEdges);
    }

    static inline int addVertex(TriangleMesh& mesh, // Mesh to add to
        float3 vertex, std::map<float3, int, float3less>& knownVertices) // Map of unique idx to vertices ID
    {
        if (knownVertices.find(vertex) != knownVertices.end()) {
            return knownVertices[vertex];
        }

        int newID             = (int) mesh.vertices.size(); // Add to the end of the list
        knownVertices[vertex] = newID; // Remember that we know this vertices

        mesh.vertices.push_back(vertex); // Locate the coord of current vertices

        return newID;
    }

    meshModel::meshModel(
        const std::vector<float3>& vertices, const std::vector<int3>& indices, Eigen::Affine3f baseTransform) {
        mesh.vertices = vertices;
        mesh.indices  = indices;
        bodyTransform = baseTransform.affine();
        for (auto& vertex : mesh.vertices) {
            transformPoint(vertex, baseTransform);
        }

        std::unordered_set<int> knownEdges;
        for (auto& idx : indices) {
            addTriangleEdges(mesh, idx.x, idx.y, idx.z, knownEdges);
        }

        obb = computeBestFittingOBB(mesh.vertices);
    }

    meshModel::meshModel(const std::string_view meshPath, float scale) {
        loadOBJ(meshPath);
        for (auto& vertex : mesh.vertices) {
            vertex *= scale;
        }
        obb = computeBestFittingOBB(mesh.vertices);
    }

    meshModel::meshModel(const std::string_view meshPath, Eigen::Affine3f baseTransform, float scale) {
        loadOBJ(meshPath);
        bodyTransform = baseTransform.affine();
        for (auto& vertex : mesh.vertices) {
            vertex *= scale;
            transformPoint(vertex, baseTransform);
        }
        // for (auto& normal : mesh.normals) {
        //     transformPoint(normal, baseTransform);
        // }

        obb = computeBestFittingOBB(mesh.vertices);
    }

    meshModel::meshModel(
        urdf::LinkConstSharedPtr urdfLink, const std::string_view meshDir, const std::string_view sphereDir) {
        const std::string_view name = std::string_view(urdfLink->name);

        // if (!(meshDir.empty()) && boost::filesystem::exists(meshDir)) {
        if (!meshDir.empty()) {
            const std::string meshTempPath = static_cast<urdf::Mesh*>(urdfLink->visual->geometry.get())->filename;
            const size_t pos               = meshTempPath.rfind("/");
            const std::string meshName     = meshTempPath.substr(pos + 1);
            const std::string meshPath     = std::string(meshDir) + "/" + meshName;

            const std::filesystem::path p(meshPath);
            if (!std::filesystem::exists(p)) {
                std::cout << "Mesh file " << meshPath << " does not exist!" << std::endl;
                exit(1);
            }

            loadOBJ(meshPath);
            // loadDummySpheres();
            std::string spherePath = std::string(sphereDir) + "/" + name.data() + ".txt";

            loadSphers(spherePath);
            sphereCount = sphereOrigins.size();

            sphereOBB = computeBestFittingOBB(sphereOrigins);
            // find the biggest sphere radii
            float maxRadii = 0;
            for (auto r : sphereRadii) {
                maxRadii = std::max(maxRadii, r);
            }
            sphereOBB.halfSize += make_float3(maxRadii, maxRadii, maxRadii);
        } else {
            std::cout << "Mesh file folder does not exist!" << std::endl;
            exit(1);
        }

        obb = computeBestFittingOBB(mesh.vertices);
    }

    void meshModel::loadOBJ(const std::string_view objFile) {
        const std::string_view modelDir = objFile.substr(0, objFile.rfind('/') + 1);

        tinyobj::attrib_t attributes;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err = "";

        bool readOK = tinyobj::LoadObj(&attributes, &shapes, &materials, &err, &err, objFile.data(), modelDir.data(),
            /* triangulate */ true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from " + std::string(objFile) + " : " + err);
        } else {
            // std::cout << "Done loading obj file - found " << shapes.size() << std::endl;
        }

        const float3* vertex_array = reinterpret_cast<float3*>(attributes.vertices.data()); // Global vertices array
        std::map<float3, int, float3less> knownVertices; // Map of unique idx to global vertices ID
        std::unordered_set<int> knownEdges; // Set of unique edges, actually should be unordered_set<int2>

        // All Shapes are stored in one TriangleMesh object, even though they may not be connected
        // This is to avoid having to create multiple GASes & multiple VertexBuffers
        // This is not a problem for us, since we only want one mesh per link in robot

        // Vertices are stored on faces, so we need to iterate over all faces
        // Faces are painted with different materials, so we need to iterate over all materials

        std::set<int> materialIDs; // Find all materials used in this shape
        for (int shapeID = 0; shapeID < (int) shapes.size(); shapeID++) { // For each shape in the obj file
            tinyobj::shape_t& shape = shapes[shapeID]; // Get the shape

            for (auto faceMatID : shape.mesh.material_ids) {
                materialIDs.insert(faceMatID);
            }

            for (int materialID : materialIDs) { // For each material used in this shape

                for (int faceID = 0; faceID < shape.mesh.material_ids.size();
                    faceID++) { // go through all faces in the shape
                    if (shape.mesh.material_ids[faceID] != materialID) {
                        continue; // only consider faces with the current material
                    }
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    float3 v0             = vertex_array[idx0.vertex_index];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    float3 v1             = vertex_array[idx1.vertex_index];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];
                    float3 v2             = vertex_array[idx2.vertex_index];

                    int globIdx0 = addVertex(mesh, v0, knownVertices); // Here we track a global idx for each vertex
                    int globIdx1 = addVertex(mesh, v1, knownVertices);
                    int globIdx2 = addVertex(mesh, v2, knownVertices);

                    mesh.indices.push_back(make_int3(globIdx0, globIdx1, globIdx2));

                    int2 edge1 = globIdx0 < globIdx1 ? make_int2(globIdx0, globIdx1) : make_int2(globIdx1, globIdx0);
                    int2 edge2 = globIdx1 < globIdx2 ? make_int2(globIdx1, globIdx2) : make_int2(globIdx2, globIdx1);
                    int2 edge3 = globIdx2 < globIdx0 ? make_int2(globIdx2, globIdx0) : make_int2(globIdx0, globIdx2);

                    mesh.edge_to_triangles[edge1].insert(mesh.indices.size() - 1);
                    mesh.edge_to_triangles[edge2].insert(mesh.indices.size() - 1);
                    mesh.edge_to_triangles[edge3].insert(mesh.indices.size() - 1);

                    addTriangleEdges(mesh, globIdx0, globIdx1, globIdx2, knownEdges);
                }
            }
        }
        auto start = std::chrono::high_resolution_clock::now();
        // true for from small index to big, false for from high to big
        std::unordered_set<int2> ordered_edges;
        std::set<int> processed_triangles;

        bool allAssigned = false;
        while (!allAssigned) {
            // loop over all triangles and check if it is in the processed set
            // if not push it to the open set and start the incremental while loop
            // if all triangles are processed, set all Assigned to true

            std::queue<int> open_triangles;

            for (int i = 0; i < mesh.indices.size(); ++i) {
                if (processed_triangles.find(i) == processed_triangles.end()) {
                    open_triangles.push(i);
                    break;
                }
            }

            if (open_triangles.empty()) {
                allAssigned = true;
                break;
            }

            while (!open_triangles.empty()) {
                int current = open_triangles.front();
                open_triangles.pop();

                if (processed_triangles.find(current) != processed_triangles.end()) {
                    continue;
                }

                // retrive the triangle
                int3 triangle = mesh.indices[current];

                std::array<int, 4> forwardLoop = {triangle.x, triangle.y, triangle.z, triangle.x};
                std::array<edgeType, 3> edges;
                std::vector<int> undefined_ids;
                int forwardCount  = 0;
                int backwardCount = 0;
                int bidirectCount = 0;

                int n_ordered_edges = 0;

                for (int i = 0; i < 3; ++i) {
                    int2 edge  = make_int2(forwardLoop[i], forwardLoop[i + 1]);
                    int2 edger = make_int2(forwardLoop[i + 1], forwardLoop[i]);

                    bool isForward  = ordered_edges.find(edge) != ordered_edges.end();
                    bool isBackward = ordered_edges.find(edger) != ordered_edges.end();

                    if (isForward && isBackward) {
                        edges[i] = edgeType::BIDIRECT;
                        forwardCount++;
                        backwardCount++;
                        bidirectCount++;
                        n_ordered_edges++;
                    } else if (isForward) {
                        edges[i] = edgeType::FORWARD;
                        forwardCount++;
                        n_ordered_edges++;
                    } else if (isBackward) {
                        edges[i] = edgeType::BACKWARD;
                        backwardCount++;
                        n_ordered_edges++;
                    } else {
                        edges[i] = edgeType::UNDEF;
                        undefined_ids.push_back(i);
                    }
                }

                switch (n_ordered_edges) {
                case 3: // all edges are assigned
                    {
                        bool isLoop = doesLoop(forwardCount, backwardCount);

                        if (!isLoop) {
                            int ruleBreaker = findRuleBreaker(edges, forwardCount, backwardCount);
                            if (edges[ruleBreaker] == edgeType::FORWARD) {
                                int2 newEdge = make_int2(forwardLoop[ruleBreaker + 1], forwardLoop[ruleBreaker]);
                                ordered_edges.insert(newEdge);
                            } else {
                                int2 newEdge = make_int2(forwardLoop[ruleBreaker], forwardLoop[ruleBreaker + 1]);
                                ordered_edges.insert(newEdge);
                            }
                        }

                        processed_triangles.insert(current);
                    }
                    break;

                case 2: // two edges are assigned
                    {
                        bool isAgree = doesAgree(forwardCount, backwardCount);

                        int edgeId = undefined_ids[0]; // only one undefined edge
                        edgeType loopDirc;

                        if (!isAgree) { // has to be one forward and one backward
                            for (int i = 0; i < 3; ++i) {
                                if (edges[i] != edgeType::UNDEF && edges[i] != edgeType::BIDIRECT) {
                                    bool isEdgeForward = edges[i] == edgeType::FORWARD;

                                    int2 newEdge = isEdgeForward ? make_int2(forwardLoop[i + 1], forwardLoop[i])
                                                                 : make_int2(forwardLoop[i], forwardLoop[i + 1]);
                                    ordered_edges.insert(newEdge);

                                    loopDirc = isEdgeForward ? edgeType::BACKWARD : edgeType::FORWARD;
                                    break;
                                }
                            }
                            int2 newEdge = loopDirc == edgeType::FORWARD
                                             ? make_int2(forwardLoop[edgeId], forwardLoop[edgeId + 1])
                                             : make_int2(forwardLoop[edgeId + 1], forwardLoop[edgeId]);
                            ordered_edges.insert(newEdge);
                        } else { // both forward, both backward, one bidirectional
                            for (int i = 0; i < 3; ++i) {
                                if (edges[i] != edgeType::BIDIRECT && edges[i] != edgeType::UNDEF) {
                                    // we look for the first assigned but non-bidirectional edge
                                    loopDirc = edges[i];
                                    break;
                                }
                            }
                            int2 newEdge = loopDirc == edgeType::FORWARD
                                             ? make_int2(forwardLoop[edgeId], forwardLoop[edgeId + 1])
                                             : make_int2(forwardLoop[edgeId + 1], forwardLoop[edgeId]);
                            ordered_edges.insert(newEdge);
                        }
                    }
                    break;
                case 1: // two unassigned edges
                    {
                        // find the edge that is assigned, use it as loop direction, if bidirectional (no chance, but
                        // for robustness) just make the loop forward
                        int assignedEdgeId;
                        for (int i = 0; i < 3; ++i) {
                            if (edges[i] != edgeType::UNDEF) {
                                assignedEdgeId = i;
                                break;
                            }
                        }
                        edgeType loopDirc = edges[assignedEdgeId];
                        for (int& i : undefined_ids) {
                            int2 newEdge = loopDirc == edgeType::FORWARD
                                             ? make_int2(forwardLoop[i], forwardLoop[i + 1])
                                             : make_int2(forwardLoop[i + 1], forwardLoop[i]);
                            ordered_edges.insert(newEdge);
                        }
                    }
                    break;
                case 0:
                    {
                        // Just make all edges forward
                        for (int i = 0; i < 3; ++i) {
                            int2 newEdge = make_int2(forwardLoop[i], forwardLoop[i + 1]);
                            ordered_edges.insert(newEdge);
                        }
                    }
                    break;
                default:
                    break;
                }


                processed_triangles.insert(current);

                // add adjacent triangles to open
                for (int i = 0; i < 3; ++i) {
                    int2 edge = forwardLoop[i] < forwardLoop[i + 1] ? make_int2(forwardLoop[i], forwardLoop[i + 1])
                                                                    : make_int2(forwardLoop[i + 1], forwardLoop[i]);
                    for (int tri : mesh.edge_to_triangles[edge]) {
                        if (tri != current && processed_triangles.find(tri) == processed_triangles.end()) {
                            open_triangles.push(tri);
                        }
                    }
                }
            }
        }

        // loop through all triangles and see if they are all processed and make a loop
        for (int i = 0; i < mesh.indices.size(); ++i) {

            int3 triangle                  = mesh.indices[i];
            std::array<int, 4> forwardLoop = {triangle.x, triangle.y, triangle.z, triangle.x};

            std::array<edgeType, 3> edges;
            int forwardCount  = 0;
            int backwardCount = 0;
            int bidirectCount = 0;

            for (int j = 0; j < 3; ++j) {
                int2 edge  = make_int2(forwardLoop[j], forwardLoop[j + 1]);
                int2 edger = make_int2(forwardLoop[j + 1], forwardLoop[j]);

                bool isForward  = ordered_edges.find(edge) != ordered_edges.end();
                bool isBackward = ordered_edges.find(edger) != ordered_edges.end();

                if (isForward && isBackward) {
                    edges[j] = edgeType::BIDIRECT;
                    forwardCount++;
                    backwardCount++;
                    bidirectCount++;
                } else if (isForward) {
                    edges[j] = edgeType::FORWARD;
                    forwardCount++;
                } else if (isBackward) {
                    edges[j] = edgeType::BACKWARD;
                    backwardCount++;
                } else {
                    edges[j] = edgeType::UNDEF;
                    throw std::runtime_error(std::string(objFile) + "Triangle " + std::to_string(i)
                                             + " is not processed! The mesh is not watertight!");
                }
            }

            bool isLoop = doesLoop(forwardCount, backwardCount);
            if (!isLoop) {
                throw std::runtime_error(
                    "Triangle " + std::to_string(i) + " is not processed! The mesh is not watertight!");
            }
        }
        auto end                              = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Time to process the mesh: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms" << std::endl;

        for (int i = 0; i < mesh.indices.size(); ++i) {
            int3 triangle = mesh.indices[i];

            std::array<int, 4> forwardLoop = {triangle.x, triangle.y, triangle.z, triangle.x};
            for (int i = 0; i < 3; ++i) {
                if (ordered_edges.find(make_int2(forwardLoop[i], forwardLoop[i + 1])) != ordered_edges.end()) {
                    mesh.loopEdgeIndices.push_back(make_int2(forwardLoop[i], forwardLoop[i + 1]));
                    ordered_edges.erase(make_int2(forwardLoop[i], forwardLoop[i + 1]));
                }
                if (ordered_edges.find(make_int2(forwardLoop[i + 1], forwardLoop[i])) != ordered_edges.end()) {
                    mesh.loopEdgeIndices.push_back(make_int2(forwardLoop[i + 1], forwardLoop[i]));
                    ordered_edges.erase(make_int2(forwardLoop[i + 1], forwardLoop[i]));
                }
            }
        }

        std::cout << "Mesh " << objFile << " has " << mesh.vertices.size() << " vertices and " << mesh.indices.size()
                  << " triangles" << " and " << mesh.uniqueEdgeIndices.size() << " unique edges." << " and "
                  << mesh.loopEdgeIndices.size() << " loop edges." << std::endl;

        // for (auto & vertex : mesh.vertices) {
        // 	bounds.include(vertex);
        // }

        // TODO: compute bounding box
        // for (auto mesh : model->meshes)
        //	for (auto vtx : mesh->vertices)
        //		model->bounds.extend(vtx);

        // std::cout << "Mesh " << " has " << mesh.vertices.size() << " vertices and " << mesh.indices.size()
        //           << " triangles" << " and " << mesh.uniqueEdgeIndices.size() << " unique edges." << std::endl;
    }

    void meshModel::uploadToDevice() {
        if (uploaded) {
            return;
        }
        vertexDeviceBuffer.alloc_and_upload(mesh.vertices);
        indexDeviceBuffer.alloc_and_upload(mesh.indices);
        edgeDeviceBuffer.alloc_and_upload(mesh.uniqueEdgeIndices);
        uploaded = true;
    }

    void meshModel::uploadSphereToDevice() {
        if (sphereUploaded) {
            return;
        }
        sphereDeviceBuffer.alloc_and_upload(sphereOrigins);
        sphereRadiiDeviceBuffer.alloc_and_upload(sphereRadii);
        sphereUploaded = true;
    }

    void meshModel::buildGAS(OptixDeviceContext optixContext, CUDABuffer& tempBuffer) {
        if (wrapped) {
            return;
        }

        if (!uploaded) {
            uploadToDevice();
        }

        CUdeviceptr d_vertices;
        CUdeviceptr d_indices;
        uint32_t triangleInputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices = vertexDeviceBuffer.d_pointer();
        d_indices  = indexDeviceBuffer.d_pointer();

        triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        triangleInput.triangleArray.numVertices         = (int) mesh.vertices.size();
        triangleInput.triangleArray.vertexBuffers       = &d_vertices;

        triangleInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes = sizeof(int3);
        triangleInput.triangleArray.numIndexTriplets   = (int) mesh.indices.size();
        triangleInput.triangleArray.indexBuffer        = d_indices;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput.triangleArray.flags                       = &triangleInputFlags;
        triangleInput.triangleArray.numSbtRecords               = 1;
        triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0;
        triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
        triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &triangleInput, 1, &blasBufferSizes));

        // ==================================================================
        // execute build (main stage)
        // ==================================================================

        if (tempBuffer.sizeInBytes < blasBufferSizes.tempSizeInBytes) {
            tempBuffer.free();
            tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
        }

        CUDABuffer tempASBuffer;
        asBuffer.alloc(blasBufferSizes.outputSizeInBytes);


        CUDABuffer compactSizeBuffer;
        compactSizeBuffer.alloc(sizeof(size_t));
        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = compactSizeBuffer.d_pointer();

        OPTIX_CHECK(optixAccelBuild(optixContext, 0, /*Stream*/
            &accelOptions, &triangleInput, 1, tempBuffer.d_pointer(), tempBuffer.sizeInBytes, asBuffer.d_pointer(),
            asBuffer.sizeInBytes, &traversableHandle, &emitProperty, 1));


        size_t compactedSize;
        compactSizeBuffer.download(&compactedSize, 1);

        if (compactedSize < blasBufferSizes.outputSizeInBytes) {
            CUDABuffer compactedGASBuffer;
            compactedGASBuffer.alloc(compactedSize);
            OPTIX_CHECK(optixAccelCompact(
                optixContext, 0, traversableHandle, compactedGASBuffer.d_pointer(), compactedSize, &traversableHandle));
            asBuffer.free();
            asBuffer.swap(compactedGASBuffer);
        }

        wrapped = true;
    }

    template <BuildType BUILD>
    OptixTraversableHandle meshModel::buildGAS(const OptixDeviceContext optixContext, CUDABuffer& tempBuffer,
        const unsigned int buildFlags, const unsigned int inputFlags) {
        if (wrapped) {
            return traversableHandle;
        }

        if (sphereWrapped) {
            throw std::runtime_error("Sphere GAS has already been built!");
        }

        if (!uploaded) {
            uploadToDevice();
        }

        CUdeviceptr vertexPtr = vertexDeviceBuffer.d_pointer();

        OptixBuildInput triangleInput                           = {};
        triangleInput.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangleInput.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes         = sizeof(float3);
        triangleInput.triangleArray.numVertices                 = mesh.vertices.size();
        triangleInput.triangleArray.vertexBuffers               = &vertexPtr;
        triangleInput.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes          = sizeof(int3);
        triangleInput.triangleArray.numIndexTriplets            = mesh.indices.size();
        triangleInput.triangleArray.indexBuffer                 = indexDeviceBuffer.d_pointer();
        triangleInput.triangleArray.flags                       = &inputFlags;
        triangleInput.triangleArray.numSbtRecords               = 1;
        triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0;
        triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
        triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = buildFlags;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes GASBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &triangleInput, 1, &GASBufferSizes));

        if (GASBufferSizes.tempSizeInBytes > tempBuffer.sizeInBytes) {
            tempBuffer.resize(GASBufferSizes.tempSizeInBytes);
        }

        asBuffer.alloc(GASBufferSizes.outputSizeInBytes);

        if constexpr (BUILD == BuildType::COMPACT) {
            CUDABuffer compactSizeBuffer;
            compactSizeBuffer.alloc(sizeof(size_t));
            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result             = compactSizeBuffer.d_pointer();

            OPTIX_CHECK(optixAccelBuild(optixContext, 0, &accelOptions, &triangleInput, 1, tempBuffer.d_pointer(),
                GASBufferSizes.tempSizeInBytes, asBuffer.d_pointer(), GASBufferSizes.outputSizeInBytes,
                &traversableHandle, &emitProperty, 1));

            size_t compactedSize;
            compactSizeBuffer.download(&compactedSize, 1);
            if (compactedSize < GASBufferSizes.outputSizeInBytes) {
                CUDABuffer compactedGASBuffer;
                compactedGASBuffer.alloc(compactedSize);
                OPTIX_CHECK(optixAccelCompact(optixContext, 0, traversableHandle, compactedGASBuffer.d_pointer(),
                    compactedSize, &traversableHandle));
                asBuffer.free();
                asBuffer.swap(compactedGASBuffer);
            }
        } else {
            OPTIX_CHECK(optixAccelBuild(optixContext, 0, &accelOptions, &triangleInput, 1, tempBuffer.d_pointer(),
                GASBufferSizes.tempSizeInBytes, asBuffer.d_pointer(), GASBufferSizes.outputSizeInBytes,
                &traversableHandle, nullptr, 0));
        }
        CUDA_SYNC_CHECK();

        wrapped = true;
        return traversableHandle;
    }
    // void meshModel::buildGAS(OptixDeviceContext optixContext) {
    // 	meshModel::buildGAS(optixContext, localTempBuffer);
    // }
    void meshModel::loadSphers(const std::string sphereFile) {
        std::ifstream sphereFileIn(sphereFile);
        if (!sphereFileIn.is_open()) {
            std::cout << "Could not open sphere file " << sphereFile << std::endl;
            exit(1);
        }

        std::string line;
        while (std::getline(sphereFileIn, line)) {
            // Check for comment and skip the line if it starts with '#'
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream iss(line);
            float x, y, z, r;
            if (!(iss >> x >> y >> z >> r)) {
                std::cout << "Could not read sphere file " << sphereFile << std::endl;
                exit(1);
            }
            sphereOrigins.push_back(make_float3(x, y, z));
            sphereRadii.push_back(r);
        }
        sphereFileIn.close();
    }


    template <BuildType BUILD>
    OptixTraversableHandle meshModel::buildSphereGAS(const OptixDeviceContext optixContext, CUDABuffer& tempBuffer,
        const unsigned int buildFlags, const unsigned int inputFlags) {

        if (sphereWrapped) {
            return traversableHandle;
        }

        if (wrapped) {
            throw std::runtime_error("Mesh GAS has already been built!");
        }

        if (!sphereUploaded) {
            uploadSphereToDevice();
        }


        CUdeviceptr spherePtr = sphereDeviceBuffer.d_pointer();
        CUdeviceptr radiiPtr  = sphereRadiiDeviceBuffer.d_pointer();

        OptixBuildInput sphereInput           = {};
        sphereInput.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphereInput.sphereArray.vertexBuffers = &spherePtr;
        sphereInput.sphereArray.numVertices   = sphereCount;
        sphereInput.sphereArray.radiusBuffers = &radiiPtr;

        sphereInput.sphereArray.numSbtRecords = 1;
        sphereInput.sphereArray.flags         = &inputFlags;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = buildFlags;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes GASBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &sphereInput, 1, &GASBufferSizes));

        if (GASBufferSizes.tempSizeInBytes > tempBuffer.sizeInBytes) {
            tempBuffer.resize(GASBufferSizes.tempSizeInBytes);
        }

        asBuffer.alloc(GASBufferSizes.outputSizeInBytes);

        if constexpr (BUILD == BuildType::COMPACT) {
            CUDABuffer compactSizeBuffer;
            compactSizeBuffer.alloc(sizeof(size_t));
            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result             = compactSizeBuffer.d_pointer();

            OPTIX_CHECK(optixAccelBuild(optixContext, 0, &accelOptions, &sphereInput, 1, tempBuffer.d_pointer(),
                GASBufferSizes.tempSizeInBytes, asBuffer.d_pointer(), GASBufferSizes.outputSizeInBytes,
                &traversableHandle, &emitProperty, 1));

            size_t compactedSize;
            compactSizeBuffer.download(&compactedSize, 1);
            if (compactedSize < GASBufferSizes.outputSizeInBytes) {
                CUDABuffer compactedGASBuffer;
                compactedGASBuffer.alloc(compactedSize);
                OPTIX_CHECK(optixAccelCompact(optixContext, 0, traversableHandle, compactedGASBuffer.d_pointer(),
                    compactedSize, &traversableHandle));
                asBuffer.free();
                asBuffer.swap(compactedGASBuffer);
            }
        } else {
            OPTIX_CHECK(optixAccelBuild(optixContext, 0, &accelOptions, &sphereInput, 1, tempBuffer.d_pointer(),
                GASBufferSizes.tempSizeInBytes, asBuffer.d_pointer(), GASBufferSizes.outputSizeInBytes,
                &traversableHandle, nullptr, 0));
        }

        CUDA_SYNC_CHECK();

        sphereWrapped = true;
        return traversableHandle;
    }
} // namespace RTCD
