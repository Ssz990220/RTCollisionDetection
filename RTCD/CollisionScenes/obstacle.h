#pragma once
#include <Meshes/mesh.h>
#include <vector>

namespace RTCD {
    class obstacle : public meshModel {
    private:
        // std::vector<float3> doubleVertices;
        std::vector<float3> singleVertices;
        // CUDABuffer dDoubleVertices;
        CUDABuffer dSingleVertices;

    public:
        obstacle();
        obstacle(std::string filename, float scale, bool loop = false);
        obstacle(std::string filename, Eigen::Affine3f& transform, float scale, bool loop = false);
        obstacle(std::vector<float3>& vertices, std::vector<int3>& indices, Eigen::Affine3f& transform);

        // Gets
        inline OptixTraversableHandle getGASHandle() { return traversableHandle; };
        inline CUdeviceptr getdSgVerts() { return dSingleVertices.d_pointer(); }
        inline unsigned int getSgRayCnt() { return singleVertices.size() / 2; };
        inline unsigned int getNumSingleVertices() { return singleVertices.size(); };
        inline std::vector<float3>& getSgVerts() { return singleVertices; };
    };

    constexpr float cubeMeshSize = 0.05f;

    class cube : public obstacle {
    public:
        cube();
        cube(float3 dim, float3 pos);
        cube(float x, float y, float z, float px, float py, float pz);
        cube(float x, float y, float z, Eigen::Affine3f transform = Eigen::Affine3f::Identity());
    };

    static obstacle genCube(float x, float y, float z, Eigen::Affine3f transform) {
        std::vector<float3> vertices;
        std::vector<int3> indices;
        int xCount  = x / cubeMeshSize + 1;
        int yCount  = y / cubeMeshSize + 1;
        int zCount  = z / cubeMeshSize + 1;
        float xStep = x / xCount;
        float yStep = y / yCount;
        float zStep = z / zCount;
        // x-y surface
        for (int i = 0; i < xCount; i++) {
            int vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2 + i * xStep, -y / 2, -z / 2));
            vertices.push_back(make_float3(-x / 2 + i * xStep, y / 2, -z / 2));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep, -y / 2, -z / 2));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep, y / 2, -z / 2));
            indices.push_back(make_int3(vertCount + 0, vertCount + 1, vertCount + 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 3));

            vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2 + i * xStep, -y / 2, z / 2));
            vertices.push_back(make_float3(-x / 2 + i * xStep, y / 2, z / 2));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep, -y / 2, z / 2));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep, y / 2, z / 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 0));
            indices.push_back(make_int3(vertCount + 2, vertCount + 3, vertCount + 1));
        }

        // same for y-z
        for (int i = 0; i < yCount; i++) {
            int vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2, -y / 2 + i * yStep, -z / 2));
            vertices.push_back(make_float3(-x / 2, -y / 2 + i * yStep, z / 2));
            vertices.push_back(make_float3(-x / 2, -y / 2 + (i + 1) * yStep, -z / 2));
            vertices.push_back(make_float3(-x / 2, -y / 2 + (i + 1) * yStep, z / 2));
            indices.push_back(make_int3(vertCount + 0, vertCount + 1, vertCount + 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 3));

            vertCount = vertices.size();
            vertices.push_back(make_float3(x / 2, -y / 2 + i * yStep, -z / 2));
            vertices.push_back(make_float3(x / 2, -y / 2 + i * yStep, z / 2));
            vertices.push_back(make_float3(x / 2, -y / 2 + (i + 1) * yStep, -z / 2));
            vertices.push_back(make_float3(x / 2, -y / 2 + (i + 1) * yStep, z / 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 0));
            indices.push_back(make_int3(vertCount + 2, vertCount + 3, vertCount + 1));
        }

        // same for x-z
        for (int i = 0; i < zCount; i++) {
            int vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2, -y / 2, -z / 2 + i * zStep));
            vertices.push_back(make_float3(x / 2, -y / 2, -z / 2 + i * zStep));
            vertices.push_back(make_float3(-x / 2, -y / 2, -z / 2 + (i + 1) * zStep));
            vertices.push_back(make_float3(x / 2, -y / 2, -z / 2 + (i + 1) * zStep));
            indices.push_back(make_int3(vertCount + 0, vertCount + 1, vertCount + 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 3));

            vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2, y / 2, -z / 2 + i * zStep));
            vertices.push_back(make_float3(x / 2, y / 2, -z / 2 + i * zStep));
            vertices.push_back(make_float3(-x / 2, y / 2, -z / 2 + (i + 1) * zStep));
            vertices.push_back(make_float3(x / 2, y / 2, -z / 2 + (i + 1) * zStep));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 0));
            indices.push_back(make_int3(vertCount + 2, vertCount + 3, vertCount + 1));
        }
        Eigen::Affine3f T = Eigen::Affine3f::Identity();
        return obstacle(vertices, indices, T);
    }

    static obstacle genCube(float x, float y, float z, float px, float py, float pz) {
        std::vector<float3> vertices;
        std::vector<int3> indices;


        int xCount  = x / cubeMeshSize + 1;
        int yCount  = y / cubeMeshSize + 1;
        int zCount  = z / cubeMeshSize + 1;
        float xStep = x / xCount;
        float yStep = y / yCount;
        float zStep = z / zCount;
        // x-y surface
        for (int i = 0; i < xCount; i++) {
            int vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2 + i * xStep + px, -y / 2 + py, -z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + i * xStep + px, y / 2 + py, -z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep + px, -y / 2 + py, -z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep + px, y / 2 + py, -z / 2 + pz));
            indices.push_back(make_int3(vertCount + 0, vertCount + 1, vertCount + 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 3));

            vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2 + i * xStep + px, -y / 2 + py, z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + i * xStep + px, y / 2 + py, z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep + px, -y / 2 + py, z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + (i + 1) * xStep + px, y / 2 + py, z / 2 + pz));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 0));
            indices.push_back(make_int3(vertCount + 2, vertCount + 3, vertCount + 1));
        }

        // same for y-z
        for (int i = 0; i < yCount; i++) {
            int vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2 + px, -y / 2 + i * yStep + py, -z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + px, -y / 2 + i * yStep + py, z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + px, -y / 2 + (i + 1) * yStep + py, -z / 2 + pz));
            vertices.push_back(make_float3(-x / 2 + px, -y / 2 + (i + 1) * yStep + py, z / 2 + pz));
            indices.push_back(make_int3(vertCount + 0, vertCount + 1, vertCount + 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 3));

            vertCount = vertices.size();
            vertices.push_back(make_float3(x / 2 + px, -y / 2 + i * yStep + py, -z / 2 + pz));
            vertices.push_back(make_float3(x / 2 + px, -y / 2 + i * yStep + py, z / 2 + pz));
            vertices.push_back(make_float3(x / 2 + px, -y / 2 + (i + 1) * yStep + py, -z / 2 + pz));
            vertices.push_back(make_float3(x / 2 + px, -y / 2 + (i + 1) * yStep + py, z / 2 + pz));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 0));
            indices.push_back(make_int3(vertCount + 2, vertCount + 3, vertCount + 1));
        }

        // same for x-z
        for (int i = 0; i < zCount; i++) {
            int vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2 + px, -y / 2 + py, -z / 2 + i * zStep + pz));
            vertices.push_back(make_float3(x / 2 + px, -y / 2 + py, -z / 2 + i * zStep + pz));
            vertices.push_back(make_float3(-x / 2 + px, -y / 2 + py, -z / 2 + (i + 1) * zStep + pz));
            vertices.push_back(make_float3(x / 2 + px, -y / 2 + py, -z / 2 + (i + 1) * zStep + pz));
            indices.push_back(make_int3(vertCount + 0, vertCount + 1, vertCount + 2));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 3));

            vertCount = vertices.size();
            vertices.push_back(make_float3(-x / 2 + px, y / 2 + py, -z / 2 + i * zStep + pz));
            vertices.push_back(make_float3(x / 2 + px, y / 2 + py, -z / 2 + i * zStep + pz));
            vertices.push_back(make_float3(-x / 2 + px, y / 2 + py, -z / 2 + (i + 1) * zStep + pz));
            vertices.push_back(make_float3(x / 2 + px, y / 2 + py, -z / 2 + (i + 1) * zStep + pz));
            indices.push_back(make_int3(vertCount + 2, vertCount + 1, vertCount + 0));
            indices.push_back(make_int3(vertCount + 2, vertCount + 3, vertCount + 1));
        }

        Eigen::Affine3f T = Eigen::Affine3f::Identity();
        return obstacle(vertices, indices, T);
    }
} // namespace RTCD


// Function definitations
namespace RTCD {
    obstacle::obstacle() : meshModel() {
        // Do nothing
    }

    obstacle::obstacle(std::string filename, float scale, bool loop) : meshModel(filename, scale) {
        // doubleVertices.clear();
        std::vector<int2>& edgeIndices = loop ? mesh.loopEdgeIndices : mesh.uniqueEdgeIndices;
        singleVertices.clear();
        for (int2 idx : edgeIndices) {
            singleVertices.push_back(mesh.vertices[idx.x]);
            singleVertices.push_back(mesh.vertices[idx.y]);
        }
        // dDoubleVertices.alloc_and_upload(doubleVertices);
        dSingleVertices.alloc_and_upload(singleVertices);
    }

    obstacle::obstacle(std::string filename, Eigen::Affine3f& transform, float scale, bool loop)
        : meshModel(filename, transform, scale) {
        std::vector<int2>& edgeIndices = loop ? mesh.loopEdgeIndices : mesh.uniqueEdgeIndices;
        // doubleVertices.clear();
        singleVertices.clear();
        for (int2 idx : edgeIndices) {
            singleVertices.push_back(mesh.vertices[idx.x]);
            singleVertices.push_back(mesh.vertices[idx.y]);
        }
        // dDoubleVertices.alloc_and_upload(doubleVertices);
        dSingleVertices.alloc_and_upload(singleVertices);
    }

    obstacle::obstacle(std::vector<float3>& vertices, std::vector<int3>& indices, Eigen::Affine3f& transform)
        : meshModel(vertices, indices, transform) {
        for (int2 idx : mesh.uniqueEdgeIndices) {
            singleVertices.push_back(mesh.vertices[idx.x]);
            singleVertices.push_back(mesh.vertices[idx.y]);
        }
        // dDoubleVertices.alloc_and_upload(doubleVertices);
        dSingleVertices.alloc_and_upload(singleVertices);
    }

    cube::cube() : obstacle() {
        // Do nothing
    }

    cube::cube(float x, float y, float z, Eigen::Affine3f transform) : obstacle(genCube(x, y, z, transform)) {}

    cube::cube(float3 dim, float3 pos) : obstacle(genCube(dim.x, dim.y, dim.z, pos.x, pos.y, pos.z)) {}

    cube::cube(float x, float y, float z, float px, float py, float pz) : obstacle(genCube(x, y, z, px, py, pz)) {}
} // namespace RTCD
