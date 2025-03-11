#pragma once

#include <CollisionScenes/obstacle.h>
#include <Meshes/meshTypes.h>
#include <memory>
#include <numeric>

namespace RTCD {
    class scene {
    private:
        bool uploaded = false;
        bool wrapped  = false;
        size_t nObs   = 0;
        size_t nVerts = 0;
        std::vector<std::unique_ptr<obstacle>> obsPtrs;
        std::vector<float3> obsSgVerts;
        std::vector<uint> obsVertCnt;
        std::vector<uint> whereInEdge;

        CUDABuffer dObsSgVerts; // single vertices
        CUDABuffer dObsVertCnt;
        CUDABuffer dWhereInEdge;

        // Bounding Box
        std::vector<OBB> obbs;
        CUDABuffer dOBBs;

        // Optix AS
        OptixBuildInput buildInput;
        std::vector<OptixInstance> instances;
        CUDABuffer dInstances;
        OptixAccelBufferSizes bufferSizesIAS;
        OptixTraversableHandle sceneTraversableHandle;
        CUDABuffer dIASOutputBuffer;
        CUDABuffer tempBuffer;

    public:
        scene() = default;

        // Getters
        inline size_t getNumObstacles() const { return nObs; };
        inline CUdeviceptr getdSgVerts() const { return dObsSgVerts.d_pointer(); }
        inline OptixTraversableHandle getTraversable() const { return sceneTraversableHandle; };
        inline CUdeviceptr getdOBBs() const { return dOBBs.d_pointer(); }
        inline size_t getSgRayCnt() const { return obsSgVerts.size() / 2; }
        inline OptixTraversableHandle getHandle() const { return sceneTraversableHandle; }
        inline size_t getSBTSize() const { return nObs; }
        inline const std::vector<OptixInstance>& getInstances() const { return instances; }
        inline size_t getInstancesBufferSize() const { return dInstances.sizeInBytes; }
        inline CUdeviceptr getInstancesBufferPtr() const { return dInstances.d_pointer(); }
        inline size_t getEdgeBufferSize() const { return dObsSgVerts.sizeInBytes; }
        inline CUdeviceptr getEdgesBufferPtr() const { return dObsSgVerts.d_pointer(); }
        inline OptixAccelBufferSizes getIASBufferSize() const { return bufferSizesIAS; }
        inline CUdeviceptr getObsVertCntPtr() const { return dObsVertCnt.d_pointer(); }
        inline CUdeviceptr getWhereInEdgePtr() const { return dWhereInEdge.d_pointer(); }
        inline std::vector<uint> getObsVertCnt() const { return obsVertCnt; }
        inline std::vector<float3> getSgVerts() const { return obsSgVerts; }
        inline const std::vector<OBB>& getOBBs() const { return obbs; }
        // Public Functions
        void addObstacle(std::unique_ptr<obstacle> obs);
        void uploadToDevice();
        OptixTraversableHandle wrapScene(OptixDeviceContext context, bool build = true);
    };

    void scene::addObstacle(std::unique_ptr<obstacle> obs) {
        assert(!uploaded && "Scene already uploaded to device");
        obsPtrs.push_back(std::move(obs));
        obsSgVerts.insert(obsSgVerts.end(), obsPtrs.back()->getSgVerts().begin(), obsPtrs.back()->getSgVerts().end());
        obsVertCnt.push_back(obsPtrs.back()->getSgVerts().size());
        obbs.emplace_back(obsPtrs.back()->getOBB());
        nObs++;
    }

    void scene::uploadToDevice() {
        if (uploaded) {
            return;
        }
        dObsSgVerts.alloc_and_upload(obsSgVerts);
        dOBBs.alloc_and_upload(obbs);
        dObsVertCnt.alloc_and_upload(obsVertCnt);
        whereInEdge.resize(nObs);
        std::exclusive_scan(obsVertCnt.begin(), obsVertCnt.end(), whereInEdge.begin(), 0, std::plus<uint>());
        dWhereInEdge.alloc_and_upload(whereInEdge);
        std::cout << "Uploaded " << nObs << " obstacles with " << obsSgVerts.size() / 2 << "edges" << std::endl;
        uploaded = true;
    }

    inline constexpr std::array<float, 12> defaultTransform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    static inline OptixInstance defaultInstance(unsigned int sbtOffset) {
        OptixInstance instance  = {};
        instance.flags          = OPTIX_INSTANCE_FLAG_NONE;
        instance.instanceId     = 0;
        instance.visibilityMask = 255;
        instance.sbtOffset      = sbtOffset;
        memcpy(instance.transform, defaultTransform.data(), sizeof(float) * 12);
        return instance;
    }


    OptixTraversableHandle scene::wrapScene(OptixDeviceContext context, bool build) {
        if (wrapped) {
            return sceneTraversableHandle;
        } else {
            wrapped = true;
        }
        unsigned int sbtOffset = 0;
        for (auto& obs : obsPtrs) {
            obs->buildGAS(context, tempBuffer);
            OptixInstance instance     = defaultInstance(sbtOffset);
            instance.traversableHandle = obs->getGASHandle();
            instance.sbtOffset         = sbtOffset;
            memcpy(instance.transform, defaultTransform.data(), sizeof(float) * 12);
            instances.push_back(instance);
        }

        dInstances.alloc_and_upload(instances);

        OptixBuildInput buildInput = {};

        buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances    = dInstances.d_pointer();
        buildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

        OptixAccelBuildOptions accelBuildOptions = {};
        accelBuildOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelBuildOptions, &buildInput,
            1, // Number of build inputs
            &bufferSizesIAS));

        if (tempBuffer.sizeInBytes < bufferSizesIAS.tempSizeInBytes) {
            tempBuffer.alloc(bufferSizesIAS.tempSizeInBytes);
        }
        dIASOutputBuffer.alloc(bufferSizesIAS.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(context,
            0, // CUDA stream
            &accelBuildOptions, &buildInput,
            1, // num build inputs
            tempBuffer.d_pointer(), bufferSizesIAS.tempSizeInBytes, dIASOutputBuffer.d_pointer(),
            bufferSizesIAS.outputSizeInBytes, &sceneTraversableHandle,
            nullptr, // emitted property list
            0)); // num emitted properties
        return 0;
    }
} // namespace RTCD
