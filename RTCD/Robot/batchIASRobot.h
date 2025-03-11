#pragma once

#include <Robot/baseBatchRobot.h>
#include <Robot/batchRobotConfig.h>
#include <Robot/robot.h>
#include <Utils/cuda/cudaAlgorithms.cuh>
namespace RTCD // Batch Robot Classes Declaration
{

    // optixLink
    // A class to represent a batch of links. Each link has one optixLink attached to it.
    //
    //
    template <BuildType BUILD>
    struct optixLink {
        std::shared_ptr<meshModel> model;
        OptixTraversableHandle handle;
        uint32_t InputFlags{OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    };

    template <BuildType BUILD>
    struct optixMeshLink final : public optixLink<BUILD> {
        // optixMeshLink(std::shared_ptr<meshModel> model);

        optixMeshLink(std::shared_ptr<meshModel> model, unsigned int buildFlags, OptixDeviceContext optixContext,
            CUDABuffer& tempBuffer);

        void buildGAS(unsigned int buildFlags, OptixDeviceContext optixContext, CUDABuffer& tempBuffer);
    };

    template <BuildType BUILD>
    struct optixSphereLink final : public optixLink<BUILD> {
        optixSphereLink(std::shared_ptr<meshModel> model, unsigned int buildFlags, OptixDeviceContext optixContext,
            CUDABuffer& tempBuffer);

        void buildGAS(unsigned int buildFlags, OptixDeviceContext optixContext, CUDABuffer& tempBuffer);
    };


    // StreamUniqueRobot
    //
    // @brief A class to manage robots on a single stream
    template <batchRobotConfig CFG>
    struct streamUniqueRobot {
        std::shared_ptr<robot<CFG.DOF>> robotPtr;
        cudaStream_t stream;
        OptixDeviceContext context;
        OptixTraversableHandle IASHandle;
        OptixBuildInput buildInput{};
        OptixAccelBuildOptions buildOptions{};
        CUDABuffer IASBuffer;
        size_t IASBufferSize;
        CUDABuffer tempBuffer;
        CUDABuffer instancesBuffer;

        OptixAccelBuildOptions updateOptions;

        CUDABuffer linkTransformsBuffer;

        CUdeviceptr linkobbTemplate; // points to memory that contains DOF + 1 obbs
        CUdeviceptr indexMapTemplate; // points to memory that contains index map to copy from
        CUDABuffer linkOBBsBuffer; // The link OBBs are transformed version of obbs in linkobbTempalte
        CUDABuffer indexMap; // selected link to pose index map, V2.5.
        CUDABuffer batchInstancesBuffer; // selected instances might be in collision, V2.5

        CUDABuffer dWhereInDst; // stream compact result, V2.5
        std::array<OptixInstance, (CFG.DOF + 1) * CFG.BATCHSIZE> instances;
        std::array<std::unique_ptr<optixLink<CFG.BUILD>>, CFG.DOF + 1> batchLinks;


        // cudaEvent_t event;
        bool graphCreated = false;
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        CUDABuffer dCount;
        size_t selectTmpSize;
        uint maskCnt = 0;

        streamUniqueRobot() = default;
        streamUniqueRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr, cudaStream_t stream, OptixDeviceContext context);

        inline OptixTraversableHandle getTraversable() const { return IASHandle; }

        // buildAS
        //
        // @brief Build the GAS & IAS for the robot array for the first time
        void buildAS();

        // updateOBB
        //
        // @brief Update the OBBs of the robot with the current poses
        // inline void updateOBB(const CUdeviceptr linkTransforms);
        inline void updateOBB(const size_t nPoses);


        // inline void fkine(const CUdeviceptr posees, const size_t idx);
        inline void fkine(const CUdeviceptr poses, const size_t nPoses);
        // updateIAS
        //
        // @brief Update the IAS with the current poses of the robot
        // contains only GPU function calls.
        inline void updateIAS(const size_t nPoses);

        // updateWithMask
        //
        // @brief Update the IAS with only part of the links.
        // OptixTraversableHandle updateWithMask(const CUdeviceptr mask);
        bool updateWithMask(const CUdeviceptr mask, const size_t nPoses);
    };

    // batchIASRobot
    //
    // @brief A class to manage a batch of IAS Robots on different streams
    // @tparam batchSize The number of robots in each batch running on a stream
    // @tparam CFG.NSTREAM The number of streams to run the batch on.
    // @tparam CFG.DOF The number of degrees of freedom of the robot
    // @tparam CFG.BUILD The build type of the robot: COMPACT, FAST_BUILD, FAST_TRACE
    // @tparam CFG.TYPE The link type of the robot: MESH, SPHERE
    template <batchRobotConfig CFG>
    class batchIASRobot final : public baseBatchRobot<CFG.NSTREAM, CFG.TYPE> {
    public:
        batchIASRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr, std::array<cudaStream_t, CFG.NSTREAM> streams,
            OptixDeviceContext context);
        void fkine(const CUdeviceptr poses, const size_t idx) override;
        void fkine(const CUdeviceptr poses, const size_t idx, const size_t nPoses) override;
        void update(const CUdeviceptr poses, const size_t idx) override;
        void update(const CUdeviceptr poses, const size_t idx, const size_t nPoses) override;
        bool updateWithMask(const CUdeviceptr mask, const size_t idx) override;
        bool updateWithMask(const CUdeviceptr mask, const size_t idx, const size_t nPoses) override;
        void buildTraversables() override;
        constexpr unsigned int getBuildFlags() const override;
        constexpr size_t getSBTSize() const override;
        constexpr size_t getBatchSize() const override;
        constexpr size_t getTrajSize() const override { return 1; }
        constexpr size_t getDOF() const override { return CFG.DOF; }
        CUdeviceptr getOBBs(const size_t idx) const override { return batchRobots[idx].linkOBBsBuffer.d_pointer(); }
        constexpr size_t getNOBBs() const override { return CFG.BATCHSIZE * (CFG.DOF + 1); }
        constexpr size_t getNOBBs(size_t nPoses) const override { return nPoses * (CFG.DOF + 1); }

        inline CUdeviceptr getMapIndex(const size_t idx) const override {
            return batchRobots[idx].indexMap.d_pointer();
        }

        const OptixTraversableHandle getHandle(const size_t idx) const override { return batchRobots[idx].IASHandle; }

        void resetGraph() override;

#if defined(_DEBUG) || defined(DEBUG)
        void downloadOBB(std::vector<OBB>& obbs, const size_t idx) {
            obbs.resize(CFG.BATCHSIZE * (CFG.DOF + 1));
            batchRobots[idx].linkOBBsBuffer.download(obbs);
        }

        void downloadTF(std::vector<float>& tfs, const size_t idx) {
            batchRobots[idx].linkTransformsBuffer.download(tfs);
        }
#endif
    protected:
        inline void updateOBBs(const size_t idx) override;
        inline void updateOBBs(const size_t idx, const size_t nPoses) override;

    private:
        std::array<streamUniqueRobot<CFG>, CFG.NSTREAM> batchRobots;
    };

} // namespace RTCD

namespace RTCD // BatchIASRobot Implementation
{
    template <batchRobotConfig CFG>
    batchIASRobot<CFG>::batchIASRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr,
        std::array<cudaStream_t, CFG.NSTREAM> streams, OptixDeviceContext context)
        : baseBatchRobot<CFG.NSTREAM, CFG.TYPE>(streams, context) {

        baseBatchRobot<CFG.NSTREAM, CFG.TYPE>::useOBB = CFG.useOBB;

        if constexpr (CFG.TYPE == LinkType::MESH) {
            robotPtr->uploadOBBs();
        } else {
            robotPtr->uploadSphereOBBs();
        }

        std::array<uint, CFG.BATCHSIZE*(CFG.DOF + 1)> mapIndex;
        for (uint i = 0; i < CFG.BATCHSIZE; i++) {
            for (uint j = 0; j < CFG.DOF + 1; j++) {
                mapIndex[i * (CFG.DOF + 1) + j] = i;
            }
        }
        baseBatchRobot<CFG.NSTREAM, CFG.TYPE>::mapIndex.alloc_and_upload(mapIndex);


        for (size_t i = 0; i < CFG.NSTREAM; i++) {
            streamUniqueRobot<CFG> r(robotPtr, streams[i], context);
            batchRobots[i]                  = std::move(r);
            batchRobots[i].indexMapTemplate = baseBatchRobot<CFG.NSTREAM, CFG.TYPE>::mapIndex.d_pointer();
        }
        buildTraversables();

        CUDA_SYNC_CHECK();
    }

    template <batchRobotConfig CFG>
    void batchIASRobot<CFG>::buildTraversables() {
        for (size_t i = 0; i < CFG.NSTREAM; i++) {
            batchRobots[i].buildAS();
            baseBatchRobot<CFG.NSTREAM, CFG.TYPE>::ASHandles[i] = batchRobots[i].getTraversable();
        }
    }

    template <batchRobotConfig CFG>
    constexpr unsigned int batchIASRobot<CFG>::getBuildFlags() const {
        if constexpr (CFG.BUILD == BuildType::COMPACT) {
            return OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        }
        if constexpr (CFG.BUILD == BuildType::FAST_BUILD) {
            return OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        }
        if constexpr (CFG.BUILD == BuildType::FAST_TRACE) {
            return OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        }
    }

    template <batchRobotConfig CFG>
    constexpr size_t batchIASRobot<CFG>::getSBTSize() const {
        return CFG.BATCHSIZE * (CFG.DOF + 1);
    }

    template <batchRobotConfig CFG>
    constexpr size_t batchIASRobot<CFG>::getBatchSize() const {
        return CFG.BATCHSIZE;
    }

    template <batchRobotConfig CFG>
    void batchIASRobot<CFG>::update(const CUdeviceptr poses, const size_t idx) {
        fkine(poses, idx, CFG.BATCHSIZE);
        batchRobots[idx].updateIAS(CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    void batchIASRobot<CFG>::update(const CUdeviceptr poses, const size_t idx, const size_t nPoses) {
        fkine(poses, idx, nPoses);
        batchRobots[idx].updateIAS(nPoses);
    }

    template <batchRobotConfig CFG>
    void batchIASRobot<CFG>::fkine(const CUdeviceptr poses, const size_t idx) {
        batchRobots[idx].fkine(poses, CFG.BATCHSIZE);
        updateOBBs(idx, CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    void batchIASRobot<CFG>::fkine(const CUdeviceptr poses, const size_t idx, const size_t nPoses) {
        batchRobots[idx].fkine(poses, nPoses);
        updateOBBs(idx, nPoses);
    }

    template <batchRobotConfig CFG>
    bool batchIASRobot<CFG>::updateWithMask(const CUdeviceptr mask, const size_t idx) {
        return batchRobots[idx].updateWithMask(mask, CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    bool batchIASRobot<CFG>::updateWithMask(const CUdeviceptr mask, const size_t idx, const size_t nPoses) {
        return batchRobots[idx].updateWithMask(mask, nPoses);
    }

    template <batchRobotConfig CFG>
    inline void batchIASRobot<CFG>::updateOBBs(const size_t idx) {
        batchRobots[idx].updateOBB(CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    inline void batchIASRobot<CFG>::updateOBBs(const size_t idx, const size_t nPoses) {
        batchRobots[idx].updateOBB(nPoses);
    }

    template <batchRobotConfig CFG>
    void batchIASRobot<CFG>::resetGraph() {
        for (size_t i = 0; i < CFG.NSTREAM; i++) {
            batchRobots[i].graphCreated = false;
        }
    }
} // namespace RTCD

namespace RTCD // StreamUniqueRobot Implementation
{
    template <batchRobotConfig CFG>
    streamUniqueRobot<CFG>::streamUniqueRobot(
        std::shared_ptr<robot<CFG.DOF>> robotPtr, cudaStream_t stream, OptixDeviceContext context)
        : robotPtr(robotPtr), stream(stream), context(context) {
        linkobbTemplate = robotPtr->getOBBs();
        linkOBBsBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(OBB));

        updateOptions.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        updateOptions.motionOptions.numKeys = 1;
        updateOptions.operation             = OPTIX_BUILD_OPERATION_UPDATE;

        buildOptions.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        buildOptions.motionOptions.numKeys = 1;
        buildOptions.operation             = OPTIX_BUILD_OPERATION_BUILD;

        linkTransformsBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * 12 * sizeof(float));

        indexMap.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(uint));
        dWhereInDst.alloc(roundUpLog2(CFG.BATCHSIZE * (CFG.DOF + 1)) * sizeof(uint));
        dCount.alloc(sizeof(int));
    }

    template <batchRobotConfig CFG>
    void streamUniqueRobot<CFG>::buildAS() {

        unsigned int buildFlags = 0;
        if constexpr (CFG.BUILD == BuildType::COMPACT) {
            buildFlags |= (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        }
        if constexpr (CFG.BUILD == BuildType::FAST_BUILD) {
            buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        }
        if constexpr (CFG.BUILD == BuildType::FAST_TRACE) {
            buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        }

        constexpr std::array<float, 12> defaultTransform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

        for (int i = 0; i < CFG.DOF + 1; i++) {
            if constexpr (CFG.TYPE == LinkType::MESH) {
                batchLinks[i] = std::make_unique<optixMeshLink<CFG.BUILD>>(
                    robotPtr->getOrderedLinks()[i], buildFlags, context, tempBuffer);
            } else if constexpr (CFG.TYPE == LinkType::SPHERES) {
                batchLinks[i] = std::make_unique<optixSphereLink<CFG.BUILD>>(
                    robotPtr->getOrderedLinks()[i], buildFlags, context, tempBuffer);
            }
            // batchLinks[i]->setModel(robotPtr->getOrderedLinks()[i]);
            // batchLinks[i]->buildGAS(buildFlags, context, tempBuffer);
        }

        // Allocate enough memory for stream compact
        tempBuffer.expandIfNotEnough(
            (((CFG.DOF + 1) * CFG.BATCHSIZE
                 + ((CFG.DOF + 1) * CFG.BATCHSIZE + 4 * THREADBLOCK_SIZE - 1) / 4 * THREADBLOCK_SIZE)
                * sizeof(uint)));


        for (int j = 0; j < CFG.BATCHSIZE; j++) {
            for (int i = 0; i < CFG.DOF + 1; i++) {
                instances[j * (CFG.DOF + 1) + i]                   = {};
                instances[j * (CFG.DOF + 1) + i].visibilityMask    = 255;
                instances[j * (CFG.DOF + 1) + i].flags             = OPTIX_INSTANCE_FLAG_NONE;
                instances[j * (CFG.DOF + 1) + i].traversableHandle = batchLinks[i]->handle;
                instances[j * (CFG.DOF + 1) + i].sbtOffset         = j * (CFG.DOF + 1) + i;
                memcpy(instances[j * (CFG.DOF + 1) + i].transform, defaultTransform.data(), sizeof(float) * 12);
            }
        }

        instancesBuffer.alloc_and_upload(instances);
        batchInstancesBuffer.alloc_and_upload(instances);

        buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances    = batchInstancesBuffer.d_pointer();
        buildInput.instanceArray.numInstances = (CFG.DOF + 1) * CFG.BATCHSIZE;

        buildIAS<BuildType::FAST_TRACE>(
            context, buildOptions, buildInput, tempBuffer, IASBuffer, IASHandle, IASBufferSize, 1.5);
    }

    template <batchRobotConfig CFG>
    inline void streamUniqueRobot<CFG>::updateIAS(const size_t nPoses) {
        upInsPoses(batchInstancesBuffer.d_pointer(), linkTransformsBuffer.d_pointer(), nPoses * (CFG.DOF + 1),
            (CFG.DOF + 1) * CFG.BATCHSIZE, stream);
        OPTIX_CHECK(optixAccelBuild(context, stream, &updateOptions, &buildInput, 1, tempBuffer.d_pointer(),
            tempBuffer.sizeInBytes, IASBuffer.d_pointer(), IASBuffer.sizeInBytes, &IASHandle, nullptr, 0));
    }

    template <batchRobotConfig CFG>
    inline bool streamUniqueRobot<CFG>::updateWithMask(const CUdeviceptr mask, const size_t nPoses) {

        upInsPoses(instancesBuffer.d_pointer(), linkTransformsBuffer.d_pointer(), nPoses * (CFG.DOF + 1),
            (CFG.DOF + 1) * CFG.BATCHSIZE, stream);

        selectInsIdx(batchInstancesBuffer.d_pointer(), indexMap.d_pointer(), instancesBuffer.d_pointer(),
            indexMapTemplate, mask, tempBuffer, nPoses * (CFG.DOF + 1), selectTmpSize, dCount.d_pointer(), stream);
        dCount.downloadAsync(&maskCnt, 1, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        buildInput.instanceArray.numInstances = maskCnt;

        if (buildInput.instanceArray.numInstances == 0) {
            return false;
        }

        OPTIX_CHECK(optixAccelBuild(context, stream, &buildOptions, &buildInput, 1, tempBuffer.d_pointer(),
            tempBuffer.sizeInBytes, IASBuffer.d_pointer(), IASBuffer.sizeInBytes, &IASHandle, nullptr, 0));

        return true;
    }

    template <batchRobotConfig CFG>
    inline void streamUniqueRobot<CFG>::fkine(const CUdeviceptr poses, const size_t nPoses) {
        batchFkineAdj(poses, linkTransformsBuffer.d_pointer(), CFG.DOF, nPoses, stream);
    }


    template <batchRobotConfig CFG>
    inline void streamUniqueRobot<CFG>::updateOBB(const size_t nPoses) {
        assert(nPoses <= CFG.BATCHSIZE && "The number of poses should be smaller than the batch size");
        batchOBBTransform(
            linkTransformsBuffer.d_pointer(), linkobbTemplate, linkOBBsBuffer.d_pointer(), CFG.DOF, nPoses, stream);
    }
} // namespace RTCD

namespace RTCD // BatchLink Implementation
{
    template <BuildType BUILD>
    optixMeshLink<BUILD>::optixMeshLink(std::shared_ptr<meshModel> model, unsigned int buildFlags,
        OptixDeviceContext optixContext, CUDABuffer& tempBuffer) {
        optixLink<BUILD>::model = model;
        optixLink<BUILD>::model->uploadToDevice();
        optixLink<BUILD>::handle = optixLink<BUILD>::model->template buildGAS<BUILD>(
            optixContext, tempBuffer, buildFlags, optixLink<BUILD>::InputFlags);
    }

    template <BuildType BUILD>
    optixSphereLink<BUILD>::optixSphereLink(std::shared_ptr<meshModel> model, unsigned int buildFlags,
        OptixDeviceContext optixContext, CUDABuffer& tempBuffer) {
        optixLink<BUILD>::model = model;
        optixLink<BUILD>::model->uploadSphereToDevice();
        optixLink<BUILD>::handle = optixLink<BUILD>::model->template buildGAS<BUILD>(
            optixContext, tempBuffer, buildFlags, optixLink<BUILD>::InputFlags);
    }
} // namespace RTCD
