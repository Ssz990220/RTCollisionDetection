#pragma once
#include <Robot/baseBatchRobot.h>
#include <Robot/batchRobotConfig.h>
#include <Robot/robot.h>
#include <Robot/streamRobot.h>
#include <memory>

namespace RTCD {

    template <batchRobotConfig CFG>
    struct streamSphereGASRobot final : public streamGASRobot<CFG> {
        streamSphereGASRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr, cudaStream_t stream, OptixDeviceContext context,
            CUdeviceptr radiusPtr);
        OptixTraversableHandle updateAS(const float* poses) override;
        OptixTraversableHandle updateAS(const float* poses, const size_t nPoses) override;
        void preAllocSphereBuffer();

        CUdeviceptr radiusPtr;
        size_t nSpherePerRobot;
        size_t nSpheresPerBatch;
    };

    template <batchRobotConfig CFG>
    class batchGASRobot final : public baseBatchRobot<CFG.NSTREAM, CFG.TYPE> {
    public:
        batchGASRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr, std::array<cudaStream_t, CFG.NSTREAM> streams,
            OptixDeviceContext context);
        OptixTraversableHandle update(const float* poses, const size_t idx) override;
        OptixTraversableHandle update(const float* poses, const size_t idx, const size_t nPoses) override;
        void buildTraversables() override;
        constexpr unsigned int getBuildFlags() const override;
        constexpr size_t getSBTSize() const override;
        constexpr size_t getBatchSize() const override;
        constexpr size_t getTrajSize() const override { return 1; }
        constexpr size_t getDOF() const override { return CFG.DOF; }

    private:
        std::shared_ptr<robot<CFG.DOF>> robotPtr;
        std::vector<float> radius;
        CUDABuffer radiusBuffer;
        CUdeviceptr radiusPtr;
        std::array<std::unique_ptr<streamGASRobot<CFG>>, CFG.NSTREAM> robots;
    };
} // namespace RTCD

namespace RTCD { // implementation of batchGASRobot

    template <batchRobotConfig CFG>
    batchGASRobot<CFG>::batchGASRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr,
        std::array<cudaStream_t, CFG.NSTREAM> streams, OptixDeviceContext context)
        : baseBatchRobot<CFG.NSTREAM, CFG.TYPE>(streams, context), robotPtr(robotPtr) {
        if constexpr (CFG.TYPE == LinkType::SPHERE_GAS) {
            robotPtr->uploadSphereArrays();
        }

        size_t nSpheresPerBatch = robotPtr->getRobotTotalSphereCount() * CFG.BATCHSIZE;
        radius.resize(nSpheresPerBatch);
        std::vector<int> mapIndex(CFG.BATCHSIZE * (robotPtr->getRobotTotalSphereCount() + 1));
        std::array<int, CFG.BATCHSIZE> localIdxMapping;
        std::iota(localIdxMapping.begin(), localIdxMapping.end(), 0);
        for (int i = 0; i < robotPtr->getRobotTotalSphereCount(); ++i) {
            std::fill(radius.begin() + i * CFG.BATCHSIZE, radius.begin() + (i + 1) * CFG.BATCHSIZE,
                robotPtr->getSphereRadii()[i]);
            std::copy(localIdxMapping.begin(), localIdxMapping.end(), mapIndex.begin() + i * CFG.BATCHSIZE);
        }
        baseBatchRobot<CFG.NSTREAM, CFG.TYPE>::mapIndex.alloc_and_upload(mapIndex);

        radiusBuffer.alloc_and_upload(radius);
        radiusPtr = radiusBuffer.d_pointer();


        for (size_t i = 0; i < CFG.NSTREAM; ++i) {
            if constexpr (CFG.TYPE == LinkType::SPHERE_GAS) {
                robots[i] = std::make_unique<streamSphereGASRobot<CFG>>(robotPtr, streams[i], context, radiusPtr);
            } else {
                static_assert("Invalid LinkType for GAS Robot");
            }
        }

        CUDA_SYNC_CHECK();

        buildTraversables();
    }

    template <batchRobotConfig CFG>
    OptixTraversableHandle batchGASRobot<CFG>::update(const float* poses, const size_t idx) {
        return robots[idx]->updateAS(poses);
    }


    template <batchRobotConfig CFG>
    OptixTraversableHandle batchGASRobot<CFG>::update(const float* poses, const size_t idx, const size_t nPoses) {
        return robots[idx]->updateAS(poses, nPoses);
    }

    template <batchRobotConfig CFG>
    void batchGASRobot<CFG>::buildTraversables() {
        CUDABuffer tempPoses;
        tempPoses.alloc(CFG.BATCHSIZE * CFG.DOF * sizeof(float));
        for (size_t i = 0; i < CFG.NSTREAM; ++i) {
            robots[i]->updateAS(reinterpret_cast<float*>(tempPoses.d_pointer()));
            baseBatchRobot<CFG.NSTREAM, CFG.TYPE>::ASHandles[i] = robots[i]->getTraversable();
        }
    };

    template <batchRobotConfig CFG>
    constexpr unsigned int batchGASRobot<CFG>::getBuildFlags() const {
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
    constexpr size_t batchGASRobot<CFG>::getSBTSize() const {
        return 1;
    }

    template <batchRobotConfig CFG>
    constexpr size_t batchGASRobot<CFG>::getBatchSize() const {
        return CFG.BATCHSIZE;
    }

} // namespace RTCD

namespace RTCD { // implementation of streamGASRobot

    template <batchRobotConfig CFG>
    streamSphereGASRobot<CFG>::streamSphereGASRobot(
        std::shared_ptr<robot<CFG.DOF>> robotPtr, cudaStream_t stream, OptixDeviceContext context, CUdeviceptr rPtr)
        : streamGASRobot<CFG>(robotPtr, stream, context), radiusPtr(rPtr) {

        using Base = streamGASRobot<CFG>;

        nSpherePerRobot  = robotPtr->getRobotTotalSphereCount();
        nSpheresPerBatch = nSpherePerRobot * CFG.BATCHSIZE;

        // Sphere from the same template sphere are stored next to each other
        Base::transformBuffer.alloc((CFG.DOF + 1) * CFG.BATCHSIZE * sizeof(float) * 12);
        Base::pointsBuffer.alloc(nSpheresPerBatch * 3 * sizeof(float));
        Base::pointsPtr                            = Base::pointsBuffer.d_pointer();
        Base::InputFlags                           = OPTIX_GEOMETRY_FLAG_NONE;
        Base::buildInput.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        Base::buildInput.sphereArray.vertexBuffers = &(Base::pointsPtr);
        Base::buildInput.sphereArray.numVertices   = nSpheresPerBatch;
        Base::buildInput.sphereArray.radiusBuffers = &radiusPtr;
        Base::buildInput.sphereArray.numSbtRecords = 1;
        Base::buildInput.sphereArray.flags         = &(Base::InputFlags);


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
        Base::accelOptions.buildFlags = buildFlags;
        Base::accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

        preAllocSphereBuffer();
    }

    template <batchRobotConfig CFG>
    void streamSphereGASRobot<CFG>::preAllocSphereBuffer() {

        using Base = streamGASRobot<CFG>;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(Base::context, &(Base::accelOptions), &(Base::buildInput),
            1, // Number of build inputs
            &gasBufferSizes));

        Base::GASBuffer.alloc(gasBufferSizes.outputSizeInBytes);
        Base::tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

        // std::cout << "Sphere GAS Memory Usage: " << gasBufferSizes.outputSizeInBytes << " bytes and "
        //           << gasBufferSizes.tempSizeInBytes << " bytes as temp buffer." << std::endl;
    }

    template <batchRobotConfig CFG>
    OptixTraversableHandle streamSphereGASRobot<CFG>::updateAS(const float* poses) {

        using Base = streamGASRobot<CFG>;

        batchFkineAdj(reinterpret_cast<CUdeviceptr>(poses), Base::transformBuffer.d_pointer(), CFG.DOF, CFG.BATCHSIZE,
            Base::stream);
        batchSphrCntrAdj(reinterpret_cast<float*>(Base::transformBuffer.d_pointer()),
            reinterpret_cast<float*>(Base::robotPtr->getSphereCenters()),
            reinterpret_cast<float*>(Base::pointsBuffer.d_pointer()), CFG.BATCHSIZE,
            reinterpret_cast<int*>(Base::robotPtr->getSphereIdxMap()), Base::robotPtr->getRobotTotalSphereCount(),
            CFG.DOF, Base::stream);

        OPTIX_CHECK(optixAccelBuild(Base::context,
            Base::stream, // CUDA stream
            &(Base::accelOptions), &(Base::buildInput),
            1, // num build inputs
            Base::tempBuffer.d_pointer(), Base::tempBuffer.sizeInBytes, Base::GASBuffer.d_pointer(),
            Base::GASBuffer.sizeInBytes, &(Base::GASHandle),
            nullptr, // emitted property list
            0)); // num emitted properties
        return Base::GASHandle;
    }


    template <batchRobotConfig CFG>
    OptixTraversableHandle streamSphereGASRobot<CFG>::updateAS(const float* poses, const size_t nPoses) {

        using Base = streamGASRobot<CFG>;

        batchFkineAdj(reinterpret_cast<CUdeviceptr>(poses), Base::transformBuffer.d_pointer(), CFG.DOF, CFG.BATCHSIZE,
            Base::stream);
        batchSphrCntrAdj(reinterpret_cast<float*>(Base::transformBuffer.d_pointer()),
            reinterpret_cast<float*>(Base::robotPtr->getSphereCenters()),
            reinterpret_cast<float*>(Base::pointsBuffer.d_pointer()), CFG.BATCHSIZE,
            reinterpret_cast<int*>(Base::robotPtr->getSphereIdxMap()), Base::robotPtr->getRobotTotalSphereCount(),
            CFG.DOF, Base::stream);

        Base::buildInput.sphereArray.numVertices = nPoses * nSpherePerRobot;

        OPTIX_CHECK(optixAccelBuild(Base::context,
            Base::stream, // CUDA stream
            &(Base::accelOptions), &(Base::buildInput),
            1, // num build inputs
            Base::tempBuffer.d_pointer(), Base::tempBuffer.sizeInBytes, Base::GASBuffer.d_pointer(),
            Base::GASBuffer.sizeInBytes, &(Base::GASHandle),
            nullptr, // emitted property list
            0)); // num emitted properties
        return Base::GASHandle;
    }
} // namespace RTCD
