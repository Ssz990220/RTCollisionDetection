#pragma once
#include <Robot/batchRobotConfig.h>

namespace RTCD {

    template <batchRobotConfig CFG>
    struct streamGASRobot {
        std::shared_ptr<robot<CFG.DOF>> robotPtr;
        cudaStream_t stream;
        OptixDeviceContext context;
        OptixTraversableHandle GASHandle;
        OptixAccelBuildOptions accelOptions = {};
        OptixBuildInput buildInput          = {};
        uint32_t InputFlags;
        CUDABuffer GASBuffer;
        CUDABuffer tempBuffer;

        CUDABuffer posesBuffer;
        CUDABuffer transformBuffer;

        CUDABuffer pointsBuffer;
        CUdeviceptr pointsPtr;

        streamGASRobot() = default;
        streamGASRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr, cudaStream_t stream, OptixDeviceContext context)
            : robotPtr(robotPtr), stream(stream), context(context){};
        virtual ~streamGASRobot() = default;

        virtual void updateAS(const CUdeviceptr poses)                      = 0;
        virtual void updateAS(const CUdeviceptr poses, const size_t nPoses) = 0;
        inline OptixTraversableHandle getTraversable() const { return GASHandle; }
    };
} // namespace RTCD
