#pragma once
#include <CollisionScenes/batchSceneConfig.h>
#include <memory>

namespace RTCD {
    // baseBatchScene
    // @brief Type erasure class for batchScene.
    template <size_t NSTREAM>
    class baseBatchScene {
    public:
        baseBatchScene()          = default;
        virtual ~baseBatchScene() = default;

        virtual bool updateASWithMask(const CUdeviceptr tfs, const CUdeviceptr mask, const size_t idx) = 0;
        virtual bool updateASWithMask(
            const CUdeviceptr tfs, const CUdeviceptr mask, const size_t idx, const size_t nPoses) = 0;
        // virtual void updateEdgeWithMask(const CUdeviceptr mask, const size_t idx) = 0;

        virtual size_t getSBTSize() const                                          = 0;
        virtual size_t getNumObstacles() const                                     = 0;
        virtual size_t getNumEdges() const                                         = 0;
        virtual CUdeviceptr getOBBs() const                                        = 0;
        virtual size_t getSgRayCnt() const                                         = 0;
        virtual CUdeviceptr getSgVerts() const                                     = 0;
        virtual meshRayInfo getRayInfo() const                                     = 0;
        virtual const CUdeviceptr getMapIndex(const size_t idx) const              = 0;
        virtual const std::vector<meshRayInfo>& getRayInfo(const size_t idx) const = 0;
        virtual const OptixTraversableHandle getHandle() const                     = 0;
        virtual const OptixTraversableHandle getHandle(const size_t idx) const     = 0;
    };
} // namespace RTCD
