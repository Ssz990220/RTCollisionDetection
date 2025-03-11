#pragma once
#include <Meshes/meshTypes.h>
#include <Utils/CUDABuffer.h>
#include <Utils/optix7.h>

namespace RTCD {
    template <size_t NSTREAM, LinkType TYPE>
    class baseBatchRobot {
    public:
        baseBatchRobot(std::array<cudaStream_t, NSTREAM> streams, OptixDeviceContext context)
            : streams(streams), context(context) {};
        virtual ~baseBatchRobot() = default;
        // fkine
        // @brief Forward kinematics for a batch poses of default size, also update the OBBs of links
        // V2.5 API
        virtual void fkine(const CUdeviceptr poses, const size_t idx) = 0;
        // fkine
        // @brief Forward kinematics for a batch poses of size `nPoses`, also update the OBBs of links
        // V2.5 API
        virtual void fkine(const CUdeviceptr poses, const size_t idx, const size_t nPoses) = 0;
        // update(const float* poses, const size_t idx)
        // @brief Update the robot with a batch of poses of default size
        // Fkine is computed inside this function
        // V2.0 API, use fkine V2.5 since V2.5
        virtual void update(const CUdeviceptr poses, const size_t idx) = 0;
        // update(const CUdeviceptr poses, const size_t idx, const size_t nPoses)
        // @brief Update the robot with a batch of poses of size `nPoses`
        // Fkine is computed inside this function
        // V2.0 API, use fkine V2.5 since V2.5
        virtual void update(const CUdeviceptr poses, const size_t idx, const size_t nPoses) = 0;
        // update(const bool* mask, const float* poses, const size_t idx)
        // @brief Update the robot with a mask, only update the links that are true in the mask array
        // The mask is computed by OBB collision checking
        // V2.5 API
        virtual bool updateWithMask(const CUdeviceptr mask, const size_t idx)                      = 0;
        virtual bool updateWithMask(const CUdeviceptr mask, const size_t idx, const size_t nPoses) = 0;

        // virtual void updateWithRSMask(const size_t obsId, const CUdeviceptr mask, const size_t idx) = 0;
        // virtual void updateWithRSMask(
        //     const size_t obsId, const CUdeviceptr mask, const size_t idx, const size_t nPoses) = 0;

        virtual void buildTraversables() = 0;

        virtual void resetGraph() = 0;

        // Getters...

        virtual CUdeviceptr getOBBs(const size_t idx) const = 0; // V2.5 API
        virtual size_t getNOBBs() const                     = 0; // V2.5 API
        virtual size_t getNOBBs(size_t nPoses) const        = 0; // V2.5 API
        virtual size_t getBatchSize() const                 = 0;
        virtual size_t getTrajSize() const                  = 0;
        virtual size_t getSBTSize() const                   = 0;
        virtual size_t getDOF() const                       = 0;
        virtual unsigned int getBuildFlags() const          = 0;

        virtual const OptixTraversableHandle getHandle(const size_t idx) const = 0;
        // virtual const OptixTraversableHandle getHandle(const size_t obsId, const size_t idx) const = 0;

    protected:
        // updateOBBs
        // @brief Update the OBBs of the robot, enforcing the implementation in derived classes
        // V2.5 API
        virtual void updateOBBs(const size_t idx) = 0;

        // updateOBBs
        // @brief Update the OBBs of the robot, enforcing the implementation in derived classes
        // V2.5 API
        virtual void updateOBBs(const size_t idx, const size_t nPoses) = 0;

    public:
        OptixDeviceContext context;
        std::array<cudaStream_t, NSTREAM> streams;
        std::array<OptixTraversableHandle, NSTREAM> ASHandles;
        CUDABuffer mapIndex;
        bool useOBB;

    public:
        CUdeviceptr getMapIndex() const { return mapIndex.d_pointer(); }

        virtual CUdeviceptr getMapIndex(const size_t idx) const = 0; // V2.5 API
    };

    template <size_t NSTREAM, LinkType TYPE>
    class baseRayRobot {
    public:
        baseRayRobot(std::array<cudaStream_t, NSTREAM> streams, OptixDeviceContext context)
            : streams(streams), context(context) {
            static_assert((TYPE == LinkType::RAY_STATIC || TYPE == LinkType::RAY_ONLINE)
                          && "This class is only for ray robots"); // remove RAY support in V2.5
        };
        virtual ~baseRayRobot() = default;

        virtual void fkine(const CUdeviceptr poses, const size_t idx)                              = 0;
        virtual void fkine(const CUdeviceptr poses, const size_t idx, const size_t nPoses)         = 0;
        virtual void fkine2(const CUdeviceptr poses, const size_t idx)                             = 0;
        virtual void fkine2(const CUdeviceptr poses, const size_t idx, const size_t nPoses)        = 0;
        virtual void update(const CUdeviceptr poses, const size_t idx)                             = 0;
        virtual void update(const CUdeviceptr poses, const size_t idx, const size_t nPoses)        = 0;
        virtual bool updateWithMask(const CUdeviceptr mask, const size_t idx)                      = 0;
        virtual bool updateWithMask(const CUdeviceptr mask, const size_t idx, const size_t nPoses) = 0;

        virtual size_t getBatchSize() const                             = 0;
        virtual size_t getTrajSize() const                              = 0;
        virtual CUdeviceptr getOBBs(const size_t idx) const             = 0; // V2.5
        virtual size_t getNOBBs() const                                 = 0; // V2.5 API
        virtual size_t getNOBBs(size_t nPoses) const                    = 0; // V2.5 API
        virtual const std::array<CUdeviceptr, NSTREAM>& getRays() const = 0;
        virtual size_t getRayCount() const                              = 0;
        virtual size_t getDOF() const                                   = 0;
        virtual int* getRayMap() const                                  = 0; // updated in V2.5
        virtual std::vector<float3> getRayVertices() const              = 0;
        virtual const std::vector<meshRayInfo>& getRayInfo() const      = 0; // V2.5 API
        virtual const meshRayInfo getRayInfo(const size_t idx) const    = 0; // V2.6 API
        virtual const CUdeviceptr getLinkTfs(const size_t idx) const    = 0; // V2.5 API

        virtual void resetGraph() = 0;

    protected:
        virtual void updateOBBs(const size_t idx)                       = 0;
        virtual void updateOBBs(const size_t idx, const size_t nPoses)  = 0;
        virtual void updateOBBs2(const size_t idx)                      = 0;
        virtual void updateOBBs2(const size_t idx, const size_t nPoses) = 0;

    public:
        OptixDeviceContext context;
        std::array<cudaStream_t, NSTREAM> streams;
        bool useOBB;
    };
} // namespace RTCD
