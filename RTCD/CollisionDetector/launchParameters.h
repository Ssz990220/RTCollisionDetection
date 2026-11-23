#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

typedef unsigned int uint;

enum class rayType {
    ROBOT,
    LINK,
    MESH,
};

// RAY_ONLINE: one launch index per (pose, ray) of the batch, launch width = nPoses * raysPerRobot (host-known, so no
// device->host count read-back is needed). Rays of links whose OBB is free (linkMask == 0) exit before tracing.
struct rayRobotCfg {
    uint raysPerRobot; // rays of one robot (all links), launch index = poseIdx * raysPerRobot + rayIdx
    uint linksPerRobot; // DOF + 1
    uint skipHitPoses; // EXPERIMENT: skip rays of poses already in collision
    const uint* linkMask; // [nPoses * linksPerRobot] OBB result per (pose, link), > 0 means trace the link's rays
    const int* rayLinkMap; // [raysPerRobot] ray index within the robot -> link index
    const float3* rays; // [raysPerRobot * 2] robot-local ray endpoints (origin, target), ordered by link
    const float* lkTfs; // [nPoses * linksPerRobot * 12] link transforms (row-major 3x4)

    rayRobotCfg()
        : raysPerRobot(0), linksPerRobot(0), skipHitPoses(1), linkMask(nullptr), rayLinkMap(nullptr), rays(nullptr), lkTfs(nullptr) {}
};

// launch info for shooting rays from one link to obs
struct rayLinkCfg {
    uint poseIdx;
    float* lkTfs;

    rayLinkCfg() : poseIdx(0), lkTfs(nullptr) {}
};

// launch info for shooting rays from obs to ro
struct rayMeshCfg {
    uint* primIdxToPoseIdx;

    rayMeshCfg() : primIdxToPoseIdx(nullptr) {}
};

struct LaunchParams {
    rayType type;
    union {
        rayRobotCfg robot;
        rayLinkCfg link;
        rayMeshCfg mesh;
        char pad[64];
    };

    float3* verticesBuffer;
    uint totalRayCount;
    OptixTraversableHandle traversable;
    int* hitBuffer;

    LaunchParams() : verticesBuffer(nullptr), totalRayCount(0), traversable(0), hitBuffer(nullptr) {}
};

namespace RTCD {
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void* data;
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void* data;
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void* data;
    };
} // namespace RTCD
