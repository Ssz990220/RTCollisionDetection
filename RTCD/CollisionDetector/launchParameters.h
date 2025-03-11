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

struct rayRobotCfg {
    uint linkCnt; // how many compacted links
    uint* lkStarts; // prefix sum of each compacted link's ray count, in order to locate which link this ray belongs to
                    // given a ray ID
    uint* lkPosMap; // map linkIdx to poseIdx
    float3** lkRays; // pointers to each link's device rays (float3 * 2)
    float* lkTfs; // compacted link transforms

    rayRobotCfg() : linkCnt(0), lkStarts(nullptr), lkPosMap(nullptr), lkRays(nullptr), lkTfs(nullptr) {}
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
