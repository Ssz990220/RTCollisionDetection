#pragma once
#include <Utils/cuUtils.cuh>

typedef unsigned int uint;

namespace RTCD {

    // subset of the launch parameter required for RAY_ONLINE robot
    // refer to launchParameters.h for full definition
    struct rayRobotInfo {
        uint linkCnt;
        CUdeviceptr lkStarts;
        CUdeviceptr lkPosMap;
        CUdeviceptr lkRays;
        CUdeviceptr lkTfs;

        rayRobotInfo() = default;
        rayRobotInfo(uint linkCnt, CUdeviceptr lkStarts, CUdeviceptr lkPosMap, CUdeviceptr lkRays, CUdeviceptr lkTfs)
            : linkCnt(linkCnt), lkStarts(lkStarts), lkPosMap(lkPosMap), lkRays(lkRays), lkTfs(lkTfs) {}
    };

    struct meshRayInfo {
        uint meshGlbIdx; // Mesh Idx in all mesh
        uint nRays;
        CUdeviceptr meshRays;
        rayRobotInfo robotInfo;

        meshRayInfo() = default;

        meshRayInfo(uint meshGlbIdx, uint nRays, CUdeviceptr meshRays)
            : meshGlbIdx(meshGlbIdx), nRays(nRays), meshRays(meshRays) {}

        meshRayInfo(uint meshGlbIdx, uint nRays, CUdeviceptr meshRays, rayRobotInfo robotInfo)
            : meshGlbIdx(meshGlbIdx), nRays(nRays), meshRays(meshRays), robotInfo(robotInfo) {}
    };

    enum class LinkType : int {
        MESH            = 0, // Each link is a mesh GAS, each robot is a single IAS
        LINEAR_CURVE    = 1, // KEEP for rendering
        QUADRATIC_CURVE = 2, // KEEP for rendering
        CUBIC_CURVE     = 3, // KEEP for rendering
        SPHERES         = 4, // Each link is a sphere array GAS, each robot is a single IAS
        SPHERE_GAS      = 5, // Each robot is an array of sphere as a single GAS
        RAY             = 6, // Each link outlined by rays shooting from vertices, obstacle as single GAS
        RAY_ONLINE = 7, // Each link outlined by rays shooting from vertices, transformed online obstacle as single GAS
        RAY_STATIC = 8, // Each link outlined by rays, transform obs relative to the links, only shoot once per link for
                        // all poses
    };
    enum class BuildType { FAST_BUILD, FAST_TRACE, COMPACT, NONE };

    struct OBB {
        float3 center   = make_float3(0);
        float3 halfSize = make_float3(0);
        float3 orientation[3];
        OBB() = default;
    };
} // namespace RTCD
