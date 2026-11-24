#pragma once
// rtcd/limits.h -- the ONLY place for true compile-time constants of the new stack, and for every struct whose
// byte layout is shared between host code and device code (kernels, OptiX programs).
//
// Includable from:
//   * host C++20 translation units (MSVC, no CUDA headers needed),
//   * nvcc device and host code (CUDA C++17 or later),
//   * OptiX program translation units.
// Rules for this header:
//   * only <cstdint> / <cstddef> are included; no CUDA, OptiX, STL containers or <span>;
//   * every struct here is trivially copyable, has a fixed layout, and its size is static_asserted, so host and
//     device agree byte for byte. Device pointers are stored as `unsigned long long` device addresses (same
//     representation as CUdeviceptr on 64-bit); device code reinterpret_casts them.
//   * nothing here is a runtime option. Runtime options live in config.h; capacities are given at construction.
//
// Ownership: frozen contract (Stage 0-B). Changing a layout here requires changing every program / kernel that
// reads it; request changes through main.

#include <cstddef>
#include <cstdint>

// Spelling of the launch-parameter variable for OptiX program sources (must equal rtcd::kLaunchParamsName):
//   extern "C" __constant__ rtcd::EdgeRayParams rtcdLaunchParams;
#define RTCD_LAUNCH_PARAMS_NAME rtcdLaunchParams

namespace rtcd {

    // ---------------------------------------------------------------------------------------------------------
    // Robot size limits
    // ---------------------------------------------------------------------------------------------------------

    // Maximum number of actuated (revolute / continuous) joints of a robot. Sizes the kinematics table.
    inline constexpr std::uint32_t kMaxDof = 16;
    // Maximum number of geometry links (link 0 = root, link j + 1 = child of joint j). Always kMaxDof + 1.
    inline constexpr std::uint32_t kMaxLinks = kMaxDof + 1;
    // Floats of one rigid transform: row-major 3x4 [R | t], rows (r00 r01 r02 tx, r10 r11 r12 ty, r20 r21 r22 tz).
    inline constexpr std::uint32_t kTransformFloats = 12;

    // ---------------------------------------------------------------------------------------------------------
    // Curve (continuous) mode limits
    // ---------------------------------------------------------------------------------------------------------

    // Supported B-spline degrees of the swept-sphere curves: 1 = round linear, 2 = round quadratic B-spline,
    // 3 = round cubic B-spline. Kernels that need the degree at compile time are explicitly instantiated for
    // exactly these values inside their .cu file.
    inline constexpr std::uint32_t kMinCurveDegree = 1;
    inline constexpr std::uint32_t kMaxCurveDegree = 3;
    constexpr bool isSupportedCurveDegree(std::uint32_t degree) {
        return degree >= kMinCurveDegree && degree <= kMaxCurveDegree;
    }
    // Capacity of the spline weight table (control points x trajectory points), floats. Same limit as the old
    // __constant__ splineWeights[512]: controlPoints * trajectoryPoints must be <= this.
    inline constexpr std::uint32_t kMaxSplineWeights = 512;
    // Largest trajectory point count the control-point reduction supports (powOf2(nTrajPts) / 2 <= 256).
    inline constexpr std::uint32_t kMaxTrajectoryPoints = 512;

    // ---------------------------------------------------------------------------------------------------------
    // OptiX launch limits
    // ---------------------------------------------------------------------------------------------------------

    // OptiX limits width * height * depth of one launch to 2^30.
    inline constexpr std::uint64_t kMaxLaunchSize = 1ull << 30;
    // Upper bound of one launch-parameter block (bytes). The detector core allocates one block of this size per
    // stream; LaunchPlan::paramsBytes must be <= this.
    inline constexpr std::uint32_t kMaxLaunchParamsBytes = 256;
    // Name of the launch-parameter variable every OptiX program module of the new stack declares:
    //   extern "C" __constant__ <ParamsStruct> rtcdLaunchParams;
    // (OptixPipelineCompileOptions::pipelineLaunchParamsVariableName; RTCD_LAUNCH_PARAMS_NAME below).
    inline constexpr const char* kLaunchParamsName = "rtcdLaunchParams";
    // SBT record header size / alignment. Equal to OPTIX_SBT_RECORD_HEADER_SIZE / OPTIX_SBT_RECORD_ALIGNMENT;
    // src/core static_asserts the equality where optix_types.h is available.
    inline constexpr std::uint32_t kSbtRecordHeaderSize = 32;
    inline constexpr std::uint32_t kSbtRecordAlignment  = 16;

    // ---------------------------------------------------------------------------------------------------------
    // Shared layouts
    // ---------------------------------------------------------------------------------------------------------

    // Oriented bounding box. Byte-identical to the old RTCD::OBB (float3 center, float3 halfSize,
    // float3 orientation[3]); axes[i] is the unit direction of the box's i-th local axis in the parent frame and
    // halfSize[i] the half extent along it. Kernels read it as 15 consecutive floats:
    //   [0..2] center, [3..5] halfSize, [6..8] axes[0], [9..11] axes[1], [12..14] axes[2].
    struct Obb {
        float center[3];
        float halfSize[3];
        float axes[3][3];
    };
    static_assert(sizeof(Obb) == 15 * sizeof(float), "Obb must be 15 packed floats (old RTCD::OBB layout)");

    // Kinematics of one serial-chain robot in the form the kinematics kernels consume. Replaces the old
    // __constant__ cMask / sMask / oMask / baseAffine globals of RTCD/Robot/robot.cu: one table per robot, kept in
    // a device buffer owned by whoever launches the kernels (no globals; several robots coexist).
    //
    // Semantics (exactly the old kernels):
    //   local_j(q) = cos(q_j) * cosMask[j] + sin(q_j) * sinMask[j] + oneMask[j]      (elementwise, 12 floats)
    //   link_0     = base
    //   link_{j+1} = link_j * local_j(q_j)                                            (3x4 affine composition)
    // with cos/sin computed by __sincosf in float. All transforms row-major 3x4 (kTransformFloats).
    // Entries for j >= dof are zero. numLinks is always dof + 1.
    struct KinematicsTable {
        std::uint32_t dof;       // actuated joints, 1 .. kMaxDof
        std::uint32_t numLinks;  // dof + 1
        std::uint32_t reserved[2];
        float base[kTransformFloats];                // world pose of link 0 (robot base pose * root fixed chain)
        float cosMask[kMaxDof][kTransformFloats];
        float sinMask[kMaxDof][kTransformFloats];
        float oneMask[kMaxDof][kTransformFloats];
    };
    static_assert(sizeof(KinematicsTable) == 16 + 4 * (kTransformFloats + 3 * kMaxDof * kTransformFloats),
        "KinematicsTable layout changed");

    // SBT record without per-record data (all programs of the new stack read only the launch parameters).
    // `data` is always 0; with the explicit tail padding it keeps the old 48-byte record size.
    struct alignas(kSbtRecordAlignment) SbtRecord {
        char header[kSbtRecordHeaderSize];
        unsigned long long data;
        unsigned long long reserved; // explicit tail padding to the 16-byte record alignment
    };
    static_assert(sizeof(SbtRecord) == 48, "SbtRecord must be 48 bytes");

    // Launch parameters of the edge-ray programs: rays along scene obstacle edges traced against a robot
    // acceleration structure. Used by the mesh backend (robot IAS, any-hit reads optixGetInstanceIndex()) and the
    // curve backend (curve GAS, any-hit reads optixGetPrimitiveIndex()).
    // Ray r (launch index x, 0 <= r < edgeCount): origin = edgeVertices[2r], target = edgeVertices[2r + 1]
    // (float3 each); direction = normalize(target - origin), tmin = 0, tmax = |target - origin|.
    // Any-hit: atomicAdd(hitCounts + primToItem[index], 1); optixIgnoreIntersection().
    struct EdgeRayParams {
        unsigned long long traversable;   // OptixTraversableHandle of the robot AS of this batch
        unsigned long long edgeVertices;  // const float* : 6 floats per ray (origin xyz, target xyz)
        unsigned long long primToItem;    // const uint32_t* : instance (mesh) / primitive (curve) index -> item
        unsigned long long hitCounts;     // uint32_t* [itemCount] : BatchView::hitCounts
        std::uint32_t edgeCount;          // launch width
        std::uint32_t reserved;
    };
    static_assert(sizeof(EdgeRayParams) == 40, "EdgeRayParams layout changed");
    static_assert(sizeof(EdgeRayParams) <= kMaxLaunchParamsBytes, "EdgeRayParams too large");

    // Launch parameters of the robot-ray program (RAY mode, RobotToObs): rays along robot link mesh edges,
    // transformed by the link poses of each item, traced against the scene IAS.
    // Ray id = item * raysPerRobot + rayInRobot, 0 <= id < totalRayCount = itemCount * raysPerRobot. The launch
    // width may be smaller (capped); launch index x walks ids x, x + width, x + 2 width, ... (grid stride).
    // A ray is traced only if linkMask[item * linksPerRobot + rayLink[rayInRobot]] != 0 and, when skipHitItems
    // != 0, hitCounts[item] == 0 (volatile read). Hit: atomicAdd(hitCounts + item, 1).
    struct RobotRayParams {
        unsigned long long traversable;   // OptixTraversableHandle of the scene IAS
        unsigned long long rays;          // const float* : 6 floats per robot ray, robot-local, ordered by link
        unsigned long long rayLink;       // const int32_t* [raysPerRobot] : ray -> link index
        unsigned long long linkMask;      // const uint32_t* [itemCount * linksPerRobot] : OBB overlap count per link
        unsigned long long linkPoses;     // const float* [itemCount * linksPerRobot * 12] : pose-major link poses
        unsigned long long hitCounts;     // uint32_t* [itemCount] : BatchView::hitCounts
        std::uint32_t totalRayCount;
        std::uint32_t raysPerRobot;
        std::uint32_t linksPerRobot;
        std::uint32_t skipHitItems;
    };
    static_assert(sizeof(RobotRayParams) == 64, "RobotRayParams layout changed");
    static_assert(sizeof(RobotRayParams) <= kMaxLaunchParamsBytes, "RobotRayParams too large");

} // namespace rtcd
