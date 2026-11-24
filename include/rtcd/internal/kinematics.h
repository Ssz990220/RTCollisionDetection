#pragma once
// rtcd/internal/kinematics.h -- forward-kinematics kernels over a KinematicsTable (rtcd/limits.h), shared by the
// mesh, ray and curve backends. Library rtcd_robot (src/robot, Wave 1 d, which also owns the table loader).
// INTERNAL contract.
//
// The table lives in a DEVICE buffer (one per robot per owner; uploadKinematics() creates it), never in
// __constant__ globals, so robots coexist. Every kernel must produce results BIT-IDENTICAL to the old kernels
// named below (same operation order, __sincosf, same affine-multiply order) -- the goldens depend on it.
// All functions enqueue on `stream`, return immediately, never allocate, accept count 0 (no launch).
//
// Layouts (12 floats per transform, row-major 3x4):
//   q (poses):          [pose][dof]                                   (rtcd/types.h item layout)
//   q (trajectories):   [traj][point][dof]
//   pose-major poses:   linkPoses[(pose * numLinks + link) * 12 + k]          (old "Adj" layout)
//   trajectory poses:   linkPoses[((traj * numLinks + link) * points + point) * 12 + k]  (old fkineTraj layout)
//   OBBs:               obbs[pose * numLinks + link]

#include <rtcd/device_buffer.h>
#include <rtcd/limits.h>
#include <rtcd/types.h>

#include <cstdint>

namespace rtcd::internal {

    // Both views of ONE table: `host` is the RobotModel's table (launch configuration reads dof / numLinks from
    // it; an implementation may also pass it by value as a __grid_constant__ kernel parameter), `device` its copy
    // made by uploadKinematics(). Both must describe the same table and outlive the enqueued work (`device`) /
    // the call (`host`). Which one the kernels read is an implementation choice of rtcd_robot (see
    // docs/design/interfaces.md, open question Q1); callers always pass both.
    struct KinematicsRef {
        const KinematicsTable* host = nullptr;
        DevicePtr device            = 0;
    };

    // Device copy of one table (synchronous upload).
    DeviceBuffer uploadKinematics(const KinematicsTable& table);

    // Pose-major link poses of nPoses poses. Old batchFkineAdj.
    void linkPoses(const KinematicsRef& table, DeviceSpan<const float> q, std::uint32_t nPoses,
        DeviceSpan<float> linkPosesOut, StreamHandle stream);

    // linkPoses + link OBBs in one launch: obbsOut[p * numLinks + l] = obbTemplates[l] transformed by link pose
    // (p, l). Old batchFkineAdjOBB (fused kernel, a Level-2 optimization to keep).
    void linkPosesAndObbs(const KinematicsRef& table, DeviceSpan<const float> q, std::uint32_t nPoses,
        DeviceSpan<const Obb> obbTemplates, DeviceSpan<float> linkPosesOut, DeviceSpan<Obb> obbsOut,
        StreamHandle stream);

    // OBB transform only, from pose-major link poses. Old batchOBBTransform.
    void transformObbs(DeviceSpan<const float> linkPoses, std::uint32_t numLinks, std::uint32_t nPoses,
        DeviceSpan<const Obb> obbTemplates, DeviceSpan<Obb> obbsOut, StreamHandle stream);

    // Trajectory-layout link poses of nTraj trajectories of `points` poses. Old batchFkineTraj.
    void trajectoryLinkPoses(const KinematicsRef& table, DeviceSpan<const float> q, std::uint32_t points,
        std::uint32_t nTraj, DeviceSpan<float> linkPosesOut, StreamHandle stream);

} // namespace rtcd::internal
