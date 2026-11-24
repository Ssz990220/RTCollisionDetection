#pragma once
// Private header of rtcd_backend_curve: host wrappers of the curve kernels (curve_kernels.cu).
//
// Ports of the old curve kernels, bit-identical outputs (tests/backend_curve/test_curve_kernels compares them with
// the old stack's rtcd_legacy_kernels):
//   fusedCurvePoints      old batchFkineSphrTraj / fkineSphrTrajKern (RTCD/Robot/robot.cu): FK + sphere centers
//                         (+ B-spline control points) of whole trajectories in one launch
//   sphereCentersTraj     old batchSphrCntrTraj / genSphrTrajKer (robot.cu), after trajectoryLinkPoses
//   controlPoints         old batchGenControlPoints / genCtrlPtsKern (RTCD/Robot/sweptVolume.cu)
//   curveSegmentObbs      old genCurveOBB<degree> / genCurveOBBKern (sweptVolume.cu/.inl)
//   compactSegments       old genIndex + 2x compactCpyCurve (streamCompact.inl) of the curve OBB path, plus the
//                         compacted segment -> item map the old path was missing (design doc Q5) and the count
// The only difference to the old kernels: the kinematics table and the spline weights are read from device
// buffers (no __constant__ robot data), so several robots / checkers coexist.
//
// Layouts (floats, packed):
//   q            [traj][point][dof]                                    (BatchView::items)
//   link poses   [traj][link][point][12]                               (internal::trajectoryLinkPoses)
//   centers      [traj][sphere][point][3]                              (linear: these are the curve vertices)
//   ctrl points  [traj][sphere][ctrlPoint][3]
//   weights      [ctrlPoint][trajPoint]
// All wrappers enqueue on `stream`, return immediately, never allocate, throw CudaError on a launch error and do
// nothing for a zero count.

#include <rtcd/limits.h>
#include <rtcd/types.h>

#include <cstddef>
#include <cstdint>

namespace rtcd::curve {

    // Old fkineSphrTrajSupported minus its DOF <= 8 condition (that limit was the size of the old __constant__
    // fkine masks; the table here holds kMaxDof joints). ctrlPoints == 0 means linear (no control points).
    bool fusedCurvePointsSupported(std::uint32_t dof, std::uint32_t trajPoints, std::uint32_t nSpheres,
        std::uint32_t ctrlPoints);
    // Dynamic shared memory of the fused kernel (old fkineSphrTrajSmem).
    std::size_t fusedCurvePointsSmem(std::uint32_t dof, std::uint32_t trajPoints, std::uint32_t nSpheres, bool ctrl);

    // table: device KinematicsTable; sphereCenters: [nSpheres][3] link frame; sphereLink: [nSpheres] int32;
    // weights: [ctrlPoints][trajPoints] (unused if ctrlPoints == 0). out: centers (ctrlPoints == 0) or control
    // points. Caller checks fusedCurvePointsSupported() first (InvalidArgument otherwise).
    void fusedCurvePoints(DevicePtr table, std::uint32_t dof, DevicePtr q, DevicePtr sphereCenters,
        DevicePtr sphereLink, DevicePtr weights, DevicePtr out, std::uint32_t trajPoints, std::uint32_t nTraj,
        std::uint32_t nSpheres, std::uint32_t ctrlPoints, StreamHandle stream);

    void sphereCentersTraj(DevicePtr linkPoses, DevicePtr sphereCenters, DevicePtr sphereLink, DevicePtr out,
        std::uint32_t trajPoints, std::uint32_t nTraj, std::uint32_t nSpheres, std::uint32_t dof, StreamHandle stream);

    // nCurves = nTraj * nSpheres. trajPoints <= kMaxTrajectoryPoints.
    void controlPoints(DevicePtr trajPointsIn, DevicePtr ctrlPointsOut, DevicePtr weights, std::uint32_t nCurves,
        std::uint32_t ctrlPoints, std::uint32_t trajPoints, StreamHandle stream);

    // degree 1 or 2 (the old stack had no cubic OBB). segStart: [nSegs] first vertex of each segment (the curve
    // index buffer), radii: per vertex. obbs: [nSegs].
    void curveSegmentObbs(std::uint32_t degree, DevicePtr vertices, DevicePtr segStart, DevicePtr radii,
        DevicePtr obbs, std::uint32_t nSegs, StreamHandle stream);

    // For every segment i with flags[i] != 0 (flags 0/1, where = exclusive scan of flags), d = where[i]:
    //   verticesOut[d * vps + k] = vertices[segStart[i] + k], radiiOut[d * vps + k] = radii[segStart[i] + k]
    //   (k < vps = vertices per segment), indexOut[d] = d * vps, mapOut[d] = segToItem[i];
    // and *count = where[nSegs - 1] + flags[nSegs - 1].
    void compactSegments(DevicePtr vertices, DevicePtr radii, DevicePtr segStart, DevicePtr segToItem,
        DevicePtr flags, DevicePtr where, std::uint32_t nSegs, std::uint32_t verticesPerSegment, DevicePtr verticesOut,
        DevicePtr radiiOut, DevicePtr indexOut, DevicePtr mapOut, DevicePtr count, StreamHandle stream);

} // namespace rtcd::curve
