#pragma once
// Private kernels of rtcd_backend_mesh (instance-buffer updates of the per-stream robot IAS). Host-callable
// wrappers; every function enqueues on `stream`, never synchronizes, never allocates, accepts n == 0 (no launch)
// and throws CudaError on a launch failure.
//
// Instances are OptixInstance (80 bytes, 16-byte aligned) in template order (pose * numLinks + link); link poses
// are the pose-major 3x4 row-major transforms of rtcd/internal/kinematics.h (12 floats per (pose, link)).

#include <rtcd/limits.h>
#include <rtcd/types.h>

#include <cstddef>
#include <cstdint>

namespace rtcd::mesh_detail {

    // Old upInsVisibility (RTCD/Robot/iasRobotKernels.cu): for i < n, instances[i].transform = linkPoses[i] and
    // instances[i].visibilityMask = (counts == 0 || counts[i] > 0) ? 255 : 0. counts == 0 sets every mask to 255.
    void writeInstanceTransformsAndMasks(
        DevicePtr instances, DevicePtr linkPoses, DevicePtr counts, std::size_t n, StreamHandle stream);

    // Masked path, fused (G3): for i < linkObbs.count, instances[i].transform = linkPoses[i] and
    // instances[i].visibilityMask = 255 if linkObbs[i] overlaps any of sceneObbs, else 0 -- the same decision as
    // internal::obbOverlapCounts(...)[i] > 0 (shared device code core/obb_sat.cuh), without the count buffer.
    void writeInstanceTransformsAndOverlapMasks(DevicePtr instances, DevicePtr linkPoses, DeviceSpan<const Obb> linkObbs,
        DeviceSpan<const Obb> sceneObbs, StreamHandle stream);

    // Old upInsPoses (RTCD/Robot/robot.cu): for i < total, instances[i].transform = i < nActive ? linkPoses[i] :
    // park (the old __constant__ safeTransform, passed by value here). Visibility masks are not touched.
    struct ParkTransform {
        float m[kTransformFloats];
    };
    void writeInstanceTransformsParked(DevicePtr instances, DevicePtr linkPoses, std::size_t nActive, std::size_t total,
        const ParkTransform& park, StreamHandle stream);

} // namespace rtcd::mesh_detail
