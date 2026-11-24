#pragma once
// src/core (private, device code only): the OBB separating-axis test behind internal::obbOverlapCounts
// (algorithms.cu), shared with kernels of other modules that must take exactly the same overlap decision (the
// mesh backend's fused overlap + instance-write kernel). Include it only from .cu files compiled with the project's
// regular CUDA flags (no fast math), as algorithms.cu is: the decision is then bit-identical.
//
// Bit-identical port of RTCD/CollisionDetector/BBCollisionDetection.cu (isSeparatingAxis, checkOBBCollision) with
// the float3 helpers of RTCD/Utils/cuUtils.cuh written out in the same expression order (same rounding / FMA
// contraction as the old translation unit, which is also compiled without fast math).

#include <rtcd/limits.h>

#include <cuda_runtime.h>

namespace rtcd::internal::obb_sat {

    struct ObbF3 {
        float3 center;
        float3 halfSize;
        float3 orientation[3];
    };
    static_assert(sizeof(ObbF3) == sizeof(Obb), "rtcd::Obb must have the old RTCD::OBB layout");

    __forceinline__ __device__ float dot3(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    __forceinline__ __device__ float3 cross3(const float3& a, const float3& b) {
        return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    __forceinline__ __device__ float3 sub3(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __forceinline__ __device__ bool isSeparatingAxis(const ObbF3& obb1, const ObbF3& obb2, const float3& axis) {
        float3 u[3] = {obb1.orientation[0], obb1.orientation[1], obb1.orientation[2]};
        float3 v[3] = {obb2.orientation[0], obb2.orientation[1], obb2.orientation[2]};

        float ra = obb1.halfSize.x * fabsf(dot3(axis, u[0])) + obb1.halfSize.y * fabsf(dot3(axis, u[1]))
                 + obb1.halfSize.z * fabsf(dot3(axis, u[2]));

        float rb = obb2.halfSize.x * fabsf(dot3(axis, v[0])) + obb2.halfSize.y * fabsf(dot3(axis, v[1]))
                 + obb2.halfSize.z * fabsf(dot3(axis, v[2]));

        float3 t = sub3(obb2.center, obb1.center);
        float d  = fabsf(dot3(t, axis));

        return d > ra + rb;
    }

    inline __device__ bool checkObbCollision(const ObbF3& obb1, const ObbF3& obb2) {
        float3 u[3] = {obb1.orientation[0], obb1.orientation[1], obb1.orientation[2]};
        float3 v[3] = {obb2.orientation[0], obb2.orientation[1], obb2.orientation[2]};

        bool isColliding = true;
        for (int i = 0; i < 3; ++i) {
            if (isSeparatingAxis(obb1, obb2, u[i]) || isSeparatingAxis(obb1, obb2, v[i])
                || isSeparatingAxis(obb1, obb2, cross3(u[i], v[0]))
                || isSeparatingAxis(obb1, obb2, cross3(u[i], v[1]))
                || isSeparatingAxis(obb1, obb2, cross3(u[i], v[2]))) {
                isColliding = false;
                break;
            }
        }
        return isColliding;
    }

} // namespace rtcd::internal::obb_sat
