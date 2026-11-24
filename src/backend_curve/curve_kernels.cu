// Curve kernels of rtcd_backend_curve (declarations and layouts: curve_kernels.h).
//
// Ports of RTCD/Robot/robot.cu (fkineSphrTrajKern, genSphrTrajKer) and RTCD/Robot/sweptVolume.cu/.inl
// (genCtrlPtsKern, genCurveOBBKern, genIndexKern) plus RTCD/Utils/cuda/streamCompact.inl (dataCpyCurveKern). Thread
// layouts, launch configurations and the operation order of every float expression are kept exactly (FMA
// contraction follows the expression tree), so the outputs are bit-identical to the old kernels; the only change is
// that the kinematics table and the spline weights are read from device buffers instead of __constant__ globals.
// Compiled with the project's default CUDA flags (no fast math), like the old rtcd_legacy_kernels.
// Do not "clean up" an expression here without re-running tests/backend_curve/test_curve_kernels.
#include "curve_kernels.h"

#include <rtcd/errors.h>

#include <cuda_runtime.h>

#include <string>

namespace rtcd::curve {

    namespace {

        // ---- float3 helpers, exactly the old RTCD/Utils/cuUtils.cuh definitions ----
        __forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }
        __forceinline__ __device__ float3 operator-(const float3& a, const float3& b) {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }
        __forceinline__ __device__ float3 operator*(const float3& a, const float s) {
            return make_float3(a.x * s, a.y * s, a.z * s);
        }
        __forceinline__ __device__ float3 operator/(const float3& a, const float s) {
            float inv = 1.0f / s;
            return a * inv;
        }
        __forceinline__ __device__ float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        __forceinline__ __device__ float3 cross(const float3& a, const float3& b) {
            return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        }
        __forceinline__ __device__ float length(const float3& v) { return sqrtf(dot(v, v)); }
        __forceinline__ __device__ float3 normalize(const float3& v) {
            float invLen = 1.0f / sqrtf(dot(v, v));
            return v * invLen;
        }

        // Old RTCD::OBB with float3 members; same 15-float layout as rtcd::Obb.
        struct ObbF3 {
            float3 center;
            float3 halfSize;
            float3 orientation[3];
        };
        static_assert(sizeof(ObbF3) == sizeof(Obb), "ObbF3 must match rtcd::Obb");

        // Constant index tables of the 3x4 affine product (not robot data; old m1Indices / m2Indices).
        __constant__ int kM1Indices[9]  = {0, 1, 2, 4, 5, 6, 8, 9, 10};
        __constant__ int kM2Indices[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};

        // Old affineMult: 12 threads per transform; thread tid computes entry tid of m1 * m2.
        __forceinline__ __device__ void affineMult(float* m1, float* m2, float* result) {
            const size_t tid = threadIdx.x % 12;
            const size_t idx = tid / 4 * 3;
            const size_t idy = tid % 4 * 3;
            result[tid]      = m1[kM1Indices[idx]] * m2[kM2Indices[idy]] + m1[kM1Indices[idx + 1]] * m2[kM2Indices[idy + 1]]
                        + m1[kM1Indices[idx + 2]] * m2[kM2Indices[idy + 2]];
            __syncthreads();
            if (tid % 4 == 3) {
                result[tid] += m1[tid];
            }
        }

        // Old baseShiftParallel.
        __forceinline__ __device__ void baseShiftParallel(const KinematicsTable* tab, float* base) {
            const size_t tid = threadIdx.x % 12;
            base[tid]        = tab->base[tid];
        }

        // Old shiftCenterParallel (the row is taken from the y thread index).
        __forceinline__ __device__ void shiftCenterParallel(const float* p, const float* T, float* result) {
            const size_t pointIdx     = (blockIdx.y * blockDim.y + threadIdx.y) % 3;
            const size_t multTableIdx = pointIdx * 4;
            *result = T[multTableIdx] * p[0] + T[multTableIdx + 1] * p[1] + T[multTableIdx + 2] * p[2] + T[multTableIdx + 3];
        }

        __host__ __device__ inline std::size_t powOf2(std::size_t v) { // old sweptVolume.cu powOf2
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;
            return v;
        }

        std::size_t powOf2Host(std::size_t v) { // old robot.cu powOf2Host
            std::size_t p = 1;
            while (p < v) {
                p <<= 1;
            }
            return p;
        }

        // ---------------------------------------------------------------------------------------------------------
        // Old fkineSphrTrajKern: one block per trajectory, blockDim.x == trajPoints * 12. H = powOf2(nTPts) / 2.
        // ---------------------------------------------------------------------------------------------------------
        template <bool CTRL, int H>
        __global__ void fkineSphrTrajKern(const KinematicsTable* __restrict__ tab, const float* q,
            const float* sphereTemplate, const int* sphereLink, const float* curveSplineWeights, float* out,
            const int DOF, const int nPoPerTraj, const int nSpheres, const int nCPts) {
            extern __shared__ float smem[];
            const float* cMask = &tab->cosMask[0][0];
            const float* sMask = &tab->sinMask[0][0];
            const float* oMask = &tab->oneMask[0][0];
            float* localTf     = smem;                                   // [nPoPerTraj][DOF][12]
            float* tfs         = localTf + nPoPerTraj * DOF * 12;        // [DOF + 1][nPoPerTraj][12]
            float* tPts        = tfs + (DOF + 1) * nPoPerTraj * 12;      // CTRL only: [nSpheres][nPoPerTraj][3]

            const size_t trajIdx = blockIdx.x;

            // ---- forward kinematics (fkineTrajKernel) ----
            {
                const size_t poIdxInTraj       = threadIdx.x / 12;
                const size_t idxInPo           = threadIdx.x % 12;
                float* localFirstLinkTransform = localTf + poIdxInTraj * DOF * 12;
                const float* qPose             = q + (trajIdx * nPoPerTraj + poIdxInTraj) * DOF;
                for (int i = 0; i < DOF; i++) {
                    float sq, cq;
                    float qNow = *(qPose + i);
                    __sincosf(qNow, &sq, &cq);
                    size_t padding                   = i * 12 + idxInPo;
                    localFirstLinkTransform[padding] = cq * cMask[padding] + sq * sMask[padding] + oMask[padding];
                }
                const size_t TPadding = nPoPerTraj * 12;
                float* baseTransform  = tfs + poIdxInTraj * 12;
                baseShiftParallel(tab, baseTransform);
                __syncthreads();
                for (int i = 0; i < DOF; i++) {
                    affineMult(
                        baseTransform + i * TPadding, localFirstLinkTransform + i * 12, baseTransform + (i + 1) * TPadding);
                    __syncthreads();
                }
            }

            // ---- sphere centers (genSphrTrajKer); element e = (sphere * nPoPerTraj + pose) * 3 + xyz ----
            const int nTPtsElem = nSpheres * nPoPerTraj * 3;
            float* centers      = CTRL ? tPts : out + trajIdx * nTPtsElem;
            for (int e = threadIdx.x; e < nTPtsElem; e += blockDim.x) {
                const int xyz          = e % 3;
                const int pose         = (e / 3) % nPoPerTraj;
                const int sphere       = e / (3 * nPoPerTraj);
                const int linkIdx      = sphereLink[sphere];
                const float* T         = tfs + 12 * pose + 12 * (linkIdx * nPoPerTraj);
                const float* p         = sphereTemplate + sphere * 3;
                const int multTableIdx = xyz * 4;
                centers[e] = T[multTableIdx] * p[0] + T[multTableIdx + 1] * p[1] + T[multTableIdx + 2] * p[2]
                           + T[multTableIdx + 3];
            }

            if constexpr (CTRL) {
                __syncthreads();
                // ---- control points (genCtrlPtsKern), element e = (sphere * nCPts + ctrlPt) * 3 + xyz ----
                // genCtrlPtsKern: lane t of a group of H sums points t and t + H, then a halving tree reduction;
                // replayed here serially in the same order.
                const int nTPts     = nPoPerTraj;
                const int nCPtsElem = nSpheres * nCPts * 3;
                float* cOut         = out + trajIdx * nCPtsElem;
                for (int e = threadIdx.x; e < nCPtsElem; e += blockDim.x) {
                    const int xyz            = e % 3;
                    const int pointIdxInTraj = (e / 3) % nCPts;
                    const int sphere         = e / (3 * nCPts);
                    const float* g_idata     = tPts + sphere * nTPts * 3;
                    const float* w           = curveSplineWeights + pointIdxInTraj * nTPts;
                    float sdata[H];
#pragma unroll
                    for (int tid = 0; tid < H; tid++) {
                        float partial = g_idata[tid * 3 + xyz] * w[tid];
                        const int i2  = tid + H;
                        if (i2 < nTPts) {
                            partial += g_idata[i2 * 3 + xyz] * w[i2];
                        }
                        sdata[tid] = partial;
                    }
#pragma unroll
                    for (int offset = H / 2; offset > 0; offset /= 2) {
#pragma unroll
                        for (int tid = 0; tid < offset; tid++) {
                            sdata[tid] += sdata[tid + offset];
                        }
                    }
                    cOut[e] = sdata[0];
                }
            }
        }

        template <bool CTRL, int H>
        void launchFused(DevicePtr table, DevicePtr q, DevicePtr sphereCenters, DevicePtr sphereLink, DevicePtr weights,
            DevicePtr out, std::uint32_t dof, std::uint32_t trajPoints, std::uint32_t nTraj, std::uint32_t nSpheres,
            std::uint32_t ctrlPoints, StreamHandle stream) {
            fkineSphrTrajKern<CTRL, H><<<nTraj, trajPoints * 12, fusedCurvePointsSmem(dof, trajPoints, nSpheres, CTRL),
                stream>>>(reinterpret_cast<const KinematicsTable*>(table), reinterpret_cast<const float*>(q),
                reinterpret_cast<const float*>(sphereCenters), reinterpret_cast<const int*>(sphereLink),
                reinterpret_cast<const float*>(weights), reinterpret_cast<float*>(out), static_cast<int>(dof),
                static_cast<int>(trajPoints), static_cast<int>(nSpheres), static_cast<int>(ctrlPoints));
        }

        // ---------------------------------------------------------------------------------------------------------
        // Old genSphrTrajKer: x = pose of a trajectory, y = coordinate of a sphere center.
        // ---------------------------------------------------------------------------------------------------------
        __global__ void genSphrTrajKer(const float* TransformArray, const float* PointsArray, float* ShiftedPointsArray,
            const size_t nPoPerTraj, const size_t nTraj, const int* pointIdxToLinkIdx, const size_t totalPointCount,
            const int DOF) {
            size_t idx_x       = blockIdx.x * blockDim.x + threadIdx.x;
            size_t idx_y       = blockIdx.y * blockDim.y + threadIdx.y;
            size_t trajIdx     = idx_x / nPoPerTraj;
            size_t poIdxInTraj = idx_x % nPoPerTraj;
            if (idx_x >= nPoPerTraj * nTraj) {
                return;
            }
            if (idx_y >= totalPointCount * 3) {
                return;
            }
            size_t linkIdx = pointIdxToLinkIdx[idx_y / 3];
            const float* currentTransform =
                TransformArray + 12 * poIdxInTraj + 12 * (linkIdx * nPoPerTraj) + trajIdx * 12 * (nPoPerTraj * (DOF + 1));
            const float* currentTemplatePoint = PointsArray + idx_y / 3 * 3;
            float* currentResultPoints        = ShiftedPointsArray + trajIdx * nPoPerTraj * totalPointCount * 3
                                       + (idx_y / 3 * nPoPerTraj) * 3 + poIdxInTraj * 3 + idx_y % 3;
            shiftCenterParallel(currentTemplatePoint, currentTransform, currentResultPoints);
        }

        // ---------------------------------------------------------------------------------------------------------
        // Old genCtrlPtsKern (the fixed version: guarded tree reduction, no divergent barrier, no out-of-bounds).
        // ---------------------------------------------------------------------------------------------------------
        __global__ void genCtrlPtsKern(
            const float* tPts, float* cPts, const float* splineWeights, size_t nCPts, size_t nTPts, size_t nCurves) {
            const size_t idx_x             = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t rndNTPts          = powOf2(nTPts);
            const size_t nThreadsPerXYZ    = rndNTPts / 2;
            const size_t nThreadsPerCPoint = nThreadsPerXYZ * 3;

            extern __shared__ float sdataBlock[];

            const size_t globalPoint    = idx_x / nThreadsPerCPoint;
            const bool active           = globalPoint < nCPts * nCurves;
            const size_t trajIdx        = globalPoint / nCPts;
            const size_t pointIdxInTraj = globalPoint % nCPts;
            const size_t xyz            = idx_x % nThreadsPerCPoint / nThreadsPerXYZ;
            const size_t tid            = idx_x % nThreadsPerXYZ;

            float* sdata = sdataBlock + (threadIdx.x / nThreadsPerCPoint) * nThreadsPerCPoint + xyz * nThreadsPerXYZ;

            float partial = 0.0f;
            if (active) {
                const float* g_idata = tPts + trajIdx * nTPts * 3;
                const float* w       = splineWeights + pointIdxInTraj * nTPts;
                partial              = g_idata[tid * 3 + xyz] * w[tid];
                const size_t i2      = tid + nThreadsPerXYZ;
                if (i2 < nTPts) {
                    partial += g_idata[i2 * 3 + xyz] * w[i2];
                }
            }
            sdata[tid] = partial;

            for (size_t offset = nThreadsPerXYZ / 2; offset > 0; offset /= 2) {
                __syncthreads();
                if (tid < offset) {
                    sdata[tid] += sdata[tid + offset];
                }
            }

            if (active && tid == 0) {
                cPts[trajIdx * nCPts * 3 + pointIdxInTraj * 3 + xyz] = sdata[0];
            }
        }

        // ---------------------------------------------------------------------------------------------------------
        // Old curve segment OBBs (sweptVolume.cu). OBB_3p is kept exactly as in the old code, including its anchor
        // point choice (see the report / design doc Q5); its __syncthreads() inside divergent code is dropped (it
        // guarded no shared memory, so no value changes).
        // ---------------------------------------------------------------------------------------------------------
        __device__ void OBB_1p(const float3& pt, const float r, ObbF3& obb) {
            obb.center         = pt;
            obb.halfSize       = make_float3(r, r, r);
            obb.orientation[0] = make_float3(1, 0, 0);
            obb.orientation[1] = make_float3(0, 1, 0);
            obb.orientation[2] = make_float3(0, 0, 1);
        }

        __device__ void OBB_2p(const float3& p1, const float3& p2, const float r, ObbF3& obb) {
            obb.center = (p1 + p2) / 2.0f;
            float3 e1  = p2 - p1;
            float3 e1n = normalize(e1);
            float l1   = length(e1) / 2;
            float3 n1  = cross(e1n, make_float3(1, 0, 0));
            if (length(n1) == 0) {
                n1 = cross(e1n, make_float3(0, 1, 0));
                if (length(n1) == 0) {
                    n1 = cross(e1n, make_float3(0, 0, 1));
                }
            }
            n1                 = normalize(n1);
            float3 n2          = cross(e1n, n1);
            obb.halfSize       = make_float3(l1 + r, r, r);
            obb.orientation[0] = e1n;
            obb.orientation[1] = n1;
            obb.orientation[2] = n2;
        }

        __device__ void OBB_3p(const float3& p1, const float3& p2, const float3& p3, const float r, ObbF3& obb) {
            float3 e1 = p2 - p1;
            float3 e2 = p3 - p1;
            float3 e3 = p3 - p2;

            float l1 = length(e1);
            float l2 = length(e2);
            float l3 = length(e3);

            float3 edge1, edge2, point1;
            float length1;

            if (l1 > l2 && l1 > l3) {
                edge1   = e1;
                length1 = l1;
                edge2   = e2;
                point1  = p1;
            } else if (l2 > l1 && l2 > l3) {
                edge1   = e2;
                edge2   = e3;
                length1 = l2;
                point1  = p2;
            } else {
                edge1   = e3;
                edge2   = e1;
                length1 = l3;
                point1  = p3;
            }

            float3 x = normalize(edge1);
            float ex = length1 + 2 * r;

            float3 y = normalize(edge2 - x * dot(edge2, x));
            float ey = dot(edge2, y) + 2 * r;

            float3 z = cross(x, y);
            float ez = 2 * r;

            obb.center         = point1 + x * ((ex - 2 * r) / 2) + y * ((ey - 2 * r) / 2);
            obb.halfSize       = make_float3(ex / 2, ey / 2, ez / 2);
            obb.orientation[0] = x;
            obb.orientation[1] = y;
            obb.orientation[2] = z;
        }

        template <int degree>
        __global__ void genCurveOBBKern(const float3* cPts, const unsigned int* segs, const float* radii, ObbF3* obb,
            const size_t nSegs);

        template <>
        __global__ void genCurveOBBKern<1>(
            const float3* cPts, const unsigned int* segs, const float* radii, ObbF3* obb, const size_t nSegs) {
            size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx_x >= nSegs) {
                return;
            }
            const int segIdx = static_cast<int>(segs[idx_x]);
            const float3 p1  = cPts[segIdx];
            const float3 p2  = cPts[segIdx + 1];

            float l = length(p2 - p1);
            if (l == 0) {
                OBB_1p(p1, radii[segIdx], obb[idx_x]);
            } else {
                OBB_2p(p1, p2, radii[segIdx], obb[idx_x]);
            }
        }

        template <>
        __global__ void genCurveOBBKern<2>(
            const float3* cPts, const unsigned int* segs, const float* radii, ObbF3* obb, const size_t nSegs) {
            size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx_x >= nSegs) {
                return;
            }
            const int segIdx = static_cast<int>(segs[idx_x]);
            const float3 p1  = cPts[segIdx];
            const float3 p2  = cPts[segIdx + 1];
            const float3 p3  = cPts[segIdx + 2];

            float l1 = length(p2 - p1);
            float l2 = length(p3 - p2);
            float l3 = length(p1 - p3);

            if (l1 == 0) {
                if (l2 == 0) {
                    OBB_1p(p3, radii[segIdx], obb[idx_x]);
                } else {
                    OBB_2p(p1, p3, radii[segIdx], obb[idx_x]);
                }
            } else {
                if (l2 == 0) {
                    OBB_2p(p1, p2, radii[segIdx], obb[idx_x]);
                } else {
                    if (l3 == 0) {
                        OBB_2p(p2, p3, radii[segIdx], obb[idx_x]);
                    } else {
                        OBB_3p(p1, p2, p3, radii[segIdx], obb[idx_x]);
                    }
                }
            }
        }

        // ---------------------------------------------------------------------------------------------------------
        // Compaction of the OBB-surviving segments (old genIndexKern + dataCpyCurveKern x2, plus the item map and
        // the count the old path read back separately).
        // ---------------------------------------------------------------------------------------------------------
        __global__ void compactSegmentsKern(const float* vertices, const float* radii, const unsigned int* segStart,
            const unsigned int* segToItem, const unsigned int* flags, const unsigned int* where, unsigned int nSegs,
            unsigned int vps, float* verticesOut, float* radiiOut, unsigned int* indexOut, unsigned int* mapOut,
            unsigned int* count) {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= nSegs) {
                return;
            }
            const unsigned int f = flags[i];
            const unsigned int d = where[i];
            if (f > 0) {
                const unsigned int s = segStart[i];
                for (unsigned int k = 0; k < vps; ++k) {
                    verticesOut[(d * vps + k) * 3 + 0] = vertices[(s + k) * 3 + 0];
                    verticesOut[(d * vps + k) * 3 + 1] = vertices[(s + k) * 3 + 1];
                    verticesOut[(d * vps + k) * 3 + 2] = vertices[(s + k) * 3 + 2];
                    radiiOut[d * vps + k]              = radii[s + k];
                }
                indexOut[d] = d * vps;
                mapOut[d]   = segToItem[i];
            }
            if (i == nSegs - 1) {
                *count = d + (f > 0 ? 1u : 0u);
            }
        }

        void checkLaunch() { RTCD_CUDA_CHECK(cudaGetLastError()); }

    } // namespace

    std::size_t fusedCurvePointsSmem(std::uint32_t dof, std::uint32_t trajPoints, std::uint32_t nSpheres, bool ctrl) {
        const std::size_t DOF = dof, n = trajPoints, s = nSpheres;
        return sizeof(float) * (n * DOF * 12 + (DOF + 1) * n * 12 + (ctrl ? s * n * 3 : 0));
    }

    bool fusedCurvePointsSupported(std::uint32_t dof, std::uint32_t trajPoints, std::uint32_t nSpheres,
        std::uint32_t ctrlPoints) {
        const bool ctrl     = ctrlPoints > 0;
        const std::size_t H = powOf2Host(trajPoints) / 2;
        return dof >= 1 && dof <= kMaxDof && std::size_t(trajPoints) * 12 <= 1024
            && fusedCurvePointsSmem(dof, trajPoints, nSpheres, ctrl) <= 48 * 1024 && (!ctrl || (H >= 1 && H <= 256));
    }

    void fusedCurvePoints(DevicePtr table, std::uint32_t dof, DevicePtr q, DevicePtr sphereCenters,
        DevicePtr sphereLink, DevicePtr weights, DevicePtr out, std::uint32_t trajPoints, std::uint32_t nTraj,
        std::uint32_t nSpheres, std::uint32_t ctrlPoints, StreamHandle stream) {
        if (nTraj == 0) {
            return;
        }
        if (!fusedCurvePointsSupported(dof, trajPoints, nSpheres, ctrlPoints)) {
            throw InvalidArgument("rtcd::curve::fusedCurvePoints: configuration not supported by the fused kernel (dof "
                                  + std::to_string(dof) + ", trajectory points " + std::to_string(trajPoints) + ", spheres "
                                  + std::to_string(nSpheres) + ")");
        }
        if (ctrlPoints == 0) {
            launchFused<false, 1>(table, q, sphereCenters, sphereLink, weights, out, dof, trajPoints, nTraj, nSpheres, 0,
                stream);
            checkLaunch();
            return;
        }
        switch (powOf2Host(trajPoints) / 2) {
#define RTCD_CURVE_FUSED_CASE(h)                                                                                       \
    case h:                                                                                                            \
        launchFused<true, h>(table, q, sphereCenters, sphereLink, weights, out, dof, trajPoints, nTraj, nSpheres,      \
            ctrlPoints, stream);                                                                                       \
        break;
            RTCD_CURVE_FUSED_CASE(1)
            RTCD_CURVE_FUSED_CASE(2)
            RTCD_CURVE_FUSED_CASE(4)
            RTCD_CURVE_FUSED_CASE(8)
            RTCD_CURVE_FUSED_CASE(16)
            RTCD_CURVE_FUSED_CASE(32)
            RTCD_CURVE_FUSED_CASE(64)
            RTCD_CURVE_FUSED_CASE(128)
            RTCD_CURVE_FUSED_CASE(256)
#undef RTCD_CURVE_FUSED_CASE
        default: throw InvalidArgument("rtcd::curve::fusedCurvePoints: unsupported reduction width");
        }
        checkLaunch();
    }

    void sphereCentersTraj(DevicePtr linkPoses, DevicePtr sphereCenters, DevicePtr sphereLink, DevicePtr out,
        std::uint32_t trajPoints, std::uint32_t nTraj, std::uint32_t nSpheres, std::uint32_t dof, StreamHandle stream) {
        if (nTraj == 0 || trajPoints == 0 || nSpheres == 0) {
            return;
        }
        // old batchSphrCntrTraj
        dim3 numThreads(32, 16, 1);
        dim3 numBlocks(static_cast<unsigned>((std::size_t(trajPoints) * nTraj + numThreads.x - 1) / numThreads.x),
            static_cast<unsigned>((std::size_t(nSpheres) * 3 + numThreads.y - 1) / numThreads.y), 1);
        genSphrTrajKer<<<numBlocks, numThreads, 0, stream>>>(reinterpret_cast<const float*>(linkPoses),
            reinterpret_cast<const float*>(sphereCenters), reinterpret_cast<float*>(out), trajPoints, nTraj,
            reinterpret_cast<const int*>(sphereLink), nSpheres, static_cast<int>(dof));
        checkLaunch();
    }

    void controlPoints(DevicePtr trajPointsIn, DevicePtr ctrlPointsOut, DevicePtr weights, std::uint32_t nCurves,
        std::uint32_t ctrlPoints, std::uint32_t trajPoints, StreamHandle stream) {
        if (nCurves == 0 || ctrlPoints == 0) {
            return;
        }
        if (trajPoints < 1 || trajPoints > kMaxTrajectoryPoints) {
            throw InvalidArgument("rtcd::curve::controlPoints: trajectory points " + std::to_string(trajPoints)
                                  + " out of [1, kMaxTrajectoryPoints]");
        }
        // old batchGenControlPoints
        const std::size_t nThreadsPerCPoint = powOf2(trajPoints) / 2 * 3;
        const std::size_t nPointsPerBlock   = 512 / nThreadsPerCPoint;
        const std::size_t dimBlock          = nThreadsPerCPoint * nPointsPerBlock;
        const std::size_t dimGrid           = (std::size_t(ctrlPoints) * nCurves + nPointsPerBlock - 1) / nPointsPerBlock;
        const std::size_t smemSize          = nThreadsPerCPoint * sizeof(float) * nPointsPerBlock;
        genCtrlPtsKern<<<static_cast<unsigned>(dimGrid), static_cast<unsigned>(dimBlock), smemSize, stream>>>(
            reinterpret_cast<const float*>(trajPointsIn), reinterpret_cast<float*>(ctrlPointsOut),
            reinterpret_cast<const float*>(weights), ctrlPoints, trajPoints, nCurves);
        checkLaunch();
    }

    void curveSegmentObbs(std::uint32_t degree, DevicePtr vertices, DevicePtr segStart, DevicePtr radii,
        DevicePtr obbs, std::uint32_t nSegs, StreamHandle stream) {
        if (nSegs == 0) {
            return;
        }
        // old genCurveOBB<degree>
        const std::size_t nThreads = 128;
        const std::size_t nBlocks  = (nSegs + nThreads - 1) / nThreads;
        const auto* p              = reinterpret_cast<const float3*>(vertices);
        const auto* s              = reinterpret_cast<const unsigned int*>(segStart);
        const auto* r              = reinterpret_cast<const float*>(radii);
        auto* o                    = reinterpret_cast<ObbF3*>(obbs);
        if (degree == 1) {
            genCurveOBBKern<1><<<static_cast<unsigned>(nBlocks), static_cast<unsigned>(nThreads), 0, stream>>>(p, s, r, o, nSegs);
        } else if (degree == 2) {
            genCurveOBBKern<2><<<static_cast<unsigned>(nBlocks), static_cast<unsigned>(nThreads), 0, stream>>>(p, s, r, o, nSegs);
        } else {
            throw UnsupportedError("curve segment OBBs are implemented for linear and quadratic curves only (degree "
                                   + std::to_string(degree) + ")");
        }
        checkLaunch();
    }

    void compactSegments(DevicePtr vertices, DevicePtr radii, DevicePtr segStart, DevicePtr segToItem,
        DevicePtr flags, DevicePtr where, std::uint32_t nSegs, std::uint32_t verticesPerSegment, DevicePtr verticesOut,
        DevicePtr radiiOut, DevicePtr indexOut, DevicePtr mapOut, DevicePtr count, StreamHandle stream) {
        if (nSegs == 0) {
            return;
        }
        const unsigned int nThreads = 256;
        const unsigned int nBlocks  = (nSegs + nThreads - 1) / nThreads;
        compactSegmentsKern<<<nBlocks, nThreads, 0, stream>>>(reinterpret_cast<const float*>(vertices),
            reinterpret_cast<const float*>(radii), reinterpret_cast<const unsigned int*>(segStart),
            reinterpret_cast<const unsigned int*>(segToItem), reinterpret_cast<const unsigned int*>(flags),
            reinterpret_cast<const unsigned int*>(where), nSegs, verticesPerSegment, reinterpret_cast<float*>(verticesOut),
            reinterpret_cast<float*>(radiiOut), reinterpret_cast<unsigned int*>(indexOut),
            reinterpret_cast<unsigned int*>(mapOut), reinterpret_cast<unsigned int*>(count));
        checkLaunch();
    }

} // namespace rtcd::curve
