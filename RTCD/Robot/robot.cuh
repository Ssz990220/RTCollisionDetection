#pragma once

#include <Meshes/meshTypes.h>
#include <Utils/CUDABuffer.h>
#include <Utils/cuUtils.cuh>
#include <array>
#include <cuda.h>
#include <vector>

void gatherCntData(CUdeviceptr mask, CUdeviceptr maskSum, CUdeviceptr cnt, CUdeviceptr cntSum, CUdeviceptr output,
    const size_t arrayLength, cudaStream_t stream);

void upInsPoses(CUdeviceptr instances, CUdeviceptr poses, size_t nLinks, size_t totalLinks, cudaStream_t stream);
void upInsMasked(
    CUdeviceptr instances, CUdeviceptr poses, CUdeviceptr mask, size_t nLinks, size_t totalLinks, cudaStream_t stream);
void setFkineMasks(const float* cosMask, const float* sinMask, const float* oneMask, size_t d);

void setBaseAffine(std::array<float, 12>& baseTransform);
void setSafeTransform(std::array<float, 12>& transform);

void batchFkine(CUdeviceptr qs, CUdeviceptr TransformArray, size_t DOF, size_t nPoses, cudaStream_t stream = 0);
void batchFkineAdj(CUdeviceptr qs, CUdeviceptr TransformArray, size_t DOF, size_t nPoses, cudaStream_t stream = 0);
// batchFkineAdj + batchOBBTransform fused into a single launch (same outputs: TransformArray in the Adj layout,
// outOBBs[pose * (DOF + 1) + link] = obbTemplate[link] transformed by the link pose).
void batchFkineAdjOBB(CUdeviceptr qs, CUdeviceptr TransformArray, CUdeviceptr obbTemplate, CUdeviceptr outOBBs,
    size_t DOF, size_t nPoses, cudaStream_t stream = 0);
void batchFkineTraj(
    CUdeviceptr qs, CUdeviceptr TransformArray, size_t DOF, size_t nPoPerTraj, size_t nTraj, cudaStream_t stream = 0);

void batchSphrCntr(float* TransformArray, float* PointsArray, float* ShiftedPointsArray, const size_t nPoses,
    const int* pointIdxToLinkIdx, const size_t totalTemplatePointCount, const size_t DOF, const cudaStream_t stream);

void batchSphrCntrTraj(float* TransformArray, float* PointsArray, float* ShiftedPointsArray, const size_t nPoses,
    const size_t nTraj, const int* pointIdxToLinkIdx, const size_t totalTemplatePointCount, const size_t DOF,
    const cudaStream_t stream);

void batchSphrCntrAdj(float* TransformArray, float* PointsArray, float* ShiftedPointsArray, const size_t nPoses,
    const int* pointIdxToLinkIdx, const size_t totalTemplatePointCount, const size_t DOF, const cudaStream_t stream);

void batchRayVertices(float* TransformArray, float3* templatePts, float3* shiftedPts, int* rayLinkMap,
    size_t rayPerRobot, size_t DOF, size_t BATCHSIZE, const cudaStream_t stream);

void batchOBBTransform(const CUdeviceptr TransformArray, const CUdeviceptr obbs, CUdeviceptr output, size_t nPoses,
    size_t DOF, const cudaStream_t stream);

void batchOBBTransform2(const CUdeviceptr TransformArray, const CUdeviceptr obbs, CUdeviceptr output, size_t nPoses,
    size_t DOF, const cudaStream_t stream);

// tmpSize is computed once (when it is 0); capacityN = largest N ever passed, so the cached size stays valid.
void selectInsIdx(CUdeviceptr instances, CUdeviceptr idxMap, CUdeviceptr templateIns, CUdeviceptr templateMap,
    CUdeviceptr mask, RTCD::CUDABuffer& tmp, size_t N, size_t& tmpSize, CUdeviceptr count, cudaStream_t stream,
    size_t capacityN = 0);
// Curve robot: fused forward kinematics + sphere centers (+ quadratic/cubic B-spline control points when nCPts > 0)
// for nTraj trajectories, one launch. Output layout equals batchSphrCntrTraj's (nCPts == 0) or
// batchGenControlPoints' (nCPts > 0) output, bit-identical. Check fkineSphrTrajSupported() before calling.
cudaError_t setCurveSplineWeights(const float* weights, size_t nElements);
bool fkineSphrTrajSupported(size_t DOF, size_t nPoPerTraj, size_t nSpheres, size_t nCPts);
void batchFkineSphrTraj(CUdeviceptr qs, const float* sphereTemplate, const int* sphereLink, float* out, size_t DOF,
    size_t nPoPerTraj, size_t nTraj, size_t nSpheres, size_t nCPts, cudaStream_t stream);
