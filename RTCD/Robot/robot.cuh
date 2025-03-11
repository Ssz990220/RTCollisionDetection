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

void selectInsIdx(CUdeviceptr instances, CUdeviceptr idxMap, CUdeviceptr templateIns, CUdeviceptr templateMap,
    CUdeviceptr mask, RTCD::CUDABuffer& tmp, size_t N, size_t& tmpSize, CUdeviceptr count, cudaStream_t stream);