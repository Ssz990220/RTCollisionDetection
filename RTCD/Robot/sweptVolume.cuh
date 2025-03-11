#pragma once
#include <Meshes/meshTypes.h>
#include <Robot/batchRobotConfig.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace RTCD;

cudaError_t fillWeight(float* weights, size_t nWeights);

__global__ void genCtrlPtsKern(
    float* trajPoints, float* controlPoints, size_t nCurves, size_t nCtrlPoints, size_t nTrajPts);

void batchGenControlPoints(
    float* trajPoints, float* controlPoints, size_t nCurves, cudaStream_t stream, size_t nCPts, size_t nTPts);

template <int degree>
void genCurveOBB(const CUdeviceptr cPts, const CUdeviceptr segs, const CUdeviceptr radii, CUdeviceptr obb,
    const size_t nSegs, const cudaStream_t stream);

template <int degree>
__global__ void genCurveOBBKern(const float3* cPts, const int* segs, const float* radii, OBB* obb, const size_t nSegs);

void genIndex(CUdeviceptr dst, const CUdeviceptr flags, const CUdeviceptr whereInDst, const uint segSize,
    const uint nEle, const cudaStream_t stream);

#if defined(__CUDACC__)
#include "sweptVolume.inl"
#endif
