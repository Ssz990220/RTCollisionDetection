#include "sweptVolume.cuh"
#include <Utils/cuUtils.cuh>

__constant__ float splineWeights[512]; // Limit the product of N_CTRL_PTS * N_TRAJ_PTS to 512


cudaError_t fillWeight(float* weights, size_t nElements) {
    return cudaMemcpyToSymbol(splineWeights, weights, nElements * sizeof(float));
}

__inline__ __device__ __host__ size_t powOf2(size_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__global__ void genCtrlPtsKern(float* tPts, float* cPts, size_t nCPts, size_t nTPts) {
    size_t idx_x             = blockIdx.x * blockDim.x + threadIdx.x;
    size_t rndNTPts          = powOf2(nTPts);
    size_t nThreadsPerCPoint = rndNTPts / 2 * 3; // 96
    size_t nThreadsPerXYZ    = rndNTPts / 2; // 32
    size_t nSMemPerPoint     = rndNTPts / 2 * 3;

    float* sdataBlock = SharedMemory<float>();
    ///////////////////////////
    // point level variables //
    ///////////////////////////
    // global idx info
    size_t trajIdx = idx_x / (nCPts * nThreadsPerCPoint);
    // 0...48000, group by (64 / 2) * 3 * 16
    size_t pointIdxInTraj = idx_x % (nCPts * nThreadsPerCPoint) / (nThreadsPerCPoint);
    // 0...15, group by (64 / 2) * 3
    size_t weightPadding = pointIdxInTraj * nTPts;
    // 0, 64, 128, ...
    float* g_idata = tPts + trajIdx * nTPts * 3;
    // 64 * 3 floats of tPts per traj
    float* g_odata = cPts + trajIdx * nCPts * 3 + pointIdxInTraj * 3;
    // 16 * 3 floats of cPts per traj, 3 floats per point
    float* splineWeightsLocal = splineWeights + weightPadding;
    // 64 floats of splineWeights per traj

    // local idx info
    size_t pointIdxInBlock = threadIdx.x / nThreadsPerCPoint; // 0,1,2,3
    float* sdataPoint      = sdataBlock + pointIdxInBlock * nSMemPerPoint;
    ///////////////////////////
    //  xyz level variables  //
    ///////////////////////////
    // cg::thread_block_tile<nThreadsPerXYZ> tileXYZ = cg::tiled_partition<nThreadsPerXYZ>(cg::this_thread_block());
    size_t xyz = idx_x % nThreadsPerCPoint / nThreadsPerXYZ; // 0,1,2
    // size_t tid = tileXYZ.thread_rank();
    size_t tid   = idx_x % nThreadsPerXYZ; // 0...31
    float* sdata = sdataPoint + xyz * nThreadsPerXYZ;

    ////////////////////////////
    //         REDUCE         //
    ////////////////////////////

    sdata[tid] = g_idata[tid * 3 + xyz] * splineWeightsLocal[tid]
               + g_idata[(tid + nThreadsPerXYZ) * 3 + xyz] * splineWeightsLocal[tid + nThreadsPerXYZ]
                     * (tid + nThreadsPerXYZ < nTPts);

    __syncthreads();

    if (tid < rndNTPts / 4) {
        for (int offset = rndNTPts / 4; offset > 0; offset /= 2) {
            sdata[tid] = sdata[tid] + sdata[tid + offset];
            __syncthreads();
        }
    } else {
        return;
    }


    // write result for this block to global mem
    if (tid == 0) {
        g_odata[xyz] = sdata[0];
    }
}

void batchGenControlPoints(float* tPts, float* cPts, size_t nCurves, cudaStream_t stream, size_t nCPts, size_t nTPts) {

    // divide by 2 because we load 2 points into one shared_mem slot
    size_t nThreadsPerCPoint = powOf2(nTPts) / 2 * 3;
    size_t nPointsPerBlock   = 512 / nThreadsPerCPoint;
    size_t dimBlock          = nThreadsPerCPoint * nPointsPerBlock;
    size_t dimGrid           = (nCPts * nCurves + nPointsPerBlock - 1) / nPointsPerBlock;
    int smemSize             = nThreadsPerCPoint * sizeof(float) * nPointsPerBlock;
    // divide by 2 for what mentioned above

    genCtrlPtsKern<<<dimGrid, dimBlock, smemSize, stream>>>(tPts, cPts, nCPts, nTPts);
}

__device__ void OBB_1p(const float3& pt, const float r, OBB& obb) {
    obb.center         = pt;
    obb.halfSize       = make_float3(r, r, r);
    obb.orientation[0] = make_float3(1, 0, 0);
    obb.orientation[1] = make_float3(0, 1, 0);
    obb.orientation[2] = make_float3(0, 0, 1);
}

__device__ void OBB_2p(const float3& p1, const float3& p2, const float r, OBB& obb) {
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

__device__ void OBB_3p(const float3& p1, const float3& p2, const float3& p3, const float r, OBB& obb) {
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

    __syncthreads();

    float3 x = normalize(edge1);
    float ex = length1 + 2 * r;

    float3 y = normalize(edge2 - x * dot(edge2, x));
    float ey = dot(edge2, y) + 2 * r;

    float3 z = cross(x, y);
    float ez = 2 * r;

    obb.center         = point1 + (ex - 2 * r) / 2 * x + (ey - 2 * r) / 2 * y;
    obb.halfSize       = make_float3(ex / 2, ey / 2, ez / 2);
    obb.orientation[0] = x;
    obb.orientation[1] = y;
    obb.orientation[2] = z;
}

template <>
__global__ void genCurveOBBKern<1>(
    const float3* cPts, const int* segs, const float* radii, OBB* obb, const size_t nSegs) {
    size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x >= nSegs) {
        return;
    }
    const int segIdx = segs[idx_x];
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
    const float3* cPts, const int* segs, const float* radii, OBB* obb, const size_t nSegs) {
    size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x >= nSegs) {
        return;
    }
    const int segIdx = segs[idx_x];
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

template void genCurveOBB<1>(const CUdeviceptr cPts, const CUdeviceptr segs, const CUdeviceptr radii, CUdeviceptr obb,
    const size_t nSegs, const cudaStream_t stream);


template void genCurveOBB<2>(const CUdeviceptr cPts, const CUdeviceptr segs, const CUdeviceptr radii, CUdeviceptr obb,
    const size_t nSegs, const cudaStream_t stream);

__global__ void genIndexKern(
    uint* dst, const uint* flags, const uint* whereInDst, const uint segSize, const uint nEle) {
    size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x >= nEle) {
        return;
    }
    if (flags[idx_x] > 0) {
        dst[whereInDst[idx_x]] = whereInDst[idx_x] * segSize;
    }
}

void genIndex(CUdeviceptr dst, const CUdeviceptr flags, const CUdeviceptr whereInDst, const uint segSize,
    const uint nEle, const cudaStream_t stream) {
    const size_t nThreadsPerBlock = 512;
    const size_t nBlocks          = (nEle + nThreadsPerBlock - 1) / nThreadsPerBlock;
    genIndexKern<<<nBlocks, nThreadsPerBlock, 0, stream>>>(reinterpret_cast<uint*>(dst), reinterpret_cast<uint*>(flags),
        reinterpret_cast<uint*>(whereInDst), segSize, nEle);
}
