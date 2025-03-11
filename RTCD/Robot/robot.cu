#include <Robot/robot.cuh>
#include <Utils/optix7.h>
#include <cub/cub.cuh>

__global__ void gatherCntKern(const uint* mask, const uint* maskSum, const uint* cnt, const uint* cntSum, uint* output,
    const size_t arrayLength) {
    output[0] = mask[arrayLength - 1];
    output[1] = maskSum[arrayLength - 1];
    output[2] = cnt[arrayLength - 1];
    output[3] = cntSum[arrayLength - 1];
}

void gatherCntData(CUdeviceptr mask, CUdeviceptr maskSum, CUdeviceptr cnt, CUdeviceptr cntSum, CUdeviceptr output,
    const size_t arrayLength, cudaStream_t stream) {
    gatherCntKern<<<1, 1, 0, stream>>>(reinterpret_cast<uint*>(mask), reinterpret_cast<uint*>(maskSum),
        reinterpret_cast<uint*>(cnt), reinterpret_cast<uint*>(cntSum), reinterpret_cast<uint*>(output), arrayLength);
}

// we use 9 threads to compute the affine matrix multiplication
// because the rotation part have 9 elements
// then we use 3 of the 9 threads to compute the translation part
// because the translation part is different which violates the SIMT principle
// The indices of entries are read from a table.
__constant__ float sMask[8 * 12];
__constant__ float oMask[8 * 12];
__constant__ float cMask[8 * 12];

void setFkineMasks(const float* cosMask, const float* sinMask, const float* oneMask, size_t d) {
    CUDA_CHECK(cudaMemcpyToSymbol(cMask, cosMask, sizeof(float) * d));
    CUDA_CHECK(cudaMemcpyToSymbol(sMask, sinMask, sizeof(float) * d));
    CUDA_CHECK(cudaMemcpyToSymbol(oMask, oneMask, sizeof(float) * d));
}

__constant__ int m1Indices[9]  = {0, 1, 2, 4, 5, 6, 8, 9, 10};
__constant__ int m2Indices[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
__forceinline__ __device__ void affineMult(float* m1, float* m2, float* result) {

    const size_t tid = threadIdx.x % 12; // at least 12 threads are invoked for a single robot
    const size_t idx = tid / 4 * 3;
    const size_t idy = tid % 4 * 3;
    result[tid]      = m1[m1Indices[idx]] * m2[m2Indices[idy]] + m1[m1Indices[idx + 1]] * m2[m2Indices[idy + 1]]
                + m1[m1Indices[idx + 2]] * m2[m2Indices[idy + 2]];
    __syncthreads();
    if (tid % 4 == 3) {
        result[tid] += m1[tid];
    }
}

// baseShiftParallel
__constant__ float baseAffine[12];

void setBaseAffine(std::array<float, 12>& baseTransform) {
    cudaMemcpyToSymbol(baseAffine, baseTransform.data(), sizeof(float) * 12);
}

__forceinline__ __device__ void baseShiftParallel(float* base, int robotIdx, int bodyIdx = 0) {
    const size_t tid = threadIdx.x % 12;
    base[tid]        = baseAffine[tid];
}


__constant__ float4 safeTransform[3];

void setSafeTransform(std::array<float, 12>& transform) {
    cudaMemcpyToSymbol(safeTransform, transform.data(), sizeof(float) * 12);
}

__global__ void upInsPosesKer(OptixInstance* instances, float4* poses, size_t nLinks, size_t totalLinks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalLinks) {
        return;
    }
    float4* pose;
    if (idx >= nLinks) {
        pose = safeTransform;
    } else {
        pose = poses + idx * 3;
    }
    memcpy(instances[idx].transform, (void*) pose, sizeof(float) * 12);
}


void upInsPoses(CUdeviceptr instances, CUdeviceptr poses, size_t nLinks, size_t totalLinks, cudaStream_t stream) {
    const unsigned int numThreads = 128;
    const unsigned int numBlocks  = (totalLinks + numThreads - 1) / numThreads;
    upInsPosesKer<<<numBlocks, numThreads, 0, stream>>>(
        reinterpret_cast<OptixInstance*>(instances), reinterpret_cast<float4*>(poses), nLinks, totalLinks);
}

__global__ void upInsMaskedKer(OptixInstance* instances, float4* poses, uint* mask, size_t nLinks, size_t totalLinks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalLinks) {
        return;
    }
    float4* pose;
    if (mask[idx] == 0 || idx >= nLinks) {
        pose = safeTransform;
    } else {
        pose = poses + idx * 3;
    }
    memcpy(instances[idx].transform, (void*) pose, sizeof(float) * 12);
}


void upInsMasked(
    CUdeviceptr instances, CUdeviceptr poses, CUdeviceptr mask, size_t nLinks, size_t totalLinks, cudaStream_t stream) {
    const unsigned int numThreads = 128;
    const unsigned int numBlocks  = (totalLinks + numThreads - 1) / numThreads;
    upInsMaskedKer<<<numBlocks, numThreads, 0, stream>>>(reinterpret_cast<OptixInstance*>(instances),
        reinterpret_cast<float4*>(poses), reinterpret_cast<uint*>(mask), nLinks, totalLinks);
}

__global__ void fkineKernel(float* q, float* TransformArray, size_t DOF, size_t nPoses) {
    extern __shared__ float
        sharedTransforms[]; // DOF * 12 * nPosesPerBlock = 12 * nThreadsPerBlock, nPosesPerBlock = nThreadsPerBlock / 6
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // const size_t nLinks = DOF + 1;              // we presume that there is no branches in the kinematic tree
    if (idx >= nPoses * 12) {
        return; // since we manually assigned the size of block, this might be unnecessary
    }
    size_t robotGlobalIdx          = idx / 12;
    size_t localIdx                = idx % 12;
    size_t robotLocalIdx           = threadIdx.x / 12;
    float* localFirstLinkTransform = sharedTransforms + robotLocalIdx * DOF * 12;
    for (int i = 0; i < DOF; i++) {
        float sq, cq;
        float qNow = *(q + robotGlobalIdx * DOF + i);
        __sincosf(qNow, &sq, &cq);
        size_t padding = i * 12 + localIdx;
        // Compute local transform
        localFirstLinkTransform[padding] = cq * cMask[padding] + sq * sMask[padding] + oMask[padding];
    }
    // Compute forward kinematics on the first thread of each six-threads bundle
    // size_t robotID = threadIdx.x / 12 % nRobots;
    size_t TPadding      = nPoses * 12;
    float* baseTransform = TransformArray + robotGlobalIdx * 12;
    baseShiftParallel(baseTransform, 0); // TODO: check if this is correct
    __syncthreads();
    for (int i = 0; i < DOF; i++) {
        affineMult(baseTransform + i * TPadding, localFirstLinkTransform + i * 12, baseTransform + (i + 1) * TPadding);
        __syncthreads();
    }
}
// Transform of one link is stored in a continuous memory
// Link1[tf0, tf1, ..., tfnPoses], Link2[tf0, tf1, ..., tfnPoses], ...
void batchFkine(CUdeviceptr qs, CUdeviceptr TransformArray, size_t DOF, size_t nPoses, cudaStream_t stream) {
    const size_t numPosePerBlock = nPoses < 42 ? nPoses : 42;
    const size_t numThreads      = numPosePerBlock * 12;
    const size_t numBlocks       = (nPoses + numPosePerBlock - 1) / numPosePerBlock;

    fkineKernel<<<numBlocks, numThreads, DOF * 12 * numPosePerBlock * sizeof(float), stream>>>(
        reinterpret_cast<float*>(qs), reinterpret_cast<float*>(TransformArray), DOF, nPoses);
}

__global__ void fkineTrajKernel(float* q, float* TransformArray, size_t DOF, size_t nPoPerTraj, size_t nTraj) {
    extern __shared__ float
        sharedTransforms[]; // DOF * 12 * nPosesPerBlock = 12 * nThreadsPerBlock, nPosesPerBlock = nThreadsPerBlock / 6
    size_t idx     = blockIdx.x * blockDim.x + threadIdx.x;
    size_t trajIdx = idx / (nPoPerTraj * 12);
    // const size_t nLinks = DOF + 1;              // we presume that there is no branches in the kinematic tree
    if (idx >= nPoPerTraj * nTraj * 12) {
        return; // since we manually assigned the size of block, this might be unnecessary
    }
    size_t poIdxInTraj             = (idx - trajIdx * nPoPerTraj * 12) / 12;
    size_t idxInPo                 = idx % 12;
    size_t poIdxInBlk              = threadIdx.x / 12;
    float* localFirstLinkTransform = sharedTransforms + poIdxInBlk * DOF * 12;
    for (int i = 0; i < DOF; i++) {
        float sq, cq;
        float qNow = *(q + idx / 12 * DOF + i);
        __sincosf(qNow, &sq, &cq);
        size_t padding = i * 12 + idxInPo;
        // Compute local transform
        localFirstLinkTransform[padding] = cq * cMask[padding] + sq * sMask[padding] + oMask[padding];
    }
    // Compute forward kinematics on the first thread of each six-threads bundle
    // size_t robotID = threadIdx.x / 12 % nRobots;
    size_t TPadding      = nPoPerTraj * 12;
    float* baseTransform = TransformArray + poIdxInTraj * 12 + trajIdx * 12 * (DOF + 1) * nPoPerTraj;
    baseShiftParallel(baseTransform, 0); // TODO: check if this is correct
    __syncthreads();
    for (int i = 0; i < DOF; i++) {
        affineMult(baseTransform + i * TPadding, localFirstLinkTransform + i * 12, baseTransform + (i + 1) * TPadding);
        __syncthreads();
    }
}

// Transform of one link in one traj is stored in a continuous memory, and the transform of one traj is stored in a
// continuous memory
// Traj1[Link0[tf0, tf1, ... tfDOF], Link1[tf0, tf1, ... tfDOF], ...], Traj2[Link0[tf0, tf1, ... tfDOF], Link1[tf0,
// tf1, ... tfDOF], ...], ...
void batchFkineTraj(
    CUdeviceptr qs, CUdeviceptr TransformArray, size_t DOF, size_t nPoPerTraj, size_t nTraj, cudaStream_t stream) {
    const size_t numPosePerBlock = nPoPerTraj < 42 ? nPoPerTraj : 42;
    const size_t numThreads      = numPosePerBlock * 12;
    const size_t numBlocks       = (nPoPerTraj * nTraj + numPosePerBlock - 1) / numPosePerBlock;

    fkineTrajKernel<<<numBlocks, numThreads, DOF * 12 * numPosePerBlock * sizeof(float), stream>>>(
        reinterpret_cast<float*>(qs), reinterpret_cast<float*>(TransformArray), DOF, nPoPerTraj, nTraj);
}

__global__ void fkineKernelAdj(float* q, float* TransformArray, unsigned int DOF, unsigned int nPoses) {
    extern __shared__ float
        sharedTransforms[]; // DOF * 12 * nPosesPerBlock = 12 * nThreadsPerBlock, nPosesPerBlock = nThreadsPerBlock / 6
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // const size_t nLinks = DOF + 1;              // we presume that there is no branches in the kinematic tree
    if (idx >= nPoses * 12) {
        return; // since we manually assigned the size of block, this might be unnecessary
    }
    size_t robotGlobalIdx          = idx / 12;
    size_t localIdx                = idx % 12;
    size_t robotLocalIdx           = threadIdx.x / 12;
    float* localFirstLinkTransform = sharedTransforms + robotLocalIdx * DOF * 12;
    for (int i = 0; i < DOF; i++) {
        float sq, cq;
        float qNow = *(q + robotGlobalIdx * DOF + i);
        __sincosf(qNow, &sq, &cq);
        size_t padding = i * 12 + localIdx;
        // Compute local transform
        localFirstLinkTransform[padding] = cq * cMask[padding] + sq * sMask[padding] + oMask[padding];
    }
    // Compute forward kinematics on the first thread of each six-threads bundle
    // size_t robotID = threadIdx.x / 12 % nRobots;
    float* baseTransform = TransformArray + robotGlobalIdx * 12 * (DOF + 1);
    baseShiftParallel(baseTransform, 0); // TODO: check if this is correct
    __syncthreads();
    for (int i = 0; i < DOF; i++) {
        affineMult(baseTransform + i * 12, localFirstLinkTransform + i * 12, baseTransform + (i + 1) * 12);
        __syncthreads();
    }
}

// Transform of one robot pose is stored in continuous memory
// Pose1[tf0, tf1, ... tfDOF], Pose2[tf0, tf1, ... tfDOF], ...
void batchFkineAdj(CUdeviceptr qs, CUdeviceptr TransformArray, size_t DOF, size_t nPoses, cudaStream_t stream) {
    const size_t numPosePerBlock = nPoses < 32 ? nPoses : 32;
    const size_t numThreads      = numPosePerBlock * 12;
    const size_t numBlocks       = (nPoses + numPosePerBlock - 1) / numPosePerBlock;
    fkineKernelAdj<<<numBlocks, numThreads, DOF * 12 * numPosePerBlock * sizeof(float), stream>>>(
        reinterpret_cast<float*>(qs), reinterpret_cast<float*>(TransformArray), DOF, nPoses);
}

// TODO1: 2D kernel vs 1D kernel
__forceinline__ __device__ void shiftCenterParallel(float* p, float* T, float* result) {
    const size_t pointIdx     = (blockIdx.y * blockDim.y + threadIdx.y) % 3;
    const size_t multTableIdx = pointIdx * 4;
    *result = T[multTableIdx] * p[0] + T[multTableIdx + 1] * p[1] + T[multTableIdx + 2] * p[2] + T[multTableIdx + 3];
}


__global__ void genShiftedCentersParallelKernel(float* TransformArray, float* PointsArray, float* ShiftedPointsArray,
    const size_t nPoses, const int* pointIdxToLinkIdx, const size_t totalPointCount) {
    size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x >= nPoses) {
        return;
    }
    if (idx_y >= totalPointCount * 3) {
        return;
    }

    // Find out which link this point belongs to
    // size_t coveredLinkPointCount = 0;
    size_t linkIdx = pointIdxToLinkIdx[idx_y / 3];

    // Get the corresponding affine transformation matrix
    float* currentTransform     = TransformArray + 12 * (linkIdx * nPoses + idx_x);
    float* currentTemplatePoint = PointsArray + idx_y / 3 * 3;
    float* currentResultPoints  = ShiftedPointsArray + (idx_y / 3 * nPoses + idx_x) * 3 + idx_y % 3;

    shiftCenterParallel(currentTemplatePoint, currentTransform, currentResultPoints);
}

void batchSphrCntr(float* TransformArray, float* PointsArray, float* ShiftedPointsArray, const size_t nPoses,
    const int* pointIdxToLinkIdx, const size_t totalTemplatePointCount, const size_t DOF, const cudaStream_t stream) {
    dim3 numThreads(32, 32, 1);
    dim3 numBlocks(
        (nPoses + numThreads.x - 1) / numThreads.x, (totalTemplatePointCount * 3 + numThreads.y - 1) / numThreads.y, 1);
    genShiftedCentersParallelKernel<<<numBlocks, numThreads, 0, stream>>>(
        TransformArray, PointsArray, ShiftedPointsArray, nPoses, pointIdxToLinkIdx, totalTemplatePointCount);
}

// x-dim for each pose in each trajectory, y-dim for each point (x, y, z) for each point on the robot
__global__ void genSphrTrajKer(float* TransformArray, float* PointsArray, float* ShiftedPointsArray,
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

    // Find out which link this point belongs to
    // size_t coveredLinkPointCount = 0;
    size_t linkIdx = pointIdxToLinkIdx[idx_y / 3];

    // Get the corresponding affine transformation matrix
    float* currentTransform =
        TransformArray + 12 * poIdxInTraj + 12 * (linkIdx * nPoPerTraj) + trajIdx * 12 * (nPoPerTraj * (DOF + 1));
    float* currentTemplatePoint = PointsArray + idx_y / 3 * 3;
    float* currentResultPoints  = ShiftedPointsArray + trajIdx * nPoPerTraj * totalPointCount * 3
                               + (idx_y / 3 * nPoPerTraj) * 3 + poIdxInTraj * 3 + idx_y % 3;

    shiftCenterParallel(currentTemplatePoint, currentTransform, currentResultPoints);
}

void batchSphrCntrTraj(float* TransformArray, float* PointsArray, float* ShiftedPointsArray, const size_t nPoPerTraj,
    const size_t nTraj, const int* pointIdxToLinkIdx, const size_t totalTemplatePointCount, const size_t DOF,
    const cudaStream_t stream) {
    dim3 numThreads(32, 16, 1);
    dim3 numBlocks((nPoPerTraj * nTraj + numThreads.x - 1) / numThreads.x,
        (totalTemplatePointCount * 3 + numThreads.y - 1) / numThreads.y, 1);
    genSphrTrajKer<<<numBlocks, numThreads, 0, stream>>>(TransformArray, PointsArray, ShiftedPointsArray, nPoPerTraj,
        nTraj, pointIdxToLinkIdx, totalTemplatePointCount, DOF);
}

__global__ void genShftCntrAdj(float* TransformArray, float* PointsArray, float* ShiftedPointsArray,
    const size_t nPoses, const int* pointIdxToLinkIdx, const size_t totalPointCount, const size_t DOF) {
    size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x >= nPoses) {
        return;
    }
    if (idx_y >= totalPointCount * 3) {
        return;
    }

    // Find out which link this point belongs to
    // size_t coveredLinkPointCount = 0;
    size_t linkIdx = pointIdxToLinkIdx[idx_y / 3];

    // Get the corresponding affine transformation matrix
    float* currentTransform     = TransformArray + 12 * (idx_x * (DOF + 1) + linkIdx);
    float* currentTemplatePoint = PointsArray + idx_y / 3 * 3;
    float* currentResultPoints  = ShiftedPointsArray + idx_y + idx_x * totalPointCount * 3;

    shiftCenterParallel(currentTemplatePoint, currentTransform, currentResultPoints);
}

void batchSphrCntrAdj(float* TransformArray, float* PointsArray, float* ShiftedPointsArray, const size_t nPoses,
    const int* pointIdxToLinkIdx, const size_t totalTemplatePointCount, const size_t DOF, const cudaStream_t stream) {
    dim3 numThreads(32, 32, 1);
    dim3 numBlocks(
        (nPoses + numThreads.x - 1) / numThreads.x, (totalTemplatePointCount * 3 + numThreads.y - 1) / numThreads.y);
    genShftCntrAdj<<<numBlocks, numThreads, 0, stream>>>(
        TransformArray, PointsArray, ShiftedPointsArray, nPoses, pointIdxToLinkIdx, totalTemplatePointCount, DOF);
}


__forceinline__ __device__ float3 shiftPoint(float3 p, float* T) {
    float3 result;
    result.x = T[0] * p.x + T[1] * p.y + T[2] * p.z + T[3];
    result.y = T[4] * p.x + T[5] * p.y + T[6] * p.z + T[7];
    result.z = T[8] * p.x + T[9] * p.y + T[10] * p.z + T[11];
    return result;
}

// Each Thread handle one ray (two points)
__global__ void shiftPointKernel(float* TransformArray, float3* templatePts, float3* shiftedPts, int* rayLinkMap,
    size_t rayPerRobot, size_t DOF, size_t BATCHSIZE, size_t nBlocksPerRobot) {
    extern __shared__ float sharedTransforms[];
    size_t idx              = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nThreadsPerRobot = blockDim.x * nBlocksPerRobot;
    size_t idxInRobot       = idx % nThreadsPerRobot;
    int robotIdx            = blockIdx.x / nBlocksPerRobot;

#pragma unroll
    // load all the transforms for the robot into shared memory
    for (int i = threadIdx.x; i < 12 * (DOF + 1); i += blockDim.x) {
        if (i < 12 * (DOF + 1)) { // Check to avoid out-of-bounds access
            sharedTransforms[i] = TransformArray[robotIdx * (DOF + 1) * 12 + i];
        }
    }

    if (idxInRobot >= rayPerRobot) {
        return;
    }
    float3* currentTemplatePoint = templatePts + idxInRobot * 2;
    float3* currentResultPoints  = shiftedPts + robotIdx * rayPerRobot * 2 + idxInRobot * 2;


    __syncthreads();

    // Find out which link this point belongs to
    int linkIdx = rayLinkMap[idxInRobot];

    currentResultPoints[0] = shiftPoint(currentTemplatePoint[0], sharedTransforms + linkIdx * 12);
    currentResultPoints[1] = shiftPoint(currentTemplatePoint[1], sharedTransforms + linkIdx * 12);
}

void batchRayVertices(float* TransformArray, float3* templatePts, float3* shiftedPts, int* rayLinkMap,
    size_t rayPerRobot, size_t DOF, size_t BATCHSIZE, const cudaStream_t stream) {
    constexpr size_t numThreads  = 256;
    const size_t nBlocksPerRobot = (rayPerRobot + numThreads - 1) / numThreads;
    const size_t numBlocks       = BATCHSIZE * nBlocksPerRobot;
    const size_t sharedMemSize   = (DOF + 1) * 12 * sizeof(float);

    shiftPointKernel<<<numBlocks, numThreads, sharedMemSize, stream>>>(
        TransformArray, templatePts, shiftedPts, rayLinkMap, rayPerRobot, DOF, BATCHSIZE, nBlocksPerRobot);
}

__forceinline__ __device__ void transformOBB(const float4* tf, const RTCD::OBB* input, RTCD::OBB* output) {
    size_t workerID = threadIdx.z % 3;

    const float* centerIn       = reinterpret_cast<const float*>(input);
    const float* orientationXIn = centerIn + 6;
    const float* orientationYIn = orientationXIn + 3;
    const float* orientationZIn = orientationYIn + 3;

    float* center       = reinterpret_cast<float*>(output);
    float* orientationX = center + 6;
    float* orientationY = orientationX + 3;
    float* orientationZ = orientationY + 3;

    // Deal with center first, each worker handle one dimension
    center[workerID] =
        tf[workerID].x * centerIn[0] + tf[workerID].y * centerIn[1] + tf[workerID].z * centerIn[2] + tf[workerID].w;
    // now orientation
    orientationX[workerID] =
        tf[workerID].x * orientationXIn[0] + tf[workerID].y * orientationXIn[1] + tf[workerID].z * orientationXIn[2];
    orientationY[workerID] =
        tf[workerID].x * orientationYIn[0] + tf[workerID].y * orientationYIn[1] + tf[workerID].z * orientationYIn[2];
    orientationZ[workerID] =
        tf[workerID].x * orientationZIn[0] + tf[workerID].y * orientationZIn[1] + tf[workerID].z * orientationZIn[2];

    // Copy the halfsize to target
    output->halfSize = input->halfSize;
}

__global__ void obbTfKern(
    const float4* TransformArray, const RTCD::OBB* obbs, RTCD::OBB* output, const size_t nPoses, const size_t DOF) {
    size_t poseId   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t linkId   = blockIdx.y * blockDim.y + threadIdx.y;
    size_t workerId = threadIdx.z % 3; // Three worker for one transformation

    if (poseId >= nPoses || linkId >= DOF + 1 || workerId >= 3) {
        return;
    }

    extern __shared__ float4 sharedTf[];

    const float4* tf = TransformArray + (poseId * (DOF + 1) + linkId) * 3; // global tf index
    float4* tfLink   = sharedTf + (threadIdx.x * 8 + threadIdx.y) * 3; // local shared memory index
    tfLink[workerId] = tf[workerId]; // copy tf to shared memory

    const RTCD::OBB* obb = obbs + linkId;
    RTCD::OBB* out       = output + poseId * (DOF + 1) + linkId;

    transformOBB(tfLink, obb, out);
}

void batchOBBTransform(const CUdeviceptr TransformArray, const CUdeviceptr obbs, CUdeviceptr output, const size_t DOF,
    const size_t nPoses, const cudaStream_t stream) {
    const dim3 numThreads(32, 8, 3); // 32 Poses, DOF + 1 OBBs, 3 threads per OBB transformation
    const dim3 numBlocks((nPoses + numThreads.x - 1) / numThreads.x, (DOF + 1 + numThreads.y - 1) / numThreads.y, 1);
    obbTfKern<<<numBlocks, numThreads, 8 * 32 * 12 * sizeof(float), stream>>>(
        reinterpret_cast<const float4*>(TransformArray), reinterpret_cast<const RTCD::OBB*>(obbs),
        reinterpret_cast<RTCD::OBB*>(output), nPoses, DOF);
}

__global__ void obbTfKern2(
    const float4* TransformArray, const RTCD::OBB* obbs, RTCD::OBB* output, const size_t nPoses, const size_t DOF) {
    size_t poseId   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t linkId   = blockIdx.y * blockDim.y + threadIdx.y;
    size_t workerId = threadIdx.z % 3; // Three worker for one transformation

    if (poseId >= nPoses || linkId >= DOF + 1 || workerId >= 3) {
        return;
    }

    extern __shared__ float4 sharedTf[];

    const float4* tf = TransformArray + (linkId * nPoses + poseId) * 3; // global tf index
    float4* tfLink   = sharedTf + (threadIdx.x + threadIdx.y * 32) * 3; // local shared memory index
    tfLink[workerId] = tf[workerId]; // copy tf to shared memory

    const RTCD::OBB* obb = obbs + linkId;
    RTCD::OBB* out       = output + linkId * nPoses + poseId;

    transformOBB(tfLink, obb, out);
}

void batchOBBTransform2(const CUdeviceptr TransformArray, const CUdeviceptr obbs, CUdeviceptr output, const size_t DOF,
    const size_t nPoses, const cudaStream_t stream) {
    const dim3 numThreads(32, 8, 3); // 32 Poses, DOF + 1 OBBs, 3 threads per OBB transformation
    const dim3 numBlocks((nPoses + numThreads.x - 1) / numThreads.x, (DOF + 1 + numThreads.y - 1) / numThreads.y, 1);
    obbTfKern2<<<numBlocks, numThreads, 8 * 32 * 12 * sizeof(float), stream>>>(
        reinterpret_cast<const float4*>(TransformArray), reinterpret_cast<const RTCD::OBB*>(obbs),
        reinterpret_cast<RTCD::OBB*>(output), nPoses, DOF);
}


struct is_hit {
    __host__ __device__ bool operator()(int flag) const { return flag > 0; }
};

void selectInsIdx(CUdeviceptr instances, CUdeviceptr idxMap, CUdeviceptr templateIns, CUdeviceptr templateMap,
    CUdeviceptr mask, RTCD::CUDABuffer& tmp, size_t N, size_t& tmpSize, CUdeviceptr count, cudaStream_t stream) {
    if (tmpSize == 0) { // if first time,
        size_t tmpSizeIns;
        cub::DeviceSelect::FlaggedIf(nullptr, tmpSizeIns, reinterpret_cast<OptixInstance*>(templateIns),
            reinterpret_cast<uint*>(mask), reinterpret_cast<OptixInstance*>(instances), reinterpret_cast<int*>(count),
            N, is_hit(), stream);

        size_t tmpSizeMap;
        cub::DeviceSelect::FlaggedIf(nullptr, tmpSizeMap, reinterpret_cast<uint*>(templateMap),
            reinterpret_cast<uint*>(mask), reinterpret_cast<uint*>(idxMap), reinterpret_cast<int*>(count), N, is_hit(),
            stream);

        tmpSize = tmpSizeIns > tmpSizeMap ? tmpSizeIns : tmpSizeMap;
        if (tmp.sizeInBytes < tmpSize) {
            tmp.freeAsync(stream);
            tmp.allocAsync(tmpSize, stream);
        }
    }

    cub::DeviceSelect::FlaggedIf(reinterpret_cast<void*>(tmp.d_pointer()), tmpSize,
        reinterpret_cast<OptixInstance*>(templateIns), reinterpret_cast<uint*>(mask),
        reinterpret_cast<OptixInstance*>(instances), reinterpret_cast<int*>(count), N, is_hit(), stream);

    cub::DeviceSelect::FlaggedIf(reinterpret_cast<void*>(tmp.d_pointer()), tmpSize,
        reinterpret_cast<uint*>(templateMap), reinterpret_cast<uint*>(mask), reinterpret_cast<uint*>(idxMap),
        reinterpret_cast<int*>(count), N, is_hit(), stream);
}
