// Forward-kinematics kernels of include/rtcd/internal/kinematics.h.
//
// Ports of the old RTCD/Robot/robot.cu kernels (fkineKernelAdj, fkineObbKernelAdj, obbTfKern, fkineTrajKernel)
// with ONE change: the kinematics table is read from a per-robot device buffer (KinematicsTable) instead of the
// __constant__ cMask / sMask / oMask / baseAffine globals (open question Q1: device-buffer read first). Thread
// layouts, launch configurations, operation order of every float expression and the __sincosf calls are kept
// exactly, so the outputs are bit-identical (tests/robot/test_kinematics_kernels compares against the old kernels).
// Do not "clean up" an expression here without re-running that test: FMA contraction follows the expression tree.
#include <rtcd/errors.h>
#include <rtcd/internal/kinematics.h>

#include <cuda_runtime.h>

#include <span>
#include <string>

namespace rtcd::internal {

    namespace {

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

        // Old fkineKernelAdj (pose-major layout).
        __global__ void fkineKernelAdj(
            const KinematicsTable* __restrict__ tab, const float* q, float* TransformArray, unsigned int DOF, unsigned int nPoses) {
            extern __shared__ float sharedTransforms[];
            const float* cMask = &tab->cosMask[0][0];
            const float* sMask = &tab->sinMask[0][0];
            const float* oMask = &tab->oneMask[0][0];
            size_t idx         = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nPoses * 12) {
                return;
            }
            size_t robotGlobalIdx          = idx / 12;
            size_t localIdx                = idx % 12;
            size_t robotLocalIdx           = threadIdx.x / 12;
            float* localFirstLinkTransform = sharedTransforms + robotLocalIdx * DOF * 12;
            for (int i = 0; i < DOF; i++) {
                float sq, cq;
                float qNow = *(q + robotGlobalIdx * DOF + i);
                __sincosf(qNow, &sq, &cq);
                size_t padding                   = i * 12 + localIdx;
                localFirstLinkTransform[padding] = cq * cMask[padding] + sq * sMask[padding] + oMask[padding];
            }
            float* baseTransform = TransformArray + robotGlobalIdx * 12 * (DOF + 1);
            baseShiftParallel(tab, baseTransform);
            __syncthreads();
            for (int i = 0; i < DOF; i++) {
                affineMult(baseTransform + i * 12, localFirstLinkTransform + i * 12, baseTransform + (i + 1) * 12);
                __syncthreads();
            }
        }

        // Old transformOBBRow (rows of the 3x4 transform, worker = row). Obb = 15 floats, old RTCD::OBB layout.
        __forceinline__ __device__ void transformObbRow(
            const float4* tf, const Obb* input, Obb* output, const unsigned int workerID) {
            const float* centerIn       = reinterpret_cast<const float*>(input);
            const float* orientationXIn = centerIn + 6;
            const float* orientationYIn = orientationXIn + 3;
            const float* orientationZIn = orientationYIn + 3;

            float* center       = reinterpret_cast<float*>(output);
            float* orientationX = center + 6;
            float* orientationY = orientationX + 3;
            float* orientationZ = orientationY + 3;

            const float4 row       = tf[workerID];
            center[workerID]       = row.x * centerIn[0] + row.y * centerIn[1] + row.z * centerIn[2] + row.w;
            orientationX[workerID] = row.x * orientationXIn[0] + row.y * orientationXIn[1] + row.z * orientationXIn[2];
            orientationY[workerID] = row.x * orientationYIn[0] + row.y * orientationYIn[1] + row.z * orientationYIn[2];
            orientationZ[workerID] = row.x * orientationZIn[0] + row.y * orientationZIn[1] + row.z * orientationZIn[2];

            if (workerID == 0) {
                output->halfSize[0] = input->halfSize[0];
                output->halfSize[1] = input->halfSize[1];
                output->halfSize[2] = input->halfSize[2];
            }
        }

        // Old fkineObbKernelAdj: fkineKernelAdj + link OBB transform in one launch.
        __global__ void fkineObbKernelAdj(const KinematicsTable* __restrict__ tab, const float* q, float* TransformArray,
            const Obb* obbs, Obb* outOBBs, unsigned int DOF, unsigned int nPoses) {
            extern __shared__ float sharedTransforms[];
            const float* cMask = &tab->cosMask[0][0];
            const float* sMask = &tab->sinMask[0][0];
            const float* oMask = &tab->oneMask[0][0];
            size_t idx         = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nPoses * 12) {
                return;
            }
            size_t robotGlobalIdx          = idx / 12;
            size_t localIdx                = idx % 12;
            size_t robotLocalIdx           = threadIdx.x / 12;
            float* localFirstLinkTransform = sharedTransforms + robotLocalIdx * DOF * 12;
            for (int i = 0; i < DOF; i++) {
                float sq, cq;
                float qNow = *(q + robotGlobalIdx * DOF + i);
                __sincosf(qNow, &sq, &cq);
                size_t padding                   = i * 12 + localIdx;
                localFirstLinkTransform[padding] = cq * cMask[padding] + sq * sMask[padding] + oMask[padding];
            }
            float* baseTransform = TransformArray + robotGlobalIdx * 12 * (DOF + 1);
            baseShiftParallel(tab, baseTransform);
            __syncthreads();
            for (int i = 0; i < DOF; i++) {
                affineMult(baseTransform + i * 12, localFirstLinkTransform + i * 12, baseTransform + (i + 1) * 12);
                __syncthreads();
            }

            const float4* poseTfs = reinterpret_cast<const float4*>(baseTransform);
            Obb* poseOBBs         = outOBBs + robotGlobalIdx * (DOF + 1);
            for (unsigned int w = localIdx; w < 3 * (DOF + 1); w += 12) {
                const unsigned int linkId = w / 3;
                transformObbRow(poseTfs + linkId * 3, obbs + linkId, poseOBBs + linkId, w % 3);
            }
        }

        // Old transformOBB (worker = threadIdx.z % 3).
        __forceinline__ __device__ void transformObb(const float4* tf, const Obb* input, Obb* output) {
            size_t workerID = threadIdx.z % 3;

            const float* centerIn       = reinterpret_cast<const float*>(input);
            const float* orientationXIn = centerIn + 6;
            const float* orientationYIn = orientationXIn + 3;
            const float* orientationZIn = orientationYIn + 3;

            float* center       = reinterpret_cast<float*>(output);
            float* orientationX = center + 6;
            float* orientationY = orientationX + 3;
            float* orientationZ = orientationY + 3;

            center[workerID] =
                tf[workerID].x * centerIn[0] + tf[workerID].y * centerIn[1] + tf[workerID].z * centerIn[2] + tf[workerID].w;
            orientationX[workerID] =
                tf[workerID].x * orientationXIn[0] + tf[workerID].y * orientationXIn[1] + tf[workerID].z * orientationXIn[2];
            orientationY[workerID] =
                tf[workerID].x * orientationYIn[0] + tf[workerID].y * orientationYIn[1] + tf[workerID].z * orientationYIn[2];
            orientationZ[workerID] =
                tf[workerID].x * orientationZIn[0] + tf[workerID].y * orientationZIn[1] + tf[workerID].z * orientationZIn[2];

            output->halfSize[0] = input->halfSize[0];
            output->halfSize[1] = input->halfSize[1];
            output->halfSize[2] = input->halfSize[2];
        }

        // Old obbTfKern (pose-major link poses).
        __global__ void obbTfKern(const float4* TransformArray, const Obb* obbs, Obb* output, const size_t nPoses, const size_t DOF) {
            size_t poseId   = blockIdx.x * blockDim.x + threadIdx.x;
            size_t linkId   = blockIdx.y * blockDim.y + threadIdx.y;
            size_t workerId = threadIdx.z % 3;

            if (poseId >= nPoses || linkId >= DOF + 1 || workerId >= 3) {
                return;
            }

            extern __shared__ float4 sharedTf[];

            const float4* tf = TransformArray + (poseId * (DOF + 1) + linkId) * 3;
            float4* tfLink   = sharedTf + (threadIdx.x * 8 + threadIdx.y) * 3;
            tfLink[workerId] = tf[workerId];

            const Obb* obb = obbs + linkId;
            Obb* out       = output + poseId * (DOF + 1) + linkId;

            transformObb(tfLink, obb, out);
        }

        // Old fkineTrajKernel (trajectory layout).
        __global__ void fkineTrajKernel(const KinematicsTable* __restrict__ tab, const float* q, float* TransformArray,
            size_t DOF, size_t nPoPerTraj, size_t nTraj) {
            extern __shared__ float sharedTransforms[];
            const float* cMask = &tab->cosMask[0][0];
            const float* sMask = &tab->sinMask[0][0];
            const float* oMask = &tab->oneMask[0][0];
            size_t idx         = blockIdx.x * blockDim.x + threadIdx.x;
            size_t trajIdx     = idx / (nPoPerTraj * 12);
            if (idx >= nPoPerTraj * nTraj * 12) {
                return;
            }
            size_t poIdxInTraj             = (idx - trajIdx * nPoPerTraj * 12) / 12;
            size_t idxInPo                 = idx % 12;
            size_t poIdxInBlk              = threadIdx.x / 12;
            float* localFirstLinkTransform = sharedTransforms + poIdxInBlk * DOF * 12;
            for (int i = 0; i < DOF; i++) {
                float sq, cq;
                float qNow = *(q + idx / 12 * DOF + i);
                __sincosf(qNow, &sq, &cq);
                size_t padding                   = i * 12 + idxInPo;
                localFirstLinkTransform[padding] = cq * cMask[padding] + sq * sMask[padding] + oMask[padding];
            }
            size_t TPadding      = nPoPerTraj * 12;
            float* baseTransform = TransformArray + poIdxInTraj * 12 + trajIdx * 12 * (DOF + 1) * nPoPerTraj;
            baseShiftParallel(tab, baseTransform);
            __syncthreads();
            for (int i = 0; i < DOF; i++) {
                affineMult(baseTransform + i * TPadding, localFirstLinkTransform + i * 12, baseTransform + (i + 1) * TPadding);
                __syncthreads();
            }
        }

        // ---- host-side argument checks (InvalidArgument before any GPU work) ----

        void requireTable(const KinematicsRef& t, const char* fn) {
            if (t.host == nullptr || t.device == 0) {
                throw InvalidArgument(std::string(fn) + ": KinematicsRef needs both the host table and its device copy");
            }
            if (t.host->dof < 1 || t.host->dof > kMaxDof || t.host->numLinks != t.host->dof + 1) {
                throw InvalidArgument(std::string(fn) + ": invalid kinematics table (dof " + std::to_string(t.host->dof)
                                      + ", numLinks " + std::to_string(t.host->numLinks) + ")");
            }
        }

        template <typename T>
        void requireSpan(const DeviceSpan<T>& s, std::size_t need, const char* fn, const char* name) {
            if (s.count < need || (need > 0 && s.ptr == 0)) {
                throw InvalidArgument(std::string(fn) + ": " + name + " holds " + std::to_string(s.count)
                                      + " elements, " + std::to_string(need) + " needed");
            }
        }

        template <typename T>
        void requireAligned16(const DeviceSpan<T>& s, const char* fn, const char* name) {
            if (s.ptr % 16 != 0) {
                throw InvalidArgument(std::string(fn) + ": " + name + " must be 16-byte aligned (read as float4)");
            }
        }

        const KinematicsTable* dev(const KinematicsRef& t) { return reinterpret_cast<const KinematicsTable*>(t.device); }

    } // namespace

    DeviceBuffer uploadKinematics(const KinematicsTable& table) {
        return DeviceBuffer::fromHost(std::span<const KinematicsTable>(&table, 1));
    }

    void linkPoses(const KinematicsRef& table, DeviceSpan<const float> q, std::uint32_t nPoses,
        DeviceSpan<float> linkPosesOut, StreamHandle stream) {
        constexpr const char* fn = "rtcd::internal::linkPoses";
        requireTable(table, fn);
        if (nPoses == 0) {
            return;
        }
        const std::size_t DOF = table.host->dof;
        requireSpan(q, std::size_t(nPoses) * DOF, fn, "q");
        requireSpan(linkPosesOut, std::size_t(nPoses) * (DOF + 1) * 12, fn, "linkPosesOut");
        // old batchFkineAdj
        const size_t numPosePerBlock = nPoses < 32 ? nPoses : 32;
        const size_t numThreads      = numPosePerBlock * 12;
        const size_t numBlocks       = (nPoses + numPosePerBlock - 1) / numPosePerBlock;
        fkineKernelAdj<<<static_cast<unsigned>(numBlocks), static_cast<unsigned>(numThreads),
            DOF * 12 * numPosePerBlock * sizeof(float), stream>>>(dev(table), reinterpret_cast<const float*>(q.ptr),
            reinterpret_cast<float*>(linkPosesOut.ptr), static_cast<unsigned>(DOF), nPoses);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    void linkPosesAndObbs(const KinematicsRef& table, DeviceSpan<const float> q, std::uint32_t nPoses,
        DeviceSpan<const Obb> obbTemplates, DeviceSpan<float> linkPosesOut, DeviceSpan<Obb> obbsOut, StreamHandle stream) {
        constexpr const char* fn = "rtcd::internal::linkPosesAndObbs";
        requireTable(table, fn);
        if (nPoses == 0) {
            return;
        }
        const std::size_t DOF = table.host->dof;
        requireSpan(q, std::size_t(nPoses) * DOF, fn, "q");
        requireSpan(obbTemplates, DOF + 1, fn, "obbTemplates");
        requireSpan(linkPosesOut, std::size_t(nPoses) * (DOF + 1) * 12, fn, "linkPosesOut");
        requireSpan(obbsOut, std::size_t(nPoses) * (DOF + 1), fn, "obbsOut");
        requireAligned16(linkPosesOut, fn, "linkPosesOut");
        // old batchFkineAdjOBB
        const size_t numPosePerBlock = nPoses < 32 ? nPoses : 32;
        const size_t numThreads      = numPosePerBlock * 12;
        const size_t numBlocks       = (nPoses + numPosePerBlock - 1) / numPosePerBlock;
        fkineObbKernelAdj<<<static_cast<unsigned>(numBlocks), static_cast<unsigned>(numThreads),
            DOF * 12 * numPosePerBlock * sizeof(float), stream>>>(dev(table), reinterpret_cast<const float*>(q.ptr),
            reinterpret_cast<float*>(linkPosesOut.ptr), reinterpret_cast<const Obb*>(obbTemplates.ptr),
            reinterpret_cast<Obb*>(obbsOut.ptr), static_cast<unsigned>(DOF), nPoses);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    void transformObbs(DeviceSpan<const float> linkPoses, std::uint32_t numLinks, std::uint32_t nPoses,
        DeviceSpan<const Obb> obbTemplates, DeviceSpan<Obb> obbsOut, StreamHandle stream) {
        constexpr const char* fn = "rtcd::internal::transformObbs";
        if (numLinks < 1 || numLinks > kMaxLinks) {
            throw InvalidArgument(std::string(fn) + ": numLinks " + std::to_string(numLinks) + " out of [1, kMaxLinks]");
        }
        if (nPoses == 0) {
            return;
        }
        requireSpan(linkPoses, std::size_t(nPoses) * numLinks * 12, fn, "linkPoses");
        requireSpan(obbTemplates, numLinks, fn, "obbTemplates");
        requireSpan(obbsOut, std::size_t(nPoses) * numLinks, fn, "obbsOut");
        requireAligned16(linkPoses, fn, "linkPoses");
        // old batchOBBTransform (DOF = numLinks - 1)
        const size_t DOF = numLinks - 1;
        const dim3 numThreads(32, 8, 3);
        const dim3 numBlocks(static_cast<unsigned>((nPoses + numThreads.x - 1) / numThreads.x),
            static_cast<unsigned>((DOF + 1 + numThreads.y - 1) / numThreads.y), 1);
        obbTfKern<<<numBlocks, numThreads, 8 * 32 * 12 * sizeof(float), stream>>>(
            reinterpret_cast<const float4*>(linkPoses.ptr), reinterpret_cast<const Obb*>(obbTemplates.ptr),
            reinterpret_cast<Obb*>(obbsOut.ptr), nPoses, DOF);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

    void trajectoryLinkPoses(const KinematicsRef& table, DeviceSpan<const float> q, std::uint32_t points,
        std::uint32_t nTraj, DeviceSpan<float> linkPosesOut, StreamHandle stream) {
        constexpr const char* fn = "rtcd::internal::trajectoryLinkPoses";
        requireTable(table, fn);
        if (nTraj == 0 || points == 0) {
            return;
        }
        const std::size_t DOF = table.host->dof;
        requireSpan(q, std::size_t(nTraj) * points * DOF, fn, "q");
        requireSpan(linkPosesOut, std::size_t(nTraj) * points * (DOF + 1) * 12, fn, "linkPosesOut");
        // old batchFkineTraj
        const size_t nPoPerTraj      = points;
        const size_t numPosePerBlock = nPoPerTraj < 42 ? nPoPerTraj : 42;
        const size_t numThreads      = numPosePerBlock * 12;
        const size_t numBlocks       = (nPoPerTraj * nTraj + numPosePerBlock - 1) / numPosePerBlock;
        fkineTrajKernel<<<static_cast<unsigned>(numBlocks), static_cast<unsigned>(numThreads),
            DOF * 12 * numPosePerBlock * sizeof(float), stream>>>(dev(table), reinterpret_cast<const float*>(q.ptr),
            reinterpret_cast<float*>(linkPosesOut.ptr), DOF, nPoPerTraj, nTraj);
        RTCD_CUDA_CHECK(cudaGetLastError());
    }

} // namespace rtcd::internal
