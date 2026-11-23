#pragma once
#include "BBCollisionDetection.cuh"
#include "CCRobotWrapper.h"
#include "launchParameters.h"
#include "threadPool.h"
#undef max
#undef min
#include <CollisionScenes/baseBatchScene.h>
#include <Utils/OptiXIRUtils.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cstring>
#include <vector>

inline constexpr bool PEROBS = false;

namespace RTCD {

    template <size_t NSTREAM, LinkType TYPE>
    class CollisionDetector {

    public:
        CollisionDetector() = default;
        // Errors are ignored here: global detectors can be destroyed during process exit, after CUDA teardown.
        ~CollisionDetector() { destroyOptix(false); }
        // Owns OptiX handles (pipelines, program groups, modules): not copyable.
        CollisionDetector(const CollisionDetector&)            = delete;
        CollisionDetector& operator=(const CollisionDetector&) = delete;

        CollisionDetector(const OptixDeviceContext& context, const std::array<cudaStream_t, NSTREAM> streams);

        CollisionDetector(const OptixDeviceContext& context, const std::array<cudaStream_t, NSTREAM> streams,
            const std::shared_ptr<CCWrappedRobot<NSTREAM, TYPE>>& robot, std::shared_ptr<baseBatchScene<NSTREAM>> s);
        CollisionDetector(const OptixDeviceContext& context, const std::array<cudaStream_t, NSTREAM> streams,
            const std::shared_ptr<CCRayRobot<NSTREAM, TYPE>>& robot, std::shared_ptr<baseBatchScene<NSTREAM>> s);

        void setupScene(const std::shared_ptr<baseBatchScene<NSTREAM>> s);
        void setupRobot(const std::shared_ptr<CCWrappedRobot<NSTREAM, TYPE>>& robot);
        void setupRobot(const std::shared_ptr<CCRayRobot<NSTREAM, TYPE>>& robot);
        void resetRobot();
        void resetSBT();
        void setupOptix();

        void updateASHandle(const OptixTraversableHandle handle, const size_t frameID);
        void setNCheck(const uint nCheck);

        // warmup()
        // @brief Warm up the pipeline and create the cuda graph for best system performance.
        void warmup(const int nPoses);

        // check()
        // @brief Check the collision between the robot and the scene
        //
        // Determines which pipeline to use based on the template parameter
        // Determines the size of each batch based on experiment performance results
        void check();

        // Collision Detection
        void detect(const size_t frameId);
        void detect(const size_t frameId, const int nPoses);
        void detectWithOBB(const size_t frameId); // Placeholder for future versions
        void detectWithOBB(const size_t frameId, const int nPoses); // Placeholder for future versions

        void resetGraph();

        // downloadResult()
        // @brief download the result to a cpu uint vector
        inline void downloadResult(std::vector<uint>& result) { resultBuffer.download(result); }

        // clearResult()
        // @brief clear the result buffer
        inline void clearResult();

        void buildSBT();

        // Function for headline picture generation
        void resizeHitBuffer(const int size);

        inline CUdeviceptr getHitBuffer(const int streamID) { return streamDatas[streamID].hitBuffer.d_pointer(); }

    private:
        std::vector<std::tuple<uint, uint, uint>> batchSizeInfo; // <streamID, batchSize>
#if defined(USE_MULTI_THREADING)
        ThreadPool<NSTREAM> pool;
        std::function<void(std::tuple<uint, uint, uint>)> task;
#endif

        std::shared_ptr<baseBatchScene<NSTREAM>> sc;
        std::shared_ptr<CCWrappedRobot<NSTREAM, TYPE>> meshRobot;
        std::shared_ptr<CCRayRobot<NSTREAM, TYPE>> rayRobot;
        size_t batchSize = 0; // TODO: I don't want it here
        int dof;
        OptixDeviceContext optixContext                = 0;
        OptixPipelineLinkOptions pipelineLinkOptions   = {};
        OptixModuleCompileOptions moduleCompileOptions = {};

        // OBB related
        size_t nRobotOBBs = 0;
        size_t nSceneOBBs = 0;

        // static poses related

        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixModule Module                                 = nullptr;
        OptixModule sphereModule                           = nullptr;
        OptixModule curveModule                            = nullptr;

        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};
        bool sbtBuilt               = false;

        // Types that trace against OptiX built-in primitives (spheres / curves) need a built-in IS module whose
        // build flags match the robot GAS (see getBuildFlags()).
        static constexpr bool usesBuiltinIS = TYPE == LinkType::SPHERES || TYPE == LinkType::SPHERE_GAS
                                           || TYPE == LinkType::LINEAR_CURVE || TYPE == LinkType::QUADRATIC_CURVE
                                           || TYPE == LinkType::CUBIC_CURVE;
        unsigned int moduleBuildFlags = 0; // flags the current modules were created with
        bool optixCreated             = false;


        struct perStreamData {
            cudaStream_t stream;
            LaunchParams launchParams;
            std::vector<LaunchParams> launchParamsVec;
            OptixPipeline pipeline = nullptr;
            CUDABuffer launchParamsBuffer;
            // Byte copy of what launchParamsBuffer currently holds, so unchanged params are not re-uploaded every
            // launch (each cudaMemcpyAsync costs ~10-30 us of host time on WDDM). Invalidate whenever
            // launchParamsBuffer is reallocated or written outside uploadLaunchParams().
            alignas(LaunchParams) unsigned char lpUploaded[sizeof(LaunchParams)];
            bool lpUploadedValid = false;
            CUDABuffer hitBuffer;

            // OBBs
            CUdeviceptr sceneOBBs; // shared between streams
            CUdeviceptr robotOBBs; // per stream robot obbs
            CUDABuffer rOBBInCol; // per stream obb cc result
            CUDABuffer sOBBInCol; // per stream obb cc result
            CUDABuffer rsOBBInCol;


            bool graphCreated = false;
            cudaGraph_t graph;
            cudaGraphExec_t instance;
        };
        std::array<perStreamData, NSTREAM> streamDatas;
        std::array<cudaStream_t, NSTREAM> streams;

        CUDABuffer resultBuffer;
        CUdeviceptr resultPtr;

    private:
        void destroyOptix(bool checkErrors = true);
        void rebuildOptixIfFlagsChanged();
        void createModule(const unsigned int buildFlags);
        void createRaygenPrograms();
        void createMissPrograms();
        void createHitgroupPrograms();
        void createPipeline();

        size_t getHitBufferSize(const int streamID) { return streamDatas[streamID].hitBuffer.sizeInBytes; }

        inline CUdeviceptr getRobotMask(const int streamID) { return streamDatas[streamID].rOBBInCol.d_pointer(); }
        inline CUdeviceptr getSceneMask(const int streamID) { return streamDatas[streamID].sOBBInCol.d_pointer(); }
        inline CUdeviceptr getRSMask(const int streamID) { return streamDatas[streamID].rsOBBInCol.d_pointer(); }

        inline unsigned int getBuildFlags();
        inline void setMapInfo(const size_t frameId);

        // Async-upload sd.launchParams to the device unless the device copy is already identical.
        inline void uploadLaunchParams(perStreamData& sd) {
            if (sd.lpUploadedValid && std::memcmp(sd.lpUploaded, &sd.launchParams, sizeof(LaunchParams)) == 0) {
                return;
            }
            sd.launchParamsBuffer.uploadAsync(&sd.launchParams, 1, sd.stream);
            std::memcpy(sd.lpUploaded, &sd.launchParams, sizeof(LaunchParams));
            sd.lpUploadedValid = true;
        }

        void detectOBB(const size_t frameId); // Placeholder for future versions
        void detectOBB(const size_t frameId, const int nPoses); // Placeholder for future versions
    };
} // namespace RTCD

namespace RTCD { // CC Initializers

    template <size_t NSTREAM, LinkType TYPE>
    CollisionDetector<NSTREAM, TYPE>::CollisionDetector(
        const OptixDeviceContext& context, const std::array<cudaStream_t, NSTREAM> streams)
        : optixContext(context), streams(streams) {}

    template <size_t NSTREAM, LinkType TYPE>
    CollisionDetector<NSTREAM, TYPE>::CollisionDetector(const OptixDeviceContext& context,
        const std::array<cudaStream_t, NSTREAM> streams, const std::shared_ptr<CCWrappedRobot<NSTREAM, TYPE>>& r,
        std::shared_ptr<baseBatchScene<NSTREAM>> s)
        : optixContext(context), sc(s), streams(streams), batchSize(r->getBatchSize()) {

        static_assert(
            (TYPE != LinkType::RAY_ONLINE || TYPE != LinkType::RAY_STATIC) && "Use the other constructor for RayRobot");
        setupScene(s);
        setupRobot(r);
        setupOptix();
        buildSBT();
    }

    template <size_t NSTREAM, LinkType TYPE>
    CollisionDetector<NSTREAM, TYPE>::CollisionDetector(const OptixDeviceContext& context,
        const std::array<cudaStream_t, NSTREAM> streams, const std::shared_ptr<CCRayRobot<NSTREAM, TYPE>>& r,
        std::shared_ptr<baseBatchScene<NSTREAM>> s)
        : optixContext(context), sc(s), streams(streams) {

        static_assert((TYPE == LinkType::RAY_STATIC || TYPE == LinkType::RAY_ONLINE)
                      && "Use the other constructor for Robots other than Ray Robot. RAY robot deprecates since V2.5");
        setupScene(s);
        setupRobot(r);
        setupOptix();
        buildSBT();
    }


    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::setupOptix() {
        if (optixCreated) {
            destroyOptix(); // re-setup (e.g. robot with different build flags): don't leak the old objects
        }
        moduleBuildFlags = getBuildFlags();
        createModule(moduleBuildFlags);

        createRaygenPrograms();
        createMissPrograms();
        createHitgroupPrograms();
        createPipeline();
        optixCreated = true;
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::destroyOptix(bool checkErrors) {
        if (!optixCreated) {
            return;
        }
        auto check = [checkErrors](OptixResult res) {
            if (checkErrors) {
                OPTIX_CHECK(res);
            }
        };
        for (auto& sd : streamDatas) {
            if (sd.pipeline) {
                check(optixPipelineDestroy(sd.pipeline));
                sd.pipeline = nullptr;
            }
            sd.graphCreated = false;
        }
        for (auto* pgs : {&raygenPGs, &missPGs, &hitgroupPGs}) {
            for (auto pg : *pgs) {
                if (pg) {
                    check(optixProgramGroupDestroy(pg));
                }
            }
            pgs->clear();
        }
        for (auto* m : {&Module, &sphereModule, &curveModule}) {
            if (*m) {
                check(optixModuleDestroy(*m));
                *m = nullptr;
            }
        }
        optixCreated = false;
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::rebuildOptixIfFlagsChanged() {
        if constexpr (usesBuiltinIS) {
            if (optixCreated && getBuildFlags() != moduleBuildFlags) {
                const bool hadSBT = sbt.raygenRecord != 0;
                setupOptix();
                if (hadSBT) {
                    buildSBT(); // SBT records are packed from the (now replaced) program groups
                }
            }
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::setupScene(const std::shared_ptr<baseBatchScene<NSTREAM>> s) {
        sc         = s;
        nSceneOBBs = sc->getNumObstacles(); // Single layer OBB, one per obs
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::setupRobot(const std::shared_ptr<CCWrappedRobot<NSTREAM, TYPE>>& r) {
        meshRobot = r;
        meshRobot->resetGraph();

        for (int i = 0; i < NSTREAM; i++) {
            streamDatas[i].stream = streams[i];
            streamDatas[i].hitBuffer.free();
            streamDatas[i].hitBuffer.alloc(sizeof(int) * r->getBatchSize());
            // if constexpr (TYPE != LinkType::CUBIC_CURVE && TYPE != LinkType::QUADRATIC_CURVE
            //               && TYPE != LinkType::LINEAR_CURVE) {
            if (meshRobot->useOBB) {
                streamDatas[i].sceneOBBs = sc->getOBBs();
                streamDatas[i].robotOBBs = r->getOBBs(i);
                streamDatas[i].rOBBInCol.free();
                streamDatas[i].rOBBInCol.alloc(sizeof(int) * r->getNOBBs());
                streamDatas[i].sOBBInCol.free();
                streamDatas[i].sOBBInCol.alloc(sizeof(int) * sc->getNumObstacles());
                streamDatas[i].rsOBBInCol.free();
                streamDatas[i].rsOBBInCol.alloc(sizeof(int) * r->getNOBBs() * sc->getNumObstacles());
            }
            streamDatas[i].launchParamsBuffer.free();
                streamDatas[i].lpUploadedValid = false;
            streamDatas[i].launchParamsBuffer.alloc(sizeof(LaunchParams));

            streamDatas[i].graphCreated = false;

            LaunchParams& params         = streamDatas[i].launchParams;
            params.type                  = rayType::MESH;
            params.verticesBuffer        = reinterpret_cast<float3*>(sc->getSgVerts());
            params.totalRayCount         = sc->getSgRayCnt();
            params.hitBuffer             = reinterpret_cast<int*>(streamDatas[i].hitBuffer.d_pointer());
            params.mesh.primIdxToPoseIdx = reinterpret_cast<unsigned int*>(r->getMapIndex(i));

            streamDatas[i].launchParamsVec.clear();
            for (int j = 0; j < sc->getNumObstacles(); j++) {
                streamDatas[i].launchParamsVec.push_back(params);
            }
            streamDatas[i].launchParamsBuffer.upload(&streamDatas[i].launchParams, 1);
            streamDatas[i].lpUploadedValid = false; // written outside uploadLaunchParams()
        }

        batchSize = r->getBatchSize();
        dof       = meshRobot->getDOF();

        // if constexpr (TYPE != LinkType::CUBIC_CURVE && TYPE != LinkType::QUADRATIC_CURVE
        //               && TYPE != LinkType::LINEAR_CURVE) {
        if (meshRobot->useOBB) {
            nRobotOBBs = r->getNOBBs();
        }

        // setupOptix() may have run before this robot was attached (common call order); make sure the built-in IS
        // module was created with this robot's GAS build flags.
        rebuildOptixIfFlagsChanged();
    }


    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::setupRobot(const std::shared_ptr<CCRayRobot<NSTREAM, TYPE>>& r) {
        rayRobot = r;
        rayRobot->resetGraph();

        for (int i = 0; i < NSTREAM; i++) {
            streamDatas[i].stream = streams[i];
            streamDatas[i].hitBuffer.free();
            streamDatas[i].hitBuffer.alloc(sizeof(int) * r->getBatchSize());
            if (rayRobot->useOBB) {
                streamDatas[i].sceneOBBs = sc->getOBBs();
                streamDatas[i].robotOBBs = r->getOBBs(i);
                streamDatas[i].rOBBInCol.free();
                streamDatas[i].rOBBInCol.alloc(sizeof(int) * r->getNOBBs());
                streamDatas[i].sOBBInCol.free();
                streamDatas[i].sOBBInCol.alloc(sizeof(int) * sc->getNumObstacles());
                streamDatas[i].rsOBBInCol.free();
                streamDatas[i].rsOBBInCol.alloc(sizeof(int) * r->getNOBBs() * sc->getNumObstacles());
            }
            streamDatas[i].launchParamsBuffer.free();
                streamDatas[i].lpUploadedValid = false;
            streamDatas[i].launchParamsBuffer.alloc(sizeof(LaunchParams));

            streamDatas[i].graphCreated = false;


            LaunchParams& params         = streamDatas[i].launchParams;
            params.type                  = rayType::MESH;
            params.mesh.primIdxToPoseIdx = reinterpret_cast<unsigned int*>(sc->getMapIndex(i));
            params.hitBuffer             = reinterpret_cast<int*>(streamDatas[i].hitBuffer.d_pointer());

            streamDatas[i].launchParamsVec.clear();
            for (int j = 0; j < rayRobot->getDOF() + 1; j++) {
                streamDatas[i].launchParamsVec.push_back(params);
            }
            streamDatas[i].launchParamsBuffer.upload(&streamDatas[i].launchParams, 1);
            streamDatas[i].lpUploadedValid = false; // written outside uploadLaunchParams()
        }

        batchSize = r->getBatchSize();
        if (rayRobot->useOBB) {
            nRobotOBBs = r->getNOBBs();
        }
        dof = rayRobot->getDOF();
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::resetRobot() {

        if constexpr (TYPE != LinkType::RAY_ONLINE) {

            for (int i = 0; i < NSTREAM; i++) {
                streamDatas[i].hitBuffer.free();
                // if constexpr (TYPE != LinkType::CUBIC_CURVE && TYPE != LinkType::QUADRATIC_CURVE
                //               && TYPE != LinkType::LINEAR_CURVE) {
                if (meshRobot->useOBB) {
                    streamDatas[i].rOBBInCol.free();
                    streamDatas[i].sOBBInCol.free();
                    streamDatas[i].rsOBBInCol.free();
                }
                streamDatas[i].launchParamsBuffer.free();
                streamDatas[i].lpUploadedValid = false;

                streamDatas[i].graphCreated = false;

                streamDatas[i].launchParams = {};
                streamDatas[i].launchParamsVec.clear();
            }
            meshRobot = nullptr;
        } else {
            for (int i = 0; i < NSTREAM; i++) {
                streamDatas[i].stream = streams[i];
                streamDatas[i].hitBuffer.free();
                if (rayRobot->useOBB) {
                    streamDatas[i].rOBBInCol.free();
                    streamDatas[i].sOBBInCol.free();
                    streamDatas[i].rsOBBInCol.free();
                }
                streamDatas[i].launchParamsBuffer.free();
                streamDatas[i].lpUploadedValid = false;

                streamDatas[i].graphCreated = false;

                streamDatas[i].launchParams = {};

                streamDatas[i].launchParamsVec.clear();
            }

            rayRobot = nullptr;
        }
    }
} // namespace RTCD

namespace RTCD { // Key Functions

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::setNCheck(const uint nCheck) {
        if (resultBuffer.sizeInBytes / sizeof(uint) < nCheck) {
            resultBuffer.resize(nCheck * sizeof(uint));
            resultPtr = resultBuffer.d_pointer();
        }

        batchSizeInfo.clear();

        assert(batchSize != 0 && "Batch size must be set");

        if (nCheck <= batchSize) {
            batchSizeInfo.emplace_back(0, nCheck, 0);
        } else {
            // if constexpr (TYPE == LinkType::LINEAR_CURVE || TYPE == LinkType::QUADRATIC_CURVE
            //               || TYPE == LinkType::CUBIC_CURVE) {
            //     uint frameCounter = 0;
            //     for (int left = nCheck; left > 0; left -= batchSize) {
            //         batchSizeInfo.emplace_back(
            //             frameCounter, std::min(static_cast<size_t>(batchSize), static_cast<size_t>(left)));
            //         frameCounter++;
            //     }
            // } else {
            // if (nCheck < batchSize * NSTREAM) {
            //     int streamSize = (nCheck + NSTREAM - 1) / NSTREAM;
            //     for (int i = 0, left = nCheck; left > 0; left -= streamSize, i++) {
            //         batchSizeInfo.emplace_back(
            //             i, std::min(static_cast<size_t>(streamSize), static_cast<size_t>(left)));
            //     }
            // } else {
            uint frameCounter = 0;
            uint start        = 0;
            for (int left = nCheck; left > 0; left -= batchSize) {
                batchSizeInfo.emplace_back(
                    frameCounter, std::min(static_cast<size_t>(batchSize), static_cast<size_t>(left)), start);
                frameCounter++;
                start += batchSize;
                // }
            }
            // }
        }

#ifdef USE_MULTI_THREADING
        task = [&](std::tuple<uint, uint, uint> batchInfo) {
            uint frameId      = std::get<0>(batchInfo);
            uint nPosPerBatch = std::get<1>(batchInfo);
            uint start        = std::get<2>(batchInfo);
            if constexpr (TYPE == LinkType::RAY_ONLINE || TYPE == LinkType::RAY_STATIC) {
                if (rayRobot->useOBB) {
                    detectWithOBB(frameId, nPosPerBatch);
                } else {
                    detect(frameId, nPosPerBatch);
                }
            } else {
                if (meshRobot->useOBB) {
                    detectWithOBB(frameId, nPosPerBatch);
                } else {
                    detect(frameId, nPosPerBatch);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(resultBuffer.d_pointer() + start * sizeof(int)),
                reinterpret_cast<void*>(streamDatas[frameId % NSTREAM].hitBuffer.d_pointer()),
                nPosPerBatch * sizeof(int), cudaMemcpyDeviceToDevice, streams[frameId % NSTREAM]));
        };
#endif
        // }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::check() {
#ifdef USE_MULTI_THREADING

        for (int i = 0; i < batchSizeInfo.size(); i++) {
            pool.enqueueTask(task, batchSizeInfo[i]);
            if (i % NSTREAM == NSTREAM - 1) {
                // Host-side barrier only: the next round reuses the same per-stream host state (launch params,
                // build inputs). GPU work on each stream is already ordered, so no device-wide sync is needed here;
                // the single CUDA_SYNC_CHECK() at the end of check() makes the results visible.
                pool.waitForCompletion();
            }
        }
        pool.waitForCompletion();
#else
        // if constexpr (TYPE == LinkType::CUBIC_CURVE || TYPE == LinkType::QUADRATIC_CURVE
        //               || TYPE == LinkType::LINEAR_CURVE) {
        //     for (auto& batchInfo : batchSizeInfo) {
        //         uint frameId      = batchInfo.first;
        //         uint nPosPerBatch = batchInfo.second;
        //         detect(frameId, nPosPerBatch);
        //         CUDA_CHECK(cudaMemcpyAsync(
        //             reinterpret_cast<void*>(resultBuffer.d_pointer() + frameId * nPosPerBatch * sizeof(int)),
        //             reinterpret_cast<void*>(streamDatas[frameId % NSTREAM].hitBuffer.d_pointer()),
        //             nPosPerBatch * sizeof(int), cudaMemcpyDeviceToDevice, streams[frameId % NSTREAM]));
        //     }
        // } else {
        for (auto& batchInfo : batchSizeInfo) {
            uint frameId      = std::get<0>(batchInfo);
            uint nPosPerBatch = std::get<1>(batchInfo);
            uint start        = std::get<2>(batchInfo);
            if constexpr (TYPE == LinkType::RAY_ONLINE) {
                if (rayRobot->useOBB) {
                    detectWithOBB(frameId, nPosPerBatch);
                } else {
                    detect(frameId, nPosPerBatch);
                }
            } else {
                if (meshRobot->useOBB) {
                    detectWithOBB(frameId, nPosPerBatch);
                } else {
                    detect(frameId, nPosPerBatch);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(resultBuffer.d_pointer() + start * sizeof(int)),
                reinterpret_cast<void*>(streamDatas[frameId % NSTREAM].hitBuffer.d_pointer()),
                nPosPerBatch * sizeof(int), cudaMemcpyDeviceToDevice, streams[frameId % NSTREAM]));
        }
        // }
#endif
        CUDA_SYNC_CHECK();
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::clearResult() {
        cudaMemset(reinterpret_cast<void*>(resultBuffer.d_pointer()), 0, resultBuffer.sizeInBytes);
    }
} // namespace RTCD

namespace RTCD { // Optix Pipeline Setup
    template <size_t NSTREAM, LinkType TYPE>
    inline unsigned int CollisionDetector<NSTREAM, TYPE>::getBuildFlags() {
        // Built-in intersection modules (spheres / curves) must be created with exactly the build flags of the GAS
        // they intersect, otherwise OptiX faults at trace time (sticky CUDA error). So once a robot is attached,
        // use its flags; the per-type defaults below only apply before setupRobot().
        if constexpr (usesBuiltinIS) {
            if (meshRobot) {
                return meshRobot->getBuildFlags();
            }
        }
        if constexpr (TYPE == LinkType::SPHERES) {
            return OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        } else if constexpr (TYPE == LinkType::SPHERE_GAS) {
            return OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        } else if constexpr (TYPE == LinkType::CUBIC_CURVE) {
            return OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        } else if constexpr (TYPE == LinkType::QUADRATIC_CURVE) {
            return OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        } else if constexpr (TYPE == LinkType::LINEAR_CURVE) {
            return OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        } else if constexpr (TYPE == LinkType::MESH) {
            return OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        } else {
            return OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::createModule(const unsigned int buildFlags) {
        moduleCompileOptions.maxRegisterCount = 50;

        pipelineCompileOptions                = {};
        pipelineCompileOptions.usesMotionBlur = false;
        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            pipelineCompileOptions.numPayloadValues = 1;
        } else {
            pipelineCompileOptions.numPayloadValues = 0;
        }
        pipelineCompileOptions.numAttributeValues = 2;
#if defined(_DEBUG) || defined(DEBUG)
        moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#elif defined(OPTIX_PROFILE)
        std::cout << "Using Profile Model" << std::endl;
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
#else
        moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif

        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        if constexpr (TYPE == LinkType::SPHERES) {
            pipelineCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
        } else if constexpr (TYPE == LinkType::CUBIC_CURVE) {
            pipelineCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
        } else if constexpr (TYPE == LinkType::QUADRATIC_CURVE) {
            pipelineCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE;
        } else if constexpr (TYPE == LinkType::LINEAR_CURVE) {
            pipelineCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
        } else if constexpr (TYPE == LinkType::SPHERE_GAS) {
            pipelineCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
        } else {
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
#ifdef RAYDATAMODE
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                                                         | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
#endif
        }

        pipelineLinkOptions.maxTraceDepth = 2;

        size_t inputSize = 0;
        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            const char* input = getInputData("CCRay.cu", inputSize);

            char log[4096];
            size_t sizeof_log = sizeof(log);
            const OptixResult moduleRes = optixModuleCreate(optixContext, &moduleCompileOptions,
                &pipelineCompileOptions, input, inputSize, log, &sizeof_log, &Module);
            if (sizeof_log > 1) { // print the compile log before bailing out, so failures are diagnosable
                std::cout << log << std::endl;
            }
            OPTIX_CHECK(moduleRes);
        } else {
            const char* input = getInputData("CCCuda.cu", inputSize);

            char log[4096];
            size_t sizeof_log = sizeof(log);
            const OptixResult moduleRes = optixModuleCreate(optixContext, &moduleCompileOptions,
                &pipelineCompileOptions, input, inputSize, log, &sizeof_log, &Module);
            if (sizeof_log > 1) { // print the compile log before bailing out, so failures are diagnosable
                std::cout << log << std::endl;
            }
            OPTIX_CHECK(moduleRes);
        }

        OptixBuiltinISOptions builtinISOptions = {};
        builtinISOptions.buildFlags            = buildFlags;

        if constexpr (TYPE == LinkType::SPHERES || TYPE == LinkType::SPHERE_GAS) {
            builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
            OPTIX_CHECK(optixBuiltinISModuleGet(
                optixContext, &moduleCompileOptions, &pipelineCompileOptions, &builtinISOptions, &sphereModule));
        } else if constexpr (TYPE == LinkType::CUBIC_CURVE) {
            builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
            builtinISOptions.curveEndcapFlags    = 1;
            OPTIX_CHECK(optixBuiltinISModuleGet(
                optixContext, &moduleCompileOptions, &pipelineCompileOptions, &builtinISOptions, &curveModule));
        } else if constexpr (TYPE == LinkType::QUADRATIC_CURVE) {
            builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
            builtinISOptions.curveEndcapFlags    = 1;
            OPTIX_CHECK(optixBuiltinISModuleGet(
                optixContext, &moduleCompileOptions, &pipelineCompileOptions, &builtinISOptions, &curveModule));
        } else if constexpr (TYPE == LinkType::LINEAR_CURVE) {
            builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
            builtinISOptions.curveEndcapFlags    = 1;
            OPTIX_CHECK(optixBuiltinISModuleGet(
                optixContext, &moduleCompileOptions, &pipelineCompileOptions, &builtinISOptions, &curveModule));
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::createRaygenPrograms() {
        // we do a single ray gen program in this example:
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc       = {};
        pgDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module               = Module;
        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            pgDesc.raygen.entryFunctionName = "__raygen__robotRayCompact";
            // } else if constexpr (TYPE == LinkType::RAY_STATIC) {
            //     pgDesc.raygen.entryFunctionName = "__raygen__robotRayOnline";
        } else {
            pgDesc.raygen.entryFunctionName = "__raygen__oneside";
        }

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPGs[0]));
        if (sizeof_log > 1) {
            std::cout << log << std::endl;
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::createMissPrograms() {
        // we do a single ray gen program in this example:
        missPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc       = {};
        pgDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
        // pgDesc.miss.module                 = Module;
        // pgDesc.miss.entryFunctionName      = "__miss__radiance";
        pgDesc.miss.module            = nullptr;
        pgDesc.miss.entryFunctionName = nullptr;

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPGs[0]));
        if (sizeof_log > 1) {
            std::cout << log << std::endl;
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::createHitgroupPrograms() {
        // for this simple example, we set up a single hit group
        hitgroupPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc       = {};
        pgDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
#ifdef RAYDATAMODE
        if constexpr (TYPE == LinkType::CUBIC_CURVE || TYPE == LinkType::QUADRATIC_CURVE
                      || TYPE == LinkType::LINEAR_CURVE) {
            pgDesc.hitgroup.moduleIS            = curveModule;
            pgDesc.hitgroup.entryFunctionNameIS = nullptr;
        }
        pgDesc.hitgroup.moduleCH            = nullptr;
        pgDesc.hitgroup.entryFunctionNameCH = nullptr;
        pgDesc.hitgroup.moduleAH            = Module;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__RayData";
#else

        if constexpr (TYPE == LinkType::SPHERE_GAS) {
            pgDesc.hitgroup.moduleIS            = sphereModule;
            pgDesc.hitgroup.entryFunctionNameIS = nullptr;
            pgDesc.hitgroup.moduleCH            = nullptr;
            pgDesc.hitgroup.entryFunctionNameCH = nullptr;
            pgDesc.hitgroup.moduleAH            = Module;
            pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__GAS";
        } else if constexpr (TYPE == LinkType::SPHERES) {
            pgDesc.hitgroup.moduleIS            = sphereModule;
            pgDesc.hitgroup.entryFunctionNameIS = nullptr;
            pgDesc.hitgroup.moduleCH            = nullptr;
            pgDesc.hitgroup.entryFunctionNameCH = nullptr;
            pgDesc.hitgroup.moduleAH            = Module;
            pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__IAS";
        } else if constexpr (TYPE == LinkType::CUBIC_CURVE || TYPE == LinkType::QUADRATIC_CURVE
                             || TYPE == LinkType::LINEAR_CURVE) {
            pgDesc.hitgroup.moduleIS            = curveModule;
            pgDesc.hitgroup.entryFunctionNameIS = nullptr;
            pgDesc.hitgroup.moduleCH            = nullptr;
            pgDesc.hitgroup.entryFunctionNameCH = nullptr;
            pgDesc.hitgroup.moduleAH            = Module;
            pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__GAS";
            // } else if constexpr (TYPE == LinkType::RAY || TYPE == LinkType::RAY_STATIC) {
            //     pgDesc.hitgroup.moduleCH            = Module;
            //     pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
            //     pgDesc.hitgroup.moduleAH            = nullptr;
            //     pgDesc.hitgroup.entryFunctionNameAH = nullptr;
        } else if constexpr (TYPE == LinkType::RAY_ONLINE) {
            pgDesc.hitgroup.moduleCH            = Module;
            pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
            pgDesc.hitgroup.moduleAH            = nullptr;
            pgDesc.hitgroup.entryFunctionNameAH = nullptr;
        } else {
            pgDesc.hitgroup.moduleCH            = nullptr;
            pgDesc.hitgroup.entryFunctionNameCH = nullptr;
            pgDesc.hitgroup.moduleAH            = Module;
            pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__IAS";
        }
#endif
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPGs[0]));
        if (sizeof_log > 1) {
            std::cout << log << std::endl;
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::createPipeline() {
        std::vector<OptixProgramGroup> programGroups;
        for (auto pg : raygenPGs) {
            programGroups.push_back(pg);
        }
        for (auto pg : missPGs) {
            programGroups.push_back(pg);
        }
        for (auto pg : hitgroupPGs) {
            programGroups.push_back(pg);
        }

        char log[2048];
        size_t sizeof_log = sizeof(log);

        for (auto& streamData : streamDatas) {
            auto& pipeline = streamData.pipeline;
            OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions,
                programGroups.data(), (int) programGroups.size(), log, &sizeof_log, &pipeline));
#ifdef PRINTIT
            if (sizeof_log > 1) {
                std::cout << log << std::endl;
            }
#endif

            OptixStackSizes stack_sizes = {};
            for (unsigned int i = 0; i < programGroups.size(); ++i) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroups.data()[i], &stack_sizes, pipeline));
            }
            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1,
                0, // maxCCDepth
                0, // maxDCDEpth
                &direct_callable_stack_size_from_traversal, &direct_callable_stack_size_from_state,
                &continuation_stack_size));

            if constexpr (TYPE == LinkType::SPHERES) {
                OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                    direct_callable_stack_size_from_state, continuation_stack_size,
                    2 // maxTraversableDepth
                    ));
            } else if constexpr (TYPE == LinkType::SPHERE_GAS || TYPE == LinkType::CUBIC_CURVE
                                 || TYPE == LinkType::QUADRATIC_CURVE || TYPE == LinkType::LINEAR_CURVE) {
                OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                    direct_callable_stack_size_from_state, continuation_stack_size,
                    1 // maxTraversableDepth
                    ));
            } else {
                OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                    direct_callable_stack_size_from_state, continuation_stack_size,
                    2 // maxTraversableDepth
                    ));
            }
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::buildSBT() {
        size_t sbtSize;
        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            sbtSize = sc->getNumObstacles();
        } else {
            sbtSize = meshRobot->getSBTSize();
        }
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<RaygenRecord> raygenRecords;
        for (int i = 0; i < raygenPGs.size(); i++) {
            RaygenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        raygenRecordsBuffer.free();
        raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<MissRecord> missRecords;
        for (int i = 0; i < missPGs.size(); i++) {
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        missRecordsBuffer.free();
        missRecordsBuffer.alloc_and_upload(missRecords);
        sbt.missRecordBase          = missRecordsBuffer.d_pointer();
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount         = (int) missRecords.size();

        // ------------------------------------------------------------------
        // build hitgroup records
        // ------------------------------------------------------------------

        // we don't actually have any objects in this example, but let's
        // create a dummy one so the SBT doesn't have any null pointers
        // (which the sanity checks in compilation would complain about)
        std::vector<HitgroupRecord> hitgroupRecords;
        for (int i = 0; i < sbtSize; i++) {
            int objectType = 0;
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
            rec.data = nullptr;
            hitgroupRecords.push_back(rec);
        }
        hitgroupRecordsBuffer.free();
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount         = (int) hitgroupRecords.size();
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::resetSBT() {
        sbt = {};
        raygenRecordsBuffer.free();
        missRecordsBuffer.free();
        hitgroupRecordsBuffer.free();
    }
} // namespace RTCD


namespace RTCD {
    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::updateASHandle(const OptixTraversableHandle handle, const size_t frameId) {
        const int streamId        = frameId % NSTREAM;
        perStreamData& streamData = streamDatas[streamId];

        streamData.launchParams.traversable = handle;
        uploadLaunchParams(streamData);
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::resetGraph() {
        for (int i = 0; i < NSTREAM; i++) {
            streamDatas[i].graphCreated = false;
        }
        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            rayRobot->resetGraph();
        } else {
            meshRobot->resetGraph();
        }
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::detect(const size_t frameId) {
        detect(frameId, batchSize);
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::detect(const size_t frameId, const int nPoses) {

        const int streamId        = frameId % NSTREAM;
        perStreamData& streamData = streamDatas[streamId];
        LaunchParams& lp          = streamData.launchParams;
        CUDABuffer& lpBuffer      = streamData.launchParamsBuffer;

#ifndef RAYDATAMODE // handle is manually set before detection in ray data mode
        meshRobot->movePosesToStream(frameId, nPoses);
        meshRobot->update(streamId, nPoses);
        lp.traversable = meshRobot->getHandle(streamId);
#endif
        meshRayInfo rayInfo      = sc->getRayInfo();
        lp.verticesBuffer        = reinterpret_cast<float3*>(rayInfo.meshRays);
        lp.totalRayCount         = rayInfo.nRays;
        lp.mesh.primIdxToPoseIdx = reinterpret_cast<uint*>(meshRobot->getMapIndex(streamId));
        uploadLaunchParams(streamData); // lp aliases streamData.launchParams

        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(getHitBuffer(streamId)), 0, getHitBufferSize(streamId), streams[streamId]));
        OPTIX_CHECK(optixLaunch(streamData.pipeline, streamData.stream, streamData.launchParamsBuffer.d_pointer(),
            streamData.launchParamsBuffer.sizeInBytes, &sbt, streamData.launchParams.totalRayCount, 1, 1));
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::detectWithOBB(const size_t frameId) {
        detectWithOBB(frameId, batchSize);
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::warmup(const int nPoses) {
        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            rayRobot->prepareRandomData();
        } else {
            meshRobot->prepareRandomData();
        }

        setNCheck(batchSize * NSTREAM);

        check();

        clearResult();
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::detectWithOBB(const size_t frameId, const int nPoses) {
        const int streamId        = frameId % NSTREAM;
        perStreamData& streamData = streamDatas[streamId];
        cudaStream_t& stream      = streamData.stream;

        CUdeviceptr hitBuffer = streamData.hitBuffer.d_pointer();
        size_t hitBufferSize  = streamData.hitBuffer.sizeInBytes;

        LaunchParams& lp     = streamData.launchParams;
        CUDABuffer& lpBuffer = streamData.launchParamsBuffer;

        bool& graphCreated        = streamData.graphCreated;
        cudaGraph_t& graph        = streamData.graph;
        cudaGraphExec_t& instance = streamData.instance;

        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(hitBuffer), 0, hitBufferSize, stream));

        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            rayRobot->movePosesToStream(frameId, nPoses);
            rayRobot->fkine(streamId, nPoses);
            detectOBB(streamId, nPoses);
            const CUdeviceptr mask = getRobotMask(streamId);

            // No host sync: the launch covers all nPoses * raysPerRobot rays and the raygen skips links whose OBB
            // is collision free (mask read on the device). All params below are constant for a given stream and
            // nPoses, so uploadLaunchParams() normally skips the upload.
            bool go = rayRobot->updateWithMask(mask, streamId, nPoses);
            if (go) {
                const meshRayInfo rayInfo = rayRobot->getRayInfo(streamId);
                lp.type                = rayType::ROBOT;
                lp.totalRayCount       = rayInfo.nRays;
                lp.verticesBuffer      = nullptr;
                lp.traversable         = sc->getHandle();
                lp.hitBuffer           = reinterpret_cast<int*>(hitBuffer);
                lp.robot.raysPerRobot  = rayInfo.robotInfo.raysPerRobot;
                lp.robot.linksPerRobot = rayInfo.robotInfo.linksPerRobot;
                lp.robot.linkMask      = reinterpret_cast<const uint*>(rayInfo.robotInfo.linkMask);
                lp.robot.rayLinkMap    = reinterpret_cast<const int*>(rayInfo.robotInfo.rayLinkMap);
                lp.robot.rays          = reinterpret_cast<const float3*>(rayInfo.robotInfo.rays);
                lp.robot.lkTfs         = reinterpret_cast<const float*>(rayInfo.robotInfo.lkTfs);
                // EXPERIMENT knobs (to be fixed to constants): launch width cap and hit-pose skipping
                static const unsigned int expCap  = [] { const char* s = std::getenv("RTCD_RAY_CAP"); return s ? (unsigned int) std::strtoul(s, nullptr, 10) : 0u; }();
                static const unsigned int expSkip = [] { const char* s = std::getenv("RTCD_RAY_SKIPHIT"); return s ? (unsigned int) std::strtoul(s, nullptr, 10) : 1u; }();
                lp.robot.skipHitPoses = expSkip;
                const unsigned int width = (expCap != 0 && lp.totalRayCount > expCap) ? expCap : lp.totalRayCount;

                uploadLaunchParams(streamData); // lp aliases streamData.launchParams
                OPTIX_CHECK(optixLaunch(streamData.pipeline, streamData.stream, lpBuffer.d_pointer(),
                    lpBuffer.sizeInBytes, &sbt, width, 1, 1));
                CUDA_CHECK_LAST("Ray Shot");
            }
        } else { // shoot from obs
            meshRobot->movePosesToStream(frameId, nPoses);
            meshRobot->fkine(streamId, nPoses);
            detectOBB(streamId, nPoses);
            meshRayInfo rayInfo = sc->getRayInfo();
            bool go             = meshRobot->updateWithMask(getRobotMask(streamId), streamId, nPoses);
            if (go) {
                LaunchParams& lp         = streamData.launchParams;
                lp.traversable           = meshRobot->getHandle(streamId);
                lp.verticesBuffer        = reinterpret_cast<float3*>(rayInfo.meshRays);
                lp.totalRayCount         = rayInfo.nRays;
                lp.mesh.primIdxToPoseIdx = reinterpret_cast<uint*>(meshRobot->getMapIndex(streamId));

                uploadLaunchParams(streamData); // lp aliases streamData.launchParams
                OPTIX_CHECK(optixLaunch(streamData.pipeline, streamData.stream, lpBuffer.d_pointer(),
                    lpBuffer.sizeInBytes, &sbt, lp.totalRayCount, 1, 1));
            }
        }
    }
} // namespace RTCD
namespace RTCD { // Private functions

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::detectOBB(const size_t frameId) {
        return detectOBB(frameId, batchSize);
    }

    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::detectOBB(const size_t frameId, const int nPoses) {
        const int streamId        = frameId % NSTREAM;
        perStreamData& streamData = streamDatas[streamId];

        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(getRobotMask(streamId)), 0, nRobotOBBs * sizeof(int), streams[streamId]));
        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(getSceneMask(streamId)), 0, nSceneOBBs * sizeof(int), streams[streamId]));
        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(getRSMask(streamId)), 0, nRobotOBBs * nSceneOBBs * sizeof(int), streams[streamId]));
        if constexpr (TYPE == LinkType::RAY_ONLINE) {
            BBCD(streamData.robotOBBs, streamData.sceneOBBs, rayRobot->getNOBBs(nPoses), nSceneOBBs,
                getRobotMask(streamId), getSceneMask(streamId), getRSMask(streamId), streams[streamId]);
        } else {
            BBCD(streamData.robotOBBs, streamData.sceneOBBs, meshRobot->getNOBBs(nPoses), nSceneOBBs,
                getRobotMask(streamId), getSceneMask(streamId), getRSMask(streamId), streams[streamId]);
        }
    }
} // namespace RTCD

namespace RTCD { // headline image helper function
    template <size_t NSTREAM, LinkType TYPE>
    void CollisionDetector<NSTREAM, TYPE>::resizeHitBuffer(const int size) {
        for (int i = 0; i < NSTREAM; i++) {
            streamDatas[i].hitBuffer.free();
            streamDatas[i].hitBuffer.alloc(sizeof(int) * size);
            LaunchParams& params = streamDatas[i].launchParams;
            params.hitBuffer     = reinterpret_cast<int*>(streamDatas[i].hitBuffer.d_pointer());

            streamDatas[i].launchParamsBuffer.upload(&streamDatas[i].launchParams, 1);
            streamDatas[i].lpUploadedValid = false; // written outside uploadLaunchParams()
        }
    }
} // namespace RTCD
