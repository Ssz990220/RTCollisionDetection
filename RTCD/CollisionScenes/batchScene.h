#pragma once

#include <CollisionScenes/baseBatchScene.h>
#include <CollisionScenes/scene.h>
#include <CollisionScenes/sceneKern.cuh>
#include <Utils/cuda/cudaAlgorithms.cuh>
#include <memory>

namespace RTCD {
    template <batchSceneConfig CFG>
    struct streamUniqueScene {
        size_t nObs;
        size_t nEdges;

        cudaStream_t stream;
        OptixDeviceContext context;
        OptixBuildInput buildInput{};
        OptixAccelBuildOptions buildOptions{};
        OptixAccelBuildOptions updateOptions{};

        CUDABuffer IASBuffer;
        size_t IASBufferSize;
        CUDABuffer tempBuffer;

        // CUdeviceptr edgesTemplate; // complete edges
        // CUdeviceptr edgesCnt; // edge cnt each obs
        // CUdeviceptr whereInSrcEdge; // where each obs starts in the edge buffer
        // CUDABuffer edgesBuffer; // compacted edges

        CUDABuffer tfInvBuffer;
        CUDABuffer dWhereInDst; // exclusive scan result
        CUdeviceptr indexMapTmpl;
        CUDABuffer batchIndexBuffer;

        CUdeviceptr templateIns;
        CUDABuffer templateInsBuffer;
        CUDABuffer instancesBuffer; // compacted instances
        std::vector<meshRayInfo> rayInfo;
        std::vector<uint> hMask;
        std::vector<uint> nObRay;
        std::vector<CUdeviceptr> obRays;

        OptixTraversableHandle IASHandle;

        bool updateASWithMask(const CUdeviceptr tfs, const CUdeviceptr mask, const size_t nPoses);

        void setupInstances(const CUdeviceptr& Ins);
    };

    template <batchSceneConfig CFG>
    class batchScene final : public baseBatchScene<CFG.NSTREAM> {

    public:
        batchScene() = default;
        batchScene(
            std::shared_ptr<scene> scenePtr, std::array<cudaStream_t, CFG.NSTREAM> streams, OptixDeviceContext context);
        bool updateASWithMask(const CUdeviceptr tfs, const CUdeviceptr mask, const size_t idx) override;
        bool updateASWithMask(const CUdeviceptr tfs, const CUdeviceptr mask, const size_t idx, const size_t nPoses);

        inline size_t getSBTSize() const override { return nObs; }
        inline size_t getNumObstacles() const override { return nObs; }
        inline size_t getNumEdges() const override { return nEdges; }
        inline CUdeviceptr getOBBs() const override { return s->getdOBBs(); }
        inline size_t getSgRayCnt() const override { return s->getSgRayCnt(); }
        inline CUdeviceptr getSgVerts() const override { return s->getdSgVerts(); }
        inline const std::vector<meshRayInfo>& getRayInfo(const size_t idx) const override {
            return scenes[idx].rayInfo;
        }
        inline meshRayInfo getRayInfo() const override {
            return meshRayInfo{0, static_cast<uint>(nEdges), s->getdSgVerts()};
        }
        inline const OptixTraversableHandle getHandle() const override { return s->getHandle(); }
        inline const OptixTraversableHandle getHandle(const size_t idx) const override { return scenes[idx].IASHandle; }
        const CUdeviceptr getMapIndex(const size_t idx) const override {
            return scenes[idx].batchIndexBuffer.d_pointer();
        }

    private:
        std::shared_ptr<scene> s;
        std::array<cudaStream_t, CFG.NSTREAM> streams;

        size_t nObs;
        size_t nEdges;

        OptixDeviceContext context;
        CUDABuffer insBuffer;
        CUDABuffer indexMap;
        std::array<streamUniqueScene<CFG>, CFG.NSTREAM> scenes;
    };
} // namespace RTCD

namespace RTCD { // Implementation of batchScene
    template <batchSceneConfig CFG>
    batchScene<CFG>::batchScene(
        std::shared_ptr<scene> scenePtr, std::array<cudaStream_t, CFG.NSTREAM> streams, OptixDeviceContext context)
        : s(scenePtr), nObs(scenePtr->getNumObstacles()), nEdges(scenePtr->getSgRayCnt()), context(context),
          streams(streams) {

        s->uploadToDevice();
        s->wrapScene(context, false);

        if constexpr (CFG.compactable) {
            setDefaultTf();
            setSceneSafeTf();

            auto nVerts = s->getObsVertCnt();

            std::vector<uint> nObRay(nObs);
            std::transform(
                nVerts.begin(), nVerts.end(), nObRay.begin(), [](uint n) { return static_cast<uint>(n / 2); });

            std::vector<CUdeviceptr> obRays(nObs);
            std::vector<uint> obRaySize(nObs);
            std::transform(
                nObRay.begin(), nObRay.end(), obRaySize.begin(), [](uint n) { return n * sizeof(float3) * 2; });
            std::exclusive_scan(obRaySize.begin(), obRaySize.end(), obRays.begin(), s->getdSgVerts());

            const std::vector<OptixInstance>& instances = s->getInstances();
            std::vector<OptixInstance> batchIns;
            for (int i = 0; i < nObs; i++) {
                for (int j = 0; j < CFG.BATCHSIZE; j++) {
                    batchIns.push_back(instances[i]);
                }
            }
            insBuffer.free();
            insBuffer.alloc_and_upload(batchIns);

            std::vector<uint> idxMap(CFG.BATCHSIZE);
            std::iota(idxMap.begin(), idxMap.end(), 0);
            std::vector<uint> idxMapBatch;
            for (int i = 0; i < nObs; i++) {
                idxMapBatch.insert(idxMapBatch.end(), idxMap.begin(), idxMap.end());
            }
            indexMap.alloc_and_upload(idxMapBatch);

            OptixBuildInput bi;
            bi.type                         = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            bi.instanceArray.instances      = insBuffer.d_pointer();
            bi.instanceArray.numInstances   = nObs * CFG.BATCHSIZE;
            bi.instanceArray.instanceStride = sizeof(OptixInstance);

            OptixAccelBuildOptions bo;
            bo.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            bo.motionOptions.numKeys = 1;
            bo.operation             = OPTIX_BUILD_OPERATION_BUILD;


            OptixAccelBufferSizes bufferSizesIAS;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &bo, &bi, 1, &bufferSizesIAS));

            for (size_t i = 0; i < CFG.NSTREAM; i++) {
                scenes[i].nObs    = nObs;
                scenes[i].stream  = streams[i];
                scenes[i].context = context;

                scenes[i].updateOptions.buildFlags            = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
                scenes[i].updateOptions.motionOptions.numKeys = 1;
                scenes[i].updateOptions.operation             = OPTIX_BUILD_OPERATION_UPDATE;

                scenes[i].IASBuffer.alloc(bufferSizesIAS.outputSizeInBytes * 1.2);
                scenes[i].tempBuffer.alloc(bufferSizesIAS.tempSizeInBytes * 1.2);

                scenes[i].setupInstances(insBuffer.d_pointer());
                scenes[i].indexMapTmpl = indexMap.d_pointer();
                scenes[i].batchIndexBuffer.alloc(CFG.BATCHSIZE * nObs * sizeof(uint));

                scenes[i].tfInvBuffer.alloc(CFG.BATCHSIZE * sizeof(float) * 12);
                scenes[i].dWhereInDst.alloc(roundUpLog2(nObs * CFG.BATCHSIZE) * sizeof(uint));

                // scenes[i].rayInfo.resize(nObs);
                scenes[i].hMask.resize(nObs);
                scenes[i].nObRay = nObRay;
                scenes[i].obRays = obRays;
                scenes[i].rayInfo.clear();
                for (int j = 0; j < nObs; j++) {
                    scenes[i].rayInfo.emplace_back(j, nObRay[j], obRays[j]);
                }

                scenes[i].buildInput                         = bi;
                scenes[i].buildInput.instanceArray.instances = scenes[i].instancesBuffer.d_pointer();

                scenes[i].buildOptions = bo;

                // preallocation for stream compact
                scenes[i].tempBuffer.expandIfNotEnough(
                    (nObs * CFG.BATCHSIZE + (nObs * CFG.BATCHSIZE + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE)
                    * sizeof(uint));

                // // Build when initialized
                // OPTIX_CHECK(optixAccelBuild(context, 0, &bo, &bi, 1, scenes[i].tempBuffer.d_pointer(),
                //     scenes[i].tempBuffer.sizeInBytes, scenes[i].IASBuffer.d_pointer(),
                //     scenes[i].IASBuffer.sizeInBytes,
                //     &(scenes[i].IASHandle), nullptr, 0));
            }
        }
    }

    template <batchSceneConfig CFG>
    bool batchScene<CFG>::updateASWithMask(const CUdeviceptr tfs, const CUdeviceptr mask, const size_t idx) {
        return scenes[idx].updateASWithMask(tfs, mask, CFG.BATCHSIZE);
    }

    template <batchSceneConfig CFG>
    bool batchScene<CFG>::updateASWithMask(
        const CUdeviceptr tfs, const CUdeviceptr mask, const size_t idx, const size_t nPoses) {
        return scenes[idx].updateASWithMask(tfs, mask, nPoses);
    }

    // template <batchSceneConfig CFG>
    // void batchScene<CFG>::updateEdgeWithMask(const CUdeviceptr mask, const size_t idx) {
    //     scenes[idx].updateEdgeWithMask(mask, idx);
    // }
} // namespace RTCD

namespace RTCD { // Implementation of streamUniqueScene
    template <batchSceneConfig CFG>
    void streamUniqueScene<CFG>::setupInstances(const CUdeviceptr& Ins) {
        instancesBuffer.free();
        instancesBuffer.alloc(CFG.BATCHSIZE * nObs * sizeof(OptixInstance));
        templateInsBuffer.free();
        templateInsBuffer.alloc(CFG.BATCHSIZE * nObs * sizeof(OptixInstance));
        templateInsBuffer.copy(Ins, CFG.BATCHSIZE * nObs * sizeof(OptixInstance));
        templateIns = templateInsBuffer.d_pointer();
    }

    template <batchSceneConfig CFG>
    bool streamUniqueScene<CFG>::updateASWithMask(const CUdeviceptr tfs, const CUdeviceptr mask, const size_t nPoses) {
        uint nValid = 0;
#if defined(_DEBUG) || defined(DEBUG)
        std::vector<uint> hMask(nObs * CFG.BATCHSIZE);
        cudaMemcpyAsync(
            hMask.data(), (void*) mask, nObs * CFG.BATCHSIZE * sizeof(uint), cudaMemcpyDeviceToHost, stream);
        int nInCol = std::count_if(hMask.begin(), hMask.end(), [&](uint i) { return i; });
#endif
        transformIsValid<uint, true>(mask, tempBuffer.d_pointer(), nPoses * nObs, stream);
        reduceSum<uint>(tempBuffer.d_pointer(), tempBuffer.d_pointer(), nPoses * nObs, stream);
        cudaMemcpyAsync(
            &nValid, reinterpret_cast<void*>(tempBuffer.d_pointer()), sizeof(uint), cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (nValid) {
            const size_t nEle    = nObs * nPoses;
            const size_t eleSize = sizeof(OptixInstance);
            inverseTf(tfInvBuffer.d_pointer(), tfs, nPoses, stream);
            setInsTf(tfInvBuffer.d_pointer(), templateIns, CFG.BATCHSIZE, nObs, stream);
            transformIsValid<uint, true>(mask, tempBuffer.d_pointer(), nPoses * nObs, stream);
            exclusiveScan(
                dWhereInDst.d_pointer(), tempBuffer.d_pointer(), tempBuffer.d_pointer() + nEle * eleSize, nEle, stream);
            compactCpy<true>(
                instancesBuffer.d_pointer(), templateIns, mask, dWhereInDst.d_pointer(), eleSize, nEle, stream);
            compactCpy<true>(
                batchIndexBuffer.d_pointer(), indexMapTmpl, mask, dWhereInDst.d_pointer(), sizeof(uint), nEle, stream);
#if defined(_DEBUG) || defined(DEBUG)
            CUDA_SYNC_CHECK();
            std::vector<float> hTfInv;
            tfInvBuffer.download(hTfInv);

            std::vector<uint> hWhereInDst;
            dWhereInDst.download(hWhereInDst);

            std::vector<OptixInstance> hIns(nValid);
            instancesBuffer.download(hIns);

            std::vector<uint> hIdx(nValid);
            batchIndexBuffer.download(hIdx);
#endif
            buildInput.instanceArray.numInstances = nValid;

            OPTIX_CHECK(optixAccelBuild(context, stream, &buildOptions, &buildInput, 1, tempBuffer.d_pointer(),
                tempBuffer.sizeInBytes, IASBuffer.d_pointer(), IASBuffer.sizeInBytes, &IASHandle, nullptr, 0));

        } else {
            return false;
        }
        return true;
    }
} // namespace RTCD
