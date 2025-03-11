#pragma once

#include <Robot/baseBatchRobot.h>
#include <Robot/batchRobotConfig.h>
#include <Robot/robot.h>
#include <Utils/cuda/cudaAlgorithms.cuh>
#include <memory>

namespace RTCD {
    struct ORayData {
        uint lastMask;
        uint scanLinks;
        uint lastRay;
        uint scanRays;
        ORayData() = default;
    };

    template <batchRobotConfig CFG>
    struct streamRayRobot {
        cudaStream_t stream;
        cudaEvent_t event;

        size_t totalPerRoRay;
        CUdeviceptr templateRayPtr;
        CUdeviceptr rayLinkMapPtr; // Ray Robot (not for online or OBB)
        std::array<uint, CFG.DOF + 1> linkRayCnt; // V2.5
        std::array<CUdeviceptr, CFG.DOF + 1> linkInRay; // V2.5

        CUDABuffer lkTfBuffer; // Link Transform Buffer

        // V2.5
        CUdeviceptr linkobbTemplate; // points to memory that contains DOF + 1 obbs
        CUDABuffer linkOBBsBuffer; // The link OBBs are transformed version of obbs in linkobbTempalte

        // If RAY_ONLINE
        uint lkCnt; // how many compacted links
        uint totalRayCnt; // how many rays from all compacted links

        CUdeviceptr lkRayPtrsTmpl;
        CUdeviceptr lkPosMapTmpl;
        CUdeviceptr lkRayCntTmpl;

        CUDABuffer lkRayCntBuffer; // compacted link ray count
        CUDABuffer lkStartsBuffer; // prefix sum of lkRayCntBuffer link's ray count, in order to locate which link this
            // ray belongs to given a ray ID
        CUDABuffer lkPosMapBuffer; // compacted link pose map
        CUDABuffer lkTfCmpBuffer; // compacted link transforms based on mask
        CUDABuffer lkRayPtrsBuffer; // compacted link ray pointers
        CUDABuffer maskPrefixSumBuffer;
        CUDABuffer tempBuffer;

        bool graphCreated = false;
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        ORayData cntData;
        CUDABuffer cntDataBuffer;

        size_t cubTempSize = 0;

        // selected Ray shooting info // V2.5
        std::array<uint, CFG.BATCHSIZE*(CFG.DOF + 1)> h_mask;
        std::vector<meshRayInfo> rayInfo;
        // std::vector<int> rayMap(CFG.BATCHSIZE*(CFG.DOF + 1));

        streamRayRobot() = default;

        inline void fkine(const CUdeviceptr poses, const size_t nPoses);
        inline void fkine2(const CUdeviceptr poses, const size_t nPoses);
        inline void updateOBB(const size_t nPoses);
        inline void updateOBB2(const size_t nPoses);
        inline void update(const CUdeviceptr poses, const size_t nPoses);

        inline const meshRayInfo getRayInfo() const;
        inline bool updateWithMask(const CUdeviceptr mask, const size_t nPoses);
    };

    template <batchRobotConfig CFG>
    class batchRayRobot final : public baseRayRobot<CFG.NSTREAM, CFG.TYPE> {
    public:
        batchRayRobot(std::shared_ptr<robot<CFG.DOF>> robotPtr, std::array<cudaStream_t, CFG.NSTREAM> streams,
            const OptixDeviceContext context);
        void fkine(const CUdeviceptr poses, const size_t idx) override;
        void fkine(const CUdeviceptr poses, const size_t idx, const size_t nPoses) override;
        void fkine2(const CUdeviceptr poses, const size_t idx) override;
        void fkine2(const CUdeviceptr poses, const size_t idx, const size_t nPoses) override;
        void update(const CUdeviceptr poses, const size_t idx) override;
        void update(const CUdeviceptr poses, const size_t idx, const size_t nPoses) override;
        bool updateWithMask(const CUdeviceptr mask, const size_t idx) override;
        bool updateWithMask(const CUdeviceptr mask, const size_t idx, const size_t nPoses) override;
        void updateOBBs(const size_t idx) override;
        void updateOBBs(const size_t idx, const size_t nPoses) override;
        void updateOBBs2(const size_t idx) override;
        void updateOBBs2(const size_t idx, const size_t nPoses) override;


        constexpr size_t getBatchSize() const override { return CFG.BATCHSIZE; };
        constexpr size_t getDOF() const override { return CFG.DOF; }
        constexpr size_t getTrajSize() const override { return 1; }

        size_t getRayCount() const override { return totalPerRoRay; }
        const std::array<CUdeviceptr, CFG.NSTREAM>& getRays() const override { return rayOriginPtr; }

        const CUdeviceptr getLinkTfs(const size_t idx) const override {
            return batchRobots[idx].lkTfBuffer.d_pointer();
        }

        int* getRayMap() const override { return reinterpret_cast<int*>(rayLinkMapBuffer.d_pointer()); }
        std::vector<float3> getRayVertices() const override { return rayVertices; }

        const std::vector<meshRayInfo>& getRayInfo() const override { return rayInfo; }
        const meshRayInfo getRayInfo(const size_t idx) const override;
        // const std::vector<meshRayInfo>& getRayInfo(const size_t idx) const override { return
        // batchRobots[idx].rayInfo; }

        CUdeviceptr getOBBs(const size_t idx) const override { return batchRobots[idx].linkOBBsBuffer.d_pointer(); }
        constexpr size_t getNOBBs() const override { return CFG.BATCHSIZE * (CFG.DOF + 1); }
        constexpr size_t getNOBBs(size_t nPoses) const override { return nPoses * (CFG.DOF + 1); }

        void resetGraph() override;

#if defined(_DEBUG) || defined(DEBUG)
        void downloadTransforms(std::vector<float4>& transforms, const size_t idx) const {
            transforms.resize((CFG.DOF + 1) * CFG.BATCHSIZE * 3);
            batchRobots[idx]->lkTfBuffer.download(transforms);
        }

        void downloadRays(std::vector<float3>& rays, const size_t idx) const {
            rays.resize(totalPerRoRay * CFG.BATCHSIZE * 2);
            batchRobots[idx]->verticesBuffer.download(rays);
        }
#endif

    private:
        void prepareRobot();

    private:
        std::shared_ptr<robot<CFG.DOF>> robotPtr;

        size_t totalPerRoRay = 0;
        std::vector<float3> rayVertices;
        std::vector<int> rayLinkMap;
        std::array<size_t, CFG.DOF + 1> linkRayCnt;

        CUDABuffer rayVerticesBuffer;
        CUdeviceptr rayVerticesPtr;
        CUDABuffer linkRayCntBuffer;
        CUdeviceptr linkRayCntPtr;
        CUDABuffer rayLinkMapBuffer;
        CUdeviceptr rayLinkMapPtr;

        // If RAY_ONLINE
        CUDABuffer lkRayPtrs; // Pointer to link rays. Contains DOF + 1 float3* ptrs (float3**)
        CUDABuffer lkRayCntBuffer;
        CUDABuffer lkPosMapBuffer;

        std::array<CUdeviceptr, CFG.DOF + 1> linkInRay; // V2.5
        std::vector<meshRayInfo> rayInfo;
        std::array<CUdeviceptr, CFG.NSTREAM> rayOriginPtr; // backward compatability
        std::array<streamRayRobot<CFG>, CFG.NSTREAM> batchRobots;
    };
} // namespace RTCD


namespace RTCD { // Implementation of batchRayRobot
    template <batchRobotConfig CFG>
    batchRayRobot<CFG>::batchRayRobot(std::shared_ptr<robot<CFG.DOF>> r, std::array<cudaStream_t, CFG.NSTREAM> streams,
        const OptixDeviceContext context)
        : baseRayRobot<CFG.NSTREAM, CFG.TYPE>(streams, context), robotPtr(r) {

        baseRayRobot<CFG.NSTREAM, CFG.TYPE>::useOBB = CFG.useOBB;

        static_assert((CFG.TYPE == LinkType::RAY_STATIC || CFG.TYPE == LinkType::RAY_ONLINE),
            "batchRayRobot only supports ray type robot");
        static_assert(CFG.TYPE != LinkType::RAY, "Type RAY is deprecated since V2.5");
        prepareRobot();

        for (int i = 0; i < CFG.NSTREAM; ++i) {
            auto& br = batchRobots[i];
            // batchRobots[i]  = std::make_unique<streamRayRobot<CFG>>(this, streams[i]);
            br.stream          = streams[i];
            br.totalPerRoRay   = totalPerRoRay;
            br.templateRayPtr  = rayVerticesPtr;
            br.rayLinkMapPtr   = rayLinkMapPtr;
            br.linkobbTemplate = robotPtr->getOBBs();
            br.linkInRay       = linkInRay;
            std::transform(linkRayCnt.begin(), linkRayCnt.end(), br.linkRayCnt.begin(),
                [](size_t x) { return static_cast<uint>(x); });

            br.lkTfBuffer.alloc((CFG.DOF + 1) * CFG.BATCHSIZE * sizeof(float) * 12);
            br.linkOBBsBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(OBB));
            br.rayInfo.resize(CFG.BATCHSIZE * (CFG.DOF + 1));
            br.cntDataBuffer.alloc(sizeof(ORayData));
            rayOriginPtr[i] = rayVerticesPtr; // backward compatability

            if constexpr (CFG.TYPE == LinkType::RAY_ONLINE) { // RAY_ONLINE
                br.lkRayPtrsTmpl = lkRayPtrs.d_pointer();
                br.lkRayCntTmpl  = lkRayCntBuffer.d_pointer();
                br.lkPosMapTmpl  = lkPosMapBuffer.d_pointer();
                br.lkPosMapBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(uint));
                br.lkRayCntBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(uint));
                br.lkRayPtrsBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(CUdeviceptr));
                br.lkTfCmpBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(float) * 12);

                br.lkStartsBuffer.alloc(roundUpLog2(CFG.BATCHSIZE * (CFG.DOF + 1)) * sizeof(uint));
                br.maskPrefixSumBuffer.alloc(roundUpLog2(CFG.BATCHSIZE * (CFG.DOF + 1)) * sizeof(uint));

                br.tempBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(uint));
            }
        }
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::prepareRobot() {
        robotPtr->uploadOBBs();
        auto& links = robotPtr->getOrderedLinks();
        for (int i = 0; i < CFG.DOF + 1; ++i) {
            linkRayCnt[i] = links[i]->mesh.uniqueEdgeIndices.size();
            totalPerRoRay += linkRayCnt[i];
            rayLinkMap.resize(totalPerRoRay, i);
            for (int2 idx : links[i]->mesh.uniqueEdgeIndices) {
                rayVertices.push_back(links[i]->mesh.vertices[idx.x]);
                rayVertices.push_back(links[i]->mesh.vertices[idx.y]);
            }
#if defined(_DEBUG) || defined(DEBUG)
            // std::cout << "Link " << i << " has " << linkRayCnt[i] << " rays" << std::endl;
#endif
        }
        rayVerticesBuffer.alloc_and_upload(rayVertices);
        rayVerticesPtr = rayVerticesBuffer.d_pointer();
        rayLinkMapBuffer.alloc_and_upload(rayLinkMap);
        rayLinkMapPtr = rayLinkMapBuffer.d_pointer();

        std::array<uint, CFG.DOF + 1> linkRaySize;
        std::transform(
            linkRayCnt.begin(), linkRayCnt.end(), linkRaySize.begin(), [](uint x) { return x * sizeof(float3) * 2; });
        std::exclusive_scan(linkRaySize.begin(), linkRaySize.end(), linkInRay.begin(), rayVerticesPtr);

        if constexpr (CFG.TYPE == LinkType::RAY_ONLINE) { // RAY_ONLINE
            std::vector<uint> lkPosMapVec;
            std::vector<uint> lkRayCntVec;
            std::vector<CUdeviceptr> lkRayPtrsVec;
            // for (uint i = 0; i < CFG.DOF + 1; i++) {
            //     lkPosMapVec.insert(lkPosMapVec.end(), lkPosMap.begin(), lkPosMap.end());
            //     lkRayCntVec.resize((i + 1) * CFG.BATCHSIZE, linkRayCnt[i]);
            //     lkRayPtrsVec.resize((i + 1) * CFG.BATCHSIZE, linkInRay[i]);
            // }
            for (uint i = 0; i < CFG.BATCHSIZE; i++) {
                lkPosMapVec.resize((CFG.DOF + 1) * (i + 1), i);
                lkRayCntVec.insert(lkRayCntVec.end(), linkRayCnt.begin(), linkRayCnt.end());
                lkRayPtrsVec.insert(lkRayPtrsVec.end(), linkInRay.begin(), linkInRay.end());
            }
            lkPosMapBuffer.alloc_and_upload(lkPosMapVec);
            lkRayPtrs.alloc_and_upload(lkRayPtrsVec);
            lkRayCntBuffer.alloc_and_upload(lkRayCntVec);
        }


        for (int i = 0; i < CFG.DOF + 1; ++i) {
            rayInfo.emplace_back(i, linkRayCnt[i], linkInRay[i]);
        }
        // std::cout << "totalPerRoRay: " << totalPerRoRay << std::endl;
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::update(const CUdeviceptr poses, const size_t idx) {
        batchRobots[idx].update(poses, CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::update(const CUdeviceptr poses, const size_t idx, const size_t nPoses) {
        batchRobots[idx].update(poses, nPoses);
    }

    template <batchRobotConfig CFG>
    inline void batchRayRobot<CFG>::updateOBBs(const size_t idx) {
        batchRobots[idx].updateOBB(CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    inline void batchRayRobot<CFG>::updateOBBs(const size_t idx, const size_t nPoses) {
        batchRobots[idx].updateOBB(nPoses);
    }

    template <batchRobotConfig CFG>
    inline void batchRayRobot<CFG>::updateOBBs2(const size_t idx) {
        batchRobots[idx].updateOBB2(CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    inline void batchRayRobot<CFG>::updateOBBs2(const size_t idx, const size_t nPoses) {
        batchRobots[idx].updateOBB2(nPoses);
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::fkine(const CUdeviceptr poses, const size_t idx) {
        batchRobots[idx].fkine(poses, CFG.BATCHSIZE);
        updateOBBs(idx, CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::fkine(const CUdeviceptr poses, const size_t idx, const size_t nPoses) {
        batchRobots[idx].fkine(poses, nPoses);
        updateOBBs(idx, nPoses);
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::fkine2(const CUdeviceptr poses, const size_t idx) {
        batchRobots[idx].fkine2(poses, CFG.BATCHSIZE);
        updateOBBs2(idx, CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::fkine2(const CUdeviceptr poses, const size_t idx, const size_t nPoses) {
        batchRobots[idx].fkine2(poses, nPoses);
        updateOBBs2(idx, nPoses);
    }

    template <batchRobotConfig CFG>
    bool batchRayRobot<CFG>::updateWithMask(const CUdeviceptr mask, const size_t idx) {
        return batchRobots[idx].updateWithMask(mask, CFG.BATCHSIZE);
    }

    template <batchRobotConfig CFG>
    bool batchRayRobot<CFG>::updateWithMask(const CUdeviceptr mask, const size_t idx, const size_t nPoses) {
        return batchRobots[idx].updateWithMask(mask, nPoses);
    }

    template <batchRobotConfig CFG>
    const meshRayInfo batchRayRobot<CFG>::getRayInfo(const size_t idx) const {
        return batchRobots[idx].getRayInfo();
    }

    template <batchRobotConfig CFG>
    void batchRayRobot<CFG>::resetGraph() {
        for (int i = 0; i < CFG.NSTREAM; ++i) {
            batchRobots[i].graphCreated = false;
        }
    }
} // namespace RTCD

namespace RTCD { // Implementation of streamRayRobot

    template <batchRobotConfig CFG>
    void streamRayRobot<CFG>::update(const CUdeviceptr poses, const size_t nPoses) {
        batchFkineAdj(poses, lkTfBuffer.d_pointer(), CFG.DOF, nPoses, stream);
        // if constexpr (CFG.TYPE == LinkType::RAY_STATIC) {
        return;
        // }
        // batchRayVertices(reinterpret_cast<float*>(lkTfBuffer.d_pointer()),
        //     reinterpret_cast<float3*>(templateRayPtr), reinterpret_cast<float3*>(verticesPtr),
        //     reinterpret_cast<int*>(rayLinkMapPtr), totalPerRoRay, CFG.DOF, nPoses, stream);
    }

    template <batchRobotConfig CFG>
    inline void streamRayRobot<CFG>::fkine(const CUdeviceptr poses, const size_t nPoses) {
        batchFkineAdj(poses, lkTfBuffer.d_pointer(), CFG.DOF, nPoses, stream);
    }

    template <batchRobotConfig CFG>
    inline void streamRayRobot<CFG>::updateOBB(const size_t nPoses) {
        assert(nPoses <= CFG.BATCHSIZE && "The number of poses should be smaller than the batch size");
        batchOBBTransform(lkTfBuffer.d_pointer(), linkobbTemplate, linkOBBsBuffer.d_pointer(), CFG.DOF, nPoses, stream);
    }

    template <batchRobotConfig CFG>
    inline void streamRayRobot<CFG>::fkine2(const CUdeviceptr poses, const size_t nPoses) {
        batchFkine(poses, lkTfBuffer.d_pointer(), CFG.DOF, nPoses, stream);
    }

    template <batchRobotConfig CFG>
    inline void streamRayRobot<CFG>::updateOBB2(const size_t nPoses) {
        assert(nPoses <= CFG.BATCHSIZE && "The number of poses should be smaller than the batch size");
        batchOBBTransform2(
            lkTfBuffer.d_pointer(), linkobbTemplate, linkOBBsBuffer.d_pointer(), CFG.DOF, nPoses, stream);
    }

    template <batchRobotConfig CFG>
    inline const meshRayInfo streamRayRobot<CFG>::getRayInfo() const {
        rayRobotInfo info{lkCnt, lkStartsBuffer.d_pointer(), lkPosMapBuffer.d_pointer(), lkRayPtrsBuffer.d_pointer(),
            lkTfCmpBuffer.d_pointer()};
        return meshRayInfo(0, totalRayCnt, 0, info);
    }

    template <batchRobotConfig CFG>
    inline bool streamRayRobot<CFG>::updateWithMask(const CUdeviceptr mask, const size_t nPoses) {

        transformIsValid<uint, true>(mask, mask, nPoses * (CFG.DOF + 1), stream); // Turn mask into 0/1 array
        CUDA_CHECK_LAST("Transform");

        exclusiveScan(mask, maskPrefixSumBuffer.d_pointer(), tempBuffer, nPoses * (CFG.DOF + 1),


            cubTempSize, stream);
        CUDA_CHECK_LAST("Exclusive Scan");

        // compact tfs
        compactCpy<true>(lkTfCmpBuffer.d_pointer(), lkTfBuffer.d_pointer(), mask, maskPrefixSumBuffer.d_pointer(),
            sizeof(float) * 12, nPoses * (CFG.DOF + 1), stream);
        CUDA_CHECK_LAST("Compact Tf");

        // compact compacted linkId to poseId map
        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(lkPosMapBuffer.d_pointer()), 0, nPoses * (CFG.DOF + 1) * sizeof(uint), stream));
        compactCpy<true>(lkPosMapBuffer.d_pointer(), lkPosMapTmpl, mask, maskPrefixSumBuffer.d_pointer(), sizeof(uint),
            nPoses * (CFG.DOF + 1), stream);
        CUDA_CHECK_LAST("Compact LinkPose Map");

        // compact link ray counts
        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(lkRayCntBuffer.d_pointer()), 0, nPoses * (CFG.DOF + 1) * sizeof(uint), stream));
        compactCpy<true>(lkRayCntBuffer.d_pointer(), lkRayCntTmpl, mask, maskPrefixSumBuffer.d_pointer(), sizeof(uint),
            nPoses * (CFG.DOF + 1), stream);
        CUDA_CHECK_LAST("Compact Ray Count");

        // compact link ray ptrs
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(lkRayPtrsBuffer.d_pointer()), 0,
            nPoses * (CFG.DOF + 1) * sizeof(CUdeviceptr), stream));
        compactCpy<true>(lkRayPtrsBuffer.d_pointer(), lkRayPtrsTmpl, mask, maskPrefixSumBuffer.d_pointer(),
            sizeof(CUdeviceptr), nPoses * (CFG.DOF + 1), stream);
        CUDA_CHECK_LAST("Compact Ray Ptrs");

        exclusiveScan(lkRayCntBuffer.d_pointer(), lkStartsBuffer.d_pointer(), tempBuffer, nPoses * (CFG.DOF + 1),
            cubTempSize, stream);

        CUDA_CHECK_LAST("Exclusive Scan");

        gatherCntData(mask, maskPrefixSumBuffer.d_pointer(), lkRayCntTmpl, lkStartsBuffer.d_pointer(),
            cntDataBuffer.d_pointer(), nPoses * (CFG.DOF + 1), stream);

        cudaMemcpyAsync(&cntData, reinterpret_cast<void*>(cntDataBuffer.d_pointer()), sizeof(ORayData),
            cudaMemcpyDeviceToHost, stream);

        CUDA_CHECK_LAST("Memcpy");

#if defined(_DEBUG) || defined(DEBUG)
        std::vector<uint> hMasks(nPoses * (CFG.DOF + 1));
        cudaMemcpyAsync(
            hMasks.data(), (void*) mask, nPoses * (CFG.DOF + 1) * sizeof(uint), cudaMemcpyDeviceToHost, stream);

        std::vector<uint> hMaskPrefixSum(nPoses * (CFG.DOF + 1));
        maskPrefixSumBuffer.downloadAsync(hMaskPrefixSum, stream);

        std::vector<float> hTfCmp(nPoses * (CFG.DOF + 1) * 12);
        lkTfCmpBuffer.downloadAsync(hTfCmp, stream);

        std::vector<uint> hPosMap(nPoses * (CFG.DOF + 1));
        lkPosMapBuffer.downloadAsync(hPosMap, stream);

        std::vector<uint> hRayCntTemp(nPoses * (CFG.DOF + 1));
        cudaMemcpyAsync(hRayCntTemp.data(), (void*) lkRayCntTmpl, nPoses * (CFG.DOF + 1) * sizeof(uint),
            cudaMemcpyDeviceToHost, stream);
        std::vector<uint> hRayCnt(hRayCntTemp.size());
        lkRayCntBuffer.downloadAsync(hRayCnt, stream);

        std::vector<CUdeviceptr> hTempRayPtr(CFG.NSTREAM * (CFG.DOF + 1));
        cudaMemcpyAsync(hTempRayPtr.data(), (void*) lkRayPtrsTmpl, CFG.NSTREAM * (CFG.DOF + 1) * sizeof(CUdeviceptr),
            cudaMemcpyDeviceToHost, stream);

        std::vector<CUdeviceptr> hRayPtrs(hTempRayPtr.size());
        lkRayPtrsBuffer.downloadAsync(hRayPtrs, stream);

        std::vector<uint> hStarts;
        hStarts.resize(nPoses * (CFG.DOF + 1));
        lkStartsBuffer.downloadAsync(hStarts, stream);
#endif
        cudaStreamSynchronize(stream);
        CUDA_CHECK_LAST("Sync");

        lkCnt       = cntData.scanLinks + cntData.lastMask;
        totalRayCnt = cntData.scanRays + cntData.lastRay * (lkCnt == (nPoses * (CFG.DOF + 1)));
#if defined(_DEBUG) || defined(DEBUG)
        // find the max
        auto max = std::max_element(hMasks.begin(), hMasks.end());
        // find the sum
        auto sum = std::accumulate(hMasks.begin(), hMasks.end(), 0);
        auto cnt = std::count_if(hMasks.begin(), hMasks.end(), [](uint x) { return x > 0; });

        // find the first 0 in hRayCnt
        auto firstZero = std::find(hRayCnt.begin(), hRayCnt.end(), 0);
        std::cout << "First zero: " << std::distance(hRayCnt.begin(), firstZero) << std::endl;
        // decide if all the elements after the first zero are 0
        auto allZero = std::all_of(firstZero, hRayCnt.end(), [](uint x) { return x == 0; });
        std::cout << "All zero: " << allZero << std::endl;


        // compute exclusive scan on CPU as well
        std::vector<uint> hRayCntScan(hRayCnt.size());
        std::exclusive_scan(hRayCnt.begin(), hRayCnt.end(), hRayCntScan.begin(), 0);
        // compare the results
        auto isEqual = std::equal(hRayCntScan.begin(), hRayCntScan.begin() + lkCnt, hStarts.begin());
        std::cout << "Scan Equal: " << isEqual << std::endl;

#endif
        return totalRayCnt > 0;
    }
} // namespace RTCD
