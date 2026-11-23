#pragma once

#include <Robot/baseBatchRobot.h>
#include <Robot/batchRobotConfig.h>
#include <Robot/robot.h>
#include <Utils/cuda/cudaAlgorithms.cuh>
#include <memory>

namespace RTCD {
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

        // If RAY_ONLINE: the launch covers every ray of the nPoses robots of the batch (host-known count; the launch
        // width is capped and the raygen grid-strides), and the raygen skips rays of links whose OBB is collision free. Nothing is compacted, so no count has to be read
        // back to the host and the per-batch sequence has no host<->device round trip.
        CUdeviceptr linkMaskPtr = 0; // OBB link mask of the current batch (owned by the detector)
        uint totalRayCnt        = 0; // ray ids of the batch = nPoses * totalPerRoRay
        uint launchWidthCap     = 0; // see batchRayRobot::setLaunchWidthCap
        bool skipHitPoses       = true; // see batchRayRobot::setSkipHitPoses

        bool graphCreated = false;
        cudaGraph_t graph;
        cudaGraphExec_t instance;

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

        // RAY_ONLINE launch tuning (applies to all streams; read by the detector through getRayInfo(idx)).
        // The ray launch has one ray id per (pose, ray) of the batch, nPoses * getRayCount() ids. Most belong to
        // links whose OBB is collision free and are skipped in the raygen. The launch width is capped and each launch
        // index grid-strides over the ids, so skipped rays cost a loop iteration instead of a launched thread.
        // A larger cap gives more parallelism per traced ray, a smaller one fewer idle threads. 0 = no cap (one
        // launch index per ray id). Default 262144 was the best or near-best BM_RAY setting at every N from 1 to 4096
        // (RTX 5080, Panda shelf scene, batch 1024).
        void setLaunchWidthCap(const uint cap) {
            launchWidthCap = cap;
            for (auto& br : batchRobots) {
                br.launchWidthCap = cap;
            }
        }
        uint getLaunchWidthCap() const { return launchWidthCap; }

        // If true, the raygen skips rays of poses whose hit count is already > 0. The per-pose collision flag
        // (hit > 0) is unchanged. Hit counts become lower bounds, so turn this off if exact counts are needed.
        // Costs one extra global load per surviving ray.
        void setSkipHitPoses(const bool skip) {
            skipHitPoses = skip;
            for (auto& br : batchRobots) {
                br.skipHitPoses = skip;
            }
        }
        bool getSkipHitPoses() const { return skipHitPoses; }

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

        std::array<CUdeviceptr, CFG.DOF + 1> linkInRay; // V2.5
        std::vector<meshRayInfo> rayInfo;
        uint launchWidthCap = 262144;
        bool skipHitPoses   = true;
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
            br.launchWidthCap  = launchWidthCap;
            br.skipHitPoses    = skipHitPoses;
            std::transform(linkRayCnt.begin(), linkRayCnt.end(), br.linkRayCnt.begin(),
                [](size_t x) { return static_cast<uint>(x); });

            br.lkTfBuffer.alloc((CFG.DOF + 1) * CFG.BATCHSIZE * sizeof(float) * 12);
            br.linkOBBsBuffer.alloc(CFG.BATCHSIZE * (CFG.DOF + 1) * sizeof(OBB));
            br.rayInfo.resize(CFG.BATCHSIZE * (CFG.DOF + 1));
            rayOriginPtr[i] = rayVerticesPtr; // backward compatability
        }
        if constexpr (CFG.TYPE == LinkType::RAY_ONLINE) {
            // Launch index is a uint (OptiX launch width limit is 2^30).
            assert(static_cast<unsigned long long>(CFG.BATCHSIZE) * totalPerRoRay < (1ull << 30)
                   && "RAY_ONLINE launch width (BATCHSIZE * rays per robot) exceeds the OptiX limit");
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
        const uint launchWidth =
            (launchWidthCap != 0 && totalRayCnt > launchWidthCap) ? launchWidthCap : totalRayCnt;
        rayRobotInfo info{static_cast<uint>(totalPerRoRay), static_cast<uint>(CFG.DOF + 1),
            static_cast<uint>(skipHitPoses), launchWidth, linkMaskPtr, rayLinkMapPtr, templateRayPtr,
            lkTfBuffer.d_pointer()};
        return meshRayInfo(0, totalRayCnt, 0, info);
    }

    // Records the batch's OBB link mask (> 0: the link's OBB overlaps an obstacle OBB) for the ray launch. Issues no
    // GPU work and no host<->device synchronization: the raygen reads the mask directly and skips collision-free
    // links, so the launch width is the host-known nPoses * totalPerRoRay. Returns whether there is anything to
    // launch; whether any link actually survived the OBB test is only known on the device.
    template <batchRobotConfig CFG>
    inline bool streamRayRobot<CFG>::updateWithMask(const CUdeviceptr mask, const size_t nPoses) {
        assert(nPoses <= CFG.BATCHSIZE && "The number of poses should be smaller than the batch size");
        linkMaskPtr = mask;
        totalRayCnt = static_cast<uint>(nPoses * totalPerRoRay);
        return totalRayCnt > 0;
    }
} // namespace RTCD
