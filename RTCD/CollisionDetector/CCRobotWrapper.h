#pragma once
#include <Robot/baseBatchRobot.h>
#include <Robot/batchIASRobot.h>
#include <Robot/batchRayRobot.h>
#include <Robot/batchRobotConfig.h>
#include <memory>
#include <random>
// #include <scene.h>

using namespace RTCD;

template <size_t NSTREAM, LinkType TYPE>
class CCWrappedRobot {
private:
    CUDABuffer poses;
    std::shared_ptr<baseBatchRobot<NSTREAM, TYPE>> batchRobot;
    std::array<CUDABuffer, NSTREAM> posesBuffers;

public:
    bool useOBB = true;

public:
    CCWrappedRobot(std::shared_ptr<baseBatchRobot<NSTREAM, TYPE>> batchRobot) : batchRobot(batchRobot) {
        for (auto& b : posesBuffers) {
            if (b.sizeInBytes
                < batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float)) {
                b.resize(batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float));
            }
        }
        useOBB = batchRobot->useOBB;
    };

    void allocPoseBuffer(const size_t size) {
        poses.resize(size * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float));
    }

    // prepareUpdateData
    // @brief upload data to GPU memory
    void prepareUpdateData(const float* p, const size_t nPoses) {
        if (poses.sizeInBytes < nPoses * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float)) {
            poses.free();
            poses.alloc_and_upload(p, nPoses * batchRobot->getTrajSize() * batchRobot->getDOF());
        } else {
            poses.upload(p, nPoses * batchRobot->getTrajSize() * batchRobot->getDOF());
        }
    };

    // movePosesToStream
    // @brief Move poses to stream data pool, can be useful when using cudaGraph
    void movePosesToStream(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(posesBuffers[idx].d_pointer()),
            reinterpret_cast<void*>(poses.d_pointer()
                                    + frameIdx * batchRobot->getBatchSize() * batchRobot->getTrajSize()
                                          * batchRobot->getDOF() * sizeof(float)),
            batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float),
            cudaMemcpyDeviceToDevice, batchRobot->streams[idx]));
    }

    // movePosesToStream(size_t frameIdx, size_t nPoses)
    // @brief Move poses (less than BatchSize specified) to stream data pool, can be useful when using cudaGraph
    void movePosesToStream(const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(posesBuffers[idx].d_pointer()),
            reinterpret_cast<void*>(
                poses.d_pointer()
                + frameIdx * nPoses * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float)),
            nPoses * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float), cudaMemcpyDeviceToDevice,
            batchRobot->streams[idx]));
    }

    void prepareRandomData() {
        // Create a random number generator
        std::random_device rd; // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<float> dis(0.0, 1.0); // Uniform distribution between 0 and 1

        // fill the pose buffer with some random data
        std::vector<float> hPoses;
        hPoses.resize(batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * NSTREAM);
        std::generate(hPoses.begin(), hPoses.end(), [&gen, &dis]() { return dis(gen); });

        if (poses.sizeInBytes
            < batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float)) {
            poses.free();
            poses.alloc_and_upload(
                hPoses.data(), batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF());
        } else {
            poses.upload(hPoses.data(), batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF());
        }
    }

    void update(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->update(posesBuffers[idx].d_pointer(), idx);
    }

    void update(const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->update(posesBuffers[idx].d_pointer(), idx, nPoses);
    }


    void fkine(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->fkine(posesBuffers[idx].d_pointer(), idx);
    }

    void fkine(const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->fkine(posesBuffers[idx].d_pointer(), idx, nPoses);
    }

    bool updateWithMask(const CUdeviceptr mask, const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        return batchRobot->updateWithMask(mask, idx);
    }

    bool updateWithMask(const CUdeviceptr mask, const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        return batchRobot->updateWithMask(mask, idx, nPoses);
    }

    inline unsigned int getBuildFlags() const { return batchRobot->getBuildFlags(); }

    inline size_t getSBTSize() const { return batchRobot->getSBTSize(); }

    inline size_t getBatchSize() const { return batchRobot->getBatchSize(); }

    inline size_t getDOF() const { return batchRobot->getDOF(); }

    inline CUdeviceptr getOBBs(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        return batchRobot->getOBBs(idx);
    }

    inline size_t getNOBBs() const { return batchRobot->getNOBBs(); }
    inline size_t getNOBBs(size_t nPoses) const { return batchRobot->getNOBBs(nPoses); }


    inline CUdeviceptr getMapIndex() const { return batchRobot->getMapIndex(); }

    inline CUdeviceptr getMapIndex(const int idx) const { return batchRobot->getMapIndex(idx); }

    inline OptixTraversableHandle getHandle(const size_t idx) const { return batchRobot->getHandle(idx); }

    inline void resetGraph() { batchRobot->resetGraph(); }

    inline CUdeviceptr getPoseBufferPtr() { return poses.d_pointer(); }
};

template <size_t NSTREAM, LinkType TYPE>
class CCRayRobot {
private:
    CUDABuffer poses;
    std::shared_ptr<baseRayRobot<NSTREAM, TYPE>> batchRobot;
    std::array<CUDABuffer, NSTREAM> posesBuffers;

public:
    bool useOBB = true;

public: // Public Function implementation
    CCRayRobot(std::shared_ptr<baseRayRobot<NSTREAM, TYPE>> batchRobot) : batchRobot(batchRobot) {
        for (auto& b : posesBuffers) {
            if (b.sizeInBytes
                < batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float)) {
                b.resize(batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float));
            }
        }
        useOBB = batchRobot->useOBB;
    };

    inline const std::array<CUdeviceptr, NSTREAM>& getRays() { return batchRobot->getRays(); }

    inline size_t getRayCount() const { return batchRobot->getRayCount(); }

    inline size_t getBatchSize() const { return batchRobot->getBatchSize(); }

    inline size_t getDOF() const { return batchRobot->getDOF(); }

    inline float* getLinkTransform(int i) const { return batchRobot->getLinkTransform(i); }

    inline int* getRayMap() const { return batchRobot->getRayMap(); }

    inline const size_t getNOBBs() const { return batchRobot->getNOBBs(); }
    inline const size_t getNOBBs(size_t nPoses) const { return batchRobot->getNOBBs(nPoses); }

    inline void resetGraph() { batchRobot->resetGraph(); }

    void allocPoseBuffer(const size_t size) {
        poses.resize(size * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float));
    }

    void prepareUpdateData(const float* p, const size_t nPoses) {
        if (poses.sizeInBytes < nPoses * batchRobot->getDOF() * sizeof(float)) {
            poses.free();
            poses.alloc_and_upload(p, nPoses * batchRobot->getDOF());
        } else {
            poses.upload(p, nPoses * batchRobot->getDOF());
        }
    };

    void movePosesToStream(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(posesBuffers[idx].d_pointer()),
            reinterpret_cast<void*>(poses.d_pointer()
                                    + frameIdx * batchRobot->getBatchSize() * batchRobot->getTrajSize()
                                          * batchRobot->getDOF() * sizeof(float)),
            batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float),
            cudaMemcpyDeviceToDevice, batchRobot->streams[idx]));
    }

    void movePosesToStream(const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(posesBuffers[idx].d_pointer()),
            reinterpret_cast<void*>(
                poses.d_pointer()
                + frameIdx * nPoses * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float)),
            nPoses * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float), cudaMemcpyDeviceToDevice,
            batchRobot->streams[idx]));
    }

    void prepareRandomData() {
        // Create a random number generator
        std::random_device rd; // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<float> dis(0.0, 1.0); // Uniform distribution between 0 and 1

        // fill the pose buffer with some random data
        std::vector<float> hPoses;
        hPoses.resize(batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * NSTREAM);
        std::generate(hPoses.begin(), hPoses.end(), [&gen, &dis]() { return dis(gen); });

        if (poses.sizeInBytes
            < NSTREAM * batchRobot->getBatchSize() * batchRobot->getTrajSize() * batchRobot->getDOF() * sizeof(float)) {
            poses.free();
            poses.alloc_and_upload(hPoses);
        } else {
            poses.upload(hPoses.data(), hPoses.size());
        }
    }

    void update(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->update(posesBuffers[idx].d_pointer(), idx);
    }

    void update(const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->update(posesBuffers[idx].d_pointer(), idx, nPoses);
    }


    void fkine(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->fkine(posesBuffers[idx].d_pointer(), idx);
    }

    void fkine(const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->fkine(posesBuffers[idx].d_pointer(), idx, nPoses);
    }

    void fkine2(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->fkine2(posesBuffers[idx].d_pointer(), idx);
    }

    void fkine2(const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        batchRobot->fkine2(posesBuffers[idx].d_pointer(), idx, nPoses);
    }

    bool updateWithMask(const CUdeviceptr mask, const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        return batchRobot->updateWithMask(mask, idx);
    }

    bool updateWithMask(const CUdeviceptr mask, const size_t frameIdx, const size_t nPoses) {
        const size_t idx = frameIdx % NSTREAM;
        return batchRobot->updateWithMask(mask, idx, nPoses);
    }

    const std::vector<meshRayInfo>& getRayInfo() const { return batchRobot->getRayInfo(); }
    const meshRayInfo getRayInfo(const size_t idx) const { return batchRobot->getRayInfo(idx); }

    inline CUdeviceptr getOBBs(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        return batchRobot->getOBBs(idx);
    }

    inline const CUdeviceptr getLinkTfs(const size_t frameIdx) {
        const size_t idx = frameIdx % NSTREAM;
        return batchRobot->getLinkTfs(idx);
    }

    inline CUdeviceptr getPoseBufferPtr() { return poses.d_pointer(); }
};
