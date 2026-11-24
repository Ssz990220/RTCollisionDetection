// rtcd_core parity test (new_gpu) AGAINST THE OLD STACK: links rtcd_legacy_kernels; delete with the old stack.
//
// internal::obbOverlapCounts must be bit-identical to the old RTCD::BBCDRobotMask
// (RTCD/CollisionDetector/BBCollisionDetection.cu): same counts for every robot OBB, on random OBBs including
// degenerate (point) boxes and near-touching pairs right at the separation threshold.
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/internal/algorithms.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

// Declaration copied from RTCD/CollisionDetector/BBCollisionDetection.cuh (that header drags in the old include
// tree; the signature is all this test needs). Implemented in rtcd_legacy_kernels.
namespace RTCD {
    void BBCDRobotMask(const CUdeviceptr robotOBBs, const CUdeviceptr sceneOBBs, const size_t nROBBs,
        const size_t nSOBBs, CUdeviceptr robotMask, const cudaStream_t stream);
} // namespace RTCD

using namespace rtcd;

namespace {
    Obb randomObb(std::mt19937& rng, float centerRange, float maxHalf) {
        std::uniform_real_distribution<float> c(-centerRange, centerRange);
        std::uniform_real_distribution<float> h(0.0f, maxHalf);
        std::normal_distribution<float> g(0.0f, 1.0f);
        Obb o{};
        for (int i = 0; i < 3; ++i) {
            o.center[i]   = c(rng);
            o.halfSize[i] = h(rng);
        }
        // random rotation from a normalized quaternion (float, like the old OBB fit outputs)
        float q[4] = {g(rng), g(rng), g(rng), g(rng)};
        const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (float& v : q) {
            v /= n;
        }
        const float w = q[0], x = q[1], y = q[2], z = q[3];
        const float R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
        for (int a = 0; a < 3; ++a) {
            for (int r = 0; r < 3; ++r) {
                o.axes[a][r] = R[r][a]; // axis a = column a
            }
        }
        return o;
    }

    void compareWithOld(const std::vector<Obb>& a, const std::vector<Obb>& b, StreamHandle s, std::size_t* nonZero) {
        DeviceBuffer dA = DeviceBuffer::fromHost(std::span<const Obb>(a));
        DeviceBuffer dB = DeviceBuffer::fromHost(std::span<const Obb>(b));
        DeviceBuffer newCounts(a.size() * 4), oldCounts(a.size() * 4);
        newCounts.memsetAsync(0xAB, newCounts.size(), s);
        oldCounts.memsetAsync(0xCD, oldCounts.size(), s);
        internal::obbOverlapCounts(dA.span<const Obb>(), dB.span<const Obb>(), newCounts.span<std::uint32_t>(), s);
        RTCD::BBCDRobotMask(dA.ptr(), dB.ptr(), a.size(), b.size(), oldCounts.ptr(), s);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        std::vector<std::uint32_t> n(a.size()), o(a.size());
        newCounts.download(n.data(), n.size() * 4);
        oldCounts.download(o.data(), o.size() * 4);
        for (std::size_t i = 0; i < a.size(); ++i) {
            ASSERT_EQ(n[i], o[i]) << "robot OBB " << i << " (of " << a.size() << ", scene " << b.size() << ")";
            *nonZero += n[i] != 0 ? 1 : 0;
        }
    }
} // namespace

TEST(CoreLegacyParity, ObbOverlapCountsEqualsOldBBCDRobotMaskOnRandomObbs) {
    auto ctx   = Context::create();
    auto lease = ctx->acquireStreams(1);
    std::mt19937 rng(20261123);

    // Shapes of the real uses: robot links (batch * 8 or 9 links, some collapsed root links) vs a handful of scene
    // obstacles; plus larger scene sets to exercise every lane of the 32 x 8 block.
    for (std::size_t nB : {1u, 3u, 8u, 9u, 64u, 257u}) {
        for (std::size_t nA : {1u, 31u, 33u, 1024u * 9u, 4096u * 8u + 5u}) {
            std::vector<Obb> a(nA), b(nB);
            for (auto& o : a) {
                o = randomObb(rng, 1.0f, 0.3f);
            }
            for (std::size_t i = 0; i < nA; i += 9) {
                a[i].halfSize[0] = a[i].halfSize[1] = a[i].halfSize[2] = 0.0f; // collapsed root link OBB
            }
            for (auto& o : b) {
                o = randomObb(rng, 1.0f, 0.5f);
            }
            std::size_t nonZero = 0;
            compareWithOld(a, b, lease[0], &nonZero);
            if (HasFatalFailure()) {
                return;
            }
            if (nA >= 1024) {
                EXPECT_GT(nonZero, 0u);
                EXPECT_LT(nonZero, nA);
            }
        }
    }
}

TEST(CoreLegacyParity, ObbOverlapCountsEqualsOldAtTheSeparationThreshold) {
    // Pairs separated along one face axis by (ra + rb) * (1 + eps) for eps around 0: the decision "d > ra + rb"
    // is decided by the last bits of the float arithmetic, where any change of operation order would show.
    auto ctx   = Context::create();
    auto lease = ctx->acquireStreams(1);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> eps(-4e-7f, 4e-7f);
    std::vector<Obb> a, b;
    const std::size_t nPairs = 20000;
    for (std::size_t k = 0; k < nPairs; ++k) {
        Obb p = randomObb(rng, 1.0f, 0.3f);
        Obb q = p; // same orientation: separation along p's axis 0 is exact up to rounding
        const float dist = (p.halfSize[0] + q.halfSize[0]) * (1.0f + eps(rng));
        for (int r = 0; r < 3; ++r) {
            q.center[r] = p.center[r] + dist * p.axes[0][r];
        }
        a.push_back(p);
        if (k < 256) {
            b.push_back(q); // b: 256 near-touching partners (every a also meets the others at random)
        }
    }
    // also: each a against its own partner only
    std::size_t nonZero = 0;
    compareWithOld(a, b, lease[0], &nonZero);
    for (std::size_t k = 0; k < 300 &&!HasFatalFailure(); ++k) {
        Obb p = a[k];
        Obb q = p;
        const float dist = (p.halfSize[0] + q.halfSize[0]) * (1.0f + eps(rng));
        for (int r = 0; r < 3; ++r) {
            q.center[r] = p.center[r] + dist * p.axes[0][r];
        }
        std::size_t nz = 0;
        std::vector<Obb> one = {p}, other = {q};
        compareWithOld(one, other, lease[0], &nz);
        nonZero += nz;
    }
    EXPECT_GT(nonZero, 0u);
}
