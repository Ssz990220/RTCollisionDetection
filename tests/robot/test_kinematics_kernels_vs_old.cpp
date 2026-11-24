// tests/robot: the new FK kernels (include/rtcd/internal/kinematics.h) and the derived kinematics table vs the old
// ones (new_gpu).
//
// OLD-STACK PARITY TEST AND REFERENCE GENERATOR (deleted with the old stack; the stored references it writes, run
// with RTCD_WRITE_REFERENCE=<dir>, replace it: tests/data/reference/panda_kinematics.txt and kinematics_fk.*,
// compared by test_kinematics_table.cpp and test_kinematics_kernels.cpp).
//
// The URDF-derived table equals pandas.h bit for bit. Links rtcd_legacy_kernels and calls the old batchFkineAdj / batchFkineAdjOBB /
// batchOBBTransform / batchFkineTraj (RTCD/Robot/robot.cu) with the hand-written pandas.h masks loaded into their
// __constant__ globals; the new kernels get the table derived from panda.urdf in a device buffer. Every output
// must be equal BYTE FOR BYTE.
//
// Inputs: the cached regression poses tests/data/motions/Panda4096.bin (4096 x 7 floats, the IAS / RAY regression
// input) and the regression's KNN trajectories rebuilt from those poses and tests/data/motions/Panda/1111111/
// 4096_knn4_idx.bin exactly as RTCD/Utils/Test/poseUtils.h genCSpaceKNNTraj does (16 points per trajectory,
// first 4096 trajectories = the LINEAR / QUAD regression input).
#include "robot/urdf_chain.h"

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/kinematics.h>

#include "fk_reference_cases.h"

#include <support/reference.h>

#include <Robot/models/pandas.h> // old: cosMask, sinMask, oneMask, baseT, Dim
#include <Robot/robot.cuh>       // old kernels

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

using namespace rtcd;

namespace {

#define CK(call) ASSERT_EQ((call), cudaSuccess) << #call

    std::vector<float> readFloats(const char* path, std::size_t n) {
        std::vector<float> v(n);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            ADD_FAILURE() << "missing input " << path;
            return {};
        }
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
        if (static_cast<std::size_t>(in.gcount()) != n * sizeof(float)) {
            ADD_FAILURE() << "short input " << path;
            return {};
        }
        return v;
    }

    constexpr std::size_t kPoses = 4096;
    constexpr std::size_t kKnn   = 4;
    constexpr int kTrajPts       = 16;

    const std::vector<float>& regressionPoses() {
        static const std::vector<float> p = readFloats("tests/data/motions/Panda4096.bin", kPoses * Dim);
        return p;
    }

    // genCSpaceKNNTraj<7, 4096, 4>(traj, 16, ...) with the cached poses / indices: first 4096 trajectories.
    std::vector<float> regressionTrajectories() {
        const auto& P = regressionPoses();
        std::vector<int> idx(kPoses * kKnn);
        std::ifstream in("tests/data/motions/Panda/1111111/4096_knn4_idx.bin", std::ios::binary);
        if (!in || P.empty()) {
            ADD_FAILURE() << "missing KNN index cache";
            return {};
        }
        in.read(reinterpret_cast<char*>(idx.data()), static_cast<std::streamsize>(idx.size() * sizeof(int)));
        std::vector<float> traj(kPoses * kKnn * kTrajPts * Dim);
        for (std::size_t i = 0; i < kPoses; ++i) {
            for (std::size_t j = 0; j < kKnn; ++j) {
                const float* start = &P[i * Dim];
                const float* end   = &P[static_cast<std::size_t>(idx[i * kKnn + j]) * Dim];
                float span[Dim];
                for (int d = 0; d < Dim; ++d) {
                    span[d] = (end[d] - start[d]) / (float) (kTrajPts - 1);
                }
                for (int k = 0; k < kTrajPts; ++k) {
                    for (int d = 0; d < Dim; ++d) {
                        traj[((i * kKnn + j) * kTrajPts + k) * Dim + d] = start[d] + span[d] * (float) k;
                    }
                }
            }
        }
        traj.resize(kPoses * kTrajPts * Dim);
        return traj;
    }

    struct DevMem {
        void* p = nullptr;
        std::size_t bytes = 0;
        explicit DevMem(std::size_t b) : bytes(b) {
            if (cudaMalloc(&p, b) != cudaSuccess) {
                p = nullptr;
            }
            cudaMemset(p, 0xA5, b); // poison: unwritten bytes must also agree
        }
        DevMem(DevMem&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
        DevMem(const DevMem&)            = delete;
        DevMem& operator=(const DevMem&) = delete;
        ~DevMem() { cudaFree(p); }
        DevicePtr ptr() const { return reinterpret_cast<DevicePtr>(p); }
        template <typename T>
        DeviceSpan<T> span() const {
            return {ptr(), bytes / sizeof(T)};
        }
        template <typename T>
        std::vector<T> download() const {
            std::vector<T> v(bytes / sizeof(T));
            cudaMemcpy(v.data(), p, bytes, cudaMemcpyDeviceToHost);
            return v;
        }
        template <typename T>
        static DevMem from(const std::vector<T>& v) {
            DevMem m(v.size() * sizeof(T));
            cudaMemcpy(m.p, v.data(), m.bytes, cudaMemcpyHostToDevice);
            return m;
        }
    };

    std::size_t firstByteDiff(const DevMem& a, const DevMem& b) {
        const auto va = a.download<unsigned char>();
        const auto vb = b.download<unsigned char>();
        for (std::size_t i = 0; i < va.size(); ++i) {
            if (va[i] != vb[i]) {
                return i;
            }
        }
        return std::size_t(-1);
    }

    std::size_t countFloatDiffs(const DevMem& a, const DevMem& b) {
        const auto va = a.download<std::uint32_t>();
        const auto vb = b.download<std::uint32_t>();
        std::size_t n = 0;
        for (std::size_t i = 0; i < va.size(); ++i) {
            n += va[i] != vb[i];
        }
        return n;
    }

    // Random but plausible OBB templates (unit-ish axes), 15 floats each.
    std::vector<Obb> randomObbs(std::size_t n, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        std::vector<Obb> v(n);
        for (auto& o : v) {
            for (float& f : o.center) f = 0.2f * u(rng);
            for (float& f : o.halfSize) f = 0.05f + 0.1f * (u(rng) + 1.f);
            for (auto& ax : o.axes)
                for (float& f : ax) f = u(rng);
        }
        return v;
    }

    // Guards the bit comparisons against "both kernels wrote nothing" (both buffers still poisoned): the new
    // output must match the host reference FK to float tolerance. poseOf(p, l) gives the 12 floats of pose p link l.
    template <typename PoseOf>
    void expectMatchesHostFk(const KinematicsTable& t, const float* q, std::size_t nPoses, PoseOf poseOf) {
        std::vector<float> ref((t.numLinks) * 12);
        std::size_t bad = 0;
        for (std::size_t p = 0; p < nPoses; ++p) {
            robot_detail::hostLinkPoses(t, q + p * t.dof, ref.data());
            for (std::size_t l = 0; l < t.numLinks; ++l) {
                const float* g = poseOf(p, l);
                for (int e = 0; e < 12; ++e) {
                    bad += !(std::fabs(g[e] - ref[l * 12 + e]) < 1e-4f);
                }
            }
        }
        EXPECT_EQ(bad, 0u) << "GPU link poses disagree with the host reference FK";
    }

    class KinematicsKernels : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
                GTEST_SKIP() << "no CUDA device";
            }
            // Old stack: hand-written masks into the __constant__ globals of robot.cu.
            setFkineMasks(cosMask.data(), sinMask.data(), oneMask.data(), cosMask.size());
            std::array<float, 12> base = baseT;
            setBaseAffine(base);
            cudaDeviceSynchronize();
        }

        void SetUp() override {
            table_  = robot_detail::kinematicsFromChain(robot_detail::loadChain("models/franka_description/panda.urdf"));
            device_ = internal::uploadKinematics(table_);
            ref_    = internal::KinematicsRef{&table_, device_.ptr()};
        }

        KinematicsTable table_{};
        DeviceBuffer device_;
        internal::KinematicsRef ref_;
    };

    constexpr std::size_t kLinks = Dim + 1;

} // namespace

TEST_F(KinematicsKernels, UploadedTableIsTheHostTable) {
    ASSERT_EQ(device_.size(), sizeof(KinematicsTable));
    KinematicsTable back{};
    device_.download(&back, sizeof(back));
    EXPECT_EQ(std::memcmp(&back, &table_, sizeof(back)), 0);
}

TEST_F(KinematicsKernels, LinkPosesBitIdenticalToBatchFkineAdj) {
    const auto& P = regressionPoses();
    ASSERT_FALSE(P.empty());
    const DevMem q = DevMem::from(P);
    for (std::uint32_t n : {1u, 7u, 31u, 32u, 33u, 128u, 1000u, 4096u}) {
        DevMem oldOut(n * kLinks * 12 * sizeof(float)), newOut(n * kLinks * 12 * sizeof(float));
        batchFkineAdj(q.ptr(), oldOut.ptr(), Dim, n, 0);
        internal::linkPoses(ref_, q.span<const float>(), n, newOut.span<float>(), nullptr);
        CK(cudaDeviceSynchronize());
        EXPECT_EQ(countFloatDiffs(oldOut, newOut), 0u) << "nPoses " << n << ", first byte diff " << firstByteDiff(oldOut, newOut);
        const auto v = newOut.download<float>();
        expectMatchesHostFk(table_, P.data(), n, [&](std::size_t p, std::size_t l) { return &v[(p * kLinks + l) * 12]; });
    }
}

TEST_F(KinematicsKernels, LinkPosesAndObbsBitIdenticalToBatchFkineAdjOBB) {
    const auto& P = regressionPoses();
    ASSERT_FALSE(P.empty());
    const DevMem q    = DevMem::from(P);
    const DevMem tmpl = DevMem::from(randomObbs(kLinks, 7));
    for (std::uint32_t n : {1u, 31u, 33u, 128u, 4096u}) {
        DevMem oldTf(n * kLinks * 48), newTf(n * kLinks * 48);
        DevMem oldObb(n * kLinks * sizeof(Obb)), newObb(n * kLinks * sizeof(Obb));
        batchFkineAdjOBB(q.ptr(), oldTf.ptr(), tmpl.ptr(), oldObb.ptr(), Dim, n, 0);
        internal::linkPosesAndObbs(ref_, q.span<const float>(), n, tmpl.span<const Obb>(), newTf.span<float>(),
            newObb.span<Obb>(), nullptr);
        CK(cudaDeviceSynchronize());
        EXPECT_EQ(countFloatDiffs(oldTf, newTf), 0u) << "poses, nPoses " << n;
        EXPECT_EQ(countFloatDiffs(oldObb, newObb), 0u) << "obbs, nPoses " << n;
        // OBB plausibility: center = pose * template center, half sizes copied.
        const auto tf  = newTf.download<float>();
        const auto ob  = newObb.download<Obb>();
        const auto tpl = tmpl.download<Obb>();
        std::size_t bad = 0;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t l = 0; l < kLinks; ++l) {
                const float* T = &tf[(p * kLinks + l) * 12];
                const Obb& o   = ob[p * kLinks + l];
                for (int r = 0; r < 3; ++r) {
                    const float c = T[r * 4] * tpl[l].center[0] + T[r * 4 + 1] * tpl[l].center[1]
                                  + T[r * 4 + 2] * tpl[l].center[2] + T[r * 4 + 3];
                    bad += !(std::fabs(o.center[r] - c) < 1e-5f) + (o.halfSize[r] != tpl[l].halfSize[r]);
                }
            }
        }
        EXPECT_EQ(bad, 0u) << "OBB outputs implausible, nPoses " << n;
    }
}

TEST_F(KinematicsKernels, TransformObbsBitIdenticalToBatchOBBTransform) {
    const auto& P = regressionPoses();
    ASSERT_FALSE(P.empty());
    const DevMem q    = DevMem::from(P);
    const DevMem tmpl = DevMem::from(randomObbs(kLinks, 11));
    for (std::uint32_t n : {1u, 33u, 4096u}) {
        DevMem tf(n * kLinks * 48);
        batchFkineAdj(q.ptr(), tf.ptr(), Dim, n, 0);
        DevMem oldObb(n * kLinks * sizeof(Obb)), newObb(n * kLinks * sizeof(Obb)), fusedObb(n * kLinks * sizeof(Obb));
        DevMem fusedTf(n * kLinks * 48);
        batchOBBTransform(tf.ptr(), tmpl.ptr(), oldObb.ptr(), Dim, n, 0); // old argument order: (..., DOF, nPoses)
        internal::transformObbs(tf.span<const float>(), kLinks, n, tmpl.span<const Obb>(), newObb.span<Obb>(), nullptr);
        internal::linkPosesAndObbs(ref_, q.span<const float>(), n, tmpl.span<const Obb>(), fusedTf.span<float>(),
            fusedObb.span<Obb>(), nullptr);
        CK(cudaDeviceSynchronize());
        EXPECT_EQ(countFloatDiffs(oldObb, newObb), 0u) << "nPoses " << n;
        // The contract (ray backend, doc section 5) relies on fused == unfused.
        EXPECT_EQ(countFloatDiffs(newObb, fusedObb), 0u) << "fused vs unfused, nPoses " << n;
    }
}

TEST_F(KinematicsKernels, TrajectoryLinkPosesBitIdenticalToBatchFkineTraj) {
    const std::vector<float> T = regressionTrajectories();
    ASSERT_FALSE(T.empty());
    const DevMem q = DevMem::from(T);
    for (std::uint32_t nTraj : {1u, 3u, 128u, 4096u}) {
        const std::size_t bytes = std::size_t(nTraj) * kTrajPts * kLinks * 48;
        DevMem oldOut(bytes), newOut(bytes);
        batchFkineTraj(q.ptr(), oldOut.ptr(), Dim, kTrajPts, nTraj, 0);
        internal::trajectoryLinkPoses(ref_, q.span<const float>(), kTrajPts, nTraj, newOut.span<float>(), nullptr);
        CK(cudaDeviceSynchronize());
        EXPECT_EQ(countFloatDiffs(oldOut, newOut), 0u) << "nTraj " << nTraj;
        const auto v = newOut.download<float>();
        // trajectory layout: ((traj * numLinks + link) * points + point) * 12
        expectMatchesHostFk(table_, T.data(), std::size_t(nTraj) * kTrajPts, [&](std::size_t p, std::size_t l) {
            const std::size_t traj = p / kTrajPts, pt = p % kTrajPts;
            return &v[((traj * kLinks + l) * kTrajPts + pt) * 12];
        });
    }
    // A point count above the old block cap of 42 poses (blocks straddle trajectories), raw poses as waypoints.
    const auto& P = regressionPoses();
    const DevMem q2 = DevMem::from(P);
    constexpr std::uint32_t pts = 50, nTraj = 81; // 4050 <= 4096 poses
    const std::size_t bytes     = std::size_t(nTraj) * pts * kLinks * 48;
    DevMem oldOut(bytes), newOut(bytes);
    batchFkineTraj(q2.ptr(), oldOut.ptr(), Dim, pts, nTraj, 0);
    internal::trajectoryLinkPoses(ref_, q2.span<const float>(), pts, nTraj, newOut.span<float>(), nullptr);
    CK(cudaDeviceSynchronize());
    EXPECT_EQ(countFloatDiffs(oldOut, newOut), 0u) << "50-point trajectories";
}



// The table derived from panda.urdf at run time equals the hand-written pandas.h bit for bit (moved here from
// test_kinematics_table.cpp, which now compares with the stored copy of pandas.h).
TEST(KinematicsTableVsOld, PandaUrdfEqualsPandasHBitForBit) {
    const KinematicsTable tab =
        robot_detail::kinematicsFromChain(robot_detail::loadChain("models/franka_description/panda.urdf"));
    ASSERT_EQ(tab.dof, static_cast<std::uint32_t>(Dim));
    ASSERT_EQ(tab.numLinks, static_cast<std::uint32_t>(Dim + 1));
    EXPECT_EQ(std::memcmp(&tab.cosMask[0][0], cosMask.data(), cosMask.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(&tab.sinMask[0][0], sinMask.data(), sinMask.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(&tab.oneMask[0][0], oneMask.data(), oneMask.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(tab.base, baseT.data(), baseT.size() * sizeof(float)), 0);
}

// GENERATOR (runs only with RTCD_WRITE_REFERENCE=<dir>): writes the old stack's values as stored references.
//   panda_kinematics.txt     pandas.h cosMask / sinMask / oneMask rows and baseT (exact float bits)
//   kinematics_fk.txt        hashes of the old kernels' outputs on the regression poses / trajectories
//   kinematics_fk_poses.bin  old batchFkineAdj link poses of fk_reference::rawPoseIndices() (64 x 8 x 12 floats)
//   kinematics_fk_traj.bin   old batchFkineTraj link poses of fk_reference::rawTrajectoryIndices() (8 x 8 x 16 x 12)
TEST_F(KinematicsKernels, WriteStoredReferences) {
    namespace R    = rtcd::fk_reference;
    namespace sup  = rtcd::support;
    const auto dir = sup::referenceWriteDir();
    if (!dir) {
        GTEST_SKIP() << "set RTCD_WRITE_REFERENCE=<dir> to write the stored references";
    }
    static_assert(Dim == R::kDof && kLinks == R::kLinks);
    {
        sup::ReferenceWriter w(*dir / "panda_kinematics.txt",
            "Panda kinematics table of the old stack (RTCD/Robot/models/pandas.h: cosMask, sinMask, oneMask, baseT),\n"
            "written by tests/robot/test_kinematics_kernels_vs_old.cpp (RTCD_WRITE_REFERENCE). Float bit patterns.\n"
            "<mask>.<joint> = 12 floats of the row-major 3x4 matrix; rows >= dof are zero (not stored).");
        w.putInt("dof", Dim);
        for (std::size_t j = 0; j < Dim; ++j) {
            w.putFloats("cos_mask." + std::to_string(j), std::span<const float>(cosMask.data() + j * 12, 12));
            w.putFloats("sin_mask." + std::to_string(j), std::span<const float>(sinMask.data() + j * 12, 12));
            w.putFloats("one_mask." + std::to_string(j), std::span<const float>(oneMask.data() + j * 12, 12));
        }
        w.putFloats("base", std::span<const float>(baseT.data(), baseT.size()));
        w.save();
    }

    const auto& P = regressionPoses();
    ASSERT_EQ(P.size(), R::kPoses * Dim);
    const DevMem q = DevMem::from(P);
    const std::uint32_t n = static_cast<std::uint32_t>(R::kPoses);
    auto hashOf = [](const DevMem& m) { return sup::fnv1a(m.download<unsigned char>()); };
    sup::ReferenceWriter w(*dir / "kinematics_fk.txt",
        "Outputs of the old FK kernels (RTCD/Robot/robot.cu, pandas.h masks), written by\n"
        "tests/robot/test_kinematics_kernels_vs_old.cpp (RTCD_WRITE_REFERENCE). <key> <bytes> <FNV-1a 64 of the bytes>.\n"
        "Inputs: the 4096 regression poses (tests/data/motions/Panda4096.bin) and the first 4096 regression KNN\n"
        "trajectories with 16 points; OBB templates fk_reference::obbTemplates(7) / (11) (tests/robot/fk_reference_cases.h).");
    {
        DevMem tf(std::size_t(n) * kLinks * 48);
        batchFkineAdj(q.ptr(), tf.ptr(), Dim, n, 0);
        CK(cudaDeviceSynchronize());
        w.putHash("link_poses", tf.bytes, hashOf(tf));
    }
    {
        const DevMem tmpl = DevMem::from(R::obbTemplates(7));
        DevMem tf(std::size_t(n) * kLinks * 48), obb(std::size_t(n) * kLinks * sizeof(Obb));
        batchFkineAdjOBB(q.ptr(), tf.ptr(), tmpl.ptr(), obb.ptr(), Dim, n, 0);
        CK(cudaDeviceSynchronize());
        w.putHash("link_poses_and_obbs.poses", tf.bytes, hashOf(tf));
        w.putHash("link_poses_and_obbs.obbs", obb.bytes, hashOf(obb));
    }
    {
        const DevMem tmpl = DevMem::from(R::obbTemplates(11));
        DevMem tf(std::size_t(n) * kLinks * 48), obb(std::size_t(n) * kLinks * sizeof(Obb));
        batchFkineAdj(q.ptr(), tf.ptr(), Dim, n, 0);
        batchOBBTransform(tf.ptr(), tmpl.ptr(), obb.ptr(), Dim, n, 0);
        CK(cudaDeviceSynchronize());
        w.putHash("transform_obbs", obb.bytes, hashOf(obb));
    }
    const std::vector<float> T = regressionTrajectories();
    ASSERT_EQ(T.size(), R::kPoses * kTrajPts * Dim);
    {
        const DevMem qt = DevMem::from(T);
        DevMem out(std::size_t(n) * kTrajPts * kLinks * 48);
        batchFkineTraj(qt.ptr(), out.ptr(), Dim, kTrajPts, n, 0);
        CK(cudaDeviceSynchronize());
        w.putHash("trajectory_link_poses", out.bytes, hashOf(out));
    }
    {
        constexpr std::uint32_t pts = 50, nTraj = 81; // raw poses as waypoints, above the old 42-pose block cap
        DevMem out(std::size_t(nTraj) * pts * kLinks * 48);
        batchFkineTraj(q.ptr(), out.ptr(), Dim, pts, nTraj, 0);
        CK(cudaDeviceSynchronize());
        w.putHash("trajectory_link_poses_50x81", out.bytes, hashOf(out));
    }
    w.save();

    // raw subsets
    std::vector<float> sel;
    for (const std::size_t i : R::rawPoseIndices()) sel.insert(sel.end(), P.begin() + i * Dim, P.begin() + (i + 1) * Dim);
    {
        const DevMem qs = DevMem::from(sel);
        const auto ns   = static_cast<std::uint32_t>(R::rawPoseIndices().size());
        DevMem out(std::size_t(ns) * kLinks * 48);
        batchFkineAdj(qs.ptr(), out.ptr(), Dim, ns, 0);
        CK(cudaDeviceSynchronize());
        const auto v = out.download<float>();
        sup::writeBinaryFile(*dir / "kinematics_fk_poses.bin", v.data(), v.size() * sizeof(float));
    }
    std::vector<float> selT;
    for (const std::size_t t : R::rawTrajectoryIndices())
        selT.insert(selT.end(), T.begin() + t * kTrajPts * Dim, T.begin() + (t + 1) * kTrajPts * Dim);
    {
        const DevMem qs = DevMem::from(selT);
        const auto ns   = static_cast<std::uint32_t>(R::rawTrajectoryIndices().size());
        DevMem out(std::size_t(ns) * kTrajPts * kLinks * 48);
        batchFkineTraj(qs.ptr(), out.ptr(), Dim, kTrajPts, ns, 0);
        CK(cudaDeviceSynchronize());
        const auto v = out.download<float>();
        sup::writeBinaryFile(*dir / "kinematics_fk_traj.bin", v.data(), v.size() * sizeof(float));
    }
}