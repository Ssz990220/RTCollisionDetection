// tests/robot: the FK kernels (include/rtcd/internal/kinematics.h) on the table derived from panda.urdf (new_gpu).
//
// Stored references (tests/data/reference, written from the OLD stack's kernels before it was removed; see
// tests/data/README.md and fk_reference_cases.h):
//   kinematics_fk.txt        exact: FNV-1a hashes of the old kernels' complete outputs (link poses, link poses +
//                            OBBs, OBB transform, trajectory link poses) on the 4096 regression poses / trajectories;
//   kinematics_fk_poses.bin  raw old link poses of 64 regression poses, kinematics_fk_traj.bin of 8 trajectories:
//                            compared to kRawTolerance (the hashes are the bit-exact check; the raw values show
//                            where and by how much a mismatch lies).
// Plus self-contained checks: host reference FK, fused == unfused OBBs, argument errors, two tables coexisting.
#include "fk_reference_cases.h"
#include "robot/urdf_chain.h"

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/kinematics.h>

#include <support/inputs.h>
#include <support/paths.h>
#include <support/reference.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

using namespace rtcd;
namespace R   = rtcd::fk_reference;
namespace sup = rtcd::support;

namespace {

#define CK(call) ASSERT_EQ((call), cudaSuccess) << #call

    constexpr std::size_t kDof    = R::kDof;
    constexpr std::size_t kLinks  = R::kLinks;
    constexpr std::uint32_t kPts  = R::kTrajPts;

    const std::vector<float>& regressionPoses() {
        static const std::vector<float> p = sup::regressionInputs(sup::dataDir() / "motions", 1);
        return p;
    }
    const std::vector<float>& regressionTrajectories() {
        static const std::vector<float> t = sup::regressionInputs(sup::dataDir() / "motions", kPts);
        return t;
    }

    struct DevMem {
        void* p           = nullptr;
        std::size_t bytes = 0;
        explicit DevMem(std::size_t b) : bytes(b) {
            if (cudaMalloc(&p, b) != cudaSuccess) {
                p = nullptr;
            }
            cudaMemset(p, 0xA5, b); // poison: unwritten bytes show in the hash
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

    std::size_t countFloatDiffs(const DevMem& a, const DevMem& b) {
        const auto va = a.download<std::uint32_t>();
        const auto vb = b.download<std::uint32_t>();
        std::size_t n = 0;
        for (std::size_t i = 0; i < va.size(); ++i) {
            n += va[i] != vb[i];
        }
        return n;
    }

    const sup::ReferenceFile& fkReference() {
        static const sup::ReferenceFile f = sup::ReferenceFile::load(sup::referenceDir() / "kinematics_fk.txt");
        return f;
    }

    // Bytes and hash of a device buffer against "<key> <bytes> <hash>".
    void expectMatchesReference(const DevMem& m, const char* key) {
        const auto& ref = fkReference();
        ASSERT_EQ(ref.integer(key, 0), m.bytes) << key;
        EXPECT_EQ(sup::hex64(sup::fnv1a(m.download<unsigned char>())), sup::hex64(ref.hash(key, 1)))
            << key << ": output differs from the old kernels' (stored hash)";
    }

    // Raw comparison with a stored float array: values within tolerance; returns the number of non-identical bits.
    std::size_t expectNearStored(const std::vector<float>& got, const char* file) {
        const std::vector<float> want = sup::readFloatFile(sup::referenceDir() / file);
        EXPECT_EQ(got.size(), want.size()) << file;
        std::size_t far = 0, notBitEqual = 0;
        for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
            notBitEqual += std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i]);
            if (!(std::fabs(got[i] - want[i]) <= R::kRawTolerance)) {
                if (far++ < 5) {
                    ADD_FAILURE() << file << " float " << i << ": new " << got[i] << ", stored " << want[i];
                }
            }
        }
        EXPECT_EQ(far, 0u) << file;
        return notBitEqual;
    }

    // Guards against "the kernel wrote nothing": the output must match the host reference FK to float tolerance.
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
        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
                GTEST_SKIP() << "no CUDA device";
            }
            table_  = robot_detail::kinematicsFromChain(robot_detail::loadChain("models/franka_description/panda.urdf"));
            ASSERT_EQ(table_.dof, kDof);
            ASSERT_EQ(table_.numLinks, kLinks);
            device_ = internal::uploadKinematics(table_);
            ref_    = internal::KinematicsRef{&table_, device_.ptr()};
        }

        KinematicsTable table_{};
        DeviceBuffer device_;
        internal::KinematicsRef ref_;
    };

} // namespace

TEST_F(KinematicsKernels, UploadedTableIsTheHostTable) {
    ASSERT_EQ(device_.size(), sizeof(KinematicsTable));
    KinematicsTable back{};
    device_.download(&back, sizeof(back));
    EXPECT_EQ(std::memcmp(&back, &table_, sizeof(back)), 0);
}

TEST_F(KinematicsKernels, LinkPosesMatchTheStoredReference) {
    const auto& P = regressionPoses();
    ASSERT_EQ(P.size(), R::kPoses * kDof);
    const DevMem q        = DevMem::from(P);
    const std::uint32_t n = static_cast<std::uint32_t>(R::kPoses);
    DevMem out(std::size_t(n) * kLinks * 48);
    internal::linkPoses(ref_, q.span<const float>(), n, out.span<float>(), nullptr);
    CK(cudaDeviceSynchronize());
    expectMatchesReference(out, "link_poses");
    const auto v = out.download<float>();
    expectMatchesHostFk(table_, P.data(), n, [&](std::size_t p, std::size_t l) { return &v[(p * kLinks + l) * 12]; });

    // raw values of the selected poses (computed separately: a pose's result does not depend on its batch)
    std::vector<float> sel;
    for (const std::size_t i : R::rawPoseIndices()) sel.insert(sel.end(), P.begin() + i * kDof, P.begin() + (i + 1) * kDof);
    const DevMem qs       = DevMem::from(sel);
    const auto ns         = static_cast<std::uint32_t>(R::rawPoseIndices().size());
    DevMem outS(std::size_t(ns) * kLinks * 48);
    internal::linkPoses(ref_, qs.span<const float>(), ns, outS.span<float>(), nullptr);
    CK(cudaDeviceSynchronize());
    const std::size_t notBitEqual = expectNearStored(outS.download<float>(), "kinematics_fk_poses.bin");
    EXPECT_EQ(notBitEqual, 0u) << "floats not bit-identical to the old kernels";
}

TEST_F(KinematicsKernels, LinkPosesAndObbsMatchTheStoredReference) {
    const DevMem q        = DevMem::from(regressionPoses());
    const DevMem tmpl     = DevMem::from(R::obbTemplates(7));
    const std::uint32_t n = static_cast<std::uint32_t>(R::kPoses);
    DevMem tf(std::size_t(n) * kLinks * 48), obb(std::size_t(n) * kLinks * sizeof(Obb));
    internal::linkPosesAndObbs(
        ref_, q.span<const float>(), n, tmpl.span<const Obb>(), tf.span<float>(), obb.span<Obb>(), nullptr);
    CK(cudaDeviceSynchronize());
    expectMatchesReference(tf, "link_poses_and_obbs.poses");
    expectMatchesReference(obb, "link_poses_and_obbs.obbs");
    // OBB plausibility: center = pose * template center, half sizes copied.
    const auto T   = tf.download<float>();
    const auto ob  = obb.download<Obb>();
    const auto tpl = R::obbTemplates(7);
    std::size_t bad = 0;
    for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t l = 0; l < kLinks; ++l) {
            const float* M = &T[(p * kLinks + l) * 12];
            const Obb& o   = ob[p * kLinks + l];
            for (int r = 0; r < 3; ++r) {
                const float c = M[r * 4] * tpl[l].center[0] + M[r * 4 + 1] * tpl[l].center[1]
                              + M[r * 4 + 2] * tpl[l].center[2] + M[r * 4 + 3];
                bad += !(std::fabs(o.center[r] - c) < 1e-5f) + (o.halfSize[r] != tpl[l].halfSize[r]);
            }
        }
    }
    EXPECT_EQ(bad, 0u) << "OBB outputs implausible";
}

TEST_F(KinematicsKernels, TransformObbsMatchTheStoredReferenceAndTheFusedKernel) {
    const DevMem q        = DevMem::from(regressionPoses());
    const DevMem tmpl     = DevMem::from(R::obbTemplates(11));
    const std::uint32_t n = static_cast<std::uint32_t>(R::kPoses);
    DevMem tf(std::size_t(n) * kLinks * 48), fusedTf(std::size_t(n) * kLinks * 48);
    DevMem obb(std::size_t(n) * kLinks * sizeof(Obb)), fusedObb(std::size_t(n) * kLinks * sizeof(Obb));
    internal::linkPoses(ref_, q.span<const float>(), n, tf.span<float>(), nullptr);
    internal::transformObbs(tf.span<const float>(), kLinks, n, tmpl.span<const Obb>(), obb.span<Obb>(), nullptr);
    internal::linkPosesAndObbs(
        ref_, q.span<const float>(), n, tmpl.span<const Obb>(), fusedTf.span<float>(), fusedObb.span<Obb>(), nullptr);
    CK(cudaDeviceSynchronize());
    expectMatchesReference(obb, "transform_obbs");
    // The contract (ray backend, doc section 5) relies on fused == unfused.
    EXPECT_EQ(countFloatDiffs(obb, fusedObb), 0u) << "fused vs unfused";
    EXPECT_EQ(countFloatDiffs(tf, fusedTf), 0u) << "fused vs unfused link poses";
}

TEST_F(KinematicsKernels, TrajectoryLinkPosesMatchTheStoredReference) {
    const std::vector<float>& T = regressionTrajectories();
    ASSERT_EQ(T.size(), R::kPoses * kPts * kDof);
    const DevMem q            = DevMem::from(T);
    const std::uint32_t nTraj = static_cast<std::uint32_t>(R::kPoses);
    DevMem out(std::size_t(nTraj) * kPts * kLinks * 48);
    internal::trajectoryLinkPoses(ref_, q.span<const float>(), kPts, nTraj, out.span<float>(), nullptr);
    CK(cudaDeviceSynchronize());
    expectMatchesReference(out, "trajectory_link_poses");
    const auto v = out.download<float>();
    // trajectory layout: ((traj * numLinks + link) * points + point) * 12
    expectMatchesHostFk(table_, T.data(), std::size_t(nTraj) * kPts, [&](std::size_t p, std::size_t l) {
        const std::size_t traj = p / kPts, pt = p % kPts;
        return &v[((traj * kLinks + l) * kPts + pt) * 12];
    });

    std::vector<float> sel;
    for (const std::size_t t : R::rawTrajectoryIndices())
        sel.insert(sel.end(), T.begin() + t * kPts * kDof, T.begin() + (t + 1) * kPts * kDof);
    const DevMem qs = DevMem::from(sel);
    const auto ns   = static_cast<std::uint32_t>(R::rawTrajectoryIndices().size());
    DevMem outS(std::size_t(ns) * kPts * kLinks * 48);
    internal::trajectoryLinkPoses(ref_, qs.span<const float>(), kPts, ns, outS.span<float>(), nullptr);
    CK(cudaDeviceSynchronize());
    const std::size_t notBitEqual = expectNearStored(outS.download<float>(), "kinematics_fk_traj.bin");
    EXPECT_EQ(notBitEqual, 0u) << "floats not bit-identical to the old kernels";

    // A point count above the old block cap of 42 poses (blocks straddle trajectories), raw poses as waypoints.
    const DevMem q2 = DevMem::from(regressionPoses());
    constexpr std::uint32_t pts = 50, n50 = 81; // 4050 <= 4096 poses
    DevMem out50(std::size_t(n50) * pts * kLinks * 48);
    internal::trajectoryLinkPoses(ref_, q2.span<const float>(), pts, n50, out50.span<float>(), nullptr);
    CK(cudaDeviceSynchronize());
    expectMatchesReference(out50, "trajectory_link_poses_50x81");
}

TEST_F(KinematicsKernels, CountZeroLaunchesNothingAndBadArgumentsThrow) {
    DevMem out(48);
    const DevMem q(64);
    EXPECT_NO_THROW(internal::linkPoses(ref_, q.span<const float>(), 0, out.span<float>(), nullptr));
    EXPECT_NO_THROW(internal::trajectoryLinkPoses(ref_, q.span<const float>(), 16, 0, out.span<float>(), nullptr));
    EXPECT_THROW(internal::linkPoses(ref_, q.span<const float>(), 1, out.span<float>(), nullptr), InvalidArgument);
    EXPECT_THROW(internal::linkPoses(internal::KinematicsRef{}, q.span<const float>(), 1, out.span<float>(), nullptr),
        InvalidArgument);
    EXPECT_THROW(internal::transformObbs(out.span<const float>(), 0, 1, {}, {}, nullptr), InvalidArgument);
    CK(cudaDeviceSynchronize());
}

// Several robots coexist: two tables in two device buffers give each its own result (no globals).
TEST_F(KinematicsKernels, TwoTablesCoexist) {
    KinematicsTable shifted = table_;
    shifted.base[3]         = 1.0f; // base moved by 1 m in x
    DeviceBuffer dev2       = internal::uploadKinematics(shifted);
    const internal::KinematicsRef ref2{&shifted, dev2.ptr()};
    const DevMem q = DevMem::from(regressionPoses());
    constexpr std::uint32_t n = 64;
    DevMem a(n * kLinks * 48), b(n * kLinks * 48);
    internal::linkPoses(ref_, q.span<const float>(), n, a.span<float>(), nullptr);
    internal::linkPoses(ref2, q.span<const float>(), n, b.span<float>(), nullptr);
    CK(cudaDeviceSynchronize());
    const auto va = a.download<float>(), vb = b.download<float>();
    for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t l = 0; l < kLinks; ++l) {
            const float* ta = &va[(p * kLinks + l) * 12];
            const float* tb = &vb[(p * kLinks + l) * 12];
            EXPECT_EQ(ta[0], tb[0]);
            EXPECT_NEAR(tb[3] - ta[3], 1.0f, 1e-5f);
        }
    }
}
