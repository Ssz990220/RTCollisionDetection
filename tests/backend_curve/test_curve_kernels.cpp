// tests/backend_curve: the curve kernels of src/backend_curve (curve_kernels.h) (new_gpu).
//
// Stored reference tests/data/reference/curve_kernels.txt: FNV-1a hashes of the OLD kernels' outputs
// (batchFkineSphrTraj, batchFkineTraj + batchSphrCntrTraj + batchGenControlPoints, genCurveOBB<1>; written before
// the old stack was removed) on the regression trajectories, with the Panda of configs/robots/panda.json and the
// spline weights of configs/spline_weights. The new kernels must reproduce them BYTE FOR BYTE. Plus self-contained
// checks: fused == unfused, segment OBBs contain their segments (both degrees), segment compaction vs a host
// reference.
#include "backend_curve/curve_kernels.h"

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/kinematics.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>

#include <support/inputs.h>
#include <support/paths.h>
#include <support/reference.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace rtcd;
namespace fs  = std::filesystem;
namespace sup = rtcd::support;

namespace {

#define CK(call) ASSERT_EQ((call), cudaSuccess) << #call

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
        template <typename T>
        static DevMem from(std::span<const T> v) {
            DevMem m(v.size() * sizeof(T));
            cudaMemcpy(m.p, v.data(), m.bytes, cudaMemcpyHostToDevice);
            return m;
        }
    };

    std::size_t countWordDiffs(const DevMem& a, const DevMem& b, std::size_t words) {
        const auto va = a.download<std::uint32_t>();
        const auto vb = b.download<std::uint32_t>();
        std::size_t n = 0;
        for (std::size_t i = 0; i < words && i < va.size() && i < vb.size(); ++i) {
            n += va[i] != vb[i];
        }
        return n;
    }

    // Values must be real numbers (guards "the kernel wrote nothing / garbage").
    std::size_t countNonFinite(const DevMem& a, std::size_t floats) {
        const auto v  = a.download<float>();
        std::size_t n = 0;
        for (std::size_t i = 0; i < floats; ++i) {
            n += !std::isfinite(v[i]) || std::fabs(v[i]) > 100.f;
        }
        return n;
    }

    const sup::ReferenceFile& reference() {
        static const sup::ReferenceFile f = sup::ReferenceFile::load(sup::referenceDir() / "curve_kernels.txt");
        return f;
    }

    void expectMatchesReference(const std::string& key, const DevMem& m, std::size_t bytes) {
        ASSERT_TRUE(reference().has(key)) << key;
        ASSERT_EQ(reference().integer(key, 0), bytes) << key;
        auto v = m.download<unsigned char>();
        v.resize(bytes);
        EXPECT_EQ(sup::hex64(sup::fnv1a(v)), sup::hex64(reference().hash(key, 1)))
            << key << ": output differs from the old kernels' (stored hash)";
    }

    std::vector<float> readWeights(const fs::path& f) {
        std::vector<float> w;
        std::ifstream in(f);
        float v;
        while (in >> v) {
            w.push_back(v);
        }
        return w;
    }

    class CurveKernels : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
                return;
            }
            robot_ = RobotModel::load(loadRobotDesc(sup::configsDir() / "robots" / "panda.json"));
        }
        static void TearDownTestSuite() { robot_.reset(); }

        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
                GTEST_SKIP() << "no CUDA device";
            }
            ASSERT_TRUE(robot_);
            dof_     = robot_->dof();
            ASSERT_EQ(dof_, 7u);
            table_   = internal::uploadKinematics(robot_->kinematics());
            centers_ = std::make_unique<DevMem>(DevMem::from(robot_->sphereCenters()));
            links_   = std::make_unique<DevMem>(DevMem::from(robot_->sphereLink()));
            S_       = robot_->totalSpheres();
        }

        std::vector<float> trajectories(std::uint32_t points, std::size_t n) {
            // the cache holds the 4096-sample set only; take the first n of its trajectories
            std::vector<float> v = sup::regressionInputs(sup::dataDir() / "motions", points, 4096, robot_->dof());
            v.resize(n * points * robot_->dof());
            return v;
        }

        std::vector<float> loadWeights(const char* degreeDir, std::uint32_t ctrl, std::uint32_t traj) {
            const fs::path f = sup::configsDir() / "spline_weights" / degreeDir / "normal"
                             / (std::to_string(ctrl) + "_" + std::to_string(traj) + ".txt");
            std::vector<float> w = readWeights(f);
            EXPECT_EQ(w.size(), std::size_t(ctrl) * traj) << f;
            return w;
        }

        internal::KinematicsRef kin() const { return internal::KinematicsRef{&robot_->kinematics(), table_.ptr()}; }

        static inline std::shared_ptr<const RobotModel> robot_;
        DeviceBuffer table_;
        std::unique_ptr<DevMem> centers_, links_;
        std::uint32_t S_   = 0;
        std::uint32_t dof_ = 0;
    };

    constexpr std::uint32_t kT = 16;

} // namespace

TEST_F(CurveKernels, FusedLinearPointsMatchTheStoredReference) {
    ASSERT_TRUE(curve::fusedCurvePointsSupported(dof_, kT, S_, 0));
    for (std::uint32_t n : {37u, 4096u}) {
        const DevMem dq          = DevMem::from(trajectories(kT, n));
        const std::size_t floats = std::size_t(n) * S_ * kT * 3;
        DevMem nw(floats * 4);
        curve::fusedCurvePoints(table_.ptr(), dof_, dq.ptr(), centers_->ptr(), links_->ptr(), 0, nw.ptr(), kT, n, S_, 0,
            nullptr);
        CK(cudaDeviceSynchronize());
        expectMatchesReference("fused_linear_points." + std::to_string(n), nw, floats * 4);
        EXPECT_EQ(countNonFinite(nw, floats), 0u) << "n " << n;
    }
}

TEST_F(CurveKernels, UnfusedLinearMatchesTheStoredReferenceAndTheFusedKernel) {
    const std::uint32_t n      = 512;
    const DevMem dq            = DevMem::from(trajectories(kT, n));
    const std::size_t tfFloats = std::size_t(n) * (dof_ + 1) * kT * 12;
    const std::size_t floats   = std::size_t(n) * S_ * kT * 3;
    DevMem nTf(tfFloats * 4), nw(floats * 4), fused(floats * 4);
    internal::trajectoryLinkPoses(kin(), DeviceSpan<const float>{dq.ptr(), dq.bytes / 4}, kT, n,
        DeviceSpan<float>{nTf.ptr(), nTf.bytes / 4}, nullptr);
    curve::sphereCentersTraj(nTf.ptr(), centers_->ptr(), links_->ptr(), nw.ptr(), kT, n, S_, dof_, nullptr);
    curve::fusedCurvePoints(table_.ptr(), dof_, dq.ptr(), centers_->ptr(), links_->ptr(), 0, fused.ptr(), kT, n, S_, 0,
        nullptr);
    CK(cudaDeviceSynchronize());
    expectMatchesReference("unfused_link_poses.512", nTf, tfFloats * 4);
    expectMatchesReference("unfused_sphere_centers.512", nw, floats * 4);
    EXPECT_EQ(countWordDiffs(fused, nw, floats), 0u) << "fused vs unfused";
    EXPECT_EQ(countNonFinite(nw, floats), 0u);
}

TEST_F(CurveKernels, FusedAndUnfusedControlPointsMatchTheStoredReference) {
    struct Case {
        const char* dir;
        std::uint32_t ctrl, traj;
    };
    for (const Case c : {Case{"quad", 8, 16}, Case{"cubic", 8, 16}, Case{"quad", 4, 8}, Case{"quad", 6, 12},
             Case{"quad", 4, 128}}) {
        const std::string name = std::string(c.dir) + "." + std::to_string(c.ctrl) + "_" + std::to_string(c.traj);
        SCOPED_TRACE(name);
        const DevMem dw            = DevMem::from(loadWeights(c.dir, c.ctrl, c.traj));
        const std::uint32_t n      = c.traj > 16 ? 64u : 300u;
        const DevMem dq            = DevMem::from(trajectories(c.traj, n));
        const std::size_t tfFloats = std::size_t(n) * (dof_ + 1) * c.traj * 12;
        const std::size_t tFloats  = std::size_t(n) * S_ * c.traj * 3;
        const std::size_t cFloats  = std::size_t(n) * S_ * c.ctrl * 3;
        DevMem nTf(tfFloats * 4), nT(tFloats * 4), nC(cFloats * 4);
        internal::trajectoryLinkPoses(kin(), DeviceSpan<const float>{dq.ptr(), dq.bytes / 4}, c.traj, n,
            DeviceSpan<float>{nTf.ptr(), nTf.bytes / 4}, nullptr);
        curve::sphereCentersTraj(nTf.ptr(), centers_->ptr(), links_->ptr(), nT.ptr(), c.traj, n, S_, dof_, nullptr);
        curve::controlPoints(nT.ptr(), nC.ptr(), dw.ptr(), n * S_, c.ctrl, c.traj, nullptr);
        CK(cudaDeviceSynchronize());
        expectMatchesReference("control_points." + name, nC, cFloats * 4);
        EXPECT_EQ(countNonFinite(nC, cFloats), 0u);

        // the fused kernel exists exactly where the old one did (support predicate; DOF <= 8 here)
        const bool fusedOk = curve::fusedCurvePointsSupported(dof_, c.traj, S_, c.ctrl);
        EXPECT_EQ(fusedOk, reference().has("fused_control_points." + name)) << "support predicate";
        if (fusedOk) {
            DevMem nF(cFloats * 4);
            curve::fusedCurvePoints(table_.ptr(), dof_, dq.ptr(), centers_->ptr(), links_->ptr(), dw.ptr(), nF.ptr(),
                c.traj, n, S_, c.ctrl, nullptr);
            CK(cudaDeviceSynchronize());
            expectMatchesReference("fused_control_points." + name, nF, cFloats * 4);
            EXPECT_EQ(countWordDiffs(nF, nC, cFloats), 0u) << "fused vs unfused";
        }
    }
}

TEST_F(CurveKernels, SegmentObbsContainTheSegmentAndLinearOnesMatchTheStoredReference) {
    for (const bool inject : {false, true})
    for (std::uint32_t degree : {1u, 2u}) {
        SCOPED_TRACE("degree " + std::to_string(degree) + (inject ? " with injected degenerate vertices" : " real points"));
        const std::uint32_t n   = 256;
        const std::uint32_t vpc = degree == 1 ? kT : 8u;
        const DevMem dq         = DevMem::from(trajectories(kT, n));
        std::vector<float> w(1, 0.f);
        if (degree == 2) {
            w = loadWeights("quad", 8, kT);
        }
        const DevMem dw = DevMem::from(w);
        const std::size_t vFloats = std::size_t(n) * S_ * vpc * 3;
        DevMem pts(vFloats * 4);
        curve::fusedCurvePoints(table_.ptr(), dof_, dq.ptr(), centers_->ptr(), links_->ptr(), dw.ptr(), pts.ptr(), kT, n,
            S_, degree == 1 ? 0 : vpc, nullptr);
        // topology as in the backend (old curveParam)
        const std::uint32_t spc = vpc - degree;
        std::vector<std::uint32_t> segStart;
        std::vector<float> radii;
        const auto r = robot_->sphereRadii();
        for (std::uint32_t c = 0; c < n * S_; ++c) {
            for (std::uint32_t s = 0; s < spc; ++s) segStart.push_back(c * vpc + s);
            for (std::uint32_t v = 0; v < vpc; ++v) radii.push_back(r[c % S_]);
        }
        // optionally make some segments degenerate (repeated points) to cover the 1p / 2p branches
        if (inject) {
            std::vector<float> h = pts.download<float>();
            for (std::size_t v = 0; v + 1 < h.size() / 3; v += 7) {
                for (int k = 0; k < 3; ++k) h[(v + 1) * 3 + k] = h[v * 3 + k];
            }
            cudaMemcpy(pts.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
        }
        const DevMem ds = DevMem::from(segStart);
        const DevMem dr = DevMem::from(radii);
        const auto nSegs = static_cast<std::uint32_t>(segStart.size());
        DevMem nw(nSegs * sizeof(Obb));
        curve::curveSegmentObbs(degree, pts.ptr(), ds.ptr(), dr.ptr(), nw.ptr(), nSegs, nullptr);
        CK(cudaDeviceSynchronize());

        // Containment: every box must hold the balls of radius r around the segment's control points (a linear
        // segment / quadratic B-spline segment lies in the convex hull of its control points, so this makes the
        // culling conservative). Boxes must be finite (a non-finite box never tests as separated).
        const auto hp   = pts.download<float>();
        const auto ob   = nw.download<Obb>();
        std::size_t bad = 0, nonFinite = 0;
        for (std::uint32_t i = 0; i < nSegs; ++i) {
            const Obb& b   = ob[i];
            const float* f = reinterpret_cast<const float*>(&b);
            bool finite    = true;
            for (int k = 0; k < 15; ++k) finite &= std::isfinite(f[k]);
            if (!finite) {
                ++nonFinite;
                continue;
            }
            bool in       = true;
            double excess = 0;
            for (std::uint32_t k = 0; k <= degree; ++k) {
                const float* p = &hp[(segStart[i] + k) * 3];
                for (int a = 0; a < 3; ++a) {
                    double d = 0;
                    for (int x = 0; x < 3; ++x) d += double(p[x] - b.center[x]) * b.axes[a][x];
                    const double e = std::fabs(d) + radii[segStart[i]] - b.halfSize[a];
                    excess         = (std::max)(excess, e);
                    in &= e <= 1e-5;
                }
            }
            if (!in && bad < 3) {
                ADD_FAILURE() << "segment " << i << " exceeds its box by " << excess;
            }
            bad += !in;
        }
        EXPECT_EQ(bad, 0u) << "segment OBBs must contain the segment";
        EXPECT_EQ(nonFinite, 0u);
        if (degree == 1) {
            expectMatchesReference(std::string("linear_segment_obbs.") + (inject ? "injected" : "real"), nw,
                std::size_t(nSegs) * sizeof(Obb));
        }
    }
    EXPECT_THROW(curve::curveSegmentObbs(3, 0, 0, 0, 0, 1, nullptr), UnsupportedError);
}

TEST_F(CurveKernels, CompactSegmentsMatchesHostReference) {
    const std::uint32_t nSegs = 5000, vps = 3, nVerts = nSegs + 2;
    std::mt19937 rng(3);
    std::vector<float> verts(nVerts * 3), radii(nVerts);
    for (auto& v : verts) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    for (auto& v : radii) v = std::uniform_real_distribution<float>(0, 1)(rng);
    std::vector<std::uint32_t> start(nSegs), item(nSegs), flags(nSegs), where(nSegs);
    std::uint32_t acc = 0;
    for (std::uint32_t i = 0; i < nSegs; ++i) {
        start[i] = i;
        item[i]  = i / 50;
        flags[i] = (rng() % 3 == 0) ? 1u : 0u;
        where[i] = acc;
        acc += flags[i];
    }
    flags.back() = 1; // the count must include the last segment
    acc          = where.back() + 1;
    const DevMem dv = DevMem::from(verts), dr = DevMem::from(radii), dst = DevMem::from(start), dit = DevMem::from(item),
                 df = DevMem::from(flags), dwh = DevMem::from(where);
    DevMem ov(std::size_t(nSegs) * vps * 12), orr(std::size_t(nSegs) * vps * 4), oi(nSegs * 4), om(nSegs * 4), oc(4);
    curve::compactSegments(dv.ptr(), dr.ptr(), dst.ptr(), dit.ptr(), df.ptr(), dwh.ptr(), nSegs, vps, ov.ptr(),
        orr.ptr(), oi.ptr(), om.ptr(), oc.ptr(), nullptr);
    CK(cudaDeviceSynchronize());
    const auto hv = ov.download<float>();
    const auto hr = orr.download<float>();
    const auto hi = oi.download<std::uint32_t>();
    const auto hm = om.download<std::uint32_t>();
    ASSERT_EQ(oc.download<std::uint32_t>()[0], acc);
    std::size_t bad = 0;
    for (std::uint32_t i = 0; i < nSegs; ++i) {
        if (!flags[i]) continue;
        const std::uint32_t d = where[i];
        bad += hi[d] != d * vps;
        bad += hm[d] != item[i];
        for (std::uint32_t k = 0; k < vps; ++k) {
            bad += hr[d * vps + k] != radii[start[i] + k];
            for (int x = 0; x < 3; ++x) bad += hv[(d * vps + k) * 3 + x] != verts[(start[i] + k) * 3 + x];
        }
    }
    EXPECT_EQ(bad, 0u);
}
