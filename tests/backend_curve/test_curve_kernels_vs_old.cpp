// tests/backend_curve: the curve kernels of src/backend_curve (curve_kernels.h) vs the old ones (new_gpu).
//
// OLD-STACK PARITY TEST: links rtcd_legacy_kernels and calls the old batchFkineSphrTraj (fused), batchFkineTraj +
// batchSphrCntrTraj + batchGenControlPoints (unfused) and genCurveOBB<1|2> (RTCD/Robot/robot.cu, sweptVolume.cu),
// with the hand-written pandas.h masks / base in their __constant__ globals and the spline weights loaded by the old
// fillWeight. The new kernels get the table of the URDF-loaded RobotModel and the weights in device buffers.
// Outputs must be equal BYTE FOR BYTE. Both sides use the RobotModel's sphere arrays (their equality with the old
// robot<DOF> arrays is rtcd_robot's business). Delete with the old stack.
//
// REFERENCE GENERATOR: with RTCD_WRITE_REFERENCE=<dir> it writes the old kernels' output hashes to
// curve_kernels.txt, the stored reference of test_curve_kernels.cpp (which keeps the new-only checks).
#include "backend_curve/curve_kernels.h"

#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/kinematics.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>

#include <support/inputs.h>
#include <support/paths.h>
#include <support/reference.h>

#include <Robot/models/pandas.h> // old: cosMask, sinMask, oneMask, baseT, Dim
#include <Robot/robot.cuh>       // old FK / sphere-center / fused kernels

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Old sweptVolume.cu functions (declared here instead of including sweptVolume.cuh, which declares __global__
// templates that a host compiler cannot parse). Same signatures, so they link against rtcd_legacy_kernels.
cudaError_t fillWeight(float* weights, size_t nWeights);
void batchGenControlPoints(
    float* trajPoints, float* controlPoints, size_t nCurves, cudaStream_t stream, size_t nCPts, size_t nTPts);
template <int degree>
void genCurveOBB(const CUdeviceptr cPts, const CUdeviceptr segs, const CUdeviceptr radii, CUdeviceptr obb,
    const size_t nSegs, const cudaStream_t stream);

using namespace rtcd;
namespace fs = std::filesystem;

namespace {

#define CK(call) ASSERT_EQ((call), cudaSuccess) << #call

    struct DevMem {
        void* p           = nullptr;
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

    // Values must be real numbers (guards "both sides wrote nothing / garbage").
    std::size_t countNonFinite(const DevMem& a, std::size_t floats) {
        const auto v  = a.download<float>();
        std::size_t n = 0;
        for (std::size_t i = 0; i < floats; ++i) {
            n += !std::isfinite(v[i]) || std::fabs(v[i]) > 100.f;
        }
        return n;
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
            setFkineMasks(cosMask.data(), sinMask.data(), oneMask.data(), cosMask.size());
            std::array<float, 12> base = baseT;
            setBaseAffine(base);
            cudaDeviceSynchronize();
            robot_ = RobotModel::load(loadRobotDesc(support::configsDir() / "robots" / "panda.json"));
        }
        static void TearDownTestSuite() { robot_.reset(); }

        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
                GTEST_SKIP() << "no CUDA device";
            }
            ASSERT_TRUE(robot_);
            ASSERT_EQ(robot_->dof(), static_cast<std::uint32_t>(Dim));
            table_   = internal::uploadKinematics(robot_->kinematics());
            centers_ = std::make_unique<DevMem>(DevMem::from(robot_->sphereCenters()));
            links_   = std::make_unique<DevMem>(DevMem::from(robot_->sphereLink()));
            S_       = robot_->totalSpheres();
        }

        std::vector<float> trajectories(std::uint32_t points, std::size_t n) {
            // the cache holds the 4096-sample set only; take the first n of its trajectories
            std::vector<float> v = support::regressionInputs(support::dataDir() / "motions", points, 4096, robot_->dof());
            v.resize(n * points * robot_->dof());
            return v;
        }

        // Loads the weights into the old __constant__ tables and returns them for the new device buffer.
        std::vector<float> loadWeights(const char* degreeDir, std::uint32_t ctrl, std::uint32_t traj) {
            const fs::path f = support::configsDir() / "spline_weights" / degreeDir / "normal"
                             / (std::to_string(ctrl) + "_" + std::to_string(traj) + ".txt");
            std::vector<float> w = readWeights(f);
            EXPECT_EQ(w.size(), std::size_t(ctrl) * traj) << f;
            EXPECT_EQ(fillWeight(w.data(), w.size()), cudaSuccess);
            return w;
        }

        static inline std::shared_ptr<const RobotModel> robot_;
        DeviceBuffer table_;
        std::unique_ptr<DevMem> centers_, links_;
        std::uint32_t S_ = 0;
    };

    constexpr std::uint32_t kT = 16;

} // namespace

TEST_F(CurveKernels, FusedLinearBitIdenticalToBatchFkineSphrTraj) {
    ASSERT_TRUE(curve::fusedCurvePointsSupported(Dim, kT, S_, 0));
    const std::vector<float> q = trajectories(kT, 4096);
    const DevMem dq            = DevMem::from(q);
    for (std::uint32_t n : {1u, 37u, 128u, 4096u}) {
        const std::size_t floats = std::size_t(n) * S_ * kT * 3;
        DevMem o(floats * 4), nw(floats * 4);
        batchFkineSphrTraj(dq.ptr(), static_cast<const float*>(centers_->p), static_cast<const int*>(links_->p),
            static_cast<float*>(o.p), Dim, kT, n, S_, 0, nullptr);
        curve::fusedCurvePoints(table_.ptr(), Dim, dq.ptr(), centers_->ptr(), links_->ptr(), 0, nw.ptr(), kT, n, S_, 0,
            nullptr);
        CK(cudaDeviceSynchronize());
        EXPECT_EQ(countWordDiffs(o, nw, floats), 0u) << "n " << n;
        EXPECT_EQ(countNonFinite(nw, floats), 0u) << "n " << n;
    }
}

TEST_F(CurveKernels, UnfusedLinearBitIdenticalToOldAndToFused) {
    const std::vector<float> q = trajectories(kT, 512);
    const DevMem dq            = DevMem::from(q);
    for (std::uint32_t n : {1u, 37u, 512u}) {
        const std::size_t tfFloats = std::size_t(n) * (Dim + 1) * kT * 12;
        const std::size_t floats   = std::size_t(n) * S_ * kT * 3;
        DevMem oTf(tfFloats * 4), nTf(tfFloats * 4), o(floats * 4), nw(floats * 4), fused(floats * 4);
        batchFkineTraj(dq.ptr(), oTf.ptr(), Dim, kT, n, nullptr);
        batchSphrCntrTraj(static_cast<float*>(oTf.p), static_cast<float*>(centers_->p), static_cast<float*>(o.p), kT, n,
            static_cast<const int*>(links_->p), S_, Dim, nullptr);
        internal::trajectoryLinkPoses(internal::KinematicsRef{&robot_->kinematics(), table_.ptr()},
            dq.span<const float>(), kT, n, nTf.span<float>(), nullptr);
        curve::sphereCentersTraj(nTf.ptr(), centers_->ptr(), links_->ptr(), nw.ptr(), kT, n, S_, Dim, nullptr);
        curve::fusedCurvePoints(table_.ptr(), Dim, dq.ptr(), centers_->ptr(), links_->ptr(), 0, fused.ptr(), kT, n, S_,
            0, nullptr);
        CK(cudaDeviceSynchronize());
        EXPECT_EQ(countWordDiffs(oTf, nTf, tfFloats), 0u) << "link poses, n " << n;
        EXPECT_EQ(countWordDiffs(o, nw, floats), 0u) << "centers, n " << n;
        EXPECT_EQ(countWordDiffs(fused, nw, floats), 0u) << "fused vs unfused, n " << n;
        EXPECT_EQ(countNonFinite(nw, floats), 0u);
    }
}

TEST_F(CurveKernels, FusedAndUnfusedControlPointsBitIdenticalToOld) {
    struct Case {
        const char* dir;
        std::uint32_t ctrl, traj;
    };
    for (const Case c : {Case{"quad", 8, 16}, Case{"cubic", 8, 16}, Case{"quad", 4, 8}, Case{"quad", 6, 12},
             Case{"quad", 4, 128}}) {
        SCOPED_TRACE(std::string(c.dir) + " " + std::to_string(c.ctrl) + "_" + std::to_string(c.traj));
        const std::vector<float> w = loadWeights(c.dir, c.ctrl, c.traj);
        const DevMem dw            = DevMem::from(w);
        const std::uint32_t n      = c.traj > 16 ? 64u : 300u;
        const std::vector<float> q = trajectories(c.traj, n);
        const DevMem dq            = DevMem::from(q);
        const std::size_t tfFloats = std::size_t(n) * (Dim + 1) * c.traj * 12;
        const std::size_t tFloats  = std::size_t(n) * S_ * c.traj * 3;
        const std::size_t cFloats  = std::size_t(n) * S_ * c.ctrl * 3;
        DevMem oTf(tfFloats * 4), oT(tFloats * 4), oC(cFloats * 4);
        DevMem nTf(tfFloats * 4), nT(tFloats * 4), nC(cFloats * 4);
        // old unfused: batchFkineTraj -> batchSphrCntrTraj -> batchGenControlPoints
        batchFkineTraj(dq.ptr(), oTf.ptr(), Dim, c.traj, n, nullptr);
        batchSphrCntrTraj(static_cast<float*>(oTf.p), static_cast<float*>(centers_->p), static_cast<float*>(oT.p),
            c.traj, n, static_cast<const int*>(links_->p), S_, Dim, nullptr);
        batchGenControlPoints(static_cast<float*>(oT.p), static_cast<float*>(oC.p), std::size_t(n) * S_, nullptr, c.ctrl,
            c.traj);
        // new unfused
        internal::trajectoryLinkPoses(internal::KinematicsRef{&robot_->kinematics(), table_.ptr()},
            dq.span<const float>(), c.traj, n, nTf.span<float>(), nullptr);
        curve::sphereCentersTraj(nTf.ptr(), centers_->ptr(), links_->ptr(), nT.ptr(), c.traj, n, S_, Dim, nullptr);
        curve::controlPoints(nT.ptr(), nC.ptr(), dw.ptr(), n * S_, c.ctrl, c.traj, nullptr);
        CK(cudaDeviceSynchronize());
        EXPECT_EQ(countWordDiffs(oC, nC, cFloats), 0u) << "unfused control points";
        EXPECT_EQ(countNonFinite(nC, cFloats), 0u);

        const bool fusedOk = curve::fusedCurvePointsSupported(Dim, c.traj, S_, c.ctrl);
        EXPECT_EQ(fusedOk, fkineSphrTrajSupported(Dim, c.traj, S_, c.ctrl)) << "support predicate (DOF <= 8 here)";
        if (fusedOk) {
            DevMem oF(cFloats * 4), nF(cFloats * 4);
            batchFkineSphrTraj(dq.ptr(), static_cast<const float*>(centers_->p), static_cast<const int*>(links_->p),
                static_cast<float*>(oF.p), Dim, c.traj, n, S_, c.ctrl, nullptr);
            curve::fusedCurvePoints(table_.ptr(), Dim, dq.ptr(), centers_->ptr(), links_->ptr(), dw.ptr(), nF.ptr(),
                c.traj, n, S_, c.ctrl, nullptr);
            CK(cudaDeviceSynchronize());
            EXPECT_EQ(countWordDiffs(oF, nF, cFloats), 0u) << "fused control points vs old fused";
            EXPECT_EQ(countWordDiffs(nF, nC, cFloats), 0u) << "fused vs unfused";
        }
    }
}

TEST_F(CurveKernels, SegmentObbsContainTheSegmentAndMatchGenCurveOBB) {
    for (const bool inject : {false, true})
    for (std::uint32_t degree : {1u, 2u}) {
        SCOPED_TRACE("degree " + std::to_string(degree) + (inject ? " with injected degenerate vertices" : " real points"));
        const std::uint32_t n   = 256;
        const std::uint32_t vpc = degree == 1 ? kT : 8u;
        const std::vector<float> q = trajectories(kT, n);
        const DevMem dq            = DevMem::from(q);
        std::vector<float> w(1, 0.f);
        if (degree == 2) {
            w = loadWeights("quad", 8, kT);
        }
        const DevMem dw = DevMem::from(w);
        const std::size_t vFloats = std::size_t(n) * S_ * vpc * 3;
        DevMem pts(vFloats * 4);
        curve::fusedCurvePoints(table_.ptr(), Dim, dq.ptr(), centers_->ptr(), links_->ptr(), dw.ptr(), pts.ptr(), kT, n,
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
        DevMem o(nSegs * sizeof(Obb)), nw(nSegs * sizeof(Obb));
        if (degree == 1) {
            genCurveOBB<1>(pts.ptr(), ds.ptr(), dr.ptr(), o.ptr(), nSegs, nullptr);
        } else {
            genCurveOBB<2>(pts.ptr(), ds.ptr(), dr.ptr(), o.ptr(), nSegs, nullptr);
        }
        curve::curveSegmentObbs(degree, pts.ptr(), ds.ptr(), dr.ptr(), nw.ptr(), nSegs, nullptr);
        CK(cudaDeviceSynchronize());

        // Containment: every box must hold the balls of radius r around the segment's control points (a linear
        // segment / quadratic B-spline segment lies in the convex hull of its control points, so this makes the
        // culling conservative). Non-finite boxes are skipped: they never test as separated (every SAT comparison
        // with NaN is false), so their segments are always kept.
        const auto hp = pts.download<float>();
        auto notContained = [&](const DevMem& boxes, std::size_t& nonFiniteBoxes) {
            const auto ob   = boxes.download<Obb>();
            std::size_t bad = 0;
            nonFiniteBoxes  = 0;
            for (std::uint32_t i = 0; i < nSegs; ++i) {
                const Obb& b   = ob[i];
                const float* f = reinterpret_cast<const float*>(&b);
                bool finite    = true;
                for (int k = 0; k < 15; ++k) finite &= std::isfinite(f[k]);
                if (!finite) {
                    ++nonFiniteBoxes;
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
                if (!in && bad < 3 && &boxes == &nw) {
                    ADD_FAILURE() << "segment " << i << " exceeds its box by " << excess;
                }
                bad += !in;
            }
            return bad;
        };
        std::size_t nonFiniteNew = 0, nonFiniteOld = 0;
        EXPECT_EQ(notContained(nw, nonFiniteNew), 0u) << "new segment OBBs must contain the segment";
        const std::size_t oldBad = notContained(o, nonFiniteOld);
        if (degree == 1) {
            EXPECT_EQ(countWordDiffs(o, nw, std::size_t(nSegs) * 15), 0u) << "linear: bit-identical to the old kernel";
            EXPECT_EQ(nonFiniteNew, 0u);
            EXPECT_EQ(oldBad, 0u);
        } else {
            // Old OBB_3p anchored two of its three cases at the wrong vertex and broke down for near-collinear points
            // (NaN / non-orthogonal frames); documents the fixed old bugs.
            EXPECT_GT(oldBad, 0u) << "the old quadratic OBBs were expected to miss part of some segments";
            EXPECT_EQ(nonFiniteNew, 0u);
            std::cout << "[ info ] " << (inject ? "with injected degenerate vertices" : "real control points")
                      << ": quadratic segment OBBs: old kernel " << oldBad << " of " << nSegs
                      << " finite boxes not containing their segment and " << nonFiniteOld
                      << " non-finite boxes; new kernel 0 and " << nonFiniteNew << "\n";
        }
    }
    EXPECT_THROW(curve::curveSegmentObbs(3, 0, 0, 0, 0, 1, nullptr), UnsupportedError);
}

// GENERATOR (runs only with RTCD_WRITE_REFERENCE=<dir>): curve_kernels.txt from the OLD kernels.
TEST_F(CurveKernels, WriteStoredReferences) {
    namespace sup  = rtcd::support;
    const auto dir = sup::referenceWriteDir();
    if (!dir) {
        GTEST_SKIP() << "set RTCD_WRITE_REFERENCE=<dir> to write the stored references";
    }
    sup::ReferenceWriter w(*dir / "curve_kernels.txt",
        "Outputs of the OLD curve kernels (RTCD/Robot/robot.cu, sweptVolume.cu; pandas.h masks, the Panda spheres of\n"
        "the RobotModel, configs/spline_weights/<degree>/normal), written by\n"
        "tests/backend_curve/test_curve_kernels_vs_old.cpp (RTCD_WRITE_REFERENCE). <key> <bytes> <FNV-1a of the bytes>.\n"
        "Inputs: the first n regression KNN trajectories (support::regressionInputs) with the given point count.");
    auto put = [&](const std::string& key, const DevMem& m, std::size_t bytes) {
        auto v = m.download<unsigned char>();
        v.resize(bytes);
        w.putHash(key, bytes, sup::fnv1a(v));
    };
    for (std::uint32_t n : {37u, 4096u}) {
        const std::vector<float> q = trajectories(kT, n);
        const DevMem dq            = DevMem::from(q);
        const std::size_t floats   = std::size_t(n) * S_ * kT * 3;
        DevMem o(floats * 4);
        batchFkineSphrTraj(dq.ptr(), static_cast<const float*>(centers_->p), static_cast<const int*>(links_->p),
            static_cast<float*>(o.p), Dim, kT, n, S_, 0, nullptr);
        CK(cudaDeviceSynchronize());
        put("fused_linear_points." + std::to_string(n), o, floats * 4);
    }
    {
        const std::uint32_t n      = 512;
        const DevMem dq            = DevMem::from(trajectories(kT, n));
        const std::size_t tfFloats = std::size_t(n) * (Dim + 1) * kT * 12;
        const std::size_t floats   = std::size_t(n) * S_ * kT * 3;
        DevMem oTf(tfFloats * 4), o(floats * 4);
        batchFkineTraj(dq.ptr(), oTf.ptr(), Dim, kT, n, nullptr);
        batchSphrCntrTraj(static_cast<float*>(oTf.p), static_cast<float*>(centers_->p), static_cast<float*>(o.p), kT, n,
            static_cast<const int*>(links_->p), S_, Dim, nullptr);
        CK(cudaDeviceSynchronize());
        put("unfused_link_poses.512", oTf, tfFloats * 4);
        put("unfused_sphere_centers.512", o, floats * 4);
    }
    struct Case {
        const char* dir;
        std::uint32_t ctrl, traj;
    };
    for (const Case c : {Case{"quad", 8, 16}, Case{"cubic", 8, 16}, Case{"quad", 4, 8}, Case{"quad", 6, 12},
             Case{"quad", 4, 128}}) {
        const std::string name     = std::string(c.dir) + "." + std::to_string(c.ctrl) + "_" + std::to_string(c.traj);
        const std::vector<float> wv = loadWeights(c.dir, c.ctrl, c.traj);
        const std::uint32_t n      = c.traj > 16 ? 64u : 300u;
        const DevMem dq            = DevMem::from(trajectories(c.traj, n));
        const std::size_t tfFloats = std::size_t(n) * (Dim + 1) * c.traj * 12;
        const std::size_t tFloats  = std::size_t(n) * S_ * c.traj * 3;
        const std::size_t cFloats  = std::size_t(n) * S_ * c.ctrl * 3;
        DevMem oTf(tfFloats * 4), oT(tFloats * 4), oC(cFloats * 4);
        batchFkineTraj(dq.ptr(), oTf.ptr(), Dim, c.traj, n, nullptr);
        batchSphrCntrTraj(static_cast<float*>(oTf.p), static_cast<float*>(centers_->p), static_cast<float*>(oT.p),
            c.traj, n, static_cast<const int*>(links_->p), S_, Dim, nullptr);
        batchGenControlPoints(static_cast<float*>(oT.p), static_cast<float*>(oC.p), std::size_t(n) * S_, nullptr, c.ctrl,
            c.traj);
        CK(cudaDeviceSynchronize());
        put("control_points." + name, oC, cFloats * 4);
        if (fkineSphrTrajSupported(Dim, c.traj, S_, c.ctrl)) {
            DevMem oF(cFloats * 4);
            batchFkineSphrTraj(dq.ptr(), static_cast<const float*>(centers_->p), static_cast<const int*>(links_->p),
                static_cast<float*>(oF.p), Dim, c.traj, n, S_, c.ctrl, nullptr);
            CK(cudaDeviceSynchronize());
            put("fused_control_points." + name, oF, cFloats * 4);
        }
    }
    // linear segment OBBs (the old quadratic OBBs were wrong in places and are not a reference)
    for (const bool inject : {false, true}) {
        const std::uint32_t n = 256, vpc = kT;
        const DevMem dq       = DevMem::from(trajectories(kT, n));
        const std::size_t vFloats = std::size_t(n) * S_ * vpc * 3;
        DevMem pts(vFloats * 4);
        batchFkineSphrTraj(dq.ptr(), static_cast<const float*>(centers_->p), static_cast<const int*>(links_->p),
            static_cast<float*>(pts.p), Dim, kT, n, S_, 0, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<std::uint32_t> segStart;
        std::vector<float> radii;
        const auto r = robot_->sphereRadii();
        for (std::uint32_t c = 0; c < n * S_; ++c) {
            for (std::uint32_t s = 0; s < vpc - 1; ++s) segStart.push_back(c * vpc + s);
            for (std::uint32_t v = 0; v < vpc; ++v) radii.push_back(r[c % S_]);
        }
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
        DevMem o(nSegs * sizeof(Obb));
        genCurveOBB<1>(pts.ptr(), ds.ptr(), dr.ptr(), o.ptr(), nSegs, nullptr);
        CK(cudaDeviceSynchronize());
        put(std::string("linear_segment_obbs.") + (inject ? "injected" : "real"), o, std::size_t(nSegs) * sizeof(Obb));
    }
    w.save();
}