// new_unit, OLD-STACK PARITY TEST (deleted with the old stack): support/inputs.h and support/golden.h give exactly
// what the old harness computes from the same cache files:
//   * discrete inputs == RTCD randomMotions(NITEMS) (ccRegression.cpp loadInputs, non-curve branch);
//   * curve inputs    == the first NITEMS trajectories of genCSpaceKNNTraj<Dim, NITEMS, 4> (curve branch), for
//                        the regression's 16 waypoints and the benchmark variants 8 / 32;
//   * benchmark pools == randomMotions(8192) / genCSpaceKNNTraj<Dim, 8192, 4> (RTCD/Benchmark/*);
//   * goldens decode as ccRegression.cpp reads them.
// Byte comparison (memcmp), no tolerance. The old functions would generate and write missing caches, so every
// file they read is asserted to exist first. No GPU work runs (the cache path of the old code is host only).
// The old code reads only PROJECT_BASE_DIR/data (the local, gitignored folder that generated the reference data);
// the new code reads the committed copies in tests/data. Without the local folder every test here is skipped.
#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <Utils/Test/testUtils.h> // old: readBin, randomMotions, genCSpaceKNNTraj (defines the OptiX table too)

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;
namespace sup = rtcd::support;

namespace {
    constexpr size_t kDim = 7;
    // Bounds and weights of the old harness (pandas.h lowerBound / upperBound, ccRegression weight). The bounds are
    // only used by the old code to GENERATE a missing cache, which the existence asserts below rule out.
    constexpr std::array<float, kDim> kLower{-2.8973f, -1.7628f, -2.8973f, -3.0718f, -2.8973f, -0.0175f, -2.8973f};
    constexpr std::array<float, kDim> kUpper{2.8973f, 1.7628f, 2.8973f, 0.0698f, 2.8973f, 3.7525f, 2.8973f};
    constexpr std::array<float, kDim> kWeight{1, 1, 1, 1, 1, 1, 1};

    // The old code reads PROJECT_BASE_DIR/data (config.h of this build); the new code the data dir it finds.
    fs::path oldData() { return fs::path(PROJECT_BASE_DIR) / "data"; }
    fs::path oldMotions() { return oldData() / "motions"; }
    fs::path newMotions() { return sup::dataDir() / "motions"; }

#define SKIP_WITHOUT_OLD_DATA()                                                                                    \
    if (!fs::is_directory(oldMotions())) {                                                                         \
        GTEST_SKIP() << "the old stack's local data folder " << oldData() << " is not present";                  \
    }

    std::vector<char> fileBytes(const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

    void requireCache(size_t n, bool knn) {
        ASSERT_TRUE(fs::exists(oldMotions() / ("Panda" + std::to_string(n) + ".bin")));
        if (knn) {
            const fs::path d = oldMotions() / "Panda" / "1111111";
            ASSERT_TRUE(fs::exists(d / (std::to_string(n) + "_knn4_idx.bin")));
            ASSERT_TRUE(fs::exists(d / (std::to_string(n) + "_knn4_dis.bin")));
        }
    }

    template <size_t N>
    std::vector<std::array<float, kDim>> oldPoses() {
        std::vector<std::array<float, kDim>> poses(N);
        std::array<float, kDim> up = kUpper, lo = kLower;
        randomMotions(N, poses, up, lo, "Panda");
        return poses;
    }

    template <int N>
    std::vector<std::array<float, kDim>> oldTraj(int nTrajPts) {
        std::vector<std::array<float, kDim>> traj(size_t(N) * 4 * nTrajPts);
        genCSpaceKNNTraj<kDim, N, 4>(traj, nTrajPts, kUpper, kLower, "Panda", kWeight);
        return traj;
    }
} // namespace

// The committed reference inputs (tests/data) are byte copies of the old local cache.
TEST(VsOld, CommittedDataEqualsTheLocalCache) {
    SKIP_WITHOUT_OLD_DATA();
    for (const char* rel : {"motions/Panda4096.bin", "motions/Panda8192.bin", "motions/Panda/1111111/4096_knn4_idx.bin",
             "motions/Panda/1111111/8192_knn4_idx.bin", "CCTest/golden_ias.bin", "CCTest/golden_ray.bin",
             "CCTest/golden_linear.bin", "CCTest/golden_quad.bin"}) {
        ASSERT_TRUE(fs::exists(oldData() / rel)) << rel;
        ASSERT_TRUE(fs::exists(sup::dataDir() / rel)) << rel;
        const auto a = fileBytes(oldData() / rel);
        EXPECT_FALSE(a.empty()) << rel;
        EXPECT_EQ(a, fileBytes(sup::dataDir() / rel)) << rel;
    }
}

TEST(VsOld, WeightString) {
    EXPECT_EQ(concatFloatsAsInts(kWeight), sup::knnWeightString(kWeight));
}

TEST(VsOld, DiscreteRegressionInputs) {
    SKIP_WITHOUT_OLD_DATA();
    requireCache(4096, false);
    const auto old              = oldPoses<4096>();
    const std::vector<float> nw = sup::regressionInputs(newMotions(), 1);
    ASSERT_EQ(nw.size(), old.size() * kDim);
    EXPECT_EQ(std::memcmp(nw.data(), old.data(), nw.size() * sizeof(float)), 0);
}

TEST(VsOld, CurveRegressionInputs) {
    SKIP_WITHOUT_OLD_DATA();
    requireCache(4096, true);
    for (int nT : {16, 8, 32}) {
        const auto old              = oldTraj<4096>(nT);
        const std::vector<float> nw = sup::regressionInputs(newMotions(), static_cast<std::uint32_t>(nT));
        // ccRegression copies the first NITEMS trajectories
        ASSERT_EQ(nw.size(), size_t(4096) * nT * kDim);
        ASSERT_GE(old.size() * kDim, nw.size());
        EXPECT_EQ(std::memcmp(nw.data(), old.data(), nw.size() * sizeof(float)), 0) << "trajectory points " << nT;
    }
}

TEST(VsOld, BenchmarkPools) {
    SKIP_WITHOUT_OLD_DATA();
    requireCache(8192, true);
    const auto oldP              = oldPoses<8192>();
    const std::vector<float> nwP = sup::benchmarkPosePool(newMotions());
    ASSERT_EQ(nwP.size(), oldP.size() * kDim);
    EXPECT_EQ(std::memcmp(nwP.data(), oldP.data(), nwP.size() * sizeof(float)), 0);

    // genCSpaceKNNTraj<Dim, 8192, 4> would also read the 268 MB distance file element by element; the trajectory
    // arithmetic is already compared above (4096), so compare the 8192 neighbour indices it would use instead.
    std::vector<int> oldIdx;
    readBin((oldMotions() / "Panda" / "1111111" / "8192_knn4_idx.bin").string(), oldIdx);
    const std::vector<std::int32_t> nwIdx = sup::loadKnnIndices(newMotions(), "Panda", "1111111", 8192, 4);
    ASSERT_EQ(oldIdx.size(), nwIdx.size());
    EXPECT_EQ(std::memcmp(oldIdx.data(), nwIdx.data(), nwIdx.size() * sizeof(int)), 0);
    EXPECT_EQ(sup::benchmarkTrajectoryPool(newMotions(), 16).size(), size_t(8192) * 4 * 16 * kDim);
}

TEST(VsOld, GoldensDecodeIdentically) {
    SKIP_WITHOUT_OLD_DATA();
    for (const char* m : {"ias", "ray", "linear", "quad"}) {
        const fs::path p = fs::path(PROJECT_BASE_DIR) / "data" / "CCTest" / (std::string("golden_") + m + ".bin");
        ASSERT_TRUE(fs::exists(p)) << p;
        std::vector<uint8_t> old(4096); // ccRegression: Flags golden(NITEMS); read NITEMS bytes
        std::ifstream(p, std::ios::binary).read(reinterpret_cast<char*>(old.data()), old.size());
        const sup::Flags nw = sup::readFlagsFile(p, 4096);
        EXPECT_EQ(nw, old) << m;
    }
}
