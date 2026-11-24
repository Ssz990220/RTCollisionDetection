#pragma once
// Inputs of the stored FK references (tests/data/reference/kinematics_fk.txt, kinematics_fk_poses.bin,
// kinematics_fk_traj.bin): shared by the test that compares with them (test_kinematics_kernels.cpp) and the
// generator that wrote them from the old kernels (test_kinematics_kernels_vs_old.cpp, removed with the old stack).
// Everything here is deterministic and platform independent (no <random> distributions).
#include <rtcd/limits.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtcd::fk_reference {

    constexpr std::size_t kPoses     = 4096; // tests/data/motions/Panda4096.bin
    constexpr std::size_t kDof       = 7;
    constexpr std::size_t kLinks     = 8;
    constexpr std::uint32_t kTrajPts = 16; // regression trajectories (first 4096 KNN trajectories)

    // Poses whose link poses are stored raw: every 64th regression pose (64 poses).
    inline std::vector<std::size_t> rawPoseIndices() {
        std::vector<std::size_t> v;
        for (std::size_t i = 0; i < kPoses; i += 64) v.push_back(i);
        return v;
    }
    // Trajectories whose link poses are stored raw: every 512th regression trajectory (8 trajectories).
    inline std::vector<std::size_t> rawTrajectoryIndices() {
        std::vector<std::size_t> v;
        for (std::size_t i = 0; i < kPoses; i += 512) v.push_back(i);
        return v;
    }

    // Plausible OBB templates (one per link) from a fixed LCG: centers in [-0.2, 0.2], half sizes in [0.05, 0.25],
    // axes entries in [-1, 1] (not orthonormal: the kernels only transform them).
    inline std::vector<Obb> obbTemplates(std::uint32_t seed) {
        std::uint32_t x = seed;
        auto u          = [&x]() { // uniform in [-1, 1), exactly representable
            x = x * 1664525u + 1013904223u;
            return static_cast<float>(x >> 8) * (1.0f / 8388608.0f) - 1.0f;
        };
        std::vector<Obb> v(kLinks);
        for (auto& o : v) {
            for (float& f : o.center) f = 0.2f * u();
            for (float& f : o.halfSize) f = 0.05f + 0.1f * (u() + 1.f);
            for (auto& ax : o.axes)
                for (float& f : ax) f = u();
        }
        return v;
    }

    // tolerance of the raw-value comparison (the hashes are the exact check)
    constexpr float kRawTolerance = 1e-5f;

} // namespace rtcd::fk_reference
