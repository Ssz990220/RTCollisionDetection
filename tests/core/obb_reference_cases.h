#pragma once
// Deterministic, platform-independent OBB inputs of the stored obbOverlapCounts reference
// (tests/data/reference/obb_overlap.txt): shared by test_core_obb_reference.cpp (compares) and the generator in
// test_core_legacy_parity.cpp (old RTCD::BBCDRobotMask; removed with the old stack). No <random> distributions
// (their output is implementation-defined); only + - * / and sqrt, which IEEE-754 rounds exactly.
#include <rtcd/limits.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rtcd::obb_reference {

    struct Lcg {
        std::uint32_t x;
        float unit() { // [0, 1), 24-bit
            x = x * 1664525u + 1013904223u;
            return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
        }
        float range(float lo, float hi) { return lo + (hi - lo) * unit(); }
    };

    inline Obb randomObb(Lcg& g, float centerRange, float maxHalf) {
        Obb o{};
        for (int i = 0; i < 3; ++i) {
            o.center[i]   = g.range(-centerRange, centerRange);
            o.halfSize[i] = g.range(0.0f, maxHalf);
        }
        float q[4] = {g.range(-1, 1), g.range(-1, 1), g.range(-1, 1), g.range(-1, 1)};
        float n    = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (n < 1e-3f) {
            q[0] = 1;
            q[1] = q[2] = q[3] = 0;
            n                  = 1;
        }
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

    // One stored entry: obbOverlapCounts(robot, scene) for every run, the counts of all runs concatenated.
    struct Run {
        std::vector<Obb> robot, scene;
    };
    struct Case {
        std::string key;
        std::vector<Run> runs;
    };

    // Robot-link-like sets (every 9th box collapsed to a point, like a skipped root link) against scene sets of the
    // sizes of the real uses and larger ones that exercise every lane of the 32 x 8 block; then near-touching pairs
    // separated along a face axis by (ra + rb) * (1 + eps), |eps| <= 4e-7, where the decision rests on the last bits.
    inline std::vector<Case> cases() {
        std::vector<Case> out;
        Lcg g{20261123u};
        for (std::size_t nB : {1u, 3u, 8u, 9u, 64u, 257u}) {
            for (std::size_t nA : {1u, 31u, 33u, 1024u * 9u, 4096u * 8u + 5u}) {
                Run r{std::vector<Obb>(nA), std::vector<Obb>(nB)};
                for (auto& o : r.robot) o = randomObb(g, 1.0f, 0.3f);
                for (std::size_t i = 0; i < nA; i += 9) r.robot[i].halfSize[0] = r.robot[i].halfSize[1] = r.robot[i].halfSize[2] = 0.0f;
                for (auto& o : r.scene) o = randomObb(g, 1.0f, 0.5f);
                out.push_back(Case{"random." + std::to_string(nA) + "x" + std::to_string(nB), {std::move(r)}});
            }
        }
        Lcg t{99u};
        Run th;
        for (std::size_t k = 0; k < 20000; ++k) {
            const Obb p      = randomObb(t, 1.0f, 0.3f);
            Obb q            = p;
            const float dist = (p.halfSize[0] + q.halfSize[0]) * (1.0f + t.range(-4e-7f, 4e-7f));
            for (int r = 0; r < 3; ++r) q.center[r] = p.center[r] + dist * p.axes[0][r];
            th.robot.push_back(p);
            if (k < 256) th.scene.push_back(q);
        }
        // and the first 300 boxes, each against its own near-touching partner only (300 runs of 1 x 1)
        Case pairs{"threshold.pairs300", {}};
        for (std::size_t k = 0; k < 300; ++k) {
            const Obb p      = th.robot[k];
            Obb q            = p;
            const float dist = (p.halfSize[0] + q.halfSize[0]) * (1.0f + t.range(-4e-7f, 4e-7f));
            for (int r = 0; r < 3; ++r) q.center[r] = p.center[r] + dist * p.axes[0][r];
            pairs.runs.push_back(Run{{p}, {q}});
        }
        out.push_back(Case{"threshold.20000x256", {std::move(th)}});
        out.push_back(std::move(pairs));
        return out;
    }

} // namespace rtcd::obb_reference
