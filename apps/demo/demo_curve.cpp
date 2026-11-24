// rtcd_demo_curve: new-stack port of RTCD/Demo/DemoCollisionDetectorCurve.cpp (old demoLinContinuous /
// demoQuadContinuous; the mode now comes from the setup: configs/setups/demo_linear.json, demo_quad.json).
// Checks N short straight-line joint-space trajectories for N = 1, 2, 4, ..., 4096 and prints the number of
// colliding trajectories and an indicative time per check. Options and flow: demo_common.h.
//
// Trajectories: start pose uniform within the joint limits, end pose = start + a per-joint step uniform in
// [-step, step] * (joint range), clamped to the limits; curve.trajectory_points waypoints linearly interpolated
// from start to end (both included). The old demo used straight lines from cached samples to their 4 nearest
// neighbours (genCSpaceKNNTraj); local random steps play the same role without the data/ cache.
#include "demo_common.h"

#include <algorithm>

using namespace rtcd;

namespace {
    std::vector<float> randomTrajectories(
        const RobotModel& robot, const ItemLayout& layout, std::size_t count, std::mt19937& gen, float step) {
        const auto lo           = robot.lowerLimits();
        const auto hi           = robot.upperLimits();
        const std::uint32_t dof = layout.dof, pts = layout.pointsPerItem;
        std::vector<float> start(dof), end(dof), out;
        out.reserve(count * layout.floatsPerItem());
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
        for (std::size_t i = 0; i < count; ++i) {
            for (std::uint32_t j = 0; j < dof; ++j) {
                start[j] = std::uniform_real_distribution<float>(lo[j], hi[j])(gen);
                end[j]   = std::clamp(start[j] + unit(gen) * step * (hi[j] - lo[j]), lo[j], hi[j]);
            }
            for (std::uint32_t p = 0; p < pts; ++p) {
                const float t = pts > 1 ? float(p) / float(pts - 1) : 0.0f;
                for (std::uint32_t j = 0; j < dof; ++j) {
                    out.push_back(start[j] + (end[j] - start[j]) * t);
                }
            }
        }
        return out;
    }
} // namespace

int main(int argc, char** argv) {
    const demo::Options options = demo::parseOptions(argc, argv, demo::Kind::Curve);
    return demo::run(options, demo::Kind::Curve,
        [step = options.step](const RobotModel& r, const ItemLayout& l, std::size_t n, std::mt19937& g) {
            return randomTrajectories(r, l, n, g, step);
        });
}
