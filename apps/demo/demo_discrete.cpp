// rtcd_demo_discrete: new-stack port of RTCD/Demo/DemoCollisionDetectorDisc.cpp (old demoIAS / demoRAY; the mode
// now comes from the setup: apps/demo/setups/demo_ias.json, demo_ray.json). Checks N random poses drawn uniformly
// within the robot's joint limits (the old demo used its cached Halton poses within the pandas.h bounds) for
// N = 1, 2, 4, ..., 4096 and prints the number of colliding poses and an indicative time per check. Options and
// flow: demo_common.h.
#include "demo_common.h"

using namespace rtcd;

namespace {
    std::vector<float> randomPoses(const RobotModel& robot, const ItemLayout& layout, std::size_t count, std::mt19937& gen) {
        const auto lo = robot.lowerLimits();
        const auto hi = robot.upperLimits();
        std::vector<float> out;
        out.reserve(count * layout.floatsPerItem());
        for (std::size_t i = 0; i < count; ++i) {
            for (std::uint32_t j = 0; j < layout.dof; ++j) {
                out.push_back(std::uniform_real_distribution<float>(lo[j], hi[j])(gen));
            }
        }
        return out;
    }
} // namespace

int main(int argc, char** argv) {
    const demo::Options options = demo::parseOptions(argc, argv, demo::Kind::Discrete);
    return demo::run(options, demo::Kind::Discrete, randomPoses);
}
