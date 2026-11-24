// rtcd_example: the smallest complete use of the public API (README.md quotes this file).
//
//   rtcd_example [setup.json]      default: configs/setups/regression_ias.json, relative to the working directory
//
// Loads a setup file (checker config + scene + robot), builds everything, checks eight items spread between the
// robot's lower and upper joint limits and prints one collision flag per item. Any setup works: for the discrete
// modes (mesh, ray) an item is one pose, for the curve modes a trajectory of curve.trajectory_points poses.
#include <rtcd/rtcd.h>

#include <cstdio>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
    try {
        // 1. Checker config, scene description and robot description; relative paths in the JSON files are
        //    resolved against the file that contains them.
        const rtcd::Setup setup = rtcd::loadSetup(argc > 1 ? argv[1] : "configs/setups/regression_ias.json");

        // 2. GPU context, robot (URDF + link meshes [+ spheres]), scene (obstacle meshes on the GPU), checker.
        std::shared_ptr<rtcd::Context> context        = rtcd::Context::create();
        std::shared_ptr<const rtcd::RobotModel> robot = rtcd::RobotModel::load(setup.robot);
        std::shared_ptr<const rtcd::Scene> scene      = std::make_shared<rtcd::Scene>(context, setup.scene);
        rtcd::CollisionChecker checker(context, setup.checker, robot, scene);

        // 3. Items, packed as [item][waypoint][joint] float32 radians. Item i covers the stretch
        //    [i, i + 1] / numItems of the straight line from the lower to the upper joint limits (a discrete
        //    item is the midpoint of its stretch).
        const rtcd::ItemLayout layout = checker.itemLayout(); // dof, pointsPerItem (1 for discrete modes)
        const std::size_t numItems    = 8;
        std::vector<float> items;
        for (std::size_t i = 0; i < numItems; ++i) {
            for (std::uint32_t p = 0; p < layout.pointsPerItem; ++p) {
                const float u = layout.pointsPerItem > 1 ? float(p) / float(layout.pointsPerItem - 1) : 0.5f;
                const float t = (float(i) + u) / float(numItems);
                for (std::uint32_t j = 0; j < layout.dof; ++j) {
                    const float lo = robot->lowerLimits()[j], hi = robot->upperLimits()[j];
                    items.push_back(lo + t * (hi - lo));
                }
            }
        }

        // 4. Check (blocking) and print the flags.
        const rtcd::CheckResult result = checker.check(rtcd::HostItems{items, numItems});
        for (std::size_t i = 0; i < numItems; ++i) {
            std::printf("item %zu: %s\n", i, result.collides[i] ? "collision" : "free");
        }
        std::printf("%zu of %zu items collide (robot '%s', dof %u, scene '%s')\n", result.numColliding(), numItems,
            robot->name().c_str(), robot->dof(), setup.scene.name.c_str());
        return 0;
    } catch (const rtcd::Error& e) { // ConfigError, IoError, CapacityError, UnsupportedError, CudaError, ...
        std::fprintf(stderr, "rtcd_example: %s\n", e.what());
        return 1;
    }
}
