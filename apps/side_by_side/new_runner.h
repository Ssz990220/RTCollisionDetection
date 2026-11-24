#pragma once
// New-stack half of the side-by-side runner: one CollisionChecker timed on its device path (checkDevice), the
// equivalent of the old timed region (per-batch work + result copies + final sync, without host upload /
// download). Without the detector (Wave 2) in this build, newStackAvailable() is false and createNewRunner throws.

#include <rtcd/config.h>
#include <rtcd/context.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rtcd::sbs {

    class NewRunner {
    public:
        virtual ~NewRunner() = default;
        virtual const CheckerConfig& resolvedConfig() const = 0;
        virtual void upload(const float* items, std::size_t n) = 0; // untimed: host -> caller-owned device buffer
        virtual void check() = 0;                                   // timed: checkDevice
        virtual std::vector<std::uint32_t> hitCounts(std::size_t n) = 0;
    };

    bool newStackAvailable();
    std::unique_ptr<NewRunner> createNewRunner(std::shared_ptr<Context> context, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene);

} // namespace rtcd::sbs
