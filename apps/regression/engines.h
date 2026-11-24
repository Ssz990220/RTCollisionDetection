#pragma once
// How the regression app runs a check: through the public CollisionChecker, or through the test-support
// ReferenceBatchRunner + a backend factory (runs a backend without the detector core; cross-checks the core).

#include <rtcd/config.h>
#include <rtcd/context.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rtcd::regression {

    class Engine {
    public:
        virtual ~Engine() = default;
        virtual std::string name() const = 0;
        // Per-item hit counts of one check of the first n packed items.
        virtual std::vector<std::uint32_t> run(std::span<const float> items, std::size_t n) = 0;
        virtual const CheckerConfig& resolvedConfig() const = 0;
    };

    std::unique_ptr<Engine> makeCheckerEngine(std::shared_ptr<Context> ctx, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene);
    std::unique_ptr<Engine> makeReferenceEngine(std::shared_ptr<Context> ctx, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene);

} // namespace rtcd::regression
