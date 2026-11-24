#pragma once
// How the regression app runs a check: through the public CollisionChecker (Wave 2 e), or through the test-support
// ReferenceBatchRunner + a backend factory (lets the Wave 2 backend agents run the goldens before the detector core
// exists, and cross-checks the core afterwards). Which engines exist depends on what is implemented in this build
// (RTCD_HAVE_* definitions from rtcd_new_stack_definitions, tests/support/CMakeLists.txt).

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

    bool checkerEngineAvailable();
    bool referenceEngineAvailable(Mode mode);

    // Both throw UnsupportedError when the engine is not available in this build.
    std::unique_ptr<Engine> makeCheckerEngine(std::shared_ptr<Context> ctx, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene);
    std::unique_ptr<Engine> makeReferenceEngine(std::shared_ptr<Context> ctx, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene);

} // namespace rtcd::regression
