// src/detector: CollisionChecker (rtcd/checker.h) -- the public facade over DetectorCore.
//
// Construction-time checks that need the robot / scene and are not backend specific live here; the backend
// checks its own (e.g. Ray launch size, spline weight files). Everything else forwards to the core.
#include <rtcd/checker.h>

#include "detector/backend_factory.h"
#include "detector/detector_core.h"

#include <rtcd/errors.h>

#include <string>
#include <utility>

namespace rtcd {

    namespace {
        std::string modeName(Mode m) {
            switch (m) {
            case Mode::Mesh: return "mesh";
            case Mode::Ray: return "ray";
            case Mode::LinearCurve: return "linear_curve";
            case Mode::QuadraticCurve: return "quadratic_curve";
            case Mode::CubicCurve: return "cubic_curve";
            }
            return "?";
        }
    } // namespace

    struct CollisionChecker::Impl {
        std::shared_ptr<const RobotModel> robot;
        std::shared_ptr<const Scene> scene;
        detail::DetectorCore core;

        Impl(detail::DetectorCoreDesc desc)
            : robot(desc.robot), scene(desc.scene), core(std::move(desc)) {}
    };

    CollisionChecker::CollisionChecker(std::shared_ptr<Context> context, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene) {
        if (!context) {
            throw InvalidArgument("CollisionChecker: null Context");
        }
        if (!robot) {
            throw InvalidArgument("CollisionChecker: null RobotModel");
        }
        if (!scene) {
            throw InvalidArgument("CollisionChecker: null Scene");
        }
        if (scene->context().get() != context.get()) {
            throw InvalidArgument("CollisionChecker: the Scene was built on a different Context than the checker's");
        }
        detail::DetectorCoreDesc desc;
        desc.config = config.resolved(); // validate() + Auto values replaced; ConfigError lists every problem
        if (isContinuous(desc.config.mode) && !robot->hasSpheres()) {
            throw ConfigError("checker config: mode = " + modeName(desc.config.mode) + " needs robot sphere geometry, "
                                  + "but robot '" + robot->name() + "' has no spheres (robot.geometry.spheres)",
                "mode");
        }
        desc.context = std::move(context);
        desc.dof     = robot->dof();
        desc.lowerLimits.assign(robot->lowerLimits().begin(), robot->lowerLimits().end());
        desc.upperLimits.assign(robot->upperLimits().begin(), robot->upperLimits().end());
        desc.robot         = std::move(robot);
        desc.scene         = std::move(scene);
        desc.createBackend = &detail::createBackendForMode;
        impl_              = std::make_unique<Impl>(std::move(desc));
    }

    CollisionChecker::~CollisionChecker() = default;
    CollisionChecker::CollisionChecker(CollisionChecker&&) noexcept            = default;
    CollisionChecker& CollisionChecker::operator=(CollisionChecker&&) noexcept = default;

    const CheckerConfig& CollisionChecker::config() const noexcept { return impl_->core.config(); }
    ItemLayout CollisionChecker::itemLayout() const noexcept { return impl_->core.itemLayout(); }
    const RobotModel& CollisionChecker::robot() const noexcept { return *impl_->robot; }
    const Scene& CollisionChecker::scene() const noexcept { return *impl_->scene; }

    CheckResult CollisionChecker::check(const HostItems& items, ResultDetail detail) {
        CheckResult r;
        if (items.count == 0) {
            impl_->core.checkHost(items, {}, {});
            return r;
        }
        impl_->core.validateHostItems(items); // before allocating the result vectors
        r.collides.resize(items.count);
        if (detail == ResultDetail::FlagsAndHitCounts) {
            r.hitCounts.resize(items.count);
        }
        impl_->core.checkHost(items, r.collides, r.hitCounts);
        return r;
    }

    void CollisionChecker::check(
        const HostItems& items, std::span<std::uint8_t> collides, std::span<std::uint32_t> hitCounts) {
        impl_->core.checkHost(items, collides, hitCounts);
    }

    void CollisionChecker::checkDevice(const DeviceItems& items, DeviceSpan<std::uint32_t> hitCounts) {
        impl_->core.checkDevice(items, hitCounts);
    }

    void CollisionChecker::warmup() { impl_->core.warmup(); }

    CheckStats CollisionChecker::lastStats() const noexcept { return impl_->core.lastStats(); }

} // namespace rtcd
