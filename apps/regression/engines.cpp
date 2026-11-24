#include "engines.h"

#include <rtcd/errors.h>

#if RTCD_HAVE_CHECKER
#include <rtcd/checker.h>
#endif
#if RTCD_HAVE_BACKEND_MESH || RTCD_HAVE_BACKEND_RAY || RTCD_HAVE_BACKEND_CURVE
#define RTCD_REGRESSION_HAVE_REFERENCE 1
#include <rtcd/backend.h>
#include <rtcd/internal/reference_executor.h>
#else
#define RTCD_REGRESSION_HAVE_REFERENCE 0
#endif

namespace rtcd::regression {

    bool checkerEngineAvailable() { return RTCD_HAVE_CHECKER != 0; }

    bool referenceEngineAvailable(Mode mode) {
        switch (mode) {
        case Mode::Mesh: return RTCD_HAVE_BACKEND_MESH != 0;
        case Mode::Ray: return RTCD_HAVE_BACKEND_RAY != 0;
        default: return RTCD_HAVE_BACKEND_CURVE != 0;
        }
    }

#if RTCD_HAVE_CHECKER
    namespace {
        class CheckerEngine final : public Engine {
        public:
            CheckerEngine(std::shared_ptr<Context> ctx, const CheckerConfig& cfg, std::shared_ptr<const RobotModel> robot,
                std::shared_ptr<const Scene> scene)
                : checker_(std::move(ctx), cfg, std::move(robot), std::move(scene)) {}
            std::string name() const override { return "checker"; }
            const CheckerConfig& resolvedConfig() const override { return checker_.config(); }
            std::vector<std::uint32_t> run(std::span<const float> items, std::size_t n) override {
                CheckResult r = checker_.check(HostItems{items, n, 0}, ResultDetail::FlagsAndHitCounts);
                return std::move(r.hitCounts);
            }

        private:
            CollisionChecker checker_;
        };
    } // namespace

    std::unique_ptr<Engine> makeCheckerEngine(std::shared_ptr<Context> ctx, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene) {
        return std::make_unique<CheckerEngine>(std::move(ctx), config, std::move(robot), std::move(scene));
    }
#else
    std::unique_ptr<Engine> makeCheckerEngine(
        std::shared_ptr<Context>, const CheckerConfig&, std::shared_ptr<const RobotModel>, std::shared_ptr<const Scene>) {
        throw UnsupportedError("engine 'checker' is not available in this build (CollisionChecker / src/detector and "
                               "the three backends are not all implemented)");
    }
#endif

#if RTCD_REGRESSION_HAVE_REFERENCE
    namespace {
        std::unique_ptr<Backend> createBackend(const BackendCreateInfo& info) {
            switch (info.config->mode) {
#if RTCD_HAVE_BACKEND_MESH
            case Mode::Mesh: return createMeshBackend(info);
#endif
#if RTCD_HAVE_BACKEND_RAY
            case Mode::Ray: return createRayBackend(info);
#endif
#if RTCD_HAVE_BACKEND_CURVE
            case Mode::LinearCurve:
            case Mode::QuadraticCurve:
            case Mode::CubicCurve: return createCurveBackend(info);
#endif
            default: break;
            }
            throw UnsupportedError("engine 'reference': the backend of this mode is not implemented in this build");
        }

        // Members in construction order; destruction (reverse) releases runner, executor and backend before the
        // streams go back to the Context.
        class ReferenceEngine final : public Engine {
        public:
            ReferenceEngine(std::shared_ptr<Context> ctx, const CheckerConfig& cfg, std::shared_ptr<const RobotModel> robot,
                std::shared_ptr<const Scene> scene)
                : ctx_(ctx), config_(cfg.resolved()), lease_(ctx->acquireStreams(config_.numStreams)) {
                for (std::uint32_t i = 0; i < lease_.size(); ++i) {
                    streams_.push_back(StreamContext{i, lease_[i]});
                }
                BackendCreateInfo info;
                info.context = ctx_;
                info.config  = &config_;
                info.robot   = robot;
                info.scene   = scene;
                info.streams = streams_;
                backend_     = createBackend(info);
                exec_        = std::make_unique<internal::ReferenceExecutor>(ctx_, backend_->programs(), config_.optix);
                runner_      = std::make_unique<internal::ReferenceBatchRunner>(
                    ctx_, *backend_, *exec_, config_, streams_, robot->dof());
            }
            std::string name() const override { return "reference"; }
            const CheckerConfig& resolvedConfig() const override { return config_; }
            std::vector<std::uint32_t> run(std::span<const float> items, std::size_t n) override {
                return runner_->runAll(items, static_cast<std::uint32_t>(n));
            }

        private:
            std::shared_ptr<Context> ctx_;
            CheckerConfig config_;
            StreamLease lease_;
            std::vector<StreamContext> streams_;
            std::unique_ptr<Backend> backend_;
            std::unique_ptr<internal::ReferenceExecutor> exec_;
            std::unique_ptr<internal::ReferenceBatchRunner> runner_;
        };
    } // namespace

    std::unique_ptr<Engine> makeReferenceEngine(std::shared_ptr<Context> ctx, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene) {
        if (!referenceEngineAvailable(config.mode)) {
            throw UnsupportedError("engine 'reference': the backend of this mode is not implemented in this build");
        }
        return std::make_unique<ReferenceEngine>(std::move(ctx), config, std::move(robot), std::move(scene));
    }
#else
    std::unique_ptr<Engine> makeReferenceEngine(
        std::shared_ptr<Context>, const CheckerConfig&, std::shared_ptr<const RobotModel>, std::shared_ptr<const Scene>) {
        throw UnsupportedError("engine 'reference' is not available in this build (no backend is implemented)");
    }
#endif

} // namespace rtcd::regression
