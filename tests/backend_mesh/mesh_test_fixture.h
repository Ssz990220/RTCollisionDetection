#pragma once
// Shared fixture of the mesh-backend tests: the IAS regression setup (configs/setups/regression_ias.json), its
// robot and scene on one Context, and a helper that builds backend + ReferenceExecutor + ReferenceBatchRunner for
// a config variant.
#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <rtcd/backend.h>
#include <rtcd/context.h>
#include <rtcd/internal/reference_executor.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace rtcd::mesh_test {

    // Backend + executor + runner over a leased set of streams (members in destruction-safe order).
    struct Harness {
        CheckerConfig config; // resolved
        StreamLease lease;
        std::vector<StreamContext> streams;
        std::unique_ptr<Backend> backend;
        std::unique_ptr<internal::ReferenceExecutor> exec;
        std::unique_ptr<internal::ReferenceBatchRunner> runner;
    };

    class MeshBackendTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            setup_ = std::make_unique<rtcd::Setup>(loadSetup(support::configsDir() / "setups" / "regression_ias.json"));
            ctx_   = Context::create();
            robot_ = RobotModel::load(setup_->robot, ctx_->logger());
            scene_ = std::make_shared<const Scene>(ctx_, setup_->scene);
            inputs_ = support::regressionInputs(support::dataDir() / "motions", 1);
            golden_ = support::readFlagsFile(*support::goldenPath(support::dataDir(), Mode::Mesh), 4096);
        }
        static void TearDownTestSuite() {
            scene_.reset();
            robot_.reset();
            ctx_.reset();
            setup_.reset();
        }

        // Config of the setup with `overrides` applied (applyOverrides syntax), resolved.
        static CheckerConfig config(const std::string& overrides = "") {
            CheckerConfig c = setup_->checker;
            if (!overrides.empty()) {
                applyOverrides(c, overrides);
            }
            return c.resolved();
        }

        static BackendCreateInfo createInfo(const Harness& h, std::shared_ptr<const Scene> scene = nullptr) {
            BackendCreateInfo info;
            info.context = ctx_;
            info.config  = &h.config;
            info.robot   = robot_;
            info.scene   = scene ? std::move(scene) : scene_;
            info.streams = h.streams;
            return info;
        }

        static std::unique_ptr<Harness> makeHarness(
            const std::string& overrides = "", std::shared_ptr<const Scene> scene = nullptr, bool withRunner = true) {
            auto h    = std::make_unique<Harness>();
            h->config = config(overrides);
            h->lease  = ctx_->acquireStreams(h->config.numStreams);
            for (std::uint32_t i = 0; i < h->lease.size(); ++i) {
                h->streams.push_back(StreamContext{i, h->lease[i]});
            }
            h->backend = createMeshBackend(createInfo(*h, std::move(scene)));
            if (withRunner) {
                h->exec   = std::make_unique<internal::ReferenceExecutor>(ctx_, h->backend->programs(), h->config.optix);
                h->runner = std::make_unique<internal::ReferenceBatchRunner>(
                    ctx_, *h->backend, *h->exec, h->config, h->streams, robot_->dof());
            }
            return h;
        }

        static inline std::unique_ptr<rtcd::Setup> setup_;
        static inline std::shared_ptr<Context> ctx_;
        static inline std::shared_ptr<const RobotModel> robot_;
        static inline std::shared_ptr<const Scene> scene_;
        static inline std::vector<float> inputs_;
        static inline support::Flags golden_;
    };

} // namespace rtcd::mesh_test
