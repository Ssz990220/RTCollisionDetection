#include "new_runner.h"

#include <rtcd/errors.h>

#if RTCD_HAVE_CHECKER
#include <rtcd/checker.h>
#include <rtcd/device_buffer.h>
#endif

namespace rtcd::sbs {

#if RTCD_HAVE_CHECKER
    namespace {
        class Runner final : public NewRunner {
        public:
            Runner(std::shared_ptr<Context> ctx, const CheckerConfig& cfg, std::shared_ptr<const RobotModel> robot,
                std::shared_ptr<const Scene> scene)
                : checker_(ctx, cfg, robot, scene) {
                layout_ = checker_.itemLayout();
                ctx->makeCurrent();
                const std::size_t cap = checker_.config().maxItemsPerCheck;
                items_.allocate(cap * layout_.floatsPerItem() * sizeof(float));
                counts_.allocate(cap * sizeof(std::uint32_t));
                // no checker_.warmup(): the old detectors are not warmed up either (their warmup() calls are
                // commented out in the old benchmarks); the runner's warm-up repetitions treat both alike.
            }
            const CheckerConfig& resolvedConfig() const override { return checker_.config(); }
            void upload(const float* items, std::size_t n) override {
                if (n == 0 || n > checker_.config().maxItemsPerCheck) {
                    throw InvalidArgument("new runner: item count " + std::to_string(n) + " outside [1, max_items_per_check]");
                }
                items_.upload(items, n * layout_.floatsPerItem() * sizeof(float));
                n_ = n;
            }
            void check() override {
                checker_.checkDevice(DeviceItems{items_.ptr(), n_, 0}, DeviceSpan<std::uint32_t>{counts_.ptr(), n_});
            }
            std::vector<std::uint32_t> hitCounts(std::size_t n) override {
                if (n > n_) {
                    throw InvalidArgument("new runner: hitCounts(n) beyond the last check");
                }
                std::vector<std::uint32_t> v(n);
                counts_.download(v.data(), n * sizeof(std::uint32_t));
                return v;
            }

        private:
            CollisionChecker checker_;
            ItemLayout layout_{};
            DeviceBuffer items_, counts_;
            std::size_t n_ = 0;
        };
    } // namespace

    bool newStackAvailable() { return true; }
    std::unique_ptr<NewRunner> createNewRunner(std::shared_ptr<Context> context, const CheckerConfig& config,
        std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene) {
        return std::make_unique<Runner>(std::move(context), config, std::move(robot), std::move(scene));
    }
#else
    bool newStackAvailable() { return false; }
    std::unique_ptr<NewRunner> createNewRunner(
        std::shared_ptr<Context>, const CheckerConfig&, std::shared_ptr<const RobotModel>, std::shared_ptr<const Scene>) {
        throw UnsupportedError("this build has no CollisionChecker (src/detector and the three backends are not "
                               "implemented yet, Wave 2): only the old stack can run (--old-only)");
    }
#endif

} // namespace rtcd::sbs
