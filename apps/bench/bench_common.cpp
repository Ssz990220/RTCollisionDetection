#include "bench_common.h"

#include <support/options.h>
#include <support/paths.h>

#include <rtcd/errors.h>
#include <rtcd/json_config.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <optional>

namespace fs = std::filesystem;

namespace rtcd::bench {

    namespace {
        Environment g_env;
    }

    Environment& env() { return g_env; }

    CheckerConfig withEnvironment(CheckerConfig config) {
        const support::EnvSettings e = support::readEnvSettings();
        support::applyLegacyEnv(e, &config, nullptr);
        if (e.overrides) {
            applyOverrides(config, *e.overrides);
        }
        return config;
    }

    Environment& setup(int& argc, char** argv) {
        std::optional<fs::path> scenePath, robotPath, dataPath;
        int out = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if ((a == "--scene" || a == "--robot" || a == "--data")) {
                if (i + 1 >= argc) {
                    std::cerr << "missing value for " << a << "\n";
                    std::exit(2);
                }
                (a == "--scene" ? scenePath : a == "--robot" ? robotPath : dataPath) = fs::path(argv[++i]);
            } else {
                argv[out++] = argv[i];
            }
        }
        argc = out;
        try {
            const fs::path cfgs = support::configsDir();
            const SceneDesc sd  = loadSceneDesc(scenePath.value_or(cfgs / "scenes" / "shelf.json"));
            const RobotDesc rd  = loadRobotDesc(robotPath.value_or(cfgs / "robots" / "panda.json"));
            ContextOptions co   = ContextOptions::buildDefault();
            support::applyLegacyEnv(support::readEnvSettings(), nullptr, &co);
            g_env.context          = Context::create(co);
            g_env.robot            = RobotModel::load(rd, g_env.context->logger());
            g_env.scene            = std::make_shared<const Scene>(g_env.context, sd);
            g_env.motionsDir       = support::dataDir(dataPath) / "motions";
            g_env.splineWeightsDir = cfgs / "spline_weights";
            g_env.sceneName        = sd.name;
            std::cerr << "[bench] scene " << sd.name << ", robot " << g_env.robot->name() << ", device "
                      << g_env.context->deviceInfo().name << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[bench] setup failed: " << e.what() << "\n";
            std::exit(3);
        }
        return g_env;
    }

    DeviceBench::DeviceBench(const CheckerConfig& config) {
        checker_ = std::make_unique<CollisionChecker>(g_env.context, withEnvironment(config), g_env.robot, g_env.scene);
        g_env.context->makeCurrent();
        const std::size_t cap = checker_->config().maxItemsPerCheck;
        items_.allocate(cap * checker_->itemLayout().floatsPerItem() * sizeof(float));
        counts_.allocate(cap * sizeof(std::uint32_t));
    }

    void DeviceBench::upload(std::span<const float> items, std::size_t count) {
        items_.upload(items.data(), count * checker_->itemLayout().floatsPerItem() * sizeof(float));
        count_ = count;
    }

    void DeviceBench::check() {
        checker_->checkDevice(DeviceItems{items_.ptr(), count_, 0}, DeviceSpan<std::uint32_t>{counts_.ptr(), count_});
    }

    std::vector<float> samplePoses(const std::vector<float>& pool, std::size_t unitFloats, std::size_t n, std::mt19937& gen) {
        std::vector<std::size_t> idx(pool.size() / unitFloats), pick;
        std::iota(idx.begin(), idx.end(), std::size_t(0));
        std::sample(idx.begin(), idx.end(), std::back_inserter(pick), n, gen);
        std::vector<float> out;
        out.reserve(pick.size() * unitFloats);
        for (const std::size_t i : pick) {
            out.insert(out.end(), pool.begin() + std::ptrdiff_t(i * unitFloats), pool.begin() + std::ptrdiff_t((i + 1) * unitFloats));
        }
        return out;
    }

    std::vector<float> sampleTrajectories(
        const std::vector<float>& pool, std::size_t unitFloats, std::size_t n, std::mt19937& gen) {
        std::vector<std::size_t> idx(pool.size() / unitFloats);
        std::iota(idx.begin(), idx.end(), std::size_t(0));
        std::shuffle(idx.begin(), idx.end(), gen);
        std::vector<float> out;
        out.reserve(n * unitFloats);
        for (std::size_t k = 0; k < n && k < idx.size(); ++k) {
            const std::size_t i = idx[k];
            out.insert(out.end(), pool.begin() + std::ptrdiff_t(i * unitFloats), pool.begin() + std::ptrdiff_t((i + 1) * unitFloats));
        }
        return out;
    }

    std::mt19937& rng() {
        static std::mt19937 gen{std::random_device{}()};
        return gen;
    }

} // namespace rtcd::bench
