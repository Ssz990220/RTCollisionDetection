#include "demo_common.h"

#include <rtcd/context.h>
#include <rtcd/errors.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace rtcd::demo {

    namespace {

        const char* kindName(Kind k) { return k == Kind::Discrete ? "discrete" : "curve"; }

        [[noreturn]] void usage(Kind kind, const std::string& msg, int code = 2) {
            std::ostream& os = code == 0 ? std::cout : std::cerr;
            if (!msg.empty()) {
                os << "rtcd_demo_" << kindName(kind) << ": " << msg << "\n";
            }
            os << "usage: rtcd_demo_" << kindName(kind)
               << " [--setup <setup.json | name>] [--override \"path=value,...\"] [--counts 1-4096]\n"
                  "       [--repeats R] [--seed S]"
               << (kind == Kind::Curve ? " [--step F]" : "") << "\n"
               << "  --setup  a setup JSON, or a name of apps/demo/setups/<name>.json ("
               << (kind == Kind::Discrete ? "demo_ias (default), demo_ray" : "demo_linear (default), demo_quad")
               << ")\n"
                  "Times printed are indicative only (host check() incl. upload/download), not benchmark results.\n";
            std::exit(code);
        }

        // apps/demo/setups/<name>.json, searched upwards from the working directory, then from the directory of
        // argv[0] (the executable when it was started with a path, as ctest and the scripts do).
        fs::path findNamedSetup(const std::string& name, const char* argv0) {
            const fs::path rel = fs::path("apps") / "demo" / "setups" / (name + ".json");
            std::error_code ec;
            const fs::path exeDir = argv0 ? fs::absolute(fs::path(argv0), ec).parent_path() : fs::path{};
            for (fs::path start : {fs::current_path(), exeDir}) {
                for (fs::path d = start; !d.empty(); d = d.parent_path()) {
                    if (fs::exists(d / rel)) {
                        return d / rel;
                    }
                    if (d == d.parent_path()) {
                        break;
                    }
                }
            }
            throw std::runtime_error("setup '" + name + "' not found (looked for " + rel.generic_string()
                                     + " above the working directory and the executable); pass a path to --setup");
        }

        std::vector<std::size_t> parseCounts(const std::string& text) {
            std::vector<std::size_t> out;
            const auto dash = text.find('-');
            if (dash != std::string::npos) {
                const std::size_t a = std::stoull(text.substr(0, dash)), b = std::stoull(text.substr(dash + 1));
                if (a == 0 || b < a) {
                    throw std::invalid_argument("bad range");
                }
                for (std::size_t n = a; n <= b; n *= 2) {
                    out.push_back(n);
                }
                return out;
            }
            std::stringstream ss(text);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                const std::size_t n = std::stoull(tok);
                if (n == 0) {
                    throw std::invalid_argument("count 0");
                }
                out.push_back(n);
            }
            if (out.empty()) {
                throw std::invalid_argument("empty list");
            }
            return out;
        }

        const char* modeName(Mode m) {
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

    Options parseOptions(int argc, char** argv, Kind kind) {
        Options o;
        o.repeats = kind == Kind::Discrete ? 100 : 10;
        std::optional<std::string> setupArg, countsArg;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value          = [&]() -> std::string {
                if (i + 1 >= argc) {
                    usage(kind, "missing value for " + a);
                }
                return argv[++i];
            };
            try {
                if (a == "--setup") setupArg = value();
                else if (a == "--override") o.overrides = value();
                else if (a == "--counts") countsArg = value();
                else if (a == "--repeats") o.repeats = std::stoull(value());
                else if (a == "--seed") o.seed = static_cast<unsigned>(std::stoul(value()));
                else if (a == "--step" && kind == Kind::Curve) o.step = std::stof(value());
                else if (a == "--help" || a == "-h") usage(kind, "", 0);
                else usage(kind, "unknown argument " + a);
            } catch (const std::logic_error&) { // std::sto* failures
                usage(kind, "bad value for " + a);
            }
        }
        try {
            o.counts = parseCounts(countsArg.value_or("1-4096"));
        } catch (const std::logic_error&) {
            usage(kind, "bad --counts '" + *countsArg + "' (use \"1,16,256\" or \"1-4096\")");
        }
        if (!(o.step > 0.0f && o.step <= 1.0f)) {
            usage(kind, "--step must be in (0, 1]");
        }
        const std::string s = setupArg.value_or(kind == Kind::Discrete ? "demo_ias" : "demo_linear");
        const fs::path p(s);
        try {
            o.setup = (!p.has_parent_path() && !p.has_extension()) ? findNamedSetup(s, argc > 0 ? argv[0] : nullptr) : p;
        } catch (const std::exception& e) {
            usage(kind, e.what());
        }
        return o;
    }

    int run(const Options& options, Kind kind, const ItemGenerator& generate) {
        const std::string tag = std::string("[demo_") + kindName(kind) + "] ";
        try {
            Setup setup = loadSetup(options.setup);
            if (!options.overrides.empty()) {
                applyOverrides(setup.checker, options.overrides);
            }
            if (isContinuous(setup.checker.mode) != (kind == Kind::Curve)) {
                throw ConfigError(std::string("mode '") + modeName(setup.checker.mode) + "' is not a "
                                  + kindName(kind) + " mode; use rtcd_demo_"
                                  + (kind == Kind::Curve ? "discrete" : "curve") + " for this setup");
            }

            // Context, robot, scene, checker: the public API end to end.
            std::shared_ptr<Context> ctx                = Context::create();
            std::shared_ptr<const RobotModel> robot     = RobotModel::load(setup.robot, ctx->logger());
            std::shared_ptr<const Scene> scene          = std::make_shared<const Scene>(ctx, setup.scene);
            CollisionChecker checker(ctx, setup.checker, robot, scene);

            const CheckerConfig& cfg = checker.config(); // resolved
            const ItemLayout layout  = checker.itemLayout();
            std::cout << tag << "setup " << fs::absolute(options.setup).string() << "\n"
                      << tag << "device " << ctx->deviceInfo().name << ", robot " << robot->name() << " ("
                      << robot->dof() << " DOF), scene " << setup.scene.name << " ("
                      << setup.scene.obstacles.size() << " obstacles)\n"
                      << tag << "mode " << modeName(cfg.mode) << ", streams " << cfg.numStreams
                      << ", max_batch_size " << cfg.maxBatchSize << ", max_items_per_check " << cfg.maxItemsPerCheck
                      << ", obb_culling " << (cfg.obbCullingEnabled() ? "on" : "off");
            if (isContinuous(cfg.mode)) {
                std::cout << ", trajectory_points " << cfg.curve.trajectoryPoints;
                if (cfg.mode != Mode::LinearCurve) {
                    std::cout << ", control_points " << cfg.curve.controlPoints;
                }
            }
            std::cout << "\n";

            std::vector<std::size_t> counts;
            for (const std::size_t n : options.counts) {
                if (n > cfg.maxItemsPerCheck) {
                    std::cout << tag << "skipping N=" << n << " > max_items_per_check " << cfg.maxItemsPerCheck << "\n";
                } else {
                    counts.push_back(n);
                }
            }
            if (counts.empty()) {
                throw InvalidArgument("no item count <= max_items_per_check");
            }
            const std::size_t maxN = *std::max_element(counts.begin(), counts.end());

            std::mt19937 gen(options.seed);
            const std::vector<float> items = generate(*robot, layout, maxN, gen);

            checker.warmup();

            const char* unit = isContinuous(cfg.mode) ? "trajectories" : "poses";
            std::cout << tag << "time = mean of " << options.repeats
                      << " host check() calls incl. upload/download; INDICATIVE ONLY, not a benchmark result\n";
            std::cout << std::setw(8) << "N" << std::setw(12) << "colliding" << std::setw(16) << "time [us]"
                      << std::setw(18) << "per item [us]" << "   (" << unit << ")\n";

            std::vector<std::uint8_t> flags(maxN);
            for (const std::size_t n : counts) {
                const HostItems view{std::span<const float>(items.data(), n * layout.floatsPerItem()), n, 0};
                checker.check(view, std::span<std::uint8_t>(flags.data(), n));
                const std::size_t colliding = std::count(flags.begin(), flags.begin() + std::ptrdiff_t(n), 1);

                double us = 0.0;
                if (options.repeats > 0) {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (std::size_t r = 0; r < options.repeats; ++r) {
                        checker.check(view, std::span<std::uint8_t>(flags.data(), n));
                    }
                    const auto t1 = std::chrono::steady_clock::now();
                    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / double(options.repeats);
                }
                std::cout << std::setw(8) << n << std::setw(12) << colliding << std::fixed << std::setprecision(1)
                          << std::setw(16) << us << std::setprecision(3) << std::setw(18) << us / double(n) << "\n"
                          << std::defaultfloat;
            }
            return 0;
        } catch (const std::exception& e) {
            std::cerr << tag << "error: " << e.what() << std::endl;
            return 3;
        }
    }

} // namespace rtcd::demo
