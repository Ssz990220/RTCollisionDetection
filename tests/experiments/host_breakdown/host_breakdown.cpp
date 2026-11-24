// rtcd_exp_host_breakdown -- Track G2: where does the host time of a small check go?
//
// Builds a real DetectorCore (the engine behind CollisionChecker, with the real backend of the mode) with the
// benchmark configurations of rtcd_bench_discrete (mesh, ray: 4 streams, batch 1024, OBB on, link GAS
// fast_trace_compact) and rtcd_bench_curve (linear<16,16,16>, quad<8,16,16>: 16 streams, batch 16, OBB off),
// attaches a PhaseTrace (src/detector/phase_trace.h) and times checkDevice / checkHost of N items drawn from the
// benchmark pools, phase by phase. Traced and untraced checks alternate, so the cost of the tracing itself is
// visible (total_untraced vs total).
//
// Usage (from the repository root):
//   rtcd_exp_host_breakdown [--modes mesh,ray,linear,quad] [--n 1,16] [--reps 2000] [--warmup 300]
//                           [--path device|host|both] [--dispatch thread|single] [--probe-sync 0|1]
//                           [--override path=value,...]   (rtcd::applyOverrides on every checker config)
//                           [--variants "<overrides>/<overrides>/..."] [--tables 0|1]
// Variants ('/'-separated override texts, e.g. "graph_prepare=true/graph_prepare=false"; an empty one = the mode's
// config) get one checker each and are alternated rep by rep in this process on the same items; each variant's
// flags are compared with the first checker run of that repetition (reported as "flag mismatches").
// Output: one table per (mode, path, N): phase, median, p10, p90 [us], and the share of the median total.
#include "detector/backend_factory.h"
#include "detector/detector_core.h"
#include "detector/phase_trace.h"

#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/json_config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <support/inputs.h>
#include <support/paths.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace rtcd;
using rtcd::detail::Phase;
using rtcd::detail::PhaseTrace;

namespace {

    std::vector<std::string> split(const std::string& s, char sep = ',') {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, sep)) {
            if (!t.empty()) out.push_back(t);
        }
        return out;
    }

    struct Options {
        std::vector<std::string> modes{"mesh", "ray", "linear", "quad"};
        std::vector<std::size_t> ns{1, 16};
        int reps = 2000, warmup = 300;
        std::string path = "device";
        std::string dispatch = "thread";
        bool probeSync = true;
        std::string overrides;
        bool tables = true;
        // Each variant = extra rtcd::applyOverrides text on top of the mode's config ("" = as is); one checker per
        // variant, all alternated rep by rep.
        std::vector<std::string> variants{""};
    };

    struct ModeSetup {
        CheckerConfig cfg;
        std::uint32_t trajPoints = 1; // pool unit (poses: 1)
    };

    ModeSetup modeSetup(const std::string& m, const Options& o) {
        ModeSetup s;
        CheckerConfig& c = s.cfg;
        if (m == "mesh" || m == "ray") {
            c.mode              = m == "mesh" ? Mode::Mesh : Mode::Ray;
            c.numStreams        = 4;
            c.maxBatchSize      = 1024;
            c.maxItemsPerCheck  = 8192;
            c.obbCulling        = Toggle::On;
            c.mesh.linkGasBuild = BuildPreference::FastTraceCompact;
        } else if (m == "linear" || m == "quad") {
            c.mode                   = m == "linear" ? Mode::LinearCurve : Mode::QuadraticCurve;
            c.numStreams             = 16;
            c.maxBatchSize           = 16;
            c.maxItemsPerCheck       = 4096;
            c.obbCulling             = Toggle::Off;
            c.curve.trajectoryPoints = 16;
            c.curve.controlPoints    = m == "linear" ? 16 : 8;
            c.curve.gasBuild         = m == "linear" ? BuildPreference::FastBuild : BuildPreference::FastTrace;
            c.curve.splineWeightsDir = support::configsDir() / "spline_weights";
            s.trajPoints             = 16;
        } else {
            throw std::runtime_error("unknown mode " + m);
        }
        c.hostDispatch = o.dispatch == "single" ? HostDispatch::SingleThread : HostDispatch::ThreadPerStream;
        if (!o.overrides.empty()) {
            applyOverrides(c, o.overrides);
        }
        return s;
    }

    double us(std::int64_t ticks) {
        using P = std::chrono::steady_clock::period;
        return double(ticks) * double(P::num) / double(P::den) * 1e6;
    }

    struct Samples {
        std::map<std::string, std::vector<double>> v;
        std::vector<std::string> order;
        void add(const std::string& k, double x) {
            auto it = v.find(k);
            if (it == v.end()) {
                order.push_back(k);
                it = v.emplace(k, std::vector<double>{}).first;
            }
            it->second.push_back(x);
        }
    };

    double pct(std::vector<double> x, double q) {
        if (x.empty()) return 0;
        std::sort(x.begin(), x.end());
        const double pos = q * double(x.size() - 1);
        const std::size_t i = std::size_t(pos);
        const double f = pos - double(i);
        return i + 1 < x.size() ? x[i] * (1 - f) + x[i + 1] * f : x[i];
    }

    // One traced check -> named phase durations (us). Batch phases are summed over all batches of the check.
    void analyse(const PhaseTrace& t, Samples& out) {
        const auto& c = t.slots[0];
        std::int64_t tCall = 0, tDispatch = 0, tJoined = 0, tReturn = 0, prev = 0;
        double probe = 0;
        for (const auto& e : c) {
            const double d = prev ? us(e.ticks - prev) : 0;
            switch (e.phase) {
            case Phase::Call: tCall = e.ticks; break;
            case Phase::Dispatch: tDispatch = e.ticks; break;
            case Phase::Joined: tJoined = e.ticks; break; // its interval is split below (wake / batches / join)
            case Phase::Return: tReturn = e.ticks; out.add("return", d); break;
            case Phase::SyncProbe: probe = d; break;
            default: out.add(detail::phaseName(e.phase), d); break;
            }
            prev = e.ticks;
        }
        // batch slots
        std::int64_t firstStart = 0, lastEnd = 0;
        std::map<std::string, double> sums;
        int batches = 0;
        for (std::size_t s = 1; s < t.slots.size(); ++s) {
            std::int64_t p = 0;
            for (const auto& e : t.slots[s]) {
                if (e.phase == Phase::BatchStart) {
                    if (!firstStart || e.ticks < firstStart) firstStart = e.ticks;
                    ++batches;
                } else if (p) {
                    sums[detail::phaseName(e.phase)] += us(e.ticks - p);
                }
                if (e.phase == Phase::BatchEnd && e.ticks > lastEnd) lastEnd = e.ticks;
                p = e.ticks;
            }
        }
        out.add("wake (dispatch -> first batch)", firstStart ? us(firstStart - tDispatch) : 0);
        for (const char* k : {"zero", "staging", "prepare", "graph_launch", "params", "launch", "result_copy", "batch_end"}) {
            out.add(std::string("batch: ") + k, sums.count(k) ? sums[k] : 0.0);
        }
        // With several batches the batch phases overlap (several threads); "batches wall" is first start -> last end.
        out.add("batches wall (first start -> last end)", firstStart ? us(lastEnd - firstStart) : 0);
        out.add("join (last batch end -> caller resumes)", lastEnd ? us(tJoined - lastEnd) : 0);
        if (t.probeIdleSync) out.add("[probe] idle re-sync (not in total)", probe);
        out.add("TOTAL (call -> return, minus probe)", us(tReturn - tCall) - probe);
        (void) batches;
    }

    void printTable(const std::string& title, Samples& s, double untracedMedian, double untracedP10, double untracedP90) {
        std::printf("\n== %s ==\n", title.c_str());
        std::printf("  %-42s %9s %9s %9s %7s\n", "phase", "median", "p10", "p90", "share");
        double total = pct(s.v["TOTAL (call -> return, minus probe)"], 0.5);
        for (const auto& k : s.order) {
            const auto& x = s.v[k];
            const double m = pct(x, 0.5);
            if (k != "TOTAL (call -> return, minus probe)" && pct(x, 0.9) < 0.005) continue; // phase not used here
            std::printf("  %-42s %9.2f %9.2f %9.2f %6.1f%%\n", k.c_str(), m, pct(x, 0.1), pct(x, 0.9),
                total > 0 ? 100 * m / total : 0);
        }
        std::printf("  %-42s %9.2f %9.2f %9.2f\n", "total untraced (alternating, same process)", untracedMedian,
            untracedP10, untracedP90);
    }

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--modes") o.modes = split(v);
        else if (a == "--n") {
            o.ns.clear();
            for (auto& x : split(v)) o.ns.push_back(std::stoul(x));
        } else if (a == "--reps") o.reps = std::stoi(v);
        else if (a == "--warmup") o.warmup = std::stoi(v);
        else if (a == "--path") o.path = v;
        else if (a == "--dispatch") o.dispatch = v;
        else if (a == "--probe-sync") o.probeSync = v != "0";
        else if (a == "--override") o.overrides = v;
        else if (a == "--tables") o.tables = v != "0";
        else if (a == "--variants") {
            o.variants.clear();
            std::stringstream ss(v);
            std::string t;
            while (std::getline(ss, t, '/')) o.variants.push_back(t);
            if (o.variants.empty()) o.variants.push_back("");
        }
        else {
            std::cerr << "unknown option " << a << "\n";
            return 2;
        }
    }
    try {
        const auto cfgs = support::configsDir();
        auto ctx        = Context::create();
        auto robot      = RobotModel::load(loadRobotDesc(cfgs / "robots" / "panda.json"), ctx->logger());
        auto scene      = std::make_shared<const Scene>(ctx, loadSceneDesc(cfgs / "scenes" / "shelf.json"));
        const auto motions = support::dataDir() / "motions";
        std::mt19937 rng(12345);

        for (const std::string& m : o.modes) {
            ModeSetup ms = modeSetup(m, o);
            std::vector<std::unique_ptr<detail::DetectorCore>> cores;
            for (const std::string& variant : o.variants) {
                CheckerConfig c = ms.cfg;
                if (!variant.empty()) applyOverrides(c, variant);
                detail::DetectorCoreDesc d;
                d.context = ctx;
                d.config  = c.resolved();
                d.dof     = robot->dof();
                d.lowerLimits.assign(robot->lowerLimits().begin(), robot->lowerLimits().end());
                d.upperLimits.assign(robot->upperLimits().begin(), robot->upperLimits().end());
                d.robot         = robot;
                d.scene         = scene;
                d.createBackend = &detail::createBackendForMode;
                cores.push_back(std::make_unique<detail::DetectorCore>(std::move(d)));
                cores.back()->warmup();
            }
            detail::DetectorCore& core = *cores[0];
            const std::uint32_t S = core.config().numStreams;
            const std::size_t fpi = core.itemLayout().floatsPerItem();
            const std::vector<float> pool =
                ms.trajPoints == 1 ? support::benchmarkPosePool(motions) : support::benchmarkTrajectoryPool(motions, ms.trajPoints);
            const std::size_t poolItems = pool.size() / fpi;
            ctx->makeCurrent();
            DeviceBuffer items(core.config().maxItemsPerCheck * fpi * sizeof(float));
            DeviceBuffer counts(core.config().maxItemsPerCheck * sizeof(std::uint32_t));
            std::vector<std::uint8_t> flags(core.config().maxItemsPerCheck);

            PhaseTrace trace;
            trace.probeIdleSync = o.probeSync;
            const std::size_t V = o.variants.size();
            for (const std::size_t n : o.ns) {
                for (const std::string path : {std::string("device"), std::string("host")}) {
                    if (o.path != "both" && o.path != path) continue;
                    std::vector<Samples> s(V);
                    std::vector<std::vector<double>> untraced(V);
                    std::vector<float> host(n * fpi);
                    std::vector<std::uint32_t> dl(n);
                    std::vector<std::uint8_t> ref(n), got(n);
                    std::size_t flagMismatches = 0;
                    for (int r = 0; r < o.warmup + o.reps; ++r) {
                        // sample n items (untimed), as the benchmarks do
                        for (std::size_t k = 0; k < n; ++k) {
                            const std::size_t idx = rng() % poolItems;
                            std::copy_n(pool.begin() + std::ptrdiff_t(idx * fpi), fpi, host.begin() + std::ptrdiff_t(k * fpi));
                        }
                        if (path == "device") items.upload(host.data(), n * fpi * sizeof(float));
                        // every variant: one traced and one untraced check on the same items, order rotated per rep
                        for (std::size_t j = 0; j < 2 * V; ++j) {
                            const std::size_t v = (j / 2 + std::size_t(r)) % V;
                            const bool traced   = (j & 1) == 0;
                            detail::DetectorCore& cv = *cores[v];
                            trace.reset(S);
                            cv.setPhaseTrace(traced ? &trace : nullptr);
                            const auto t0 = std::chrono::steady_clock::now();
                            if (traced) trace.mark(0, Phase::Call);
                            if (path == "device") {
                                cv.checkDevice(DeviceItems{items.ptr(), n, 0}, DeviceSpan<std::uint32_t>{counts.ptr(), n});
                            } else {
                                cv.checkHost(HostItems{host, n, 0}, std::span<std::uint8_t>(flags).first(n), {});
                            }
                            if (traced) trace.mark(0, Phase::Return);
                            const auto t1 = std::chrono::steady_clock::now();
                            cv.setPhaseTrace(nullptr);
                            // flags of every variant against the first variant's on the same items (untimed)
                            if (path == "device") {
                                counts.download(dl.data(), n * sizeof(std::uint32_t));
                                for (std::size_t k = 0; k < n; ++k) got[k] = dl[k] > 0;
                            } else {
                                std::copy_n(flags.begin(), n, got.begin());
                            }
                            if (j == 0) ref = got;
                            else if (got != ref) ++flagMismatches;
                            if (r < o.warmup) continue;
                            if (traced) analyse(trace, s[v]);
                            else untraced[v].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                        }
                    }
                    for (std::size_t v = 0; v < V; ++v) {
                        std::ostringstream title;
                        title << m << " " << (path == "device" ? "checkDevice" : "checkHost") << " N=" << n << " ("
                              << S << " streams, batch " << core.config().maxBatchSize << ", dispatch " << o.dispatch
                              << ", " << o.reps << " reps) variant [" << (o.variants[v].empty() ? "as configured" : o.variants[v]) << "]";
                        if (o.tables) {
                            printTable(title.str(), s[v], pct(untraced[v], 0.5), pct(untraced[v], 0.1), pct(untraced[v], 0.9));
                        }
                    }
                    std::printf("\n-- summary %s %s N=%zu: untraced total median [p10, p90] us; flag mismatches vs "
                                "first run: %zu\n",
                        m.c_str(), path.c_str(), n, flagMismatches);
                    const double base = pct(untraced[0], 0.5);
                    for (std::size_t v = 0; v < V; ++v) {
                        const double med = pct(untraced[v], 0.5);
                        std::printf("   %-40s %9.2f [%8.2f, %8.2f]  %+7.2f us  (%.3f x)\n", (o.variants[v].empty() ? "as configured" : o.variants[v].c_str()), med,
                            pct(untraced[v], 0.1), pct(untraced[v], 0.9), med - base, base > 0 ? med / base : 0.0);
                    }
                    std::fflush(stdout);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
