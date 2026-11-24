// rtcd_exp_graph_capture -- Track G0 feasibility experiment: CUDA Graph capture of the detector's per-batch call
// sequence, including optixLaunch and optixAccelBuild (OptiX 9.1 / CUDA 13.x).
//
// The program rebuilds, outside DetectorCore, exactly the per-batch sequence of DetectorCore::runBatch
// (src/detector/detector_core.cpp) on a real backend + real OptiX pipeline (detail::OptixProgramPipeline):
//     memsetAsync(hits) ; staging copy (D2D) ; backend.prepare() ; [params upload if changed] ; optixLaunch ;
//     result copy (D2D)
// and then:
//   correctness  (a) direct run of the 4096 regression items vs the golden flags;
//                (b) the same sequence captured into a CUDA graph (cudaStreamBeginCapture .. EndCapture),
//                    instantiated, launched; flags vs golden / vs direct; replayed several times; replayed with new
//                    contents in the same input buffer (the real use) vs a direct run on the same contents;
//                (c) single-batch graphs (n = 1, 16, 128), graph update (cudaGraphExecUpdate) to another n,
//                    curve GAS refit (UPDATE) inside a graph, multi-threaded capture on several streams,
//                    and the paths that are expected NOT to be capturable (host syncs), with the exact errors.
//   timing       host enqueue time (first call .. last call returned, no sync) and end-to-end time (incl. the
//                stream sync) of: direct calls, one prebuilt graph launch, graph launch + memcpy-node updates,
//                and (ray) the fallback "CUDA part as a graph, optixLaunch + result copy direct". Variants are
//                alternated within one process; medians and p10 / p90 are printed.
//
// Usage (from the repository root):
//   rtcd_exp_graph_capture [--mode ray|mesh|mesh_nocull|linear|quad|all] [--what correctness|timing|all]
//                          [--reps N] [--dot <dir>]
#include <rtcd/backend.h>
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/json_config.h>
#include <rtcd/limits.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include "detector/optix_pipeline.h"

#include <support/golden.h>
#include <support/inputs.h>
#include <support/paths.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace rtcd;
using Clock = std::chrono::steady_clock;

namespace {

    void* toPtr(DevicePtr p) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(p)); }

    void cudaOk(cudaError_t e, const char* what) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")");
        }
    }

    std::string errName(cudaError_t e) {
        return e == cudaSuccess ? std::string("cudaSuccess") : std::string(cudaGetErrorName(e)) + " (" + cudaGetErrorString(e) + ")";
    }

    int g_failures = 0;
    void verdict(bool ok, const std::string& what) {
        std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << "\n";
        if (!ok) {
            ++g_failures;
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Mode setup
    // ------------------------------------------------------------------------------------------------------------

    struct ModeSpec {
        std::string name;
        std::string setupFile;   // configs/setups/<file>
        Mode mode;
        bool meshMasked = false; // force the mesh masked path (no host sync): compactionMinPoses = huge
        Toggle obb      = Toggle::Auto;
    };

    ModeSpec modeSpec(const std::string& name) {
        if (name == "ray") return {"ray", "regression_ray.json", Mode::Ray};
        if (name == "mesh") return {"mesh", "regression_ias.json", Mode::Mesh, true};
        if (name == "mesh_compaction") return {"mesh_compaction", "regression_ias.json", Mode::Mesh, false};
        if (name == "mesh_nocull") return {"mesh_nocull", "regression_ias.json", Mode::Mesh, false, Toggle::Off};
        if (name == "linear") return {"linear", "regression_linear.json", Mode::LinearCurve};
        if (name == "linear_obb") return {"linear_obb", "regression_linear.json", Mode::LinearCurve, false, Toggle::On};
        if (name == "quad") return {"quad", "regression_quad.json", Mode::QuadraticCurve};
        throw std::runtime_error("unknown mode " + name);
    }

    struct Env {
        std::shared_ptr<Context> ctx;
        Setup setup;
        std::shared_ptr<const RobotModel> robot;
        std::shared_ptr<const Scene> scene;
        std::vector<float> inputs; // regression inputs (4096 items)
        support::Flags golden;     // empty if no golden for this mode
    };

    Env makeEnv(const std::shared_ptr<Context>& ctx, const ModeSpec& m) {
        Env e;
        e.ctx   = ctx;
        e.setup = loadSetup(support::configsDir() / "setups" / m.setupFile);
        e.robot = RobotModel::load(e.setup.robot, ctx->logger());
        e.scene = std::make_shared<const Scene>(ctx, e.setup.scene);
        const CheckerConfig r = e.setup.checker.resolved();
        const std::uint32_t ppi = r.itemLayout(e.robot->dof()).pointsPerItem;
        const auto data         = support::dataDir();
        e.inputs = support::regressionInputs(data / "motions", ppi, support::LegacyHarness::regressionItems, e.robot->dof());
        if (auto g = support::goldenPath(data, m.mode)) {
            e.golden = support::readFlagsFile(*g, support::LegacyHarness::regressionItems);
        }
        return e;
    }

    // ------------------------------------------------------------------------------------------------------------
    // The rig: backend + pipeline + the core's per-stream buffers, and runBatch reproduced
    // ------------------------------------------------------------------------------------------------------------

    enum class ParamsUpload { Cached, AlwaysPageable, AlwaysPinned, None };

    struct Rig {
        struct PerStream {
            DeviceBuffer staging, hits, params;
            std::vector<unsigned char> uploaded = std::vector<unsigned char>(kMaxLaunchParamsBytes);
            std::uint32_t uploadedBytes = 0;
            bool uploadedValid          = false;
            PinnedHostBuffer pinnedParams;
            LaunchPlan lastPlan{};
        };

        const Env* env = nullptr;
        CheckerConfig cfg;
        ItemLayout layout{};
        std::size_t fpiBytes = 0;
        StreamLease lease;
        std::vector<StreamContext> sc;
        std::unique_ptr<Backend> backend;
        std::unique_ptr<detail::OptixProgramPipeline> pipe;
        std::vector<PerStream> ps;
        DeviceBuffer input, results;
        ParamsUpload paramsMode = ParamsUpload::Cached;
        std::atomic<std::uint64_t> calls{0}; // API calls issued by enqueueBatch (counted, for the report)

        Rig(const Env& e, const ModeSpec& m, std::uint32_t streams, std::uint32_t batchSize) : env(&e) {
            CheckerConfig c   = e.setup.checker;
            c.numStreams      = streams;
            c.maxBatchSize    = batchSize;
            c.maxItemsPerCheck = std::max<std::uint32_t>(c.maxItemsPerCheck, 4096);
            if (m.obb != Toggle::Auto) {
                c.obbCulling = m.obb;
            }
            if (m.meshMasked) {
                c.mesh.compactionMinPoses = 0x7FFFFFFF;
            }
            cfg      = c.resolved();
            layout   = cfg.itemLayout(e.robot->dof());
            fpiBytes = layout.floatsPerItem() * sizeof(float);
            lease    = e.ctx->acquireStreams(cfg.numStreams);
            for (std::uint32_t i = 0; i < lease.size(); ++i) {
                sc.push_back(StreamContext{i, lease[i]});
            }
            BackendCreateInfo info;
            info.context = e.ctx;
            info.config  = &cfg;
            info.robot   = e.robot;
            info.scene   = e.scene;
            info.streams = sc;
            switch (cfg.mode) {
            case Mode::Mesh: backend = createMeshBackend(info); break;
            case Mode::Ray: backend = createRayBackend(info); break;
            default: backend = createCurveBackend(info); break;
            }
            Logger log = e.ctx->logger();
            pipe       = std::make_unique<detail::OptixProgramPipeline>(*e.ctx, backend->programs(), cfg.optix, log);
            ps.resize(cfg.numStreams);
            for (PerStream& p : ps) {
                p.staging.allocate(std::size_t(cfg.maxBatchSize) * fpiBytes);
                p.hits.allocate(std::size_t(cfg.maxBatchSize) * sizeof(std::uint32_t));
                p.params.allocate(kMaxLaunchParamsBytes);
                p.pinnedParams.allocate(kMaxLaunchParamsBytes);
            }
            input.allocate(std::size_t(cfg.maxItemsPerCheck) * fpiBytes);
            results.allocate(std::size_t(cfg.maxItemsPerCheck) * sizeof(std::uint32_t));
        }
        ~Rig() {
            for (const StreamContext& s : sc) {
                cudaStreamSynchronize(s.stream);
            }
        }

        std::uint32_t B() const { return cfg.maxBatchSize; }
        std::uint32_t batches(std::uint32_t count) const { return (count + B() - 1) / B(); }
        StreamHandle stream(std::uint32_t s = 0) const { return sc[s].stream; }

        void uploadItems(const std::vector<float>& items, std::uint32_t count) {
            input.upload(items.data(), std::size_t(count) * fpiBytes);
        }
        void zeroResults() { results.upload(std::vector<std::uint32_t>(cfg.maxItemsPerCheck, 0xDEADBEEFu).data(), results.size()); }
        std::vector<std::uint32_t> downloadResults(std::uint32_t count) const {
            std::vector<std::uint32_t> r(count);
            results.download(r.data(), std::size_t(count) * sizeof(std::uint32_t));
            return r;
        }

        // DetectorCore::runBatch, step 7.1 .. 7.5, on stream s (default: b % streams). `skipOptix` = only the CUDA
        // part (zero, staging, prepare) -- the ray fallback experiment; then `launchAndCopy` does the rest.
        void enqueueBatch(std::uint32_t b, std::uint32_t count, int forceStream = -1) {
            enqueuePre(b, count, forceStream);
            enqueuePost(b, count, forceStream);
        }

        void enqueuePre(std::uint32_t b, std::uint32_t count, int forceStream = -1) {
            const std::uint32_t s     = forceStream >= 0 ? std::uint32_t(forceStream) : b % cfg.numStreams;
            const std::uint32_t first = b * B();
            const std::uint32_t n     = std::min(B(), count - first);
            PerStream& st             = ps[s];
            st.hits.memsetAsync(0, std::size_t(n) * sizeof(std::uint32_t), sc[s].stream);
            st.staging.copyFromAsync(input.ptr() + std::uint64_t(first) * fpiBytes, std::size_t(n) * fpiBytes, sc[s].stream);
            calls += 2;
            BatchView view;
            view.batchIndex = b;
            view.firstItem  = first;
            view.itemCount  = n;
            view.items      = DeviceSpan<const float>{st.staging.ptr(), std::size_t(n) * layout.floatsPerItem()};
            view.hitCounts  = DeviceSpan<std::uint32_t>{st.hits.ptr(), n};
            st.lastPlan     = backend->prepare(sc[s], view);
        }
        void enqueuePost(std::uint32_t b, std::uint32_t count, int forceStream = -1) {
            const std::uint32_t s     = forceStream >= 0 ? std::uint32_t(forceStream) : b % cfg.numStreams;
            const std::uint32_t first = b * B();
            const std::uint32_t n     = std::min(B(), count - first);
            PerStream& st             = ps[s];
            const LaunchPlan& plan    = st.lastPlan;
            if (plan.launch) {
                if (plan.paramsBytes > 0 && paramsMode != ParamsUpload::None) {
                    const bool same = st.uploadedValid && st.uploadedBytes == plan.paramsBytes
                                   && std::memcmp(st.uploaded.data(), plan.params, plan.paramsBytes) == 0;
                    if (!same || paramsMode != ParamsUpload::Cached) {
                        if (paramsMode == ParamsUpload::AlwaysPinned) {
                            std::memcpy(st.pinnedParams.data(), plan.params, plan.paramsBytes);
                            cudaOk(cudaMemcpyAsync(toPtr(st.params.ptr()), st.pinnedParams.data(), plan.paramsBytes,
                                       cudaMemcpyHostToDevice, sc[s].stream),
                                "params upload (pinned)");
                        } else {
                            st.params.uploadAsync(plan.params, plan.paramsBytes, sc[s].stream);
                        }
                        std::memcpy(st.uploaded.data(), plan.params, plan.paramsBytes);
                        st.uploadedBytes = plan.paramsBytes;
                        st.uploadedValid = true;
                        ++calls;
                    }
                }
                pipe->launch(sc[s].stream, st.params.ptr(), plan.paramsBytes, plan);
                ++calls;
            }
            cudaOk(cudaMemcpyAsync(toPtr(results.ptr() + std::uint64_t(first) * sizeof(std::uint32_t)), toPtr(st.hits.ptr()),
                       std::size_t(n) * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice, sc[s].stream),
                "result copy");
            ++calls;
        }

        // A whole check, every batch on stream b % S (one thread), as DetectorCore without workers.
        void enqueueCheck(std::uint32_t count, int forceStream = -1) {
            backend->beginCheck(CheckInfo{count, batches(count)});
            for (std::uint32_t b = 0; b < batches(count); ++b) {
                enqueueBatch(b, count, forceStream);
            }
        }
        void sync() {
            for (const StreamContext& s : sc) {
                cudaOk(cudaStreamSynchronize(s.stream), "cudaStreamSynchronize");
            }
        }
        std::vector<std::uint32_t> directCheck(std::uint32_t count) {
            zeroResults();
            enqueueCheck(count);
            sync();
            return downloadResults(count);
        }
    };

    // ------------------------------------------------------------------------------------------------------------
    // Capture helper
    // ------------------------------------------------------------------------------------------------------------

    struct Captured {
        bool bodyOk = true;
        std::string bodyError;          // exception thrown by the captured calls (rtcd::Error text)
        cudaStreamCaptureStatus statusAfterBody = cudaStreamCaptureStatusNone;
        cudaError_t endError    = cudaSuccess;
        cudaError_t lastError   = cudaSuccess; // cudaGetLastError right after EndCapture
        cudaError_t instError   = cudaSuccess;
        cudaGraph_t graph       = nullptr;
        cudaGraphExec_t exec    = nullptr;
        std::map<std::string, int> nodeTypes;
        std::size_t nodes = 0;
        double captureUs = 0, instantiateUs = 0;
        bool ok() const { return bodyOk && endError == cudaSuccess && instError == cudaSuccess && exec != nullptr; }
        std::string describe() const {
            std::ostringstream o;
            o << "body " << (bodyOk ? "ok" : "THREW: " + bodyError) << "; capture status after body "
              << (statusAfterBody == cudaStreamCaptureStatusActive        ? "active"
                     : statusAfterBody == cudaStreamCaptureStatusInvalidated ? "INVALIDATED"
                                                                              : "none")
              << "; EndCapture " << errName(endError) << "; last error " << errName(lastError) << "; instantiate "
              << errName(instError) << "; " << nodes << " nodes {";
            bool firstT = true;
            for (const auto& [k, v] : nodeTypes) {
                o << (firstT ? "" : ", ") << k << " " << v;
                firstT = false;
            }
            o << "}";
            return o.str();
        }
        void destroy() {
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
            exec  = nullptr;
            graph = nullptr;
        }
    };

    const char* nodeTypeName(cudaGraphNodeType t) {
        switch (t) {
        case cudaGraphNodeTypeKernel: return "kernel";
        case cudaGraphNodeTypeMemcpy: return "memcpy";
        case cudaGraphNodeTypeMemset: return "memset";
        case cudaGraphNodeTypeHost: return "host";
        case cudaGraphNodeTypeGraph: return "child_graph";
        case cudaGraphNodeTypeEmpty: return "empty";
        case cudaGraphNodeTypeWaitEvent: return "wait_event";
        case cudaGraphNodeTypeEventRecord: return "event_record";
        case cudaGraphNodeTypeExtSemaphoreSignal: return "ext_sem_signal";
        case cudaGraphNodeTypeExtSemaphoreWait: return "ext_sem_wait";
        case cudaGraphNodeTypeMemAlloc: return "mem_alloc";
        case cudaGraphNodeTypeMemFree: return "mem_free";
        case cudaGraphNodeTypeConditional: return "conditional";
        default: return "other";
        }
    }

    std::string g_dotDir;
    cudaStreamCaptureMode g_captureMode = cudaStreamCaptureModeThreadLocal;

    // CUDA driver error-log callback (CUDA 12.9+): the driver's own text for an API error, e.g. which call was
    // not permitted during capture.
    void CUDA_CB cudaLogCallback(void*, CUlogLevel level, char* message, size_t) {
        std::cout << "    [cuda log level " << int(level) << "] " << message << "\n";
    }

    Captured capture(StreamHandle stream, const std::function<void()>& body, const std::string& label = {},
        cudaStreamCaptureMode mode = cudaStreamCaptureMode(-1), bool instantiate = true) {
        if (int(mode) == -1) mode = g_captureMode;
        Captured c;
        const auto t0 = Clock::now();
        cudaOk(cudaStreamBeginCapture(stream, mode), "cudaStreamBeginCapture");
        try {
            body();
        } catch (const std::exception& e) {
            c.bodyOk    = false;
            c.bodyError = e.what();
        }
        cudaStreamIsCapturing(stream, &c.statusAfterBody);
        c.endError  = cudaStreamEndCapture(stream, &c.graph);
        c.lastError = cudaGetLastError();
        const auto t1 = Clock::now();
        c.captureUs   = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (c.graph) {
            std::size_t n = 0;
            cudaGraphGetNodes(c.graph, nullptr, &n);
            std::vector<cudaGraphNode_t> nodes(n);
            cudaGraphGetNodes(c.graph, nodes.data(), &n);
            c.nodes = n;
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType t{};
                cudaGraphNodeGetType(node, &t);
                c.nodeTypes[nodeTypeName(t)]++;
            }
            if (!g_dotDir.empty() && !label.empty()) {
                std::filesystem::create_directories(g_dotDir);
                const std::string f = (std::filesystem::path(g_dotDir) / (label + ".dot")).string();
                cudaGraphDebugDotPrint(c.graph, f.c_str(), cudaGraphDebugDotFlagsVerbose);
            }
            if (instantiate && c.bodyOk && c.endError == cudaSuccess) {
                const auto t2 = Clock::now();
                c.instError   = cudaGraphInstantiate(&c.exec, c.graph, 0);
                c.instantiateUs = std::chrono::duration<double, std::micro>(Clock::now() - t2).count();
                if (c.instError != cudaSuccess) {
                    cudaGetLastError();
                }
            }
        }
        return c;
    }

    // ------------------------------------------------------------------------------------------------------------
    // Comparisons
    // ------------------------------------------------------------------------------------------------------------

    struct Cmp {
        std::size_t flagDiffs = 0, countDiffs = 0, collisions = 0;
    };
    Cmp compare(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
        Cmp c;
        for (std::size_t i = 0; i < a.size(); ++i) {
            c.flagDiffs += (a[i] > 0) != (b[i] > 0);
            c.countDiffs += a[i] != b[i];
            c.collisions += a[i] > 0;
        }
        return c;
    }
    std::size_t goldenDiffs(const Env& e, const std::vector<std::uint32_t>& counts) {
        std::size_t d = 0;
        for (std::size_t i = 0; i < counts.size(); ++i) {
            d += (counts[i] > 0) != (e.golden[i] != 0);
        }
        return d;
    }
    std::string cmpText(const Cmp& c) {
        return std::to_string(c.flagDiffs) + " flag diffs, " + std::to_string(c.countDiffs) + " count diffs, "
             + std::to_string(c.collisions) + " collisions";
    }

    // New contents for the same buffer: the regression items in reversed item order (a different batch content for
    // every batch, same values so the scene / AS sizes are realistic).
    std::vector<float> reversedItems(const Env& e, std::size_t fpi, std::size_t count) {
        std::vector<float> r(count * fpi);
        for (std::size_t i = 0; i < count; ++i) {
            std::memcpy(&r[i * fpi], &e.inputs[(count - 1 - i) * fpi], fpi * sizeof(float));
        }
        return r;
    }
    std::vector<float> rotatedItems(const Env& e, std::size_t fpi, std::size_t count, std::size_t shift) {
        std::vector<float> r(count * fpi);
        for (std::size_t i = 0; i < count; ++i) {
            std::memcpy(&r[i * fpi], &e.inputs[((i + shift) % count) * fpi], fpi * sizeof(float));
        }
        return r;
    }

    // ------------------------------------------------------------------------------------------------------------
    // Correctness experiments
    // ------------------------------------------------------------------------------------------------------------

    void correctness(const Env& e, const ModeSpec& m) {
        constexpr std::uint32_t N = 4096;
        std::cout << "\n=== correctness: mode " << m.name << " ===\n";
        Rig rig(e, m, 1, 128);
        const std::size_t fpi = rig.layout.floatsPerItem();
        std::cout << "  config: batch " << rig.B() << ", 1 stream, obb culling " << (rig.cfg.obbCullingEnabled() ? "on" : "off")
                  << ", synchronizesHost(128) = " << rig.backend->synchronizesHost(128) << "\n";

        // (a) direct
        rig.uploadItems(e.inputs, N);
        const auto direct = rig.directCheck(N);
        const bool hasGolden = !e.golden.empty();
        if (hasGolden) {
            verdict(goldenDiffs(e, direct) == 0, "direct check of 4096 items vs golden: " + std::to_string(goldenDiffs(e, direct)) + " flag diffs");
        }

        // (a2) the smallest possible capture: optixLaunch alone (params already uploaded by the direct run)
        {
            rig.directCheck(16);
            Captured only = capture(rig.stream(), [&] {
                rig.pipe->launch(rig.stream(), rig.ps[0].params.ptr(), rig.ps[0].lastPlan.paramsBytes, rig.ps[0].lastPlan);
            }, m.name + "_launch_only");
            std::cout << "  capture of optixLaunch alone: " << only.describe() << "\n";
            only.destroy();
            cudaGetLastError();
        }

        // (b) whole check captured into one graph (32 batches on one stream)
        Captured whole = capture(rig.stream(), [&] { rig.enqueueCheck(N); }, m.name + "_whole4096");
        std::cout << "  capture of the whole 4096-item check: " << whole.describe() << "\n";
        std::cout << "    capture " << whole.captureUs << " us, instantiate " << whole.instantiateUs << " us\n";
        verdict(whole.ok(), "whole-check capture + instantiate");
        if (whole.ok()) {
            for (int rep = 0; rep < 3; ++rep) {
                rig.zeroResults();
                cudaOk(cudaGraphLaunch(whole.exec, rig.stream()), "cudaGraphLaunch");
                rig.sync();
                const auto g = rig.downloadResults(N);
                const Cmp c  = compare(g, direct);
                std::string msg = "graph replay " + std::to_string(rep) + " vs direct: " + cmpText(c);
                if (hasGolden) {
                    msg += "; vs golden " + std::to_string(goldenDiffs(e, g)) + " flag diffs";
                }
                verdict(c.flagDiffs == 0 && (!hasGolden || goldenDiffs(e, g) == 0), msg);
            }
            // (c) new contents in the same input buffer
            for (int variant = 0; variant < 2; ++variant) {
                const std::vector<float> items = variant == 0 ? reversedItems(e, fpi, N) : rotatedItems(e, fpi, N, 1000);
                rig.uploadItems(items, N);
                const auto d2 = rig.directCheck(N);
                rig.zeroResults();
                cudaOk(cudaGraphLaunch(whole.exec, rig.stream()), "cudaGraphLaunch");
                rig.sync();
                const auto g2 = rig.downloadResults(N);
                verdict(compare(g2, d2).flagDiffs == 0,
                    std::string("graph replay with new buffer contents (") + (variant == 0 ? "reversed" : "rotated")
                        + ") vs direct on the same contents: " + cmpText(compare(g2, d2)));
            }
            rig.uploadItems(e.inputs, N);
        }
        whole.destroy();

        // (d) single-batch graphs n = 1, 16, 128 (batch 0 at item 0), replayed with changing contents
        for (std::uint32_t n : {1u, 16u, 128u}) {
            rig.uploadItems(e.inputs, N);
            Captured one = capture(rig.stream(), [&] { rig.enqueueCheck(n); }, m.name + "_batch" + std::to_string(n));
            std::cout << "  capture of one batch, n = " << n << ": " << one.describe() << "\n";
            std::cout << "    capture " << one.captureUs << " us, instantiate " << one.instantiateUs << " us, "
                      << "API calls in the captured body " << "\n";
            if (!one.ok()) {
                verdict(false, "single-batch capture n = " + std::to_string(n));
                one.destroy();
                continue;
            }
            std::size_t bad = 0, compared = 0;
            for (std::uint32_t shift = 0; shift < 4096; shift += 97) {
                rig.uploadItems(rotatedItems(e, fpi, N, shift), N);
                const auto d = rig.directCheck(n);
                rig.zeroResults();
                cudaOk(cudaGraphLaunch(one.exec, rig.stream()), "cudaGraphLaunch");
                rig.sync();
                const auto g = rig.downloadResults(n);
                bad += compare(g, d).flagDiffs;
                compared += n;
            }
            verdict(bad == 0, "single-batch graph n = " + std::to_string(n) + ", 43 different contents: " + std::to_string(bad)
                                  + " flag diffs of " + std::to_string(compared));

            // (e) graph exec update to another item count (same topology expected)
            if (n == 16) {
                Captured other = capture(rig.stream(), [&] { rig.enqueueCheck(17); }, {}, cudaStreamCaptureModeThreadLocal, false);
                cudaGraphExecUpdateResultInfo info{};
                const auto t0       = Clock::now();
                const cudaError_t u = other.graph ? cudaGraphExecUpdate(one.exec, other.graph, &info) : cudaErrorInvalidValue;
                const double us     = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                cudaGetLastError();
                std::cout << "  cudaGraphExecUpdate(n=16 exec <- n=17 graph): " << errName(u) << ", result " << int(info.result)
                          << ", " << us << " us (capture of the new graph " << other.captureUs << " us)\n";
                if (u == cudaSuccess) {
                    rig.uploadItems(rotatedItems(e, fpi, N, 555), N);
                    const auto d = rig.directCheck(17);
                    rig.zeroResults();
                    cudaOk(cudaGraphLaunch(one.exec, rig.stream()), "cudaGraphLaunch");
                    rig.sync();
                    const auto g = rig.downloadResults(17);
                    verdict(compare(g, d).flagDiffs == 0, "updated exec (n = 17) vs direct: " + cmpText(compare(g, d)));
                } else {
                    verdict(false, "cudaGraphExecUpdate to n = 17");
                }
                other.destroy();
            }
            one.destroy();
        }

        // (f) curves: a refit (UPDATE) captured; the direct path just did a BUILD of the same item count.
        if (isContinuous(rig.cfg.mode) && !rig.cfg.obbCullingEnabled()) {
            const std::uint32_t n = 128;
            rig.uploadItems(e.inputs, N);
            rig.directCheck(n); // beginCheck(1 batch) -> auto refit policy; this BUILDs (builtItems was != n?)
            rig.directCheck(n); // ... and this one refits
            Captured refit = capture(rig.stream(), [&] { rig.enqueueCheck(n); }, m.name + "_refit128");
            std::cout << "  capture of a one-batch check whose GAS build is a REFIT (UPDATE): " << refit.describe() << "\n";
            if (refit.ok()) {
                std::size_t bad = 0;
                for (std::uint32_t shift = 0; shift < 4096; shift += 257) {
                    rig.uploadItems(rotatedItems(e, fpi, N, shift), N);
                    const auto d = rig.directCheck(n); // direct refit
                    rig.zeroResults();
                    cudaOk(cudaGraphLaunch(refit.exec, rig.stream()), "cudaGraphLaunch");
                    rig.sync();
                    bad += compare(rig.downloadResults(n), d).flagDiffs;
                }
                verdict(bad == 0, "refit graph vs direct refit, 16 contents: " + std::to_string(bad) + " flag diffs");
            } else {
                verdict(false, "refit capture");
            }
            refit.destroy();
        }

        // (g) pageable vs pinned param upload inside a capture (the core uses pageable uploadAsync).
        {
            rig.paramsMode = ParamsUpload::AlwaysPageable;
            Captured pg    = capture(rig.stream(), [&] { rig.enqueueCheck(16); }, {});
            std::cout << "  capture with a PAGEABLE params upload inside: " << pg.describe() << "\n";
            pg.destroy();
            rig.paramsMode = ParamsUpload::AlwaysPinned;
            Captured pn    = capture(rig.stream(), [&] { rig.enqueueCheck(16); }, {});
            std::cout << "  capture with a PINNED params upload inside: " << pn.describe() << "\n";
            pn.destroy();
            rig.paramsMode = ParamsUpload::Cached;
            rig.sync();
        }
    }

    // optixLaunch alone, dissected: does the launch execute eagerly during the capture, does the instantiated graph
    // contain the trace work, does OptiX keep working afterwards, and what does a second capture do?
    void launchOnly(const Env& e, const ModeSpec& m) {
        constexpr std::uint32_t n = 128;
        std::cout << "\n=== optixLaunch alone under capture: mode " << m.name << " ===\n";
        Rig rig(e, m, 1, 128);
        rig.uploadItems(e.inputs, 4096);
        const auto d = rig.directCheck(n); // also uploads the params of n = 128 and leaves lastPlan
        std::size_t dHits = 0;
        for (auto v : d) dHits += v;
        std::cout << "  direct: " << compare(d, d).collisions << " colliding items, " << dHits << " hits in total\n";
        auto hitsNow = [&] {
            std::vector<std::uint32_t> h(n);
            rig.ps[0].hits.download(h.data(), n * sizeof(std::uint32_t));
            std::size_t s = 0;
            for (auto v : h) s += v;
            return s;
        };
        for (int attempt = 0; attempt < 3; ++attempt) {
            rig.ps[0].hits.memsetAsync(0, n * sizeof(std::uint32_t), rig.stream());
            rig.sync();
            Captured c = capture(rig.stream(), [&] {
                rig.pipe->launch(rig.stream(), rig.ps[0].params.ptr(), rig.ps[0].lastPlan.paramsBytes, rig.ps[0].lastPlan);
            }, m.name + "_launch_only_" + std::to_string(attempt));
            std::cout << "  attempt " << attempt << ": " << c.describe() << "\n";
            cudaError_t se = cudaStreamSynchronize(rig.stream());
            std::cout << "    stream sync after EndCapture: " << errName(se) << "; hits written DURING the capture (eager "
                      << "execution): " << hitsNow() << "\n";
            if (c.ok()) {
                rig.ps[0].hits.memsetAsync(0, n * sizeof(std::uint32_t), rig.stream());
                rig.sync();
                const cudaError_t le = cudaGraphLaunch(c.exec, rig.stream());
                se                   = cudaStreamSynchronize(rig.stream());
                const std::size_t gh = hitsNow();
                std::cout << "    cudaGraphLaunch " << errName(le) << ", sync " << errName(se) << "; hits written by the GRAPH: " << gh
                          << " (direct: " << dHits << ")\n";
                verdict(gh == dHits, "graph of optixLaunch alone reproduces the direct hit total");
            }
            c.destroy();
            cudaGetLastError();
            try {
                const auto d2 = rig.directCheck(n);
                verdict(compare(d2, d).flagDiffs == 0, "direct check after the capture still correct: " + cmpText(compare(d2, d)));
            } catch (const std::exception& ex) {
                verdict(false, std::string("direct check after the capture THREW: ") + ex.what());
            }
        }
    }

    // The fallback: capture only the CUDA part before the launch (zero hits, staging copy, prepare() INCLUDING its
    // optixAccelBuild), replay it, then optixLaunch + result copy directly. Made sensitive to a dropped build:
    // before each replay the AS is left holding DIFFERENT contents (a direct check on other items).
    // refit: curves only, nullopt = config default; Never -> the captured build is a BUILD, Always -> an UPDATE.
    void fallbackCorrectness(const Env& e, const ModeSpec& m, std::optional<RefitPolicy::Kind> refit = std::nullopt) {
        constexpr std::uint32_t N = 4096;
        Rig rig(e, m, 1, 128);
        if (refit) {
            // the backend keeps a pointer to rig.cfg: changing it before any check takes effect in beginCheck
            rig.cfg.curve.refit.kind = *refit;
        }
        const std::size_t fpi = rig.layout.floatsPerItem();
        std::cout << "\n=== fallback (CUDA part incl. AS build as a graph, optixLaunch direct): mode " << m.name
                  << (refit ? (*refit == RefitPolicy::Kind::Never ? ", refit never (BUILD)" : ", refit always (UPDATE)") : "")
                  << " ===\n";
        for (std::uint32_t n : {1u, 16u, 128u}) {
            rig.uploadItems(e.inputs, N);
            rig.directCheck(n); // the stream's state as after a check of n items (curve: builtItems = n)
            const TraversableHandle before = rig.ps[0].lastPlan.traversable;
            rig.backend->beginCheck(CheckInfo{n, 1});
            Captured pre = capture(rig.stream(), [&] { rig.enqueuePre(0, n); }, m.name + "_pre" + std::to_string(n));
            std::cout << "  n = " << n << ": " << pre.describe() << "\n    traversable at capture "
                      << (pre.bodyOk && rig.ps[0].lastPlan.traversable == before ? "== " : "!= ") << "the direct one\n";
            if (!pre.ok()) {
                verdict(false, "pre-launch capture n = " + std::to_string(n));
                pre.destroy();
                continue;
            }
            const LaunchPlan capturedPlan = rig.ps[0].lastPlan;
            std::size_t bad = 0, compared = 0, collisions = 0;
            for (std::uint32_t shift = 0; shift < N; shift += 211) {
                const std::vector<float> want = rotatedItems(e, fpi, N, shift);
                const std::vector<float> other = rotatedItems(e, fpi, N, shift + 1777);
                rig.uploadItems(want, N);
                const auto ref = rig.directCheck(n);
                rig.uploadItems(other, N);
                rig.directCheck(n); // AS now holds `other`
                rig.uploadItems(want, N);
                rig.zeroResults();
                rig.ps[0].lastPlan = capturedPlan;
                cudaOk(cudaGraphLaunch(pre.exec, rig.stream()), "cudaGraphLaunch");
                rig.enqueuePost(0, n);
                rig.sync();
                const Cmp c = compare(rig.downloadResults(n), ref);
                bad += c.flagDiffs;
                compared += n;
                collisions += c.collisions;
            }
            verdict(bad == 0, "n = " + std::to_string(n) + ": graph(zero, staging, prepare incl. AS build) + direct launch vs direct, "
                                  + std::to_string(compared) + " items in 20 contents: " + std::to_string(bad) + " flag diffs ("
                                  + std::to_string(collisions) + " collisions)");
            pre.destroy();
        }
        // OptiX still healthy?
        rig.uploadItems(e.inputs, N);
        const auto d = rig.directCheck(N);
        if (!e.golden.empty()) {
            verdict(goldenDiffs(e, d) == 0, "direct 4096-item check after the captures vs golden: " + std::to_string(goldenDiffs(e, d)) + " flag diffs");
        }
    }

    // Paths that synchronize the host inside prepare(): expected to fail capture; record how.
    void expectedFailures(const std::shared_ptr<Context>& ctx) {
        for (const std::string name : {"mesh_compaction", "linear_obb"}) {
            const ModeSpec m = modeSpec(name);
            std::cout << "\n=== capture of a host-synchronizing path: " << name << " ===\n";
            const Env e = makeEnv(ctx, m);
            Rig rig(e, m, 1, 128);
            rig.uploadItems(e.inputs, 4096);
            std::cout << "  synchronizesHost(128) = " << rig.backend->synchronizesHost(128) << "\n";
            Captured c = capture(rig.stream(), [&] { rig.enqueueCheck(128); }, {});
            std::cout << "  ThreadLocal capture: " << c.describe() << "\n";
            c.destroy();
            // The context must still be usable afterwards.
            cudaGetLastError();
            try {
                const auto d = rig.directCheck(4096);
                std::cout << "  direct check after the failed capture: " << goldenDiffs(e, d) << " flag diffs vs golden\n";
            } catch (const std::exception& ex) {
                std::cout << "  direct check after the failed capture THREW: " << ex.what() << "\n";
            }
        }
    }

    // Memcpy nodes of a captured graph.
    std::vector<cudaGraphNode_t> memcpyNodes(cudaGraph_t g) {
        std::size_t n = 0;
        cudaGraphGetNodes(g, nullptr, &n);
        std::vector<cudaGraphNode_t> nodes(n), out;
        cudaGraphGetNodes(g, nodes.data(), &n);
        for (cudaGraphNode_t node : nodes) {
            cudaGraphNodeType t{};
            cudaGraphNodeGetType(node, &t);
            if (t == cudaGraphNodeTypeMemcpy) out.push_back(node);
        }
        return out;
    }
    // Multi-threaded capture of the fallback (ray: CUDA part only, no OptiX call inside a capture): S worker
    // threads, each capturing one graph per batch of its stream concurrently (ThreadLocal mode); then every thread
    // replays its batches (graph, optixLaunch, result copy) concurrently, as DetectorCore's workers would.
    void multiThreaded(const Env& e, const ModeSpec& m) {
        constexpr std::uint32_t N = 4096, S = 4;
        std::cout << "\n=== multi-threaded capture of the CUDA part (4 streams, 4 threads): mode " << m.name << " ===\n";
        if (m.mode != Mode::Ray) {
            std::cout << "  skipped: the CUDA part of this mode's prepare() contains optixAccelBuild\n";
            return;
        }
        Rig rig(e, m, S, 128);
        rig.uploadItems(e.inputs, N);
        const auto direct = rig.directCheck(N);
        const std::uint32_t nb = rig.batches(N);
        rig.backend->beginCheck(CheckInfo{N, nb});
        std::vector<Captured> caps(nb);
        std::vector<std::thread> threads;
        for (std::uint32_t s = 0; s < S; ++s) {
            threads.emplace_back([&, s] {
                e.ctx->makeCurrent();
                for (std::uint32_t b = s; b < nb; b += S) {
                    caps[b] = capture(rig.stream(s), [&] { rig.enqueuePre(b, N); });
                }
            });
        }
        for (auto& t : threads) t.join();
        bool all = true;
        for (std::uint32_t b = 0; b < nb; ++b) {
            all = all && caps[b].ok();
            if (!caps[b].ok() || b < 2) {
                std::cout << "  batch " << b << ": " << caps[b].describe() << "\n";
            }
        }
        verdict(all, "concurrent ThreadLocal captures of 32 batch graphs on 4 streams from 4 threads");
        if (!all) {
            for (auto& c : caps) c.destroy();
            return;
        }
        for (int rep = 0; rep < 3; ++rep) {
            if (rep > 0) {
                rig.uploadItems(rep == 1 ? reversedItems(e, rig.layout.floatsPerItem(), N)
                                         : rotatedItems(e, rig.layout.floatsPerItem(), N, 1234),
                    N);
            }
            const auto d = rep == 0 ? direct : rig.directCheck(N);
            rig.zeroResults();
            std::vector<std::thread> replay;
            std::vector<std::string> errors(S);
            for (std::uint32_t s = 0; s < S; ++s) {
                replay.emplace_back([&, s] {
                    try {
                        e.ctx->makeCurrent();
                        for (std::uint32_t b = s; b < nb; b += S) {
                            cudaOk(cudaGraphLaunch(caps[b].exec, rig.stream(s)), "cudaGraphLaunch");
                            rig.enqueuePost(b, N);
                        }
                    } catch (const std::exception& ex) {
                        errors[s] = ex.what();
                    }
                });
            }
            for (auto& t : replay) t.join();
            for (const auto& err : errors) {
                if (!err.empty()) std::cout << "  replay error: " << err << "\n";
            }
            rig.sync();
            const auto g = rig.downloadResults(N);
            verdict(compare(g, d).flagDiffs == 0 && (rep != 0 || goldenDiffs(e, g) == 0),
                std::string("4 threads replaying graph + optixLaunch + copy ") + (rep ? "(new contents) " : "(vs golden too) ")
                    + "vs direct: " + cmpText(compare(g, d)));
        }
        for (auto& c : caps) c.destroy();
    }

    // ------------------------------------------------------------------------------------------------------------
    // Timing
    // ------------------------------------------------------------------------------------------------------------

    double pct(std::vector<double> v, double p) {
        if (v.empty()) return 0;
        std::sort(v.begin(), v.end());
        const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(p * double(v.size() - 1) + 0.5));
        return v[i];
    }

    struct Samples {
        std::vector<double> enq, e2e;
        std::string text() const {
            char buf[200];
            std::snprintf(buf, sizeof(buf), "enqueue %7.1f us [%6.1f %6.1f]   end-to-end %7.1f us [%6.1f %6.1f]",
                pct(enq, 0.5), pct(enq, 0.1), pct(enq, 0.9), pct(e2e, 0.5), pct(e2e, 0.1), pct(e2e, 0.9));
            return buf;
        }
    };

    struct Variant {
        std::string name;
        std::function<void()> enqueue; // issues the whole check on the rig's stream; no sync
        Samples s;
        std::uint32_t callsPerCheck = 0;
    };

    void timeVariants(Rig& rig, std::vector<Variant>& vs, int reps) {
        for (int w = 0; w < 50; ++w) { // warm-up
            for (Variant& v : vs) {
                v.enqueue();
                rig.sync();
            }
        }
        for (int r = 0; r < reps; ++r) {
            for (std::size_t k = 0; k < vs.size(); ++k) {
                Variant& v = vs[(k + std::size_t(r)) % vs.size()]; // rotate the order every repetition
                rig.sync();
                const auto t0 = Clock::now();
                v.enqueue();
                const auto t1 = Clock::now();
                rig.sync();
                const auto t2 = Clock::now();
                v.s.enq.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                v.s.e2e.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
            }
        }
    }

    // Host time of each step of DetectorCore::runBatch, direct calls, one batch of n (median over reps).
    void directBreakdown(Rig& rig, std::uint32_t n, int reps) {
        const char* names[] = {"memsetAsync(hits)", "staging copy", "prepare()", "params upload (cached)", "optixLaunch",
            "result copy"};
        std::vector<std::vector<double>> t(6);
        for (int r = 0; r < reps + 50; ++r) {
            rig.sync();
            Rig::PerStream& st = rig.ps[0];
            const StreamHandle s = rig.stream();
            rig.backend->beginCheck(CheckInfo{n, 1});
            Clock::time_point c[7];
            c[0] = Clock::now();
            st.hits.memsetAsync(0, std::size_t(n) * 4, s);
            c[1] = Clock::now();
            st.staging.copyFromAsync(rig.input.ptr(), std::size_t(n) * rig.fpiBytes, s);
            c[2] = Clock::now();
            BatchView view;
            view.itemCount = n;
            view.items     = DeviceSpan<const float>{st.staging.ptr(), std::size_t(n) * rig.layout.floatsPerItem()};
            view.hitCounts = DeviceSpan<std::uint32_t>{st.hits.ptr(), n};
            const LaunchPlan plan = rig.backend->prepare(rig.sc[0], view);
            c[3] = Clock::now();
            const bool same = st.uploadedValid && st.uploadedBytes == plan.paramsBytes
                           && std::memcmp(st.uploaded.data(), plan.params, plan.paramsBytes) == 0;
            if (!same) {
                st.params.uploadAsync(plan.params, plan.paramsBytes, s);
                std::memcpy(st.uploaded.data(), plan.params, plan.paramsBytes);
                st.uploadedBytes = plan.paramsBytes;
                st.uploadedValid = true;
            }
            c[4] = Clock::now();
            if (plan.launch) rig.pipe->launch(s, st.params.ptr(), plan.paramsBytes, plan);
            c[5] = Clock::now();
            cudaMemcpyAsync(toPtr(rig.results.ptr()), toPtr(st.hits.ptr()), std::size_t(n) * 4, cudaMemcpyDeviceToDevice, s);
            c[6] = Clock::now();
            if (r >= 50) {
                for (int k = 0; k < 6; ++k) t[k].push_back(std::chrono::duration<double, std::micro>(c[k + 1] - c[k]).count());
            }
        }
        rig.sync();
        std::cout << "    per-step host time, direct, n = " << n << " (median [p10 p90] us):";
        for (int k = 0; k < 6; ++k) {
            char buf[120];
            std::snprintf(buf, sizeof(buf), "\n      %-24s %6.1f [%5.1f %6.1f]", names[k], pct(t[k], 0.5), pct(t[k], 0.1), pct(t[k], 0.9));
            std::cout << buf;
        }
        std::cout << "\n";
    }

    void timing(const Env& e, const ModeSpec& m, int reps) {
        std::cout << "\n=== timing: mode " << m.name << " (" << reps
                  << " alternating repetitions per variant; median [p10 p90]; 1 stream, batch 128, calling thread) ===\n";
        Rig rig(e, m, 1, 128);
        rig.uploadItems(e.inputs, 4096);
        for (std::uint32_t n : {1u, 16u, 128u}) {
            directBreakdown(rig, n, reps);
        }
        if (m.mode != Mode::Ray) {
            std::cout << "  (graph variants: ray only -- every other prepare() contains optixAccelBuild, which cannot be captured)\n";
            return;
        }
        for (std::uint32_t count : {1u, 16u, 128u, 1024u}) {
            const std::uint32_t nb = rig.batches(count);
            auto nOf = [&](std::uint32_t b) { return std::min(rig.B(), count - b * rig.B()); };
            rig.directCheck(count);
            rig.calls = 0;
            rig.zeroResults();
            rig.enqueueCheck(count);
            rig.sync();
            // + the kernels inside the ray prepare(): linkPoses, transformObbs, obbOverlapCounts (OBB culling on)
            const std::uint32_t directCalls =
                static_cast<std::uint32_t>(rig.calls.load()) + nb * (rig.cfg.obbCullingEnabled() ? 3u : 1u);

            // A: one CUDA-part graph per batch (offsets baked), then optixLaunch + result copy direct.
            rig.backend->beginCheck(CheckInfo{count, nb});
            std::vector<Captured> pre;
            for (std::uint32_t b = 0; b < nb; ++b) {
                pre.push_back(capture(rig.stream(), [&] { rig.enqueuePre(b, count); }));
            }
            const LaunchPlan planFull = rig.ps[0].lastPlan; // ray: identical for every batch of the same n
            // B: chained graphs: G0 = pre(0); G(b) = result copy(b - 1) + pre(b); final result copy direct.
            std::vector<Captured> chain;
            for (std::uint32_t b = 0; b < nb; ++b) {
                chain.push_back(capture(rig.stream(), [&] {
                    if (b > 0) {
                        const std::uint32_t f = (b - 1) * rig.B();
                        cudaOk(cudaMemcpyAsync(toPtr(rig.results.ptr() + std::uint64_t(f) * 4), toPtr(rig.ps[0].hits.ptr()),
                                   std::size_t(nOf(b - 1)) * 4, cudaMemcpyDeviceToDevice, rig.stream()),
                            "chain copy");
                    }
                    rig.enqueuePre(b, count);
                }));
            }
            // C: ONE CUDA-part graph for a batch of nOf(0) items, re-pointed per batch with a memcpy-node update
            Captured movable = capture(rig.stream(), [&] { rig.enqueuePre(0, count); });
            const std::vector<cudaGraphNode_t> mc = memcpyNodes(movable.graph);
            bool okAll = movable.ok() && mc.size() == 1;
            for (auto& c : pre) okAll = okAll && c.ok();
            for (auto& c : chain) okAll = okAll && c.ok();
            if (!okAll) {
                std::cout << "  count " << count << ": a capture failed: " << movable.describe() << "\n";
                continue;
            }
            auto launchDirect = [&](std::uint32_t b) {
                LaunchPlan p = planFull;
                p.width      = std::min(nOf(b) * (planFull.width / std::max(1u, nOf(0))), planFull.width);
                rig.pipe->launch(rig.stream(), rig.ps[0].params.ptr(), planFull.paramsBytes, p);
            };
            // (ray params: only hitCounts / totalRayCount vary, both constant for equal n -> no upload in the loop)
            std::vector<Variant> vs;
            vs.push_back({"direct calls (DetectorCore::runBatch sequence)", [&] { rig.enqueueCheck(count); }, {}, directCalls});
            vs.push_back({"A: CUDA-part graph per batch + optixLaunch + copy", [&] {
                              for (std::uint32_t b = 0; b < nb; ++b) {
                                  cudaGraphLaunch(pre[b].exec, rig.stream());
                                  rig.enqueuePost(b, count);
                              }
                          },
                {}, 3 * nb});
            if (nb > 1) {
                vs.push_back({"B: chained graphs [copy(b-1) + pre(b)] + optixLaunch", [&] {
                                  for (std::uint32_t b = 0; b < nb; ++b) {
                                      cudaGraphLaunch(chain[b].exec, rig.stream());
                                      launchDirect(b);
                                  }
                                  const std::uint32_t f = (nb - 1) * rig.B();
                                  cudaMemcpyAsync(toPtr(rig.results.ptr() + std::uint64_t(f) * 4), toPtr(rig.ps[0].hits.ptr()),
                                      std::size_t(nOf(nb - 1)) * 4, cudaMemcpyDeviceToDevice, rig.stream());
                              },
                    {}, 2 * nb + 1});
                vs.push_back({"C: one graph + memcpy-node update per batch + optixLaunch + copy", [&] {
                                  for (std::uint32_t b = 0; b < nb; ++b) {
                                      const std::uint32_t first = b * rig.B();
                                      cudaGraphExecMemcpyNodeSetParams1D(movable.exec, mc.front(), toPtr(rig.ps[0].staging.ptr()),
                                          toPtr(rig.input.ptr() + std::uint64_t(first) * rig.fpiBytes),
                                          std::size_t(nOf(b)) * rig.fpiBytes, cudaMemcpyDeviceToDevice);
                                      cudaGraphLaunch(movable.exec, rig.stream());
                                      rig.enqueuePost(b, count);
                                  }
                              },
                    {}, 4 * nb});
            }
            timeVariants(rig, vs, reps);

            // correctness in the timed state: each graph variant once more vs direct
            const auto d = rig.directCheck(count);
            std::vector<std::size_t> diffs;
            for (std::size_t k = 1; k < vs.size(); ++k) {
                rig.zeroResults();
                vs[k].enqueue();
                rig.sync();
                diffs.push_back(compare(rig.downloadResults(count), d).flagDiffs);
            }
            std::cout << "  N = " << count << " (" << nb << " batch" << (nb > 1 ? "es" : "") << " of <= " << rig.B()
                      << "); CUDA-part graph: " << pre[0].nodes << " nodes, capture " << pre[0].captureUs
                      << " us, instantiate " << pre[0].instantiateUs << " us; flag diffs vs direct after timing:";
            for (auto x : diffs) std::cout << " " << x;
            std::cout << "\n";
            for (const Variant& v : vs) {
                char name[90];
                std::snprintf(name, sizeof(name), "%-66s %3u calls  ", v.name.c_str(), v.callsPerCheck);
                std::cout << "    " << name << v.s.text() << "\n";
            }
            movable.destroy();
            for (auto& p : pre) p.destroy();
            for (auto& p : chain) p.destroy();
        }
    }


} // namespace

int main(int argc, char** argv) {
    std::string modeArg = "all", what = "all";
    int reps            = 2000;
    int optixLog        = 0;
    std::string outFile;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next           = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--mode") modeArg = next();
        else if (a == "--what") what = next();
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--dot") g_dotDir = next();
        else if (a == "--capture-mode") {
            const std::string v = next();
            g_captureMode = v == "global" ? cudaStreamCaptureModeGlobal : v == "relaxed" ? cudaStreamCaptureModeRelaxed : cudaStreamCaptureModeThreadLocal;
        }
        else if (a == "--optix-log") optixLog = std::stoi(next());
        else if (a == "--out") outFile = next();
        else {
            std::cout << "usage: rtcd_exp_graph_capture [--mode ray|mesh|mesh_nocull|linear|quad|all] "
                         "[--what correctness|failures|threads|timing|all] [--reps N] [--dot dir] [--capture-mode thread_local|global|relaxed] [--optix-log 0..4]\n";
            return a == "--help" ? 0 : 2;
        }
    }
    std::ofstream outStream;
    if (!outFile.empty()) {
        outStream.open(outFile);
        std::cout.rdbuf(outStream.rdbuf());
    }
    try {
        int rt = 0, drv = 0;
        cudaRuntimeGetVersion(&rt);
        cudaDriverGetVersion(&drv);
        std::cout << "CUDA runtime " << rt << ", driver API " << drv << "\n";
        const char* dl = std::getenv("OPTIX_FORCE_DEPRECATED_LAUNCHER");
        std::cout << "OPTIX_FORCE_DEPRECATED_LAUNCHER = " << (dl ? dl : "(unset)") << "\n";
        CUlogsCallbackHandle logHandle = nullptr;
        std::cout << "cuLogsRegisterCallback: " << int(cuLogsRegisterCallback(cudaLogCallback, nullptr, &logHandle)) << "\n";
        std::cout << "capture mode: " << (g_captureMode == cudaStreamCaptureModeGlobal ? "global" : g_captureMode == cudaStreamCaptureModeRelaxed ? "relaxed" : "thread_local") << "\n";
        ContextOptions co = ContextOptions::buildDefault();
        co.optixLogLevel  = optixLog;
        co.logger.level   = LogLevel::Debug;
        co.logger.sink    = [](LogLevel l, std::string_view msg) { std::cout << "    [rtcd log " << int(l) << "] " << msg << "\n"; };
        auto ctx = Context::create(co);
        std::vector<std::string> modes;
        if (modeArg == "all") modes = {"ray", "mesh", "mesh_nocull", "linear", "quad"}; // "all": what = timing / threads
        else modes = {modeArg};
        for (const std::string& name : modes) {
            const ModeSpec m = modeSpec(name);
            const Env e      = makeEnv(ctx, m);
            if (what == "launch") launchOnly(e, m);
            if (what == "fallback_update" && isContinuous(m.mode)) fallbackCorrectness(e, m, RefitPolicy::Kind::Always);
            // Experiments that put an OptiX call into a capture (launch, correctness, failures, fallback for the AS-
            // building modes) leave the OptiX context with an invalid internal event: run each in its own process.
            if (what == "fallback" || (what == "all" && m.mode == Mode::Ray)) {
                if (isContinuous(m.mode)) {
                    fallbackCorrectness(e, m, RefitPolicy::Kind::Never);
                    fallbackCorrectness(e, m, RefitPolicy::Kind::Always);
                } else {
                    fallbackCorrectness(e, m);
                }
            }
            if (what == "correctness") correctness(e, m);
            if (what == "all" || what == "threads") multiThreaded(e, m);
            if (what == "all" || what == "timing") timing(e, m, reps);
        }
        if (what == "failures") expectedFailures(ctx);
    } catch (const std::exception& ex) {
        std::cout << "FATAL: " << ex.what() << "\n";
        return 3;
    }
    std::cout << "\nfailures: " << g_failures << "\n";
    return g_failures == 0 ? 0 : 1;
}
