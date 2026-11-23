// Collision-result regression check for every detector mode (one executable per mode, selected by a
// compile definition: REG_IAS, REG_RAY, REG_LINEAR or REG_QUAD).
//
// Checks, on deterministic inputs (cached poses / KNN trajectories under data/motions):
//   1. prefix consistency: checking only the first N items must give the same per-item collision flags as
//      the first N items of a full run (catches partial-batch / stream-offset bugs, needs no ground truth);
//   2. determinism: two identical full runs agree;
//   3. golden diff: flags match data/CCTest/golden_<mode>.bin (record it with --write-golden).
// Exit code 0 = all checks passed.
#ifndef REG_SINGLE_THREAD // define to exercise the single-threaded check() path
#define USE_MULTI_THREADING
#endif

#include <CollisionDetector/CCRobotWrapper.h>
#include <CollisionDetector/CollisionDetector.h>
#include <CollisionScenes/Scene/shelf.h>
#include <CollisionScenes/batchScene.h>
#include <Robot/batchCurveRobot.h>
#include <Robot/batchIASRobot.h>
#include <Robot/batchRayRobot.h>
#include <Robot/models/pandas.h>
#include <Utils/Test/testUtils.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace RTCD;

namespace {
    constexpr size_t NSTREAM   = 4;
#ifndef REG_BATCHSIZE
#define REG_BATCHSIZE 128
#endif
    constexpr size_t BATCHSIZE = REG_BATCHSIZE; // small on purpose: 4096 items = many multi-stream rounds
    constexpr size_t NITEMS    = 4096; // poses (discrete) or trajectories (curve)
    constexpr int KNN_K        = 4;
    constexpr std::array<float, Dim> weight{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    constexpr batchSceneConfig sceneCfg(false, NSTREAM, 0);

#if defined(REG_IAS)
    constexpr LinkType TYPE  = LinkType::MESH;
    constexpr const char* MODE = "ias";
    constexpr batchRobotConfig CFG(NSTREAM, BATCHSIZE, Dim, BuildType::COMPACT, TYPE, true);
    constexpr size_t nTrajPts  = 1; // one pose per item
    constexpr bool CURVE = false;
#elif defined(REG_RAY)
    constexpr LinkType TYPE  = LinkType::RAY_ONLINE;
    constexpr const char* MODE = "ray";
    constexpr batchRobotConfig CFG(NSTREAM, BATCHSIZE, Dim, BuildType::NONE, TYPE, true);
    constexpr size_t nTrajPts  = 1; // one pose per item
    constexpr bool CURVE = false;
#elif defined(REG_LINEAR)
    constexpr LinkType TYPE    = LinkType::LINEAR_CURVE;
    constexpr const char* MODE = "linear";
    constexpr size_t nTrajPts  = 16;
    constexpr batchRobotConfig CFG{NSTREAM, BATCHSIZE, Dim, BuildType::FAST_BUILD, TYPE, nTrajPts, 16, false};
    constexpr bool CURVE = true;
#elif defined(REG_QUAD)
    constexpr LinkType TYPE    = LinkType::QUADRATIC_CURVE;
    constexpr const char* MODE = "quad";
    constexpr size_t nTrajPts  = 16;
    constexpr batchRobotConfig CFG{NSTREAM, BATCHSIZE, Dim, BuildType::FAST_BUILD, TYPE, nTrajPts, 8, false};
    constexpr bool CURVE = true;
#else
#error "Define one of REG_IAS, REG_RAY, REG_LINEAR, REG_QUAD"
#endif

    using Flags = std::vector<uint8_t>;

    // Runs one check() over the first n items of `data` and returns the per-item collision flags.
    template <typename Detector, typename Robot>
    Flags runCheck(Detector& cc, Robot& robot, const float* data, size_t n) {
        robot->prepareUpdateData(data, n);
        cc.setNCheck(static_cast<uint>(n));
        cc.check();
        CUDA_SYNC_CHECK();
        std::vector<uint> raw(n, 0);
        cc.downloadResult(raw);
        Flags flags(n);
        for (size_t i = 0; i < n; i++) {
            flags[i] = raw[i] > 0 ? 1 : 0;
        }
        return flags;
    }

    // Number of differing flags among the first n; the first few differing indices are printed.
    size_t countDiff(const Flags& a, const Flags& b, size_t n) {
        size_t d = 0;
        for (size_t i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                if (d < 8) {
                    std::cout << "    diff at item " << i << ": " << int(a[i]) << " vs " << int(b[i]) << std::endl;
                }
                d++;
            }
        }
        return d;
    }

    size_t countHits(const Flags& f) {
        size_t c = 0;
        for (auto v : f) {
            c += v;
        }
        return c;
    }

    // Deterministic inputs: cached under data/motions, generated once if missing.
    std::vector<float> loadInputs(const std::string& robotName) {
        if constexpr (CURVE) {
            std::vector<std::array<float, Dim>> traj(NITEMS * KNN_K * nTrajPts);
            genCSpaceKNNTraj<Dim, NITEMS, KNN_K>(traj, nTrajPts, upperBound, lowerBound, robotName, weight);
            std::vector<float> out(NITEMS * nTrajPts * Dim);
            std::memcpy(out.data(), traj.data(), out.size() * sizeof(float)); // first NITEMS trajectories
            return out;
        } else {
            std::vector<std::array<float, Dim>> poses(NITEMS);
            randomMotions(NITEMS, poses, upperBound, lowerBound, robotName);
            std::vector<float> out(NITEMS * Dim);
            std::memcpy(out.data(), poses.data(), out.size() * sizeof(float));
            return out;
        }
    }
} // namespace

int main(int argc, char** argv) {
    const bool writeGolden = argc > 1 && std::string(argv[1]) == "--write-golden";

    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();
    auto r                                    = std::make_shared<robot<Dim>>(Config);

#if defined(REG_IAS)
    auto robotImpl = std::make_shared<batchIASRobot<CFG>>(r, streams, optixContext);
    auto ccRobot   = std::make_shared<CCWrappedRobot<NSTREAM, TYPE>>(robotImpl);
#elif defined(REG_RAY)
    auto robotImpl = std::make_shared<batchRayRobot<CFG>>(r, streams, optixContext);
    auto ccRobot   = std::make_shared<CCRayRobot<NSTREAM, TYPE>>(robotImpl);
#else
    auto robotImpl = std::make_shared<batchOptixCurveRobot<CFG>>(r, streams, optixContext);
    auto ccRobot   = std::make_shared<CCWrappedRobot<NSTREAM, TYPE>>(robotImpl);
#endif
    ccRobot->allocPoseBuffer(NITEMS);

    std::shared_ptr<scene> s                    = buildSharedScene(CURVE); // loop edges for curves
    std::shared_ptr<baseBatchScene<NSTREAM>> bs = std::make_shared<batchScene<sceneCfg>>(s, streams, optixContext);

    CollisionDetector<NSTREAM, TYPE> cc(optixContext, streams);
    cc.setupScene(bs);
    cc.setupOptix();
    cc.setupRobot(ccRobot);
    cc.buildSBT();

    const std::vector<float> inputs = loadInputs(std::string(r->name));
    bool ok                         = true;

    // Full run, twice (determinism).
    const Flags full  = runCheck(cc, ccRobot, inputs.data(), NITEMS);
    const Flags full2 = runCheck(cc, ccRobot, inputs.data(), NITEMS);
    const size_t nondet = countDiff(full, full2, NITEMS);
    std::cout << "[" << MODE << "] full run: " << countHits(full) << "/" << NITEMS << " in collision; repeat-run diffs: "
              << nondet << std::endl;
    ok &= nondet == 0;

    // Prefix consistency, including partial batches and partial stream rounds.
    for (size_t n : {size_t(1), size_t(37), BATCHSIZE, BATCHSIZE * NSTREAM + 5, size_t(1000), NITEMS - 1}) {
        const Flags part  = runCheck(cc, ccRobot, inputs.data(), n);
        const size_t diff = countDiff(part, full, n);
        std::cout << "[" << MODE << "] prefix N=" << n << ": " << diff << " diffs vs full run" << std::endl;
        ok &= diff == 0;
    }

    // Golden comparison.
    const std::string goldenDir  = CONCAT_PATHS(PROJECT_BASE_DIR, "/data/CCTest");
    const std::string goldenPath = goldenDir + "/golden_" + MODE + ".bin";
    if (writeGolden) {
        std::filesystem::create_directories(goldenDir);
        std::ofstream(goldenPath, std::ios::binary).write(reinterpret_cast<const char*>(full.data()), full.size());
        std::cout << "[" << MODE << "] wrote golden: " << goldenPath << std::endl;
    } else if (std::filesystem::exists(goldenPath)) {
        Flags golden(NITEMS);
        std::ifstream(goldenPath, std::ios::binary).read(reinterpret_cast<char*>(golden.data()), golden.size());
        const size_t diff = countDiff(full, golden, NITEMS);
        std::cout << "[" << MODE << "] golden: " << diff << " diffs (golden has " << countHits(golden) << " hits)"
                  << std::endl;
        ok &= diff == 0;
    } else {
        std::cout << "[" << MODE << "] no golden file (" << goldenPath << "); run with --write-golden to record one"
                  << std::endl;
    }

    std::cout << "[" << MODE << "] " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
