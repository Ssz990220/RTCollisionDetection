#pragma once
// Old-stack half of the side-by-side runner: a plain C++ interface (no CUDA / OptiX / old-stack headers), so the
// new-stack code never sees the old headers. Implemented in old_profiles.cpp (the only TU that includes the old
// detector) and old_scene.cpp (the old compile-time scene, one per executable: SHELF / SHELFSIMPLE / DENSESHELF).
//
// The old stack is templated on compile-time values, so only a fixed list of configurations ("profiles") exists,
// each one an exact copy of an old regression / benchmark setup. Deleted with the old stack.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rtcd::sbs {

    enum class OldMode { Ias, Ray, Linear, Quad };
    enum class OldBuild { Compact, FastTrace, FastBuild, None };

    struct OldProfile {
        std::string name;        // e.g. "bm_ias"
        std::string origin;      // which old executable / benchmark it reproduces
        OldMode mode;
        std::size_t streams;     // NSTREAM
        std::size_t batchSize;   // BATCHSIZE
        std::size_t trajPts;     // 1 for discrete
        std::size_t ctrlPts;     // curves: control points (linear: = trajPts), else 0
        OldBuild build;          // batchRobotConfig::BUILD
        bool obb;                // batchRobotConfig::useOBB
        std::size_t maxItems;    // pose buffer capacity (items)
    };

    // All compiled-in profiles.
    const std::vector<OldProfile>& oldProfiles();
    const OldProfile* findOldProfile(const std::string& name);

    struct OldSettings {
        std::string optixConfig;                    // old RTCD_OPTIX_CFG syntax (OptixCompileConfig::setFields)
        std::optional<std::size_t> refitEvery;      // curves: setRefitPolicy(K); nullopt = library default (auto)
        std::optional<std::string> curveGeomFlag;   // curves: "none" | "single_anyhit"; nullopt = default
    };

    class OldRunner {
    public:
        virtual ~OldRunner() = default;
        virtual const OldProfile& profile() const = 0;
        // Untimed: prepareUpdateData(items, n) (+ resetGraph for the discrete detectors, as BM_IAS / BM_RAY),
        // setNCheck(n), device sync. `items` = n packed items of the old layout (= new layout).
        virtual void upload(const float* items, std::size_t n) = 0;
        // Timed: CollisionDetector::check() (ends with cudaDeviceSynchronize, the old timed region).
        virtual void check() = 0;
        // Untimed: per-item hit counts of the last check (first n).
        virtual std::vector<std::uint32_t> hitCounts(std::size_t n) = 0;
    };

    // Builds the old detector for `profileName` on the given OptiX device context (an OptixDeviceContext passed
    // as void*; the CUDA device must be current). Creates its own NSTREAM CUDA streams (default flags, as the old
    // createContextStream). Throws std::runtime_error for an unknown profile or bad settings. (The old stack exits
    // the process on CUDA / OptiX errors.)
    std::unique_ptr<OldRunner> createOldRunner(const std::string& profileName, void* optixContext, const OldSettings& settings);

    // Name of the old scene compiled into this executable ("shelf", "shelf_simple", "shelf_dense"), matching
    // configs/scenes/<name>.json.
    const char* oldSceneName();

} // namespace rtcd::sbs
