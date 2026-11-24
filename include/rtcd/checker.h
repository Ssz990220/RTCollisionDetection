#pragma once
// rtcd/checker.h -- CollisionChecker, the public facade of the new stack.
//
//   auto ctx   = rtcd::Context::create();
//   auto robot = rtcd::RobotModel::load(robotDesc);
//   std::shared_ptr<const rtcd::Scene> scene = std::make_shared<rtcd::Scene>(ctx, sceneDesc);
//   rtcd::CollisionChecker checker(ctx, config, robot, scene);
//   rtcd::CheckResult r = checker.check(rtcd::HostItems{poses, n});   // r.collides[i] == 1: item i collides
//
// Construction: resolves and validates the config, checks it against the robot / scene, leases numStreams
// streams from the Context, creates the backend for config.mode (which allocates everything), builds the OptiX
// modules / program groups / one pipeline / SBT from the backend's ProgramSet, allocates the pinned input / result
// buffers for maxItemsPerCheck, starts the worker threads (HostDispatch::ThreadPerStream). Nothing is
// allocated later. Throws ConfigError, IoError, CapacityError, UnsupportedError, CudaError, OptixError.
//
// check(): blocking. Splits the items into batches of maxBatchSize (batch b = items [b*maxBatchSize, ...),
// stream b % numStreams), runs the backend loop (rtcd/backend.h), synchronizes the checker's streams (not the
// whole device) and returns when the results are in the caller's memory. Per-item flag = hitCount > 0.
// Flags are the contract; hit counts are diagnostic (they depend on mode, geometry flags and skip-hit, and may
// be lower bounds).
//
// Thread safety: a CollisionChecker is NOT thread-safe: calls on one checker must not overlap (serialize them
// externally). Different checkers may run concurrently from different threads, also when they share a
// Context, RobotModel or Scene (each checker leases its own streams). check() itself may use numStreams
// internal worker threads; the Logger sink can be called from them.
//
// Error behaviour: argument problems (sizes, null data, count > maxItemsPerCheck, count == 0 allowed and
// returns empty results) throw InvalidArgument / CapacityError before any GPU work, the checker stays usable.
// A CUDA / OptiX failure during a check throws after all streams are drained; if CudaError::sticky() the
// checker (and its Context) is unusable.

#include <rtcd/config.h>
#include <rtcd/context.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>
#include <rtcd/types.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rtcd {

    enum class ResultDetail : std::uint8_t { Flags, FlagsAndHitCounts };

    struct CheckResult {
        std::vector<std::uint8_t> collides;   // one entry per item: 1 = in collision, 0 = free
        std::vector<std::uint32_t> hitCounts; // filled only with ResultDetail::FlagsAndHitCounts
        std::size_t numColliding() const {
            std::size_t n = 0;
            for (auto c : collides) {
                n += c;
            }
            return n;
        }
    };

    // Counters of the last completed check (host-side bookkeeping, no GPU queries).
    struct CheckStats {
        std::uint32_t items     = 0;
        std::uint32_t batches   = 0;
        std::uint32_t launches  = 0; // batches whose LaunchPlan had launch == true
        std::uint32_t hostSyncs = 0; // batches for which Backend::synchronizesHost(itemCount) was true
        std::uint32_t paramUploads = 0; // launch-parameter uploads actually issued (identical params are skipped)
    };

    class CollisionChecker {
    public:
        CollisionChecker(std::shared_ptr<Context> context, const CheckerConfig& config,
            std::shared_ptr<const RobotModel> robot, std::shared_ptr<const Scene> scene);
        ~CollisionChecker(); // drains its streams, joins workers; never throws
        CollisionChecker(CollisionChecker&&) noexcept;
        CollisionChecker& operator=(CollisionChecker&&) noexcept;
        CollisionChecker(const CollisionChecker&)            = delete;
        CollisionChecker& operator=(const CollisionChecker&) = delete;

        const CheckerConfig& config() const noexcept; // the resolved config actually used
        ItemLayout itemLayout() const noexcept;       // dof, pointsPerItem, floatsPerItem
        const RobotModel& robot() const noexcept;
        const Scene& scene() const noexcept;

        // Host items -> host results. Packs the items into a pinned host buffer that the GPU reads in place and
        // gets the counters back through a second pinned buffer (no copy-engine transfers: on Windows a small
        // H2D / D2H copy costs 20-60 us of latency, a mapped read / write by a kernel a few us;
        // CheckerConfig::mappedHostTransfer = false restores upload + download). Allocates only the returned vectors.
        CheckResult check(const HostItems& items, ResultDetail detail = ResultDetail::Flags);

        // Same, writing into caller memory (no allocation). collides.size() and, if not empty, hitCounts.size()
        // must be >= items.count (InvalidArgument otherwise).
        void check(const HostItems& items, std::span<std::uint8_t> collides, std::span<std::uint32_t> hitCounts = {});

        // Device-resident items and results, both caller-owned memory on the checker's device. hitCounts receives
        // items.count uint32 counters (item collides iff > 0). Blocks until the counters are written. This is the
        // path benchmarks time (it matches the old timed region: per-batch work + result copies + final sync,
        // without the host upload / download).
        void checkDevice(const DeviceItems& items, DeviceSpan<std::uint32_t> hitCounts);

        // One full-capacity check (numStreams * maxBatchSize items, capped at maxItemsPerCheck) on deterministic
        // pseudo-random items within the joint limits; results discarded. Warms up OptiX / driver paths.
        void warmup();

        CheckStats lastStats() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rtcd
