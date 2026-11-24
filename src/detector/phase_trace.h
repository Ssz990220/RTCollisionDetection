#pragma once
// src/detector (private): PhaseTrace -- optional host-side timestamps of one check, for the host-time breakdown
// (Track G2, tests/experiments/host_breakdown). Diagnostic only: DetectorCore holds a PhaseTrace pointer that is
// null by default (DetectorCore::setPhaseTrace); every mark point is `if (trace_) trace_->mark(...)`, so the cost
// when off is one predictable branch per mark point (about 15 per batch).
//
// Slots: slot 0 = the checker's calling thread; slot 1 + s = whatever thread issues stream s's batches (a worker
// with ThreadPerStream, the caller with SingleThread). Each slot is written by one thread at a time; the caller
// reads all slots after the check returned (the worker handoff orders the writes before that).
// Events are (phase, steady_clock ticks) appended in time order per slot. A phase event marks the END of that
// phase; the duration of a phase is the time since the previous event of the same slot, except for the marks
// listed as "mark only" below, which open a new interval.

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

namespace rtcd::detail {

    enum class Phase : std::uint8_t {
        // caller (slot 0)
        Call,          // mark only: set by the experiment right before calling the check
        Entry,         // argument checks + Context::makeCurrent
        Upload,        // checkHost: host items -> input buffer (synchronous copy)
        BeginCheck,    // counters reset + backend.beginCheck
        Dispatch,      // mark only: right before the batches are issued (workers woken / inline loop)
        Joined,        // all workers returned (StreamWorkers::run returned); inline: the batch loop ended
        SyncUsed,      // cudaStreamSynchronize of the streams that ran batches of this check
        SyncIdle,      // cudaStreamSynchronize of the checker's other streams
        SyncProbe,     // diagnostic only (PhaseTrace::probeIdleSync): the same synchronization again, on idle streams
        Stats,         // stats gathered
        Download,      // checkHost: counters -> pinned host buffer
        Flags,         // checkHost: flags / counts written to the caller's spans
        Return,        // set by the experiment right after the check returned
        // batch issuing thread (slot 1 + s)
        BatchStart,    // mark only: runBatch entered (first one per slot: the worker's wake-up)
        Zero,          // memsetAsync of the hit counters
        Staging,       // staging copy
        Prepare,       // backend.prepare()
        GraphLaunch,   // graph_prepare: (capture +) cudaGraphLaunch replacing Zero + Staging + Prepare
        Params,        // validateLaunchPlan + params compare / upload
        Launch,        // optixLaunch
        ResultCopy,    // copy of the hit counters into the check's result range
        BatchEnd,      // counters, end of runBatch
        Count
    };

    inline const char* phaseName(Phase p) {
        static constexpr std::array<const char*, static_cast<std::size_t>(Phase::Count)> names{"call", "entry",
            "upload", "begin_check", "dispatch", "joined", "sync_used", "sync_idle", "sync_probe", "stats",
            "download", "flags", "return", "batch_start", "zero", "staging", "prepare", "graph_launch", "params",
            "launch", "result_copy", "batch_end"};
        return names[static_cast<std::size_t>(p)];
    }

    struct PhaseTrace {
        struct Event {
            Phase phase;
            std::int64_t ticks; // std::chrono::steady_clock ticks (ns on MSVC / libstdc++)
        };
        static constexpr std::size_t kMaxEventsPerSlot = 4096;

        // Diagnostic: after the check's own synchronization, synchronize every stream once more (all idle by
        // then) and record it as SyncProbe, i.e. the pure host cost of the synchronization calls. Off by default.
        bool probeIdleSync = false;

        std::vector<std::vector<Event>> slots; // 1 + numStreams entries (resize before use)

        void reset(std::size_t numStreams) {
            slots.resize(1 + numStreams);
            for (auto& s : slots) {
                s.clear();
                s.reserve(kMaxEventsPerSlot);
            }
        }
        static std::int64_t now() noexcept { return std::chrono::steady_clock::now().time_since_epoch().count(); }
        void mark(std::size_t slot, Phase p) noexcept {
            if (slot < slots.size() && slots[slot].size() < slots[slot].capacity()) {
                slots[slot].push_back(Event{p, now()}); // never reallocates: capacity reserved by reset()
            }
        }
    };

} // namespace rtcd::detail
