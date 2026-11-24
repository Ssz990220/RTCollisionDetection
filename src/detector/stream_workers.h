#pragma once
// src/detector (private): persistent worker threads, one per checker stream (HostDispatch::ThreadPerStream).
//
// Replaces the old RTCD/CollisionDetector/threadPool.h (ThreadPool<NSTREAM>, a task queue drained by any thread,
// with a host barrier after every round of NSTREAM batches). Here worker i is bound to stream i and runs ALL of
// that stream's batches of a check in order, so there is no queue and no per-round barrier: the only
// synchronization is "start" (one semaphore release per active worker) and "done" (one counting semaphore).
//
// Threading: run() must not be called concurrently with itself (the checker is not thread-safe anyway).

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

namespace rtcd::detail {

    class StreamWorkers {
    public:
        using Job = std::function<void(std::uint32_t worker)>;

        explicit StreamWorkers(std::uint32_t count); // starts `count` threads (they sleep until run())
        ~StreamWorkers();                            // stops and joins; never throws
        StreamWorkers(const StreamWorkers&)            = delete;
        StreamWorkers& operator=(const StreamWorkers&) = delete;

        std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(workers_.size()); }

        // Runs job(i) on worker thread i for every i < active (active <= size()), blocks until all of them have
        // returned. An exception escaping job(i) is caught and returned in slot i (empty slots = success); the
        // other workers are not interrupted (the job itself decides whether to stop early).
        std::vector<std::exception_ptr> run(std::uint32_t active, const Job& job);

        // Busy-wait budgets before blocking on a semaphore: the caller waiting for the workers to finish (join)
        // and an idle worker waiting for its next job (wake). 0 = block at once. Call only between run()s.
        void setSpin(std::uint32_t joinSpinUs, std::uint32_t wakeSpinUs) noexcept {
            joinSpinUs_ = joinSpinUs;
            wakeSpinUs_.store(wakeSpinUs, std::memory_order_relaxed);
        }

        // Thread id of worker i (tests: which thread ran a prepare()).
        std::thread::id threadId(std::uint32_t i) const { return workers_.at(i)->thread.get_id(); }

    private:
        struct Worker {
            std::binary_semaphore go{0};
            std::exception_ptr error;
            std::thread thread;
        };
        void loop(std::uint32_t index);

        std::vector<std::unique_ptr<Worker>> workers_;
        std::counting_semaphore<> done_{0};
        const Job* job_ = nullptr; // valid while run() waits
        bool stop_      = false;   // written before the final releases (semaphore release/acquire orders it)
        std::uint32_t joinSpinUs_ = 0;
        std::atomic<std::uint32_t> wakeSpinUs_{0};
    };

} // namespace rtcd::detail
