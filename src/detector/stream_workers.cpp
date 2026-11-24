// src/detector: StreamWorkers (stream_workers.h).
#include "detector/stream_workers.h"

#include <rtcd/errors.h>

#include <chrono>
#include <string>
#include <thread>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define RTCD_CPU_RELAX() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define RTCD_CPU_RELAX() _mm_pause()
#else
#define RTCD_CPU_RELAX() std::this_thread::yield()
#endif

namespace rtcd::detail {

    namespace {
        // Acquire `sem` once: busy-wait (try_acquire) for up to `spinUs`, then block.
        template <class Semaphore>
        void acquireSpinning(Semaphore& sem, std::uint32_t spinUs) {
            if (spinUs != 0) {
                const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(spinUs);
                do {
                    for (int k = 0; k < 64; ++k) {
                        if (sem.try_acquire()) {
                            return;
                        }
                        RTCD_CPU_RELAX();
                    }
                } while (std::chrono::steady_clock::now() < until);
            }
            sem.acquire();
        }
    } // namespace

    StreamWorkers::StreamWorkers(std::uint32_t count) {
        workers_.reserve(count);
        try {
            for (std::uint32_t i = 0; i < count; ++i) {
                workers_.push_back(std::make_unique<Worker>());
                workers_.back()->thread = std::thread([this, i] { loop(i); });
            }
        } catch (...) {
            // std::thread creation failed (system_error): stop the threads already started.
            stop_ = true;
            for (auto& w : workers_) {
                if (w->thread.joinable()) {
                    w->go.release();
                    w->thread.join();
                }
            }
            throw;
        }
    }

    StreamWorkers::~StreamWorkers() {
        stop_ = true;
        for (auto& w : workers_) {
            if (w->thread.joinable()) {
                w->go.release();
            }
        }
        for (auto& w : workers_) {
            if (w->thread.joinable()) {
                w->thread.join();
            }
        }
    }

    void StreamWorkers::loop(std::uint32_t index) {
        Worker& self = *workers_[index];
        bool first   = true;
        for (;;) {
            // Spin only between jobs (after a finished job), never before the first one.
            acquireSpinning(self.go, first ? 0u : wakeSpinUs_.load(std::memory_order_relaxed));
            first = false;
            if (stop_) {
                return;
            }
            try {
                (*job_)(index);
            } catch (...) {
                self.error = std::current_exception();
            }
            done_.release();
        }
    }

    std::vector<std::exception_ptr> StreamWorkers::run(std::uint32_t active, const Job& job) {
        if (active > workers_.size()) {
            throw Error(ErrorKind::Internal, "StreamWorkers::run: " + std::to_string(active) + " active workers > "
                                                 + std::to_string(workers_.size()));
        }
        job_ = &job;
        for (std::uint32_t i = 0; i < active; ++i) {
            workers_[i]->error = nullptr;
            workers_[i]->go.release();
        }
        for (std::uint32_t i = 0; i < active; ++i) {
            acquireSpinning(done_, joinSpinUs_);
        }
        job_ = nullptr;
        std::vector<std::exception_ptr> errors(active);
        for (std::uint32_t i = 0; i < active; ++i) {
            errors[i] = std::move(workers_[i]->error);
        }
        return errors;
    }

} // namespace rtcd::detail
