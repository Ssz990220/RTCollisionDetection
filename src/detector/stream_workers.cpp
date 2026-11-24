// src/detector: StreamWorkers (stream_workers.h).
#include "detector/stream_workers.h"

#include <rtcd/errors.h>

#include <string>

namespace rtcd::detail {

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
        for (;;) {
            self.go.acquire();
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
            done_.acquire();
        }
        job_ = nullptr;
        std::vector<std::exception_ptr> errors(active);
        for (std::uint32_t i = 0; i < active; ++i) {
            errors[i] = std::move(workers_[i]->error);
        }
        return errors;
    }

} // namespace rtcd::detail
