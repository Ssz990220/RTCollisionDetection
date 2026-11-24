#pragma once
// support/timing.h -- summary statistics of wall-clock samples (microseconds) for the benchmark apps. Pure host
// code. (Agents never report timings themselves: main runs these apps.)

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace rtcd::support {

    struct TimingStats {
        std::size_t samples = 0;
        double mean = 0, median = 0, min = 0, max = 0, p10 = 0, p90 = 0, stddev = 0; // microseconds
    };

    // Summary of `us` (any order). Percentiles by linear interpolation between order statistics.
    // Empty input gives samples == 0 and zeros.
    TimingStats summarize(std::vector<double> us);

    // Steady-clock stopwatch in microseconds.
    class Stopwatch {
    public:
        void start() { t0_ = std::chrono::steady_clock::now(); }
        double elapsedUs() const {
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0_).count();
        }

    private:
        std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
    };

} // namespace rtcd::support
