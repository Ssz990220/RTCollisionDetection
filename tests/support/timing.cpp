#include <support/timing.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace rtcd::support {

    namespace {
        double percentile(const std::vector<double>& sorted, double q) {
            if (sorted.size() == 1) {
                return sorted[0];
            }
            const double pos = q * double(sorted.size() - 1);
            const auto lo    = static_cast<std::size_t>(std::floor(pos));
            const auto hi    = std::min(lo + 1, sorted.size() - 1);
            const double f   = pos - double(lo);
            return sorted[lo] * (1.0 - f) + sorted[hi] * f;
        }
    } // namespace

    TimingStats summarize(std::vector<double> us) {
        TimingStats s;
        s.samples = us.size();
        if (us.empty()) {
            return s;
        }
        std::sort(us.begin(), us.end());
        s.min    = us.front();
        s.max    = us.back();
        s.mean   = std::accumulate(us.begin(), us.end(), 0.0) / double(us.size());
        s.median = percentile(us, 0.5);
        s.p10    = percentile(us, 0.1);
        s.p90    = percentile(us, 0.9);
        double var = 0;
        for (const double v : us) {
            var += (v - s.mean) * (v - s.mean);
        }
        s.stddev = us.size() > 1 ? std::sqrt(var / double(us.size() - 1)) : 0.0;
        return s;
    }

} // namespace rtcd::support
