#include <support/inputs.h>

#include <rtcd/errors.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace rtcd::support {

    namespace {
        template <typename T>
        std::vector<T> readElements(const fs::path& file) {
            std::ifstream in(file, std::ios::binary);
            if (!in) {
                throw IoError("cannot open input cache file (the old harness generates it; see support/inputs.h)",
                    file);
            }
            in.seekg(0, std::ios::end);
            const std::streamoff bytes = in.tellg();
            in.seekg(0, std::ios::beg);
            if (bytes < 0) {
                throw IoError("cannot determine the file size", file);
            }
            std::vector<T> data(static_cast<std::size_t>(bytes) / sizeof(T));
            if (!data.empty()) {
                in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
                if (!in) {
                    throw IoError("read error", file);
                }
            }
            return data;
        }
    } // namespace

    std::vector<float> readFloatFile(const fs::path& file) { return readElements<float>(file); }
    std::vector<std::int32_t> readInt32File(const fs::path& file) { return readElements<std::int32_t>(file); }

    std::string knnWeightString(std::span<const float> weights) {
        std::ostringstream ss;
        for (const float w : weights) {
            ss << static_cast<int>(w);
        }
        return ss.str();
    }

    std::vector<float> loadCachedPoses(
        const fs::path& motionsDir, std::string_view robotName, std::size_t count, std::uint32_t dof) {
        if (dof == 0) {
            throw InvalidArgument("loadCachedPoses: dof must be >= 1");
        }
        const fs::path file = motionsDir / (std::string(robotName) + std::to_string(count) + ".bin");
        std::vector<float> data = readFloatFile(file);
        const std::size_t poses = data.size() / dof;
        if (poses != count) {
            throw IoError("holds " + std::to_string(poses) + " poses of " + std::to_string(dof) + " floats, expected "
                              + std::to_string(count),
                file);
        }
        data.resize(poses * dof); // drop a trailing partial pose, as the old readBin did
        return data;
    }

    std::vector<std::int32_t> loadKnnIndices(const fs::path& motionsDir, std::string_view robotName,
        std::string_view weightString, std::size_t samples, std::uint32_t k) {
        const fs::path file = motionsDir / std::string(robotName) / std::string(weightString)
                            / (std::to_string(samples) + "_knn" + std::to_string(k) + "_idx.bin");
        std::vector<std::int32_t> idx = readInt32File(file);
        if (idx.size() != samples * k) {
            throw IoError("holds " + std::to_string(idx.size()) + " indices, expected " + std::to_string(samples)
                              + " * " + std::to_string(k),
                file);
        }
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] < 0 || static_cast<std::size_t>(idx[i]) >= samples) {
                throw IoError("index " + std::to_string(i) + " = " + std::to_string(idx[i]) + " is out of range [0, "
                                  + std::to_string(samples) + ")",
                    file);
            }
        }
        return idx;
    }

    std::vector<float> knnTrajectories(std::span<const float> samples, std::uint32_t dof,
        std::span<const std::int32_t> knnIndices, std::uint32_t k, std::uint32_t trajectoryPoints,
        std::size_t maxTrajectories) {
        if (dof == 0 || k == 0 || trajectoryPoints < 2) {
            throw InvalidArgument("knnTrajectories: dof and k must be >= 1 and trajectoryPoints >= 2");
        }
        if (samples.size() % dof != 0) {
            throw InvalidArgument("knnTrajectories: sample array is not a whole number of poses");
        }
        const std::size_t nSamples = samples.size() / dof;
        if (knnIndices.size() != nSamples * k) {
            throw InvalidArgument("knnTrajectories: " + std::to_string(knnIndices.size()) + " indices for "
                                  + std::to_string(nSamples) + " samples and k = " + std::to_string(k));
        }
        const std::size_t total = std::min(maxTrajectories, nSamples * k);
        std::vector<float> out(total * trajectoryPoints * dof);
        const float denom = static_cast<float>(trajectoryPoints - 1); // old: (float) (nTrajPts - 1)
        std::vector<float> span(dof);
        for (std::size_t t = 0; t < total; ++t) {
            const std::size_t i     = t / k;
            const std::size_t n     = static_cast<std::size_t>(knnIndices[t]);
            if (n >= nSamples) {
                throw InvalidArgument("knnTrajectories: neighbour index out of range");
            }
            const float* start = samples.data() + i * dof;
            const float* end   = samples.data() + n * dof;
            // old ArrayMath: (end - start) materialized, then / float, each into its own array
            for (std::uint32_t d = 0; d < dof; ++d) {
                const float diff = end[d] - start[d];
                span[d]          = diff / denom;
            }
            float* dst = out.data() + t * trajectoryPoints * dof;
            for (std::uint32_t p = 0; p < trajectoryPoints; ++p) {
                const float fp = static_cast<float>(p);
                for (std::uint32_t d = 0; d < dof; ++d) {
                    // old: start + (span * (float) k): the product is stored in a temporary array first
                    volatile float prod = span[d] * fp; // volatile: never contracted into an FMA
                    dst[p * dof + d]    = start[d] + prod;
                }
            }
        }
        return out;
    }

    std::vector<float> regressionInputs(const fs::path& motionsDir, std::uint32_t pointsPerItem, std::size_t items,
        std::uint32_t dof, std::string_view robotName) {
        if (pointsPerItem == 0) {
            throw InvalidArgument("regressionInputs: pointsPerItem must be >= 1");
        }
        // Both the discrete poses and the trajectory start / end samples come from randomMotions(NITEMS).
        std::vector<float> samples = loadCachedPoses(motionsDir, robotName, items, dof);
        if (pointsPerItem == 1) {
            return samples;
        }
        const std::vector<std::int32_t> idx =
            loadKnnIndices(motionsDir, robotName, LegacyHarness::weightString, items, LegacyHarness::knnK);
        return knnTrajectories(samples, dof, idx, LegacyHarness::knnK, pointsPerItem, items);
    }

    std::vector<float> benchmarkPosePool(const fs::path& motionsDir, std::uint32_t dof, std::string_view robotName) {
        return loadCachedPoses(motionsDir, robotName, LegacyHarness::benchmarkSamples, dof);
    }

    std::vector<float> benchmarkTrajectoryPool(
        const fs::path& motionsDir, std::uint32_t trajectoryPoints, std::uint32_t dof, std::string_view robotName) {
        const std::size_t n = LegacyHarness::benchmarkSamples;
        const std::vector<float> samples = loadCachedPoses(motionsDir, robotName, n, dof);
        const std::vector<std::int32_t> idx =
            loadKnnIndices(motionsDir, robotName, LegacyHarness::weightString, n, LegacyHarness::knnK);
        return knnTrajectories(samples, dof, idx, LegacyHarness::knnK, trajectoryPoints, n * LegacyHarness::knnK);
    }

} // namespace rtcd::support
