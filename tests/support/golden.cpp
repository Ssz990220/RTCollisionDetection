#include <support/golden.h>

#include <rtcd/errors.h>

#include <algorithm>
#include <fstream>
#include <ostream>

namespace fs = std::filesystem;

namespace rtcd::support {

    std::optional<std::string> goldenModeName(Mode mode) {
        switch (mode) {
        case Mode::Mesh: return std::string("ias");
        case Mode::Ray: return std::string("ray");
        case Mode::LinearCurve: return std::string("linear");
        case Mode::QuadraticCurve: return std::string("quad");
        case Mode::CubicCurve: return std::nullopt;
        }
        return std::nullopt;
    }

    std::optional<fs::path> goldenPath(const fs::path& dataDir, Mode mode) {
        const auto name = goldenModeName(mode);
        if (!name) {
            return std::nullopt;
        }
        return dataDir / "CCTest" / ("golden_" + *name + ".bin");
    }

    Flags readFlagsFile(const fs::path& file, std::size_t expectedCount) {
        std::ifstream in(file, std::ios::binary);
        if (!in) {
            throw IoError("cannot open flag file", file);
        }
        Flags flags((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (expectedCount != 0 && flags.size() != expectedCount) {
            throw IoError("holds " + std::to_string(flags.size()) + " flags, expected " + std::to_string(expectedCount),
                file);
        }
        for (std::size_t i = 0; i < flags.size(); ++i) {
            if (flags[i] > 1) {
                throw IoError("byte " + std::to_string(i) + " = " + std::to_string(int(flags[i])) + " is not a flag (0 / 1)",
                    file);
            }
        }
        return flags;
    }

    void writeFlagsFile(const fs::path& file, std::span<const std::uint8_t> flags) {
        if (file.filename().string().rfind("golden_", 0) == 0) {
            throw InvalidArgument("refusing to write '" + file.string()
                                  + "': goldens are never re-recorded by the new harness");
        }
        if (file.has_parent_path()) {
            fs::create_directories(file.parent_path());
        }
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(flags.data()), static_cast<std::streamsize>(flags.size()));
        if (!out) {
            throw IoError("write error", file);
        }
    }

    Flags flagsFromCounts(std::span<const std::uint32_t> counts) {
        Flags f(counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) {
            f[i] = counts[i] > 0 ? 1 : 0;
        }
        return f;
    }

    std::size_t countSet(std::span<const std::uint8_t> flags) {
        std::size_t c = 0;
        for (const auto v : flags) {
            c += v != 0 ? 1 : 0;
        }
        return c;
    }

    FlagDiff diffFlags(
        std::span<const std::uint8_t> a, std::span<const std::uint8_t> b, std::size_t n, std::size_t keepFirst) {
        if (n > a.size() || n > b.size()) {
            throw InvalidArgument("diffFlags: comparing " + std::to_string(n) + " items of arrays with "
                                  + std::to_string(a.size()) + " and " + std::to_string(b.size()) + " entries");
        }
        FlagDiff d;
        d.compared = n;
        for (std::size_t i = 0; i < n; ++i) {
            if ((a[i] != 0) != (b[i] != 0)) {
                if (d.firstIndices.size() < keepFirst) {
                    d.firstIndices.push_back(i);
                }
                ++d.differing;
            }
        }
        return d;
    }

    void printFlagDiff(
        std::ostream& os, const FlagDiff& diff, std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
        for (const std::size_t i : diff.firstIndices) {
            os << "    diff at item " << i << ": " << int(a[i]) << " vs " << int(b[i]) << "\n";
        }
        if (diff.differing > diff.firstIndices.size()) {
            os << "    ... " << (diff.differing - diff.firstIndices.size()) << " more differing items\n";
        }
    }

    std::vector<std::size_t> regressionPrefixCounts(std::size_t batchSize, std::size_t streams, std::size_t items) {
        const std::size_t candidates[] = {1, 37, batchSize, batchSize * streams + 5, 1000, items - 1};
        std::vector<std::size_t> out;
        for (const std::size_t n : candidates) {
            if (n >= 1 && n <= items && std::find(out.begin(), out.end(), n) == out.end()) {
                out.push_back(n);
            }
        }
        return out;
    }

} // namespace rtcd::support
