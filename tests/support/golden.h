#pragma once
// support/golden.h -- collision-flag goldens (tests/data/CCTest/golden_<mode>.bin) and flag comparison, as in
// RTCD/CollisionDetector/Test/ccRegression.cpp.
//
// Golden format (old ccRegression --write-golden): raw bytes, one uint8 per item, 1 = in collision, 0 = free,
// item order = input order, no header. The goldens are never re-recorded (common brief, rule 1); this module has
// no golden writer. writeFlagsFile() is for diagnostic dumps and refuses names starting with "golden_".

#include <rtcd/types.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rtcd::support {

    using Flags = std::vector<std::uint8_t>;

    // "ias" / "ray" / "linear" / "quad" (the old REG_* mode names), nullopt for modes without a golden (cubic).
    std::optional<std::string> goldenModeName(Mode mode);
    // <dataDir>/CCTest/golden_<goldenModeName(mode)>.bin; nullopt if the mode has no golden.
    std::optional<std::filesystem::path> goldenPath(const std::filesystem::path& dataDir, Mode mode);

    // Reads a flag file. Throws IoError if it is missing, its size is not exactly `expectedCount` bytes
    // (expectedCount == 0: any size), or a byte is not 0 / 1.
    Flags readFlagsFile(const std::filesystem::path& file, std::size_t expectedCount = 0);
    // Writes raw flag bytes (diagnostics, e.g. --dump-flags). Throws InvalidArgument for a file name starting with
    // "golden_" (goldens are never written by the new harness), IoError on write failure.
    void writeFlagsFile(const std::filesystem::path& file, std::span<const std::uint8_t> flags);

    // flags[i] = counts[i] > 0 (CheckResult semantics; old regression: raw[i] > 0).
    Flags flagsFromCounts(std::span<const std::uint32_t> counts);
    std::size_t countSet(std::span<const std::uint8_t> flags);

    struct FlagDiff {
        std::size_t compared = 0;              // items compared
        std::size_t differing = 0;             // items whose flags differ
        std::vector<std::size_t> firstIndices; // the first few differing item indices (ascending)
        bool equal() const { return differing == 0; }
    };
    // Compares the first n items (n must be <= both sizes, InvalidArgument otherwise). Keeps up to keepFirst
    // differing indices (old regression printed 8).
    FlagDiff diffFlags(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b, std::size_t n,
        std::size_t keepFirst = 8);
    // Prints "    diff at item i: <a> vs <b>" for the kept indices (old countDiff format) and, if more differ than
    // were kept, one summary line.
    void printFlagDiff(std::ostream& os, const FlagDiff& diff, std::span<const std::uint8_t> a,
        std::span<const std::uint8_t> b);

    // Item counts of the prefix-consistency check of ccRegression.cpp: 1, 37, B, B * S + 5, 1000, items - 1
    // (B = max batch size, S = streams); values outside [1, items] are dropped, order kept, duplicates removed.
    std::vector<std::size_t> regressionPrefixCounts(std::size_t batchSize, std::size_t streams, std::size_t items);

} // namespace rtcd::support
