#pragma once
// support/reference.h -- stored reference values (tests/data/reference) and the hash they use.
//
// The stored references replace the comparisons with the old header-only stack (removed; last commit with it: tag
// old-stack-final). They were written from the OLD stack's outputs; see tests/data/README.md for which test wrote
// which file and how. Never re-record a reference to make a test pass.
//
// Text format (one value per key, file order kept):
//     # comment
//     <key> <value> [<value> ...]
// Keys have no spaces and are unique. Values are decimal integers, 16-digit hex hashes ("0x..."), or float bit
// patterns as 8-digit hex ("0x3f800000"), so every value round-trips exactly.
//
// Hash: 64-bit FNV-1a (offset basis 0xcbf29ce484222325, prime 0x100000001b3) over the raw little-endian bytes of
// the array, in element order (floats as their IEEE-754 bit patterns, integers as uint32).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rtcd::support {

    class Fnv1a {
    public:
        static constexpr std::uint64_t kOffset = 0xcbf29ce484222325ull;
        static constexpr std::uint64_t kPrime  = 0x100000001b3ull;
        Fnv1a& add(const void* data, std::size_t bytes);
        template <typename T>
        Fnv1a& add(std::span<const T> s) {
            return add(s.data(), s.size_bytes());
        }
        std::uint64_t value() const { return h_; }

    private:
        std::uint64_t h_ = kOffset;
    };

    std::uint64_t fnv1a(const void* data, std::size_t bytes);
    template <typename T>
    std::uint64_t fnv1a(std::span<const T> s) {
        return fnv1a(s.data(), s.size_bytes());
    }
    template <typename T>
    std::uint64_t fnv1a(const std::vector<T>& v) {
        return fnv1a(v.data(), v.size() * sizeof(T));
    }

    std::string hex64(std::uint64_t v);   // "0x" + 16 hex digits
    std::string hexFloat(float v);        // "0x" + 8 hex digits of the bit pattern
    std::uint64_t parseHex64(std::string_view s); // throws InvalidArgument
    float parseHexFloat(std::string_view s);      // throws InvalidArgument

    // dataDir()/reference.
    std::filesystem::path referenceDir();

    class ReferenceFile {
    public:
        // Throws IoError if the file is missing or malformed (duplicate key, key without value).
        static ReferenceFile load(const std::filesystem::path& file);

        bool has(std::string_view key) const;
        const std::vector<std::string>& values(std::string_view key) const; // throws IoError if absent
        std::uint64_t integer(std::string_view key, std::size_t index = 0) const;
        std::vector<std::uint64_t> integers(std::string_view key) const;
        std::uint64_t hash(std::string_view key, std::size_t index = 0) const;
        std::vector<float> floats(std::string_view key) const; // every value a hex float
        const std::vector<std::string>& keys() const { return order_; }
        const std::filesystem::path& path() const { return path_; }

    private:
        std::filesystem::path path_;
        std::vector<std::string> order_;
        std::unordered_map<std::string, std::vector<std::string>> map_;
    };

    // Writes a reference file. Only the generators (old-stack parity tests, run with RTCD_WRITE_REFERENCE) use it.
    class ReferenceWriter {
    public:
        ReferenceWriter(std::filesystem::path file, std::string_view headerComment);
        ReferenceWriter& put(std::string_view key, const std::vector<std::string>& values);
        ReferenceWriter& putInt(std::string_view key, std::uint64_t v);
        ReferenceWriter& putInts(std::string_view key, std::span<const std::uint32_t> v);
        ReferenceWriter& putHash(std::string_view key, std::uint64_t count, std::uint64_t hash); // "<count> <hash>"
        ReferenceWriter& putFloats(std::string_view key, std::span<const float> v);
        void save() const; // throws IoError

    private:
        std::filesystem::path file_;
        std::string text_;
    };

    // Env RTCD_WRITE_REFERENCE: directory the generators write into (nullopt: do not write).
    std::optional<std::filesystem::path> referenceWriteDir();

    // Raw binary write (generators) of a whole array.
    void writeBinaryFile(const std::filesystem::path& file, const void* data, std::size_t bytes);

} // namespace rtcd::support
