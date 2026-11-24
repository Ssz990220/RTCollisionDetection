#include <support/reference.h>

#include <support/paths.h>

#include <rtcd/errors.h>

#include <bit>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace rtcd::support {

    Fnv1a& Fnv1a::add(const void* data, std::size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            h_ ^= p[i];
            h_ *= kPrime;
        }
        return *this;
    }

    std::uint64_t fnv1a(const void* data, std::size_t bytes) { return Fnv1a().add(data, bytes).value(); }

    std::string hex64(std::uint64_t v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(v));
        return buf;
    }

    std::string hexFloat(float v) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "0x%08x", static_cast<unsigned>(std::bit_cast<std::uint32_t>(v)));
        return buf;
    }

    std::uint64_t parseHex64(std::string_view s) {
        if (s.size() < 3 || s[0] != '0' || (s[1] != 'x' && s[1] != 'X')) {
            throw InvalidArgument("not a hex value: '" + std::string(s) + "'");
        }
        std::uint64_t v = 0;
        const auto r    = std::from_chars(s.data() + 2, s.data() + s.size(), v, 16);
        if (r.ec != std::errc() || r.ptr != s.data() + s.size()) {
            throw InvalidArgument("not a hex value: '" + std::string(s) + "'");
        }
        return v;
    }

    float parseHexFloat(std::string_view s) {
        const std::uint64_t v = parseHex64(s);
        if (v > 0xffffffffull) {
            throw InvalidArgument("not a 32-bit float pattern: '" + std::string(s) + "'");
        }
        return std::bit_cast<float>(static_cast<std::uint32_t>(v));
    }

    fs::path referenceDir() { return dataDir() / "reference"; }

    ReferenceFile ReferenceFile::load(const fs::path& file) {
        std::ifstream in(file);
        if (!in) {
            throw IoError("cannot open reference file (see tests/data/README.md)", file);
        }
        ReferenceFile r;
        r.path_ = file;
        std::string line;
        std::size_t lineNo = 0;
        while (std::getline(in, line)) {
            ++lineNo;
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream ss(line);
            std::string key, v;
            ss >> key;
            std::vector<std::string> vals;
            while (ss >> v) {
                vals.push_back(v);
            }
            if (key.empty()) {
                continue;
            }
            if (vals.empty()) {
                throw IoError("key '" + key + "' has no value", file, lineNo);
            }
            if (!r.map_.emplace(key, std::move(vals)).second) {
                throw IoError("duplicate key '" + key + "'", file, lineNo);
            }
            r.order_.push_back(key);
        }
        return r;
    }

    bool ReferenceFile::has(std::string_view key) const { return map_.count(std::string(key)) != 0; }

    const std::vector<std::string>& ReferenceFile::values(std::string_view key) const {
        const auto it = map_.find(std::string(key));
        if (it == map_.end()) {
            throw IoError("reference key '" + std::string(key) + "' not found", path_);
        }
        return it->second;
    }

    std::uint64_t ReferenceFile::integer(std::string_view key, std::size_t index) const {
        const auto& v = values(key);
        if (index >= v.size()) {
            throw IoError("reference key '" + std::string(key) + "' has no value " + std::to_string(index), path_);
        }
        std::uint64_t x = 0;
        const auto r    = std::from_chars(v[index].data(), v[index].data() + v[index].size(), x, 10);
        if (r.ec != std::errc() || r.ptr != v[index].data() + v[index].size()) {
            throw IoError("reference key '" + std::string(key) + "': '" + v[index] + "' is not an integer", path_);
        }
        return x;
    }

    std::vector<std::uint64_t> ReferenceFile::integers(std::string_view key) const {
        std::vector<std::uint64_t> out;
        for (std::size_t i = 0; i < values(key).size(); ++i) {
            out.push_back(integer(key, i));
        }
        return out;
    }

    std::uint64_t ReferenceFile::hash(std::string_view key, std::size_t index) const {
        const auto& v = values(key);
        if (index >= v.size()) {
            throw IoError("reference key '" + std::string(key) + "' has no value " + std::to_string(index), path_);
        }
        return parseHex64(v[index]);
    }

    std::vector<float> ReferenceFile::floats(std::string_view key) const {
        std::vector<float> out;
        for (const std::string& s : values(key)) {
            out.push_back(parseHexFloat(s));
        }
        return out;
    }

    ReferenceWriter::ReferenceWriter(fs::path file, std::string_view headerComment) : file_(std::move(file)) {
        std::istringstream ss{std::string(headerComment)};
        std::string line;
        while (std::getline(ss, line)) {
            text_ += "# " + line + "\n";
        }
    }

    ReferenceWriter& ReferenceWriter::put(std::string_view key, const std::vector<std::string>& values) {
        text_ += std::string(key);
        for (const std::string& v : values) {
            text_ += " " + v;
        }
        text_ += "\n";
        return *this;
    }

    ReferenceWriter& ReferenceWriter::putInt(std::string_view key, std::uint64_t v) {
        return put(key, {std::to_string(v)});
    }

    ReferenceWriter& ReferenceWriter::putInts(std::string_view key, std::span<const std::uint32_t> v) {
        std::vector<std::string> s;
        for (const auto x : v) {
            s.push_back(std::to_string(x));
        }
        return put(key, s);
    }

    ReferenceWriter& ReferenceWriter::putHash(std::string_view key, std::uint64_t count, std::uint64_t hash) {
        return put(key, {std::to_string(count), hex64(hash)});
    }

    ReferenceWriter& ReferenceWriter::putFloats(std::string_view key, std::span<const float> v) {
        std::vector<std::string> s;
        for (const float x : v) {
            s.push_back(hexFloat(x));
        }
        return put(key, s);
    }

    void ReferenceWriter::save() const {
        if (file_.has_parent_path()) {
            fs::create_directories(file_.parent_path());
        }
        std::ofstream out(file_, std::ios::binary | std::ios::trunc); // '\n' line ends on every platform
        out << text_;
        if (!out) {
            throw IoError("write error", file_);
        }
    }

    std::optional<fs::path> referenceWriteDir() {
        if (auto e = getEnv("RTCD_WRITE_REFERENCE"); e && !e->empty()) {
            return fs::path(*e);
        }
        return std::nullopt;
    }

    void writeBinaryFile(const fs::path& file, const void* data, std::size_t bytes) {
        if (file.has_parent_path()) {
            fs::create_directories(file.parent_path());
        }
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!out) {
            throw IoError("write error", file);
        }
    }

} // namespace rtcd::support
