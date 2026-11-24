#include <support/paths.h>

#include <rtcd/errors.h>

#include <cstdlib>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace rtcd::support {

    namespace {
        bool isRepoRoot(const fs::path& p) {
            std::error_code ec;
            const fs::path data = p / "tests" / "data";
            return fs::is_directory(data / "motions", ec) || fs::is_directory(data / "CCTest", ec);
        }

        std::optional<fs::path> walkUp(fs::path p) {
            std::error_code ec;
            p = fs::weakly_canonical(p, ec);
            if (ec) {
                return std::nullopt;
            }
            while (!p.empty()) {
                if (isRepoRoot(p)) {
                    return p;
                }
                const fs::path parent = p.parent_path();
                if (parent == p) {
                    break;
                }
                p = parent;
            }
            return std::nullopt;
        }
    } // namespace

    std::optional<std::string> getEnv(const char* name) {
#ifdef _WIN32
        char* value     = nullptr;
        std::size_t len = 0;
        if (_dupenv_s(&value, &len, name) != 0 || value == nullptr) {
            return std::nullopt;
        }
        std::string s(value);
        std::free(value);
        return s;
#else
        const char* v = std::getenv(name);
        if (!v) {
            return std::nullopt;
        }
        return std::string(v);
#endif
    }

    fs::path executableDir() {
#ifdef _WIN32
        std::vector<wchar_t> buf(32768);
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0 || n >= buf.size()) {
            return {};
        }
        return fs::path(std::wstring(buf.data(), n)).parent_path();
#else
        std::error_code ec;
        const fs::path p = fs::read_symlink("/proc/self/exe", ec);
        return ec ? fs::path{} : p.parent_path();
#endif
    }

    fs::path findRepoRoot() {
        std::error_code ec;
        if (auto r = walkUp(fs::current_path(ec))) {
            return *r;
        }
        const fs::path exe = executableDir();
        if (!exe.empty()) {
            if (auto r = walkUp(exe)) {
                return *r;
            }
        }
        throw IoError("cannot find the repository root (a directory containing tests/data/motions or tests/data/CCTest) "
                      "above "
                      "the working directory or the executable; pass the data directory explicitly or set "
                      "RTCD_DATA_DIR",
            fs::current_path(ec));
    }

    fs::path dataDir(const std::optional<fs::path>& override) {
        if (override) {
            return fs::absolute(*override);
        }
        if (auto env = getEnv("RTCD_DATA_DIR"); env && !env->empty()) {
            return fs::absolute(fs::path(*env));
        }
        return findRepoRoot() / "tests" / "data";
    }

    fs::path configsDir() {
        const fs::path p = findRepoRoot() / "configs";
        std::error_code ec;
        if (!fs::is_directory(p, ec)) {
            throw IoError("configs directory not found", p);
        }
        return p;
    }

} // namespace rtcd::support
