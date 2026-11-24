#pragma once
// support/paths.h -- locating the repository's shared folders (tests/data/, configs/) from tests and apps.
//
// Tests run with the repository root as working directory (rtcd_add_new_test); apps may be started from
// anywhere. Nothing here is baked in at build time: the root is found at run time by walking up from the
// current directory and from the executable's directory. The reference data (cached regression / benchmark
// inputs, goldens) is committed under tests/data (see tests/data/README.md); RTCD_DATA_DIR or an explicit
// directory (apps: --data) points elsewhere, e.g. at a local folder with the same layout.

#include <filesystem>
#include <optional>
#include <string>

namespace rtcd::support {

    // First directory that contains tests/data/motions or tests/data/CCTest, walking up from the current working
    // directory, then from the directory of the running executable. Throws rtcd::IoError if none is found.
    std::filesystem::path findRepoRoot();

    // The data folder: `override` if given, else env RTCD_DATA_DIR if set, else findRepoRoot()/tests/data.
    std::filesystem::path dataDir(const std::optional<std::filesystem::path>& override = std::nullopt);

    // findRepoRoot()/configs (the JSON configs of the new stack). Throws rtcd::IoError if missing.
    std::filesystem::path configsDir();

    // Directory of the running executable (empty if it cannot be determined).
    std::filesystem::path executableDir();

    // Value of an environment variable, nullopt if unset.
    std::optional<std::string> getEnv(const char* name);

} // namespace rtcd::support
