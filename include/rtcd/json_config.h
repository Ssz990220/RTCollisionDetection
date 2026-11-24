#pragma once
// rtcd/json_config.h -- the JSON module (library rtcd_json, Wave 1 b): maps JSON files onto the plain structs of
// config.h, scene.h and robot_model.h, and back. The core libraries do not depend on JSON; only this module
// links nlohmann-json, and this header does not include it.
//
// Rules (all loaders):
//   * Unspecified fields get the struct defaults (the C++ default member initializers are the single source of
//     defaults; the JSON module never hard-codes a second copy).
//   * Unknown keys are errors (catches typos), as are wrong types and out-of-range enum strings.
//   * Every error throws: IoError(path) when a file cannot be opened / read, IoError(path, line, column) for JSON
//     syntax errors, ConfigError for semantic errors, with what() = "<file>: <dotted.json.path>: <message>".
//   * Loaded checker configs are validate()d (not resolved; the checker resolves).
//   * Relative paths inside a file are resolved against the directory of THAT file (for a setup file that
//     references other files, against each referenced file's own directory), then made absolute and
//     lexically normalized. Absolute paths are kept. Nothing is resolved against the current working
//     directory or the source tree; no path is baked into binaries.
//   * Text overloads (parse*) take `baseDir` for that resolution and `sourceName` for messages.

#include <rtcd/config.h>
#include <rtcd/robot_model.h>
#include <rtcd/scene.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace rtcd {

    // ---- checker config ---------------------------------------------------------------------------------------
    CheckerConfig loadCheckerConfig(const std::filesystem::path& file);
    CheckerConfig parseCheckerConfig(std::string_view json, const std::filesystem::path& baseDir,
        std::string_view sourceName = "<string>");

    // ---- scene description ------------------------------------------------------------------------------------
    SceneDesc loadSceneDesc(const std::filesystem::path& file);
    SceneDesc parseSceneDesc(std::string_view json, const std::filesystem::path& baseDir,
        std::string_view sourceName = "<string>");

    // ---- robot reference ---------------------------------------------------------------------------------------
    // The file (or text) is an object with one key "robot" holding the robot block.
    RobotDesc loadRobotDesc(const std::filesystem::path& file);
    RobotDesc parseRobotDesc(std::string_view json, const std::filesystem::path& baseDir,
        std::string_view sourceName = "<string>");

    // ---- setup file: everything one experiment needs ----------------------------------------------------------
    // { "checker": <checker object | "path.json">, "scene": <scene object | "path.json">,
    //   "robot": <robot block | "path.json"> }   -- all three keys required.
    struct Setup {
        CheckerConfig checker;
        SceneDesc scene;
        RobotDesc robot;
        std::filesystem::path sourceFile;
    };
    Setup loadSetup(const std::filesystem::path& file);

    // ---- overrides -------------------------------------------------------------------------------------------
    // Applies "path=value,path=value" (dotted JSON paths of the checker schema, JSON values; bare words are taken
    // as strings), e.g. "mode=ray,optix.max_register_count=32,ray.hit_mode=traverse,curve.refit={\"every\":4}".
    // Replaces the old OptixCompileConfig::setFields / RTCD_OPTIX_CFG env parsing (apps read the env var, the
    // library never does). The result is validate()d. Throws ConfigError naming the entry.
    void applyOverrides(CheckerConfig& config, std::string_view overrides);

    // ---- dumps (effective configuration, every field, pretty-printed, stable key order) --------------------------
    // Paths are written absolute. Loading a dump gives back an equal struct.
    std::string dumpCheckerConfig(const CheckerConfig& config);
    std::string dumpSceneDesc(const SceneDesc& desc);
    std::string dumpRobotDesc(const RobotDesc& desc);

} // namespace rtcd
