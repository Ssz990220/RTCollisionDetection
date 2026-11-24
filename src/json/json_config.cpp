// rtcd_json: include/rtcd/json_config.h -- JSON files <-> CheckerConfig, SceneDesc, RobotDesc, Setup.
//
// Schemas: docs/design/interfaces.md section 7. Implementation notes:
//   * One reader per schema object. Each object first checks its keys against the schema (unknown keys are
//     errors), then reads the keys present onto a default-constructed struct, so the C++ default member
//     initializers stay the only source of defaults.
//   * Errors: ConfigError "<source>: <dotted.json.path>: <message>" (field() = the dotted path), IoError for
//     unreadable files and JSON syntax errors ("<source>:<line>:<column>: <message>"). Duplicate keys in one
//     object are rejected as well (nlohmann-json would silently keep one of them).
//   * Paths: relative paths are joined to the directory of the file that contains them (parse*: `baseDir`), made
//     absolute, lexically normalized and given native separators. JSON strings are UTF-8 (std::u8string paths).
//   * Dumps use nlohmann::ordered_json (schema key order). Floats are written with the shortest decimal text that
//     reads back to the same float, so loading a dump gives back bit-identical transforms.
//   * applyOverrides works on the dump of the config: each entry sets (or, for an object value on an object,
//     merges into) one dotted path of that JSON document, which is then read back with the normal reader, so
//     overrides get exactly the loader's type and enum checks.
#include <rtcd/errors.h>
#include <rtcd/json_config.h>

#include <nlohmann/json.hpp>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rtcd {

    namespace {

        namespace fs = std::filesystem;
        using Json   = nlohmann::ordered_json;

        // ---- paths ---------------------------------------------------------------------------------------------

        fs::path pathFromUtf8(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

        std::string utf8FromPath(const fs::path& p) {
            const std::u8string u = p.generic_u8string();
            return std::string(u.begin(), u.end());
        }

        fs::path normalized(const fs::path& p) {
            fs::path n = p.lexically_normal();
            n.make_preferred();
            return n;
        }

        fs::path absoluteDir(const fs::path& dir) {
            if (dir.empty()) {
                return normalized(fs::current_path());
            }
            return normalized(fs::absolute(dir));
        }

        // ---- error context -------------------------------------------------------------------------------------

        // Where the JSON being read comes from: `name` prefixes messages, `baseDir` resolves relative paths.
        struct Source {
            std::string name;
            fs::path baseDir; // absolute
        };

        std::string join(const std::string& path, std::string_view key) {
            return path.empty() ? std::string(key) : path + "." + std::string(key);
        }
        std::string indexed(const std::string& path, std::size_t i) { return path + "[" + std::to_string(i) + "]"; }

        [[noreturn]] void fail(const Source& src, const std::string& path, const std::string& message) {
            throw ConfigError(src.name + ": " + (path.empty() ? std::string("(top level)") : path) + ": " + message, path);
        }

        std::string describe(const Json& v) {
            if (v.is_string()) {
                return "\"" + v.get<std::string>() + "\"";
            }
            if (v.is_object() || v.is_array()) {
                return std::string(v.type_name());
            }
            return v.dump();
        }

        // ---- scalar readers ------------------------------------------------------------------------------------

        bool readBool(const Json& v, const Source& src, const std::string& path) {
            if (!v.is_boolean()) {
                fail(src, path, "must be true or false, got " + describe(v));
            }
            return v.get<bool>();
        }

        std::uint64_t readUnsigned(const Json& v, const Source& src, const std::string& path, std::uint64_t max) {
            if (v.is_number_unsigned()) {
                const auto x = v.get<std::uint64_t>();
                if (x > max) {
                    fail(src, path, std::to_string(x) + " is too large (maximum " + std::to_string(max) + ")");
                }
                return x;
            }
            if (v.is_number_integer()) { // signed storage (documents built in memory, e.g. by applyOverrides)
                const auto x = v.get<std::int64_t>();
                if (x < 0) {
                    fail(src, path, v.dump() + " must be >= 0");
                }
                if (static_cast<std::uint64_t>(x) > max) {
                    fail(src, path, std::to_string(x) + " is too large (maximum " + std::to_string(max) + ")");
                }
                return static_cast<std::uint64_t>(x);
            }
            fail(src, path, "must be a non-negative integer, got " + describe(v));
        }

        std::uint32_t readU32(const Json& v, const Source& src, const std::string& path) {
            return static_cast<std::uint32_t>(readUnsigned(v, src, path, std::numeric_limits<std::uint32_t>::max()));
        }

        double readNumber(const Json& v, const Source& src, const std::string& path) {
            if (!v.is_number()) {
                fail(src, path, "must be a number, got " + describe(v));
            }
            const double d = v.get<double>();
            if (!std::isfinite(d)) {
                fail(src, path, "must be finite");
            }
            return d;
        }

        float readFloat(const Json& v, const Source& src, const std::string& path) {
            const double d = readNumber(v, src, path);
            const float f  = static_cast<float>(d);
            if (!std::isfinite(f)) {
                fail(src, path, v.dump() + " is out of the float range");
            }
            return f;
        }

        std::string readString(const Json& v, const Source& src, const std::string& path) {
            if (!v.is_string()) {
                fail(src, path, "must be a string, got " + describe(v));
            }
            return v.get<std::string>();
        }

        // A path string, resolved against src.baseDir (or `base` when given).
        fs::path readPath(const Json& v, const Source& src, const std::string& path, const fs::path* base = nullptr) {
            const std::string s = readString(v, src, path);
            if (s.empty()) {
                fail(src, path, "must be a non-empty path");
            }
            fs::path p = pathFromUtf8(s);
            if (p.is_relative()) {
                p = (base ? *base : src.baseDir) / p;
            }
            return normalized(fs::absolute(p));
        }

        template <typename E>
        using EnumTable = std::initializer_list<std::pair<const char*, E>>;

        template <typename E>
        std::string choices(EnumTable<E> table) {
            std::string s;
            for (const auto& [name, value] : table) {
                (void)value;
                s += (s.empty() ? "\"" : ", \"") + std::string(name) + "\"";
            }
            return s;
        }

        template <typename E>
        E readEnum(const Json& v, const Source& src, const std::string& path, EnumTable<E> table) {
            if (v.is_string()) {
                const std::string s = v.get<std::string>();
                for (const auto& [name, value] : table) {
                    if (s == name) {
                        return value;
                    }
                }
            }
            fail(src, path, describe(v) + " is not one of " + choices(table));
        }

        // "auto" -> nullopt, otherwise one of the table's names.
        template <typename E>
        std::optional<E> readAutoEnum(const Json& v, const Source& src, const std::string& path, EnumTable<E> table) {
            if (v.is_string()) {
                const std::string s = v.get<std::string>();
                if (s == "auto") {
                    return std::nullopt;
                }
                for (const auto& [name, value] : table) {
                    if (s == name) {
                        return value;
                    }
                }
            }
            fail(src, path, describe(v) + " is not one of \"auto\", " + choices(table));
        }

        template <typename E>
        const char* enumName(E value, EnumTable<E> table) {
            for (const auto& [name, v] : table) {
                if (v == value) {
                    return name;
                }
            }
            return nullptr;
        }

        // ---- enum spellings (docs/design/interfaces.md section 7.1) -------------------------------------------------

        const EnumTable<Mode> kModes{{"mesh", Mode::Mesh}, {"ray", Mode::Ray}, {"linear_curve", Mode::LinearCurve},
            {"quadratic_curve", Mode::QuadraticCurve}, {"cubic_curve", Mode::CubicCurve}};
        const EnumTable<HostDispatch> kDispatch{
            {"single_thread", HostDispatch::SingleThread}, {"thread_per_stream", HostDispatch::ThreadPerStream}};
        const EnumTable<EdgeSet> kEdgeSets{{"unique", EdgeSet::Unique}, {"loop", EdgeSet::Loop}};
        const EnumTable<LogLevel> kLogLevels{{"error", LogLevel::Error}, {"warning", LogLevel::Warning},
            {"info", LogLevel::Info}, {"debug", LogLevel::Debug}};
        const EnumTable<OptixOptions::DebugLevel> kDebugLevels{{"default", OptixOptions::DebugLevel::Default},
            {"none", OptixOptions::DebugLevel::None}, {"minimal", OptixOptions::DebugLevel::Minimal},
            {"moderate", OptixOptions::DebugLevel::Moderate}, {"full", OptixOptions::DebugLevel::Full}};
        const EnumTable<std::uint32_t> kExceptions{{"stack_overflow", OptixOptions::kExceptionStackOverflow},
            {"trace_depth", OptixOptions::kExceptionTraceDepth}, {"user", OptixOptions::kExceptionUser}};
        const EnumTable<BuildPreference> kBuilds{{"fast_build", BuildPreference::FastBuild},
            {"fast_trace", BuildPreference::FastTrace}, {"fast_trace_compact", BuildPreference::FastTraceCompact}};
        const EnumTable<RayHitMode> kHitModes{{"closest_hit", RayHitMode::ClosestHit},
            {"first_hit_ch", RayHitMode::FirstHitCH}, {"first_hit_miss", RayHitMode::FirstHitMiss},
            {"traverse", RayHitMode::FirstHitTraverse}};
        const EnumTable<CurveEndPoints> kEndPoints{{"normal", CurveEndPoints::Normal},
            {"repeated", CurveEndPoints::Repeated}, {"phantom", CurveEndPoints::Phantom}};
        const EnumTable<RefitPolicy::Kind> kRefitKinds{{"auto", RefitPolicy::Kind::Auto},
            {"always", RefitPolicy::Kind::Always}, {"never", RefitPolicy::Kind::Never}};
        const EnumTable<GeometryFlags> kGeometryFlags{
            {"none", kGeometryFlagNone}, {"single_anyhit", kGeometryFlagRequireSingleAnyHitCall}};
        const EnumTable<MeshSource> kMeshSources{{"visual", MeshSource::Visual}, {"collision", MeshSource::Collision}};

        // ---- objects with a fixed key set ----------------------------------------------------------------------

        class Object {
        public:
            Object(const Json& j, const Source& src, std::string path, std::initializer_list<const char*> keys)
                : j_(j), src_(src), path_(std::move(path)) {
                if (!j.is_object()) {
                    fail(src_, path_, "must be an object, got " + describe(j));
                }
                for (const auto& item : j.items()) {
                    bool known = false;
                    for (const char* k : keys) {
                        known = known || item.key() == k;
                    }
                    if (!known) {
                        std::string allowed;
                        for (const char* k : keys) {
                            allowed += (allowed.empty() ? "" : ", ") + std::string(k);
                        }
                        fail(src_, join(path_, item.key()), "unknown key (allowed here: " + allowed + ")");
                    }
                }
            }
            const Json* find(const char* key) const {
                const auto it = j_.find(key);
                return it == j_.end() ? nullptr : &*it;
            }
            const Json& required(const char* key) const {
                const Json* v = find(key);
                if (!v) {
                    fail(src_, join(path_, key), "required key is missing");
                }
                return *v;
            }
            std::string at(const char* key) const { return join(path_, key); }
            const std::string& path() const { return path_; }

        private:
            const Json& j_;
            const Source& src_;
            std::string path_;
        };

        // ---- JSON text / files ---------------------------------------------------------------------------------

        std::string readFile(const fs::path& file) {
            std::ifstream in(file, std::ios::binary);
            if (!in) {
                throw IoError(utf8FromPath(file) + ": cannot open file", file);
            }
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (in.bad()) {
                throw IoError(utf8FromPath(file) + ": cannot read file", file);
            }
            return text;
        }

        // Parses JSON text; syntax errors and duplicate keys throw (see the file comment).
        Json parseJson(std::string_view text, const std::string& name, const fs::path& errorPath) {
            std::vector<std::set<std::string>> keyStack;
            std::string duplicate;
            const Json::parser_callback_t callback = [&](int /*depth*/, nlohmann::json::parse_event_t event,
                                                         Json& parsed) {
                using E = nlohmann::json::parse_event_t;
                if (event == E::object_start) {
                    keyStack.emplace_back();
                } else if (event == E::object_end) {
                    if (!keyStack.empty()) {
                        keyStack.pop_back();
                    }
                } else if (event == E::key && !keyStack.empty() && duplicate.empty()) {
                    const std::string key = parsed.get<std::string>();
                    if (!keyStack.back().insert(key).second) {
                        duplicate = key;
                    }
                }
                return true;
            };
            Json j;
            try {
                j = Json::parse(text.begin(), text.end(), callback, /*allow_exceptions=*/true, /*ignore_comments=*/false);
            } catch (const nlohmann::json::parse_error& e) {
                const std::string what = e.what();
                std::size_t line = 0, column = 0;
                std::smatch m;
                static const std::regex lineCol(R"(at line (\d+), column (\d+))");
                if (std::regex_search(what, m, lineCol)) {
                    line   = std::stoul(m[1].str());
                    column = std::stoul(m[2].str());
                } else {
                    // older message format: derive line / column from the byte offset
                    line = 1;
                    std::size_t lineStart = 0;
                    const std::size_t end = std::min<std::size_t>(e.byte, text.size());
                    for (std::size_t i = 0; i + 1 < end; ++i) {
                        if (text[i] == '\n') {
                            ++line;
                            lineStart = i + 1;
                        }
                    }
                    column = end > lineStart ? end - lineStart : 1;
                }
                std::string message = what;
                const std::size_t pe = what.find("parse error");
                if (pe != std::string::npos) {
                    const std::size_t colon = what.find(": ", pe);
                    if (colon != std::string::npos) {
                        message = what.substr(colon + 2);
                    }
                }
                throw IoError(name + ":" + std::to_string(line) + ":" + std::to_string(column) + ": JSON syntax error: "
                                  + message,
                    errorPath, line, column);
            }
            if (!duplicate.empty()) {
                throw ConfigError(name + ": " + duplicate + ": duplicate key in one JSON object", duplicate);
            }
            return j;
        }

        // ---- pose ------------------------------------------------------------------------------------------------

        std::vector<double> readNumbers(
            const Json& v, const Source& src, const std::string& path, std::size_t count, const char* what) {
            if (!v.is_array() || v.size() != count) {
                fail(src, path, std::string("must be an array of ") + std::to_string(count) + " numbers (" + what
                                    + "), got " + describe(v)
                                    + (v.is_array() ? " of size " + std::to_string(v.size()) : std::string()));
            }
            std::vector<double> out;
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(readNumber(v[i], src, indexed(path, i)));
            }
            return out;
        }

        Transform readMatrix(const Json& v, const Source& src, const std::string& path) {
            if (!v.is_array() || v.size() != kTransformFloats) {
                fail(src, path, "must be an array of 12 numbers (3x4 row-major [R | t]), got " + describe(v));
            }
            Transform t;
            for (std::size_t i = 0; i < kTransformFloats; ++i) {
                t.m[i] = readFloat(v[i], src, indexed(path, i));
            }
            return t;
        }

        // { "translation": [x, y, z], at most one of "rpy" | "quaternion" | "matrix" (matrix excludes translation) }
        Transform readPose(const Json& v, const Source& src, const std::string& path) {
            const Object o(v, src, path, {"translation", "rpy", "quaternion", "matrix"});
            const Json* tr = o.find("translation");
            const Json* rpy = o.find("rpy");
            const Json* quat = o.find("quaternion");
            const Json* mat = o.find("matrix");
            if (int(rpy != nullptr) + int(quat != nullptr) + int(mat != nullptr) > 1) {
                fail(src, path, "at most one of \"rpy\", \"quaternion\", \"matrix\" may be given");
            }
            if (mat) {
                if (tr) {
                    fail(src, o.at("translation"), "not allowed together with \"matrix\" (the matrix holds the translation)");
                }
                return readMatrix(*mat, src, o.at("matrix"));
            }
            std::vector<double> t{0, 0, 0};
            if (tr) {
                t = readNumbers(*tr, src, o.at("translation"), 3, "x, y, z in meters");
            }
            if (rpy) {
                const auto r = readNumbers(*rpy, src, o.at("rpy"), 3, "roll, pitch, yaw in radians");
                return Transform::fromXyzRpy(t[0], t[1], t[2], r[0], r[1], r[2]);
            }
            if (quat) {
                const auto q = readNumbers(*quat, src, o.at("quaternion"), 4, "x, y, z, w");
                if (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] < 1e-24) {
                    fail(src, o.at("quaternion"), "must not be the zero quaternion");
                }
                return Transform::fromXyzQuat(t[0], t[1], t[2], q[0], q[1], q[2], q[3]);
            }
            return Transform::translation(static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2]));
        }

        // ---- checker config --------------------------------------------------------------------------------------

        void readOptix(const Json& v, const Source& src, const std::string& path, OptixOptions& o) {
            const Object obj(v, src, path,
                {"opt_level", "debug_level", "max_register_count", "exceptions", "max_trace_depth", "print_pipeline_log"});
            if (const Json* x = obj.find("opt_level")) {
                if (x->is_string() && x->get<std::string>() == "default") {
                    o.optLevel = OptixOptions::OptLevel::Default;
                } else if (x->is_number_unsigned() && x->get<std::uint64_t>() <= 3) {
                    o.optLevel = static_cast<OptixOptions::OptLevel>(
                        static_cast<unsigned>(OptixOptions::OptLevel::O0) + x->get<unsigned>());
                } else {
                    fail(src, obj.at("opt_level"), describe(*x) + " is not one of \"default\", 0, 1, 2, 3");
                }
            }
            if (const Json* x = obj.find("debug_level")) {
                o.debugLevel = readEnum(*x, src, obj.at("debug_level"), kDebugLevels);
            }
            if (const Json* x = obj.find("max_register_count")) {
                o.maxRegisterCount = readU32(*x, src, obj.at("max_register_count"));
            }
            if (const Json* x = obj.find("exceptions")) {
                const std::string p = obj.at("exceptions");
                if (!x->is_array()) {
                    fail(src, p, "must be a list of exception names, got " + describe(*x));
                }
                std::uint32_t flags = 0;
                bool none           = false;
                for (std::size_t i = 0; i < x->size(); ++i) {
                    const Json& e = (*x)[i];
                    if (e.is_string() && e.get<std::string>() == "none") {
                        none = true;
                    } else {
                        flags |= readEnum(e, src, indexed(p, i), kExceptions);
                    }
                }
                if (none && flags != 0) {
                    fail(src, p, "\"none\" cannot be combined with other exception names");
                }
                o.exceptionFlags = flags;
            }
            if (const Json* x = obj.find("max_trace_depth")) {
                o.maxTraceDepth = readU32(*x, src, obj.at("max_trace_depth"));
            }
            if (const Json* x = obj.find("print_pipeline_log")) {
                o.printPipelineLog = readBool(*x, src, obj.at("print_pipeline_log"));
            }
        }

        void readMesh(const Json& v, const Source& src, const std::string& path, MeshOptions& m) {
            const Object obj(v, src, path, {"link_gas_build", "compaction_min_poses", "park_transform"});
            if (const Json* x = obj.find("link_gas_build")) {
                m.linkGasBuild = readEnum(*x, src, obj.at("link_gas_build"), kBuilds);
            }
            if (const Json* x = obj.find("compaction_min_poses")) {
                m.compactionMinPoses = readU32(*x, src, obj.at("compaction_min_poses"));
            }
            if (const Json* x = obj.find("park_transform")) {
                m.parkTransform = readMatrix(*x, src, obj.at("park_transform"));
            }
        }

        void readRay(const Json& v, const Source& src, const std::string& path, RayOptions& r) {
            const Object obj(v, src, path, {"launch_width_cap", "skip_hit_items", "hit_mode"});
            if (const Json* x = obj.find("launch_width_cap")) {
                r.launchWidthCap = readU32(*x, src, obj.at("launch_width_cap"));
            }
            if (const Json* x = obj.find("skip_hit_items")) {
                r.skipHitItems = readBool(*x, src, obj.at("skip_hit_items"));
            }
            if (const Json* x = obj.find("hit_mode")) {
                r.hitMode = readEnum(*x, src, obj.at("hit_mode"), kHitModes);
            }
        }

        void readCurve(const Json& v, const Source& src, const std::string& path, CurveOptions& c) {
            const Object obj(v, src, path,
                {"trajectory_points", "control_points", "end_points", "gas_build", "refit", "auto_refit_max_batches",
                    "geometry_flags", "spline_weights_dir"});
            if (const Json* x = obj.find("trajectory_points")) {
                c.trajectoryPoints = readU32(*x, src, obj.at("trajectory_points"));
            }
            if (const Json* x = obj.find("control_points")) {
                c.controlPoints = readU32(*x, src, obj.at("control_points"));
            }
            if (const Json* x = obj.find("end_points")) {
                c.endPoints = readEnum(*x, src, obj.at("end_points"), kEndPoints);
            }
            if (const Json* x = obj.find("gas_build")) {
                c.gasBuild = readEnum(*x, src, obj.at("gas_build"), kBuilds);
            }
            if (const Json* x = obj.find("refit")) {
                const std::string p = obj.at("refit");
                if (x->is_object()) {
                    const Object every(*x, src, p, {"every"});
                    c.refit = RefitPolicy{RefitPolicy::Kind::Every, readU32(every.required("every"), src, every.at("every"))};
                } else if (x->is_string()) {
                    c.refit = RefitPolicy{readEnum(*x, src, p, kRefitKinds), 0};
                } else {
                    fail(src, p, describe(*x) + " is not one of \"auto\", \"always\", \"never\", {\"every\": K}");
                }
            }
            if (const Json* x = obj.find("auto_refit_max_batches")) {
                const std::string p = obj.at("auto_refit_max_batches");
                if (x->is_string()) {
                    if (x->get<std::string>() != "auto") {
                        fail(src, p, describe(*x) + " is not \"auto\" or an integer >= 0");
                    }
                    c.autoRefitMaxBatches = -1;
                } else {
                    c.autoRefitMaxBatches = static_cast<std::int32_t>(
                        readUnsigned(*x, src, p, static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())));
                }
            }
            if (const Json* x = obj.find("geometry_flags")) {
                const std::string p = obj.at("geometry_flags");
                c.geometryFlags = readAutoEnum(*x, src, p, kGeometryFlags);
            }
            if (const Json* x = obj.find("spline_weights_dir")) {
                if (x->is_null()) {
                    c.splineWeightsDir.clear();
                } else {
                    c.splineWeightsDir = readPath(*x, src, obj.at("spline_weights_dir"));
                }
            }
        }

        void readChecker(const Json& v, const Source& src, const std::string& path, CheckerConfig& c) {
            const Object obj(v, src, path,
                {"mode", "streams", "host_dispatch", "max_batch_size", "max_items_per_check", "obb_culling",
                    "collapse_root_link_obb", "scene_edges", "verbosity", "optix", "mesh", "ray", "curve"});
            if (const Json* x = obj.find("mode")) {
                c.mode = readEnum(*x, src, obj.at("mode"), kModes);
            }
            if (const Json* x = obj.find("streams")) {
                c.numStreams = readU32(*x, src, obj.at("streams"));
            }
            if (const Json* x = obj.find("host_dispatch")) {
                c.hostDispatch = readEnum(*x, src, obj.at("host_dispatch"), kDispatch);
            }
            if (const Json* x = obj.find("max_batch_size")) {
                c.maxBatchSize = readU32(*x, src, obj.at("max_batch_size"));
            }
            if (const Json* x = obj.find("max_items_per_check")) {
                c.maxItemsPerCheck = readU32(*x, src, obj.at("max_items_per_check"));
            }
            if (const Json* x = obj.find("obb_culling")) {
                if (x->is_boolean()) {
                    c.obbCulling = x->get<bool>() ? Toggle::On : Toggle::Off;
                } else if (x->is_string() && x->get<std::string>() == "auto") {
                    c.obbCulling = Toggle::Auto;
                } else {
                    fail(src, obj.at("obb_culling"), describe(*x) + " is not one of \"auto\", true, false");
                }
            }
            if (const Json* x = obj.find("collapse_root_link_obb")) {
                c.collapseRootLinkObb = readBool(*x, src, obj.at("collapse_root_link_obb"));
            }
            if (const Json* x = obj.find("scene_edges")) {
                c.sceneEdges = readAutoEnum(*x, src, obj.at("scene_edges"), kEdgeSets);
            }
            if (const Json* x = obj.find("verbosity")) {
                c.verbosity = readEnum(*x, src, obj.at("verbosity"), kLogLevels);
            }
            if (const Json* x = obj.find("optix")) {
                readOptix(*x, src, obj.at("optix"), c.optix);
            }
            if (const Json* x = obj.find("mesh")) {
                readMesh(*x, src, obj.at("mesh"), c.mesh);
            }
            if (const Json* x = obj.find("ray")) {
                readRay(*x, src, obj.at("ray"), c.ray);
            }
            if (const Json* x = obj.find("curve")) {
                readCurve(*x, src, obj.at("curve"), c.curve);
            }
        }

        // validate(), with every message line prefixed by "<source>: " and the field path by `path` (inline checker
        // objects of a setup file).
        void validateFrom(const CheckerConfig& c, const Source& src, const std::string& path) {
            try {
                c.validate();
            } catch (const ConfigError& e) {
                const std::string prefix = path.empty() ? std::string() : path + ".";
                std::istringstream lines(e.what());
                std::string line, text;
                while (std::getline(lines, line)) {
                    text += (text.empty() ? "" : "\n") + src.name + ": " + prefix + line;
                }
                throw ConfigError(text, prefix + e.field());
            }
        }

        CheckerConfig checkerFrom(const Json& v, const Source& src, const std::string& path) {
            CheckerConfig c;
            readChecker(v, src, path, c);
            validateFrom(c, src, path);
            return c;
        }

        // ---- scene ---------------------------------------------------------------------------------------------

        SceneDesc sceneFrom(const Json& v, const Source& src, const std::string& path) {
            const Object obj(v, src, path, {"name", "obstacles"});
            SceneDesc d;
            if (const Json* x = obj.find("name")) {
                d.name = readString(*x, src, obj.at("name"));
            }
            if (const Json* x = obj.find("obstacles")) {
                const std::string p = obj.at("obstacles");
                if (!x->is_array()) {
                    fail(src, p, "must be an array of obstacles, got " + describe(*x));
                }
                for (std::size_t i = 0; i < x->size(); ++i) {
                    const std::string op = indexed(p, i);
                    const Object o((*x)[i], src, op, {"name", "mesh", "scale", "pose"});
                    ObstacleDesc od;
                    od.mesh = readPath(o.required("mesh"), src, o.at("mesh"));
                    if (const Json* n = o.find("name")) {
                        od.name = readString(*n, src, o.at("name"));
                    } else {
                        od.name = utf8FromPath(od.mesh.stem());
                    }
                    if (const Json* s = o.find("scale")) {
                        od.scale = readFloat(*s, src, o.at("scale"));
                        if (!(od.scale > 0.0f)) {
                            fail(src, o.at("scale"), s->dump() + " must be > 0");
                        }
                    }
                    if (const Json* pose = o.find("pose")) {
                        od.pose = readPose(*pose, src, o.at("pose"));
                    }
                    d.obstacles.push_back(std::move(od));
                }
            }
            return d;
        }

        // ---- robot ---------------------------------------------------------------------------------------------

        GeometrySource readGeometrySource(const Json& v, const Source& src, const std::string& path, bool dirRequired,
            const char* format, MeshSource* meshSource) {
            const Object obj = meshSource ? Object(v, src, path, {"dir", "format", "source", "files"})
                                          : Object(v, src, path, {"dir", "format", "files"});
            GeometrySource g;
            g.format = format;
            if (dirRequired) {
                g.dir = readPath(obj.required("dir"), src, obj.at("dir"));
            } else if (const Json* x = obj.find("dir"); x && !x->is_null()) {
                g.dir = readPath(*x, src, obj.at("dir"));
            }
            if (const Json* x = obj.find("format")) {
                const std::string f = readString(*x, src, obj.at("format"));
                if (f != format) {
                    fail(src, obj.at("format"), "\"" + f + "\" is not supported (only \"" + format + "\")");
                }
            }
            if (meshSource) {
                if (const Json* x = obj.find("source")) {
                    *meshSource = readEnum(*x, src, obj.at("source"), kMeshSources);
                }
            }
            if (const Json* x = obj.find("files")) {
                const std::string p = obj.at("files");
                if (!x->is_object()) {
                    fail(src, p, "must be an object mapping link names to files, got " + describe(*x));
                }
                const fs::path base = g.dir.empty() ? src.baseDir : g.dir;
                for (const auto& item : x->items()) {
                    g.files[item.key()] = readPath(item.value(), src, join(p, item.key()), &base);
                }
            }
            return g;
        }

        RobotDesc robotBlockFrom(const Json& v, const Source& src, const std::string& path) {
            const Object obj(v, src, path, {"description", "geometry", "base_pose"});
            RobotDesc d;
            d.description = readPath(obj.required("description"), src, obj.at("description"));
            {
                const std::string gp = obj.at("geometry");
                const Object geo(obj.required("geometry"), src, gp, {"mesh", "spheres"});
                d.mesh = readGeometrySource(geo.required("mesh"), src, geo.at("mesh"), true, "obj", &d.meshSource);
                if (const Json* s = geo.find("spheres"); s && !s->is_null()) {
                    d.spheres = readGeometrySource(*s, src, geo.at("spheres"), false, "xyzr", nullptr);
                }
            }
            if (const Json* x = obj.find("base_pose")) {
                d.basePose = readPose(*x, src, obj.at("base_pose"));
            }
            return d;
        }

        // A robot file / text: { "robot": <block> }.
        RobotDesc robotFileFrom(const Json& v, const Source& src) {
            const Object obj(v, src, "", {"robot"});
            return robotBlockFrom(obj.required("robot"), src, "robot");
        }

        // ---- dumps ---------------------------------------------------------------------------------------------

        // Shortest decimal that reads back (through double, as nlohmann-json parses) to exactly f.
        Json floatJson(float f) {
            char buf[64];
            const auto res = std::to_chars(buf, buf + sizeof(buf), f);
            const double shortest = std::strtod(std::string(buf, res.ptr).c_str(), nullptr);
            return static_cast<float>(shortest) == f ? Json(shortest) : Json(static_cast<double>(f));
        }

        Json transformJson(const Transform& t) {
            Json a = Json::array();
            for (float v : t.m) {
                a.push_back(floatJson(v));
            }
            return a;
        }

        Json poseJson(const Transform& t) {
            Json p;
            p["matrix"] = transformJson(t);
            return p;
        }

        std::string absoluteText(const fs::path& p) {
            return p.empty() ? std::string() : utf8FromPath(normalized(fs::absolute(p)));
        }

        Json pathJson(const fs::path& p) { return p.empty() ? Json(nullptr) : Json(absoluteText(p)); }

        template <typename E>
        Json enumJson(E value, EnumTable<E> table) {
            if (const char* name = enumName(value, table)) {
                return name;
            }
            return static_cast<std::uint64_t>(value); // invalid value: written as is (does not load back)
        }

        Json checkerJson(const CheckerConfig& c) {
            Json j;
            j["mode"]                = enumJson(c.mode, kModes);
            j["streams"]             = c.numStreams;
            j["host_dispatch"]       = enumJson(c.hostDispatch, kDispatch);
            j["max_batch_size"]      = c.maxBatchSize;
            j["max_items_per_check"] = c.maxItemsPerCheck;
            j["obb_culling"]         = c.obbCulling == Toggle::Auto ? Json("auto") : Json(c.obbCulling == Toggle::On);
            j["collapse_root_link_obb"] = c.collapseRootLinkObb;
            j["scene_edges"]            = c.sceneEdges ? enumJson(*c.sceneEdges, kEdgeSets) : Json("auto");
            j["verbosity"]              = enumJson(c.verbosity, kLogLevels);

            Json o;
            if (c.optix.optLevel == OptixOptions::OptLevel::Default) {
                o["opt_level"] = "default";
            } else {
                o["opt_level"] = static_cast<unsigned>(c.optix.optLevel) - static_cast<unsigned>(OptixOptions::OptLevel::O0);
            }
            o["debug_level"]        = enumJson(c.optix.debugLevel, kDebugLevels);
            o["max_register_count"] = c.optix.maxRegisterCount;
            Json ex                 = Json::array();
            std::uint32_t rest      = c.optix.exceptionFlags;
            for (const auto& [name, bit] : kExceptions) {
                if (c.optix.exceptionFlags & bit) {
                    ex.push_back(name);
                    rest &= ~bit;
                }
            }
            if (rest != 0) {
                ex.push_back(rest); // unknown bits: written as is (does not load back)
            }
            o["exceptions"]         = ex;
            o["max_trace_depth"]    = c.optix.maxTraceDepth;
            o["print_pipeline_log"] = c.optix.printPipelineLog;
            j["optix"]              = o;

            Json m;
            m["link_gas_build"]       = enumJson(c.mesh.linkGasBuild, kBuilds);
            m["compaction_min_poses"] = c.mesh.compactionMinPoses;
            m["park_transform"]       = transformJson(c.mesh.parkTransform);
            j["mesh"]                 = m;

            Json r;
            r["launch_width_cap"] = c.ray.launchWidthCap;
            r["skip_hit_items"]   = c.ray.skipHitItems;
            r["hit_mode"]         = enumJson(c.ray.hitMode, kHitModes);
            j["ray"]              = r;

            Json cv;
            cv["trajectory_points"] = c.curve.trajectoryPoints;
            cv["control_points"]    = c.curve.controlPoints;
            cv["end_points"]        = enumJson(c.curve.endPoints, kEndPoints);
            cv["gas_build"]         = enumJson(c.curve.gasBuild, kBuilds);
            if (c.curve.refit.kind == RefitPolicy::Kind::Every) {
                Json e;
                e["every"]  = c.curve.refit.every;
                cv["refit"] = e;
            } else {
                cv["refit"] = enumJson(c.curve.refit.kind, kRefitKinds);
            }
            cv["auto_refit_max_batches"] =
                c.curve.autoRefitMaxBatches < 0 ? Json("auto") : Json(c.curve.autoRefitMaxBatches);
            cv["geometry_flags"] = c.curve.geometryFlags ? enumJson(*c.curve.geometryFlags, kGeometryFlags) : Json("auto");
            cv["spline_weights_dir"] = pathJson(c.curve.splineWeightsDir);
            j["curve"]               = cv;
            return j;
        }

        Json geometrySourceJson(const GeometrySource& g, const MeshSource* meshSource) {
            Json j;
            j["dir"]    = pathJson(g.dir);
            j["format"] = g.format;
            if (meshSource) {
                j["source"] = enumJson(*meshSource, kMeshSources);
            }
            Json files = Json::object();
            for (const auto& [link, file] : g.files) {
                files[link] = absoluteText(file);
            }
            j["files"] = files;
            return j;
        }

        // ---- overrides -----------------------------------------------------------------------------------------

        std::string_view trim(std::string_view s) {
            const auto first = s.find_first_not_of(" \t\r\n");
            if (first == std::string_view::npos) {
                return {};
            }
            const auto last = s.find_last_not_of(" \t\r\n");
            return s.substr(first, last - first + 1);
        }

        // Splits at commas outside JSON strings, arrays and objects.
        std::vector<std::string_view> splitEntries(std::string_view s) {
            std::vector<std::string_view> out;
            int depth          = 0;
            bool inString      = false;
            bool escaped       = false;
            std::size_t start  = 0;
            for (std::size_t i = 0; i < s.size(); ++i) {
                const char ch = s[i];
                if (inString) {
                    if (escaped) {
                        escaped = false;
                    } else if (ch == '\\') {
                        escaped = true;
                    } else if (ch == '"') {
                        inString = false;
                    }
                } else if (ch == '"') {
                    inString = true;
                } else if (ch == '[' || ch == '{') {
                    ++depth;
                } else if ((ch == ']' || ch == '}') && depth > 0) {
                    --depth;
                } else if (ch == ',' && depth == 0) {
                    out.push_back(s.substr(start, i - start));
                    start = i + 1;
                }
            }
            out.push_back(s.substr(start));
            return out;
        }

        void mergeInto(Json& target, const Json& value) {
            if (target.is_object() && value.is_object()) {
                for (const auto& item : value.items()) {
                    if (target.contains(item.key())) {
                        mergeInto(target[item.key()], item.value());
                    } else {
                        target[item.key()] = item.value(); // unknown key: reported by the reader
                    }
                }
            } else {
                target = value;
            }
        }

    } // namespace

    // ---- public API ----------------------------------------------------------------------------------------------

    CheckerConfig loadCheckerConfig(const std::filesystem::path& file) {
        const fs::path abs = normalized(fs::absolute(file));
        const Source src{utf8FromPath(abs), abs.parent_path()};
        return checkerFrom(parseJson(readFile(abs), src.name, abs), src, "");
    }

    CheckerConfig parseCheckerConfig(std::string_view json, const std::filesystem::path& baseDir, std::string_view sourceName) {
        const Source src{std::string(sourceName), absoluteDir(baseDir)};
        return checkerFrom(parseJson(json, src.name, fs::path(std::u8string(sourceName.begin(), sourceName.end()))), src, "");
    }

    SceneDesc loadSceneDesc(const std::filesystem::path& file) {
        const fs::path abs = normalized(fs::absolute(file));
        const Source src{utf8FromPath(abs), abs.parent_path()};
        return sceneFrom(parseJson(readFile(abs), src.name, abs), src, "");
    }

    SceneDesc parseSceneDesc(std::string_view json, const std::filesystem::path& baseDir, std::string_view sourceName) {
        const Source src{std::string(sourceName), absoluteDir(baseDir)};
        return sceneFrom(parseJson(json, src.name, fs::path(std::u8string(sourceName.begin(), sourceName.end()))), src, "");
    }

    RobotDesc loadRobotDesc(const std::filesystem::path& file) {
        const fs::path abs = normalized(fs::absolute(file));
        const Source src{utf8FromPath(abs), abs.parent_path()};
        return robotFileFrom(parseJson(readFile(abs), src.name, abs), src);
    }

    RobotDesc parseRobotDesc(std::string_view json, const std::filesystem::path& baseDir, std::string_view sourceName) {
        const Source src{std::string(sourceName), absoluteDir(baseDir)};
        return robotFileFrom(parseJson(json, src.name, fs::path(std::u8string(sourceName.begin(), sourceName.end()))), src);
    }

    Setup loadSetup(const std::filesystem::path& file) {
        const fs::path abs = normalized(fs::absolute(file));
        const Source src{utf8FromPath(abs), abs.parent_path()};
        const Json j = parseJson(readFile(abs), src.name, abs);
        const Object obj(j, src, "", {"checker", "scene", "robot"});

        Setup s;
        s.sourceFile = abs;
        // Each part: a path (resolved against the setup file, read as its own file) or the object inline.
        const Json& checker = obj.required("checker");
        if (checker.is_string()) {
            s.checker = loadCheckerConfig(readPath(checker, src, "checker"));
        } else {
            s.checker = checkerFrom(checker, src, "checker");
        }
        const Json& scene = obj.required("scene");
        if (scene.is_string()) {
            s.scene = loadSceneDesc(readPath(scene, src, "scene"));
        } else {
            s.scene = sceneFrom(scene, src, "scene");
        }
        const Json& robot = obj.required("robot");
        if (robot.is_string()) {
            s.robot = loadRobotDesc(readPath(robot, src, "robot"));
        } else {
            s.robot = robotBlockFrom(robot, src, "robot");
        }
        return s;
    }

    void applyOverrides(CheckerConfig& config, std::string_view overrides) {
        Json doc = checkerJson(config);
        const fs::path cwd = absoluteDir({});
        CheckerConfig result = config;
        for (const std::string_view rawEntry : splitEntries(overrides)) {
            const std::string_view entry = trim(rawEntry);
            if (entry.empty()) {
                continue;
            }
            const std::string entryText(entry);
            const Source src{"overrides entry \"" + entryText + "\"", cwd};
            const std::size_t eq = entry.find('=');
            if (eq == std::string_view::npos || trim(entry.substr(0, eq)).empty()) {
                throw ConfigError(src.name + ": expected <dotted.path>=<value>");
            }
            const std::string key(trim(entry.substr(0, eq)));
            const std::string valueText(trim(entry.substr(eq + 1)));

            // JSON value, or a bare word taken as a string.
            Json value;
            try {
                value = Json::parse(valueText);
            } catch (const nlohmann::json::parse_error&) {
                value = valueText;
            }

            // Walk the dotted path through the dump (which has every key of the schema).
            Json* node = &doc;
            std::string walked;
            std::size_t pos = 0;
            while (true) {
                const std::size_t dot = key.find('.', pos);
                const std::string part = key.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos);
                const std::string here = join(walked, part);
                if (part.empty()) {
                    fail(src, key, "malformed dotted path");
                }
                if (!node->is_object() || !node->contains(part)) {
                    fail(src, here, "unknown key of the checker config");
                }
                node   = &(*node)[part];
                walked = here;
                if (dot == std::string::npos) {
                    break;
                }
                pos = dot + 1;
            }
            mergeInto(*node, value);

            // Read the whole document back: type / enum / unknown-key errors name this entry.
            CheckerConfig next;
            readChecker(doc, src, "", next);
            result = next;
        }
        validateFrom(result, Source{"overrides", cwd}, "");
        config = result;
    }

    std::string dumpCheckerConfig(const CheckerConfig& config) { return checkerJson(config).dump(2); }

    std::string dumpSceneDesc(const SceneDesc& desc) {
        Json j;
        j["name"]      = desc.name;
        Json obstacles = Json::array();
        for (const ObstacleDesc& o : desc.obstacles) {
            Json jo;
            jo["name"]  = o.name;
            jo["mesh"]  = absoluteText(o.mesh);
            jo["scale"] = floatJson(o.scale);
            jo["pose"]  = poseJson(o.pose);
            obstacles.push_back(jo);
        }
        j["obstacles"] = obstacles;
        return j.dump(2);
    }

    std::string dumpRobotDesc(const RobotDesc& desc) {
        Json r;
        r["description"] = pathJson(desc.description);
        Json geometry;
        geometry["mesh"]    = geometrySourceJson(desc.mesh, &desc.meshSource);
        geometry["spheres"] = desc.spheres ? geometrySourceJson(*desc.spheres, nullptr) : Json(nullptr);
        r["geometry"]       = geometry;
        r["base_pose"]      = poseJson(desc.basePose);
        Json j;
        j["robot"] = r;
        return j.dump(2);
    }

    namespace detail {
        // Kept until main deletes tests/skeleton at the Wave-1 merge (test_skeleton_unit calls it to prove linking).
        void jsonPlaceholder() {}
    } // namespace detail

} // namespace rtcd
