# Configuration reference

Every JSON key the new stack reads, with its type, default and meaning. Sources: the C++ structs in
`include/rtcd/config.h`, `scene.h`, `robot_model.h` (their default member initializers are the only source of
defaults), the loader `src/json/json_config.cpp`, validation and resolution in `src/config/config.cpp`, and the
construction-time checks in `src/detector` and `src/backend_*`. Design-level schemas: `docs/design/interfaces.md`
section 7.

Contents: [general rules](#general-rules) - [setup file](#setup-file) - [checker config](#checker-config) -
[scene file](#scene-file) - [robot file](#robot-file) - [overrides and dumps](#overrides-and-dumps) -
[options that are not in JSON](#options-that-are-not-in-json)

## General rules
- Keys are snake_case. Every key is optional unless marked **required**; an unspecified key keeps the C++ default.
- Unknown keys, duplicate keys in one object, wrong types and enum strings that are not listed are errors.
- Errors are exceptions: `rtcd::IoError` for a file that cannot be opened and for JSON syntax errors
  (`<file>:<line>:<column>: ...`), `rtcd::ConfigError` for everything else, with the message
  `<file>: <dotted.json.path>: <message>`. A checker config reports every violated rule at once, one per line.
- JSON comments are not accepted.
- Relative paths are resolved against the directory of the file that contains them (for a setup file that
  references other files: against each referenced file's own directory), then made absolute. Nothing is resolved
  against the working directory, and no path is compiled into a binary.
- Loading a checker config validates it (`CheckerConfig::validate()`); replacing `"auto"` values by their
  effective values (`resolved()`) happens in the `CollisionChecker` constructor, which also runs the checks that
  need the robot, the scene or files.

Three kinds of checker parameters: **runtime** (fixed when the checker is constructed; a different value needs a
new checker), **capacity** (a maximum: buffers are allocated once for it and each call must stay within it, else
`CapacityError`), **fixed** (every call uses exactly this value).

## Setup file
Loaded by `rtcd::loadSetup(path)` -> `rtcd::Setup {checker, scene, robot, sourceFile}`.

| key | type | meaning |
|---|---|---|
| `checker` (**required**) | path or object | a checker config file, or the checker object inline |
| `scene` (**required**) | path or object | a scene file, or the scene object inline |
| `robot` (**required**) | path or object | a robot file (`{"robot": {...}}`), or the robot block itself inline (without the `"robot"` wrapper) |

```json
{ "checker": "../checker_quad.json", "scene": "../scenes/shelf.json", "robot": "../robots/panda.json" }
```

## Checker config
Loaded by `rtcd::loadCheckerConfig(path)` / `parseCheckerConfig(text, baseDir)` into `rtcd::CheckerConfig`.

### Top level

| key | type | default | kind | meaning |
|---|---|---|---|---|
| `mode` | `"mesh"` \| `"ray"` \| `"linear_curve"` \| `"quadratic_curve"` \| `"cubic_curve"` | `"mesh"` | runtime | collision mode (backend), see the README |
| `streams` | integer >= 1 | `4` | runtime | CUDA streams of the checker; batch b runs on stream b % streams |
| `host_dispatch` | `"single_thread"` \| `"thread_per_stream"` | `"thread_per_stream"` | runtime | `single_thread`: the calling thread issues every batch in order; `thread_per_stream`: one persistent worker thread per stream issues that stream's batches, so a host synchronization on one stream overlaps work on the others |
| `max_batch_size` | integer >= 1 | `1024` | capacity | items per batch; every batch of a check but the last is full |
| `max_items_per_check` | integer >= 1 | `8192` | capacity | items of one `check()` call; more is a `CapacityError` |
| `obb_culling` | `"auto"` \| `true` \| `false` | `"auto"` | runtime | link / segment OBB vs. obstacle OBB pre-test. `"auto"`: on for `mesh` and `ray`, off for the curve modes. `true` is rejected for `cubic_curve` |
| `collapse_root_link_obb` | boolean | `true` | runtime | shrink link 0's OBB to its center point before OBB culling (the old stack's `SKIP_BASE`; the goldens were recorded with it) |
| `scene_edges` | `"auto"` \| `"unique"` \| `"loop"` | `"auto"` | runtime | which obstacle edges become rays in `mesh` and the curve modes. `"auto"`: `unique` for `mesh`, `loop` for the curve modes. `loop` needs every obstacle to be watertight. Ignored by `ray` |
| `verbosity` | `"error"` \| `"warning"` \| `"info"` \| `"debug"` | `"warning"` | runtime | level of this checker's messages; they go to the Context's `Logger` |
| `optix` | object | see below | runtime | OptiX compile / link options |
| `mesh` | object | see below | runtime | read only when `mode` is `mesh` |
| `ray` | object | see below | runtime | read only when `mode` is `ray` |
| `curve` | object | see below | runtime / fixed | read only for the curve modes |

### `optix`

| key | type | default (Release) | meaning |
|---|---|---|---|
| `opt_level` | `"default"` \| `0` \| `1` \| `2` \| `3` | `"default"` | OptiX module optimization level |
| `debug_level` | `"default"` \| `"none"` \| `"minimal"` \| `"moderate"` \| `"full"` | `"default"` | OptiX module debug level (OptiX default = minimal) |
| `max_register_count` | integer >= 0 | `50` | `0` = OptiX default |
| `exceptions` | list of `"stack_overflow"`, `"trace_depth"`, `"user"`; or `["none"]` | `[]` | OptiX exception flags; `"none"` cannot be combined with other names |
| `max_trace_depth` | integer >= 1 | `2` | pipeline `maxTraceDepth`; stack sizes are always computed for trace depth 1 |
| `print_pipeline_log` | boolean | `false` | print the pipeline compile log to the Logger (Info) |

The defaults come from `OptixOptions::buildDefault()` of the translation unit that default-constructs the config.
In a Debug build (`_DEBUG` or `DEBUG` defined) they are those of `OptixOptions::debug()`: `opt_level` 0,
`debug_level` `"full"`, `exceptions` `["stack_overflow", "trace_depth"]`. Only Release builds have been tested.

### `mesh` (mode `mesh`)

| key | type | default | meaning |
|---|---|---|---|
| `link_gas_build` | `"fast_build"` \| `"fast_trace"` \| `"fast_trace_compact"` | `"fast_trace_compact"` | build preference of the per-link triangle GAS (built once, at construction). `fast_trace_compact` = PREFER_FAST_TRACE + ALLOW_COMPACTION, then compacted |
| `compaction_min_poses` | integer >= 0 | `128` | with OBB culling: batches with at least this many items stream-compact the surviving instances (one device-to-host count read and one stream synchronization per batch, smaller IAS build); smaller batches build the IAS over all instances with culled ones hidden by visibility mask 0 (no host round trip). `0` = always compact; larger than `max_batch_size` = never. The header documents 128 as the crossover measured with BM_IAS on an RTX 5080 |
| `park_transform` | array of 12 numbers (3x4 row-major `[R \| t]`) | scale 0.001, translation (0, 0, -0.1) | transform given to unused instance slots of a partial batch on the refit path (OBB culling off), so they cannot collide |

### `ray` (mode `ray`)

| key | type | default | meaning |
|---|---|---|---|
| `launch_width_cap` | integer >= 0 | `262144` | `optixLaunch` width = min(items x rays per robot, cap); the ray generation program grid-strides over all rays. `0` = no cap |
| `skip_hit_items` | boolean | `true` | skip the remaining rays of items already known to collide. Flags are unchanged; hit counts become lower bounds |
| `hit_mode` | `"closest_hit"` \| `"first_hit_ch"` \| `"first_hit_miss"` \| `"traverse"` | `"closest_hit"` | how a ray decides hit / no hit: closest-hit program; + terminate on first hit; terminate on first hit with closest-hit disabled and a miss program; `optixTraverse` + `optixHitObjectIsHit()`. All four give the same answer per ray and differ only in cost |

At construction the checker requires `max_batch_size * rays per robot < 2^30` (OptiX launch size limit), where
the rays per robot are the unique edges of all link meshes (`CapacityError` otherwise).

### `curve` (modes `linear_curve`, `quadratic_curve`, `cubic_curve`)

| key | type | default | kind | meaning |
|---|---|---|---|---|
| `trajectory_points` | integer, 2 .. 512 | `16` | fixed | poses per trajectory item |
| `control_points` | integer | `8` | fixed | B-spline control points fitted to each sphere's path (quadratic / cubic only; ignored for linear curves, whose vertices are the trajectory points). Must satisfy degree < `control_points` < `trajectory_points` and `control_points * trajectory_points <= 512` (with `"phantom"`: `(control_points + 2) * trajectory_points <= 512`) |
| `end_points` | `"normal"` \| `"repeated"` \| `"phantom"` | `"normal"` | runtime | end-point treatment of the B-spline fit; selects the weight file directory. `"phantom"` adds two control points per curve; no weight files exist for it |
| `gas_build` | as `mesh.link_gas_build` | `"fast_build"` | runtime | build preference of the per-batch curve GAS. ALLOW_UPDATE is added automatically when refits are possible (OBB culling off) |
| `refit` | `"auto"` \| `"always"` \| `"never"` \| `{"every": K}` (K >= 2) | `"auto"` | runtime | refit (OptiX update) vs. rebuild of the curve GAS. `auto`: refit while the check has at most `auto_refit_max_batches` batches, else rebuild; `always`: refit whenever possible; `never`: rebuild every batch; `{"every": K}`: rebuild every K-th build of a stream, refit in between. A refit only happens when the batch has the same item count as the last full build on that stream. With OBB culling on, the effective policy is `never`. Mostly a cost trade-off, but a refitted GAS can flip grazing hits (see the README, known limitations) |
| `auto_refit_max_batches` | `"auto"` \| integer >= 0 | `"auto"` | runtime | threshold of `refit: "auto"`. `"auto"`: 2 for linear, 1 for quadratic and cubic curves |
| `geometry_flags` | `"auto"` \| `"none"` \| `"single_anyhit"` | `"auto"` | runtime | geometry flag of the curve build input. `"auto"`: `single_anyhit` (REQUIRE_SINGLE_ANYHIT_CALL) for quadratic, `none` for linear and cubic curves. Changes cost and hit counts; the quadratic golden is reproduced with both values |
| `spline_weights_dir` | path or `null` | none | runtime | directory with `<quad\|cubic>/<normal\|repeated\|phantom>/<control_points>_<trajectory_points>.txt`; **required** for quadratic and cubic curves |

Spline weight files: one float per token (whitespace separated), row-major `[control point][trajectory point]`,
exactly `control points per curve * trajectory_points` values (`IoError` otherwise, also when the file is missing).
`configs/spline_weights` holds, for `quad` and `cubic` and for `normal` and `repeated`, the pairs that fit the
512-weight limit: `4_8`, `4_16`, `4_32`, `4_64`, `4_128`, `8_16`, `8_32`, `8_64`, `16_32`, and for `quad` also
`3_6`, `5_10`, `6_12`, `7_14`. (Files for larger pairs, e.g. `16_64`, exist but exceed the limit.)

Curve modes also need robot spheres (`robot.geometry.spheres`); without them the checker constructor throws
`ConfigError`.

### Validation summary (`CheckerConfig::validate()`, at load time)
- `streams`, `max_batch_size`, `max_items_per_check`, `optix.max_trace_depth` >= 1.
- Curve modes: `2 <= trajectory_points <= 512`; for quadratic / cubic: degree < `control_points` <
  `trajectory_points`, the weight count <= 512 and `spline_weights_dir` set; `refit: {"every": K}` needs K >= 2;
  `obb_culling: true` with `cubic_curve` is rejected; `geometry_flags` only `auto` / `none` / `single_anyhit`.
- `mesh.compaction_min_poses` accepts any value; `scene_edges` is accepted (and ignored) for `ray`.

### Checks at checker construction (need the robot, the scene or files)
- The scene must have been built on the checker's Context (`InvalidArgument`).
- Curve modes: the robot has spheres (`ConfigError`); the spline weight file exists and has the right size
  (`IoError`).
- Mesh and curve modes: the effective `scene_edges` set must exist; `loop` exists only if every obstacle is
  watertight (`UnsupportedError` naming the first non-watertight obstacle).
- Ray: `max_batch_size * rays per robot < 2^30`; mesh: `max_batch_size * links` instances fit one IAS; scene edge
  rays fit one launch (`CapacityError`).

### Example
`configs/checker_ray.json` (the ray regression; keys equal to their default could be omitted):
```json
{
  "mode": "ray",
  "streams": 4,
  "host_dispatch": "thread_per_stream",
  "max_batch_size": 128,
  "max_items_per_check": 4096,
  "obb_culling": "auto",
  "collapse_root_link_obb": true,
  "verbosity": "warning",
  "optix": {
    "opt_level": "default",
    "debug_level": "default",
    "max_register_count": 50,
    "exceptions": [],
    "max_trace_depth": 2,
    "print_pipeline_log": false
  },
  "ray": {
    "launch_width_cap": 262144,
    "skip_hit_items": true,
    "hit_mode": "closest_hit"
  }
}
```

## Scene file
Loaded by `rtcd::loadSceneDesc(path)` into `rtcd::SceneDesc`; built on the GPU by `rtcd::Scene(context, desc)`.

| key | type | default | meaning |
|---|---|---|---|
| `name` | string | `""` | free text |
| `obstacles` | array of obstacle objects | `[]` | obstacle order = obstacle index everywhere. Building a `Scene` with no obstacle throws `InvalidArgument` |
| `obstacles[i].mesh` (**required**) | path | -- | OBJ file (triangulated on load) |
| `obstacles[i].name` | string | mesh file name without extension | used in messages and dumps |
| `obstacles[i].scale` | number > 0 | `1` | uniform scale, applied before the pose: `world = pose * (scale * v)` |
| `obstacles[i].pose` | pose object | identity | obstacle -> world |

**Pose object** (also used by `robot.base_pose`): `"translation": [x, y, z]` (meters, default 0) plus at most one
of `"rpy": [roll, pitch, yaw]` (radians, URDF convention R = Rz(yaw) Ry(pitch) Rx(roll)), `"quaternion": [x, y, z, w]`
(normalized on load, must not be zero), or `"matrix": [12 numbers]` (row-major 3x4 `[R | t]`; then
`"translation"` is not allowed).

```json
{ "name": "bin", "mesh": "meshes/bin.obj", "scale": 0.5, "pose": { "translation": [0.0, 0.6, 0.0], "rpy": [0, 0, 1.5708] } }
```

## Robot file
Loaded by `rtcd::loadRobotDesc(path)` into `rtcd::RobotDesc`; loaded by `rtcd::RobotModel::load(desc)` (host only).
The file is an object with one key `"robot"`; in a setup file the block can also be given inline.

| key | type | default | meaning |
|---|---|---|---|
| `robot.description` (**required**) | path | -- | URDF file |
| `robot.geometry` (**required**) | object | -- | where the link geometry is |
| `robot.geometry.mesh` (**required**) | object | -- | link meshes |
| `robot.geometry.mesh.dir` (**required**) | path | -- | directory of the link mesh files |
| `robot.geometry.mesh.format` | `"obj"` | `"obj"` | only OBJ is supported |
| `robot.geometry.mesh.source` | `"visual"` \| `"collision"` | `"visual"` | which URDF element names each link's mesh |
| `robot.geometry.mesh.files` | object `{ "<link name>": path }` | `{}` | per-link mesh file overrides; relative entries are resolved against `mesh.dir` |
| `robot.geometry.spheres` | object or `null` | absent | sphere files; required by the curve modes, optional otherwise |
| `robot.geometry.spheres.dir` | path or `null` | -- | directory of the sphere files |
| `robot.geometry.spheres.format` | `"xyzr"` | `"xyzr"` | only `xyzr` is supported |
| `robot.geometry.spheres.files` | object `{ "<link name>": path }` | `{}` | per-link sphere file overrides; relative entries are resolved against `spheres.dir` (or the robot file's directory if `dir` is not set) |
| `robot.base_pose` | pose object | identity | robot root -> world |

File naming rules, URDF rules and the sphere format: README, section "Set up a new robot" (authoritative text:
the loading rules in `include/rtcd/robot_model.h`).

## Overrides and dumps
- `rtcd::applyOverrides(config, "path=value,path=value")`: dotted paths of the checker schema, JSON values; bare
  words are taken as strings, e.g. `mode=ray,optix.max_register_count=32,ray.hit_mode=traverse,curve.refit={"every":4}`.
  Each entry goes through the normal reader (same type and enum checks); the result is validated. Commas inside
  JSON objects / arrays / strings do not split entries. The regression and benchmark apps apply the environment
  variable `RTCD_OVERRIDES` this way (the library itself never reads environment variables); `rtcd_regression`
  also takes `--override "..."`.
- `rtcd::dumpCheckerConfig(config)`, `dumpSceneDesc(desc)`, `dumpRobotDesc(desc)`: every field, stable key order,
  absolute paths; loading a dump gives back an equal struct. `CollisionChecker::config()` is the resolved config
  (every `"auto"` replaced by its value), so `dumpCheckerConfig(checker.config())` shows what a checker really uses.

## Options that are not in JSON
`rtcd::ContextOptions`, passed to `rtcd::Context::create()` in C++:

| field | default | meaning |
|---|---|---|
| `device` | `0` | CUDA device ordinal |
| `optixValidation` | `false` (`true` in Debug builds via `ContextOptions::buildDefault()`) | OptiX validation mode |
| `optixLogLevel` | `0` | `0` = no OptiX log callback; `1..4` = OptiX log level, messages forwarded to `logger` |
| `logger` | level `Warning`, stderr sink | `rtcd::Logger {level, sink}` used by everything built on the Context; the sink may be called from worker threads and must be thread-safe |
