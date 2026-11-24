# RTCD new stack: interface design (Stage 0-B, amended in Stage 0-C)

> **The old stack has been removed (Stage 3-D).** The header-only code under `RTCD/` was deleted from this branch
> together with its CMake, the side-by-side runner (`apps/side_by_side`) and every parity test that called it; the
> last commit that contains it is tag **`old-stack-final`** (`632b6a5`). Its outputs survive as stored references
> (`tests/data/reference`, written from the old stack just before the deletion) and the goldens, regression and
> benchmark inputs moved from the untracked `data/` folder into the repository (`tests/data/CCTest`,
> `tests/data/motions`; `tests/data/README.md`). Everything below that names `RTCD/...` files, old classes, parity
> tests against the old stack or `data/CCTest` / `data/motions` is HISTORY: it records where each piece came from.
> The old -> new mapping tables (sections 4, 5 and 6) are kept for that reason.

> **FROZEN as of commit `b4948ee` on branch `optix9-upgrade`** (the merge of Stage 0-C; Stage 0-C was based on
> `c46667d`).
> Since then: comment-only clarifications at Wave-1 close-out, and comment-only clarifications at Wave-2 close-out
> (Stage 3, track C, based on `a605b47`: `RefitPolicy` can flip grazing hits, 3.7; the core makes the Context
> current on every thread that calls `beginCheck` / `prepare`, and the loop comment of `backend.h` lists "zero
> hit counts" before "staging copy" as in section 4, 3.10; `ReferenceExecutor::launch` threading, 3.13 -- the
> last one with a small fix in `tests/support/reference_executor.cpp`, no declaration changed). Historical
> skeleton-stage / Wave-1 material (placeholders, `tests/skeleton`, `debug_box`) is marked as such in sections 6 and 8;
> section 10 records which open questions are settled by measurement.
> **graph_prepare / prepareCapturable added 2026-11-24** (Track G0, authorized by the owner through main):
> `CheckerConfig::graphPrepare` (JSON `graph_prepare`, default true, 3.7 / 7.1) and the virtual
> `Backend::prepareCapturable(itemCount)` with a default of false (3.10, section 4 step 7); no other declaration changed.
> **BatchView address rules relaxed 2026-11-24** (Track G2, comment-only, approved by main): items may be read in
> place, hit counts may be counted into the result range, single-batch checks run on the calling thread (3.7
> `HostDispatch`, 3.10 `BatchView` / `prepareCapturable` (b), section 4 steps 3, 6, 7, 9; Q3 / Q4 settled).
> From here on the contract -- every header under `include/rtcd/` (public and `internal/`) and this document --
> is changed only by main. **Change procedure:** a module agent that needs a header change does not make it; it sends main a request naming the file, the exact declaration (old and new
> text) and the reason. Main decides, applies the change in one commit to the contract, updates this document in
> the same commit, and tells every affected agent. Implementation files, private headers under `src/<module>/`
> and tests stay with their owners (section 8).

Status: frozen contract. Stage 0-B base commit: `22d86d1`; Stage 0-C amendments: curve geometry-flag default
"auto" (sections 3.7, 7.1), one library / directory list with five Wave-1 agents (sections 2, 8), geometry I/O
seam declarations (`rtcd/geometry.h`, section 3.6), compile-only CMake targets (section 9).
Headers: `include/rtcd/`. Everything here implements section 3 of the restructuring brief (side-by-side stack,
seams only at rare calls, three kinds of parameters, JSON outside the core, runtime robot, one backend interface,
`CollisionChecker` facade, Level-2 optimizations kept).

Contents

1. Conventions
2. Modules, libraries, dependency graph
3. Headers, one section each
4. The call sequence of one `check()` (mapped to the old stack)
5. What each backend does in `prepare()` (mapped to the old stack)
6. Old-stack concept -> new-stack home
7. JSON schemas and examples
8. Wave 1 / Wave 2 assignments
9. Compile check
10. Open questions (with recommendations)
11. Requests outside this track's ownership

---

## 1. Conventions

- Namespace `rtcd` (lower case; the old stack is `RTCD`, so both can be included in one translation unit, which the
  parity tests need). Module-to-module contracts that are not user API are in `rtcd::internal`
  (`include/rtcd/internal/`).
- Public headers are plain C++20, include no CUDA or OptiX header, and compile with `/permissive- /W4 /WX`
  (enforced by CMake targets in `AllNew`, section 9). CUDA / OptiX handles are aliases of the same opaque tags (`rtcd::StreamHandle` *is*
  `cudaStream_t`, `rtcd::DevicePtr` *is* `CUdeviceptr`, `rtcd::TraversableHandle` *is* `OptixTraversableHandle`,
  `rtcd::OptixContextHandle` *is* `OptixDeviceContext`), so no casts are needed in implementation files.
  Mirrored OptiX enums (`BuildFlagBits`, `GeometryFlagBits`, `PrimitiveTypeBits`, `TraversableGraph`, exception
  bits) have the OptiX numeric values (verified, section 9).
- `include/rtcd/limits.h` is the only header for compile-time constants and host/device shared layouts. It is also
  valid CUDA C++17 device code (verified with nvcc).
- Units: meters, radians. Transforms: row-major 3x4 `[R | t]`, 12 floats.
- Every type that owns GPU resources is move-only and uses pimpl, so headers stay free of implementation types.
- Errors: exceptions only (`rtcd/errors.h`); no `exit()` in library code; CUDA / OptiX checked in every build type.
- Diagnostics: through `rtcd::Logger` (sink + level), never `std::cout` in library code.

## 2. Modules, libraries, dependency graph

Header includes (rtcd headers only; a header only includes headers listed above it):

```
limits.h
errors.h                        (none)
types.h                         <- limits.h                                   (also Logger)
geometry.h                      <- limits.h                                   (+ rtcd_geometry free functions)
device_buffer.h                 <- types.h
context.h                       <- types.h
config.h                        <- geometry.h, types.h
internal/algorithms.h           <- limits.h, types.h
internal/accel.h                <- context.h, device_buffer.h, types.h
internal/kinematics.h           <- device_buffer.h, limits.h, types.h
scene.h                         <- context.h, geometry.h, limits.h, types.h
robot_model.h                   <- geometry.h, limits.h, types.h              (host only, no Context)
backend.h                       <- config.h, context.h, limits.h, robot_model.h, scene.h, types.h
checker.h                       <- config.h, context.h, robot_model.h, scene.h, types.h
json_config.h                   <- config.h, robot_model.h, scene.h
internal/reference_executor.h   <- backend.h, config.h, context.h
rtcd.h                          umbrella of the public (non-internal) headers
```

Libraries (CMake targets as merged in Stage 0-C; `src/CMakeLists.txt` repeats this list):

| library | directory | implements | links (PUBLIC unless noted) | wave / agent |
|---|---|---|---|---|
| `rtcd_core` | `src/core` | errors.h, types.h (Logger), device_buffer.h, context.h, internal/algorithms.h, internal/accel.h | CUDA runtime + driver, OptiX (CUB from the toolkit) | 1a |
| `rtcd_config` | `src/config` | config.h (`validate`, `resolved`) | -- (no dependencies) | 1b |
| `rtcd_json` | `src/json` | json_config.h | rtcd_config; nlohmann-json (private) | 1b |
| `rtcd_geometry` | `src/geometry` | geometry.h free functions: `loadObjMesh` (reproducing the old `loadOBJ`), `computeEdges` (unique / loop edges), `fitObb`, `loadSpheres` | rtcd_core; tinyobjloader, Eigen (private) | 1c |
| `rtcd_scene` | `src/scene` | scene.h (`Scene`: obstacle GASes + IAS, both edge sets, obstacle OBBs) | rtcd_geometry | 1c |
| `rtcd_robot` | `src/robot` | robot_model.h (URDF -> chain -> kinematics table, link geometry via rtcd_geometry), internal/kinematics.h (kernels) | rtcd_geometry; urdfdom, Eigen (private) | 1d |
| `rtcd_testsupport` | `tests/support` | internal/reference_executor.h, cached inputs / goldens utilities | rtcd_core, rtcd_config, rtcd_robot, rtcd_scene | 1e |
| `rtcd_detector` | `src/detector` | checker.h (pipeline, SBT, batch loop, dispatch) | rtcd_config, rtcd_robot, rtcd_scene; the three backends (private) | 2e |
| `rtcd_backend_mesh` | `src/backend_mesh` | `createMeshBackend` + its embedded OptiX programs | rtcd_robot, rtcd_scene, rtcd_config | 2f |
| `rtcd_backend_ray` | `src/backend_ray` | `createRayBackend` + its embedded OptiX programs | rtcd_robot, rtcd_scene, rtcd_config | 2g |
| `rtcd_backend_curve` | `src/backend_curve` | `createCurveBackend` + its embedded OptiX programs and curve kernels | rtcd_robot, rtcd_scene, rtcd_config | 2h |

Umbrella INTERFACE target `rtcd` = rtcd_detector + the three backends + rtcd_config + rtcd_json (what apps and
tests link). There is no shared programs library: each backend compiles its own program `.cu` to OptiX IR and
embeds it with `rtcd_add_optix_ir(rtcd_backend_<x>_ir SOURCES ...)` (`cmake/RTCDTargets.cmake`, linked PRIVATE),
returning the bytes in `ProgramSet::moduleCode`. `rtcd_testsupport` is configured from the top-level
`CMakeLists.txt` (apps use it too), even with `BUILD_TESTS=OFF`.

No library depends on JSON except `rtcd_json`. Every compile-time-constant kernel parameter (curve degree, the
reduction width of the control-point kernel) is a template explicitly instantiated inside its `.cu`.

## 3. Headers

### 3.1 `limits.h` -- constants and shared layouts

Purpose: `kMaxDof = 16`, `kMaxLinks = 17`, curve degrees `[1, 3]`, `kMaxSplineWeights = 512` (old
`__constant__ splineWeights[512]`), `kMaxTrajectoryPoints = 512`, `kMaxLaunchSize = 2^30`,
`kMaxLaunchParamsBytes = 256`, SBT record header / alignment (32 / 16), launch-params variable name
`rtcdLaunchParams`; layouts `Obb` (byte-identical to old `RTCD::OBB`), `KinematicsTable` (replaces the
`cMask/sMask/oMask/baseAffine` globals), `SbtRecord` (48 bytes, data 0), `EdgeRayParams` (Mesh + curves),
`RobotRayParams` (Ray). All sizes static_asserted.
Ownership: frozen; changing a layout means changing every program reading it.
Threading / errors: n/a (constants and PODs).

`KinematicsTable` semantics (exactly the old kernels): `local_j(q) = cos(q_j) C_j + sin(q_j) S_j + O_j`,
`link_0 = base`, `link_{j+1} = link_j * local_j(q_j)`, cos/sin via `__sincosf`, entries for `j >= dof` zero.

### 3.2 `types.h` -- handles, enums, item views, logging

Purpose: handle aliases; `DeviceSpan<T>`; `Mode` (Mesh, Ray, LinearCurve, QuadraticCurve, CubicCurve) with
`isContinuous` / `curveDegree`; `BuildPreference` + `toBuildFlags`; build / geometry flag bits; `CurveEndPoints`;
`EdgeSet`; `RayHitMode`; `Toggle`; `ItemLayout`, `HostItems`, `DeviceItems`; `LogLevel`, `Logger`.

Pose input layout (decided): flat float32 array; item `i` starts at float `i * stride` (`stride == 0` means packed,
`stride = floatsPerItem`); a discrete item is `dof` floats; a trajectory item is `trajectoryPoints` waypoints of
`dof` floats each, waypoint-major (`[item][waypoint][joint]`); joints in chain order root -> tip
(`RobotModel::jointNames()`), radians. This is the old in-memory layout (`std::array<float, Dim>` per pose,
`nTrajPts` consecutive poses per trajectory), so the cached inputs under `data/motions` can be passed unchanged.

Ownership: views are non-owning; host memory is only read during the call. Threading: value types.
Errors: none here (the checker validates views). `Logger::defaultSink` (stderr, serialized) is in `rtcd_core`.

### 3.3 `errors.h` -- exceptions and checking policy

Purpose: `Error` (base, `kind()`), `ConfigError` (field path), `IoError` (path, line, column), `CapacityError`
(name, requested, capacity), `InvalidArgument`, `UnsupportedError`, `CudaError` (code, driver?, sticky?),
`OptixError` (code, compile log); macros `RTCD_CUDA_CHECK`, `RTCD_CU_CHECK`, `RTCD_OPTIX_CHECK`,
`RTCD_OPTIX_CHECK_LOG` (local variable `rtcd_check_result_`, never `res`).
Policy (also in the header): checked in all build types (the old `CUDA_CHECK` was a no-op in Release); kernel
launches followed by `RTCD_CUDA_CHECK(cudaGetLastError())`; no added device syncs; asynchronous faults surface
as sticky `CudaError` at the next checked call; destructors never throw; caller errors are detected before GPU
work. Ownership / threading: exception objects are values.

### 3.4 `device_buffer.h` -- `DeviceBuffer`, `PinnedHostBuffer`

Purpose: RAII device allocation (cudaMalloc) with synchronous and stream-ordered copy / memset helpers;
page-locked host buffer for asynchronous D2H copies (results, compaction counts).
Ownership: exclusive, move-only. Allocation is synchronous and happens only at construction of the owner, sized by
capacities -- never in `check()` / `prepare()`.
Threading: not internally synchronized; CUDA stream ordering applies.
Errors: `CudaError` (allocation failure names the size), `InvalidArgument` (out-of-range offsets, checked first).

### 3.5 `context.h` -- `Context`, `ContextOptions`, `StreamLease`

Purpose: owns the OptiX device context on one CUDA device (primary context retained), the stream pool and the
Logger; replaces `createContextStream<NSTREAM>()` of `RTCD/Utils/Test/cudaCtxUtils.h`. `ContextOptions` takes the
old `OptixCompileConfig::contextValidation / contextLogLevel` (validation on in Debug by `buildDefault()`).
Ownership: always `shared_ptr<Context>`; Scene, checker and backends hold it. Streams are leased per checker
(`acquireStreams(n)`), so checkers sharing a Context never share streams. Streams are created with default flags
(as the old stack).
Threading: all methods thread-safe. Errors: `CudaError` / `OptixError` from `create()`.

### 3.6 `geometry.h` -- `Transform`, `TriangleMesh`, `Sphere`

Purpose: host geometry values. `TriangleMesh` documents exactly what the loader must reproduce from the old
`meshModel::loadOBJ` (vertex dedup order, triangle order, unique-edge order and direction, loop edges), because the
ray sets and OBBs feed the goldens. `Transform::fromXyzRpy` uses the URDF convention `Rz(y) Ry(p) Rx(r)`.
Free functions of `rtcd_geometry` (added in Stage 0-C as the seam to `rtcd_scene` and `rtcd_robot`):
`loadObjMesh(file, pose = identity, scale = 1)` (dedup on the positions as read, then `pose * (scale * v)` with the
old float results; with identity pose and scale 1 the vertices are returned exactly as read, with no arithmetic,
which preserves the -0.0f coordinates of the Panda link0 / link5 / link6 meshes that the goldens depend on; any
other pose / scale applies `vertex *= scale`, then the affine transform, in float, in that order),
`computeEdges(TriangleMesh&)` (unique + loop edges + watertight from the triangles; used again
by `rtcd_robot` after merging fixed links), `fitObb(span<const Vec3>)` (= `computeBestFittingOBB`),
`loadSpheres(file, format = "xyzr")` (the "xyzr" rules of 3.9).
Ownership: values; functions are reentrant, host only. Errors: `IoError`, `InvalidArgument`, `UnsupportedError`
(non-"xyzr" sphere format), as documented per function.

### 3.7 `config.h` -- `CheckerConfig`

Purpose: all checker options as plain data, each field documented with default, kind ([runtime], [capacity],
[fixed]) and JSON key. Nested: `OptixOptions` (old `OptixCompileConfig` compile/link part with
`release()/debug()/profile()/buildDefault()`), `MeshOptions` (link GAS build, `compactionMinPoses = 128`,
`parkTransform` = old safeTransform), `RayOptions` (`launchWidthCap = 262144`, `skipHitItems = true`, `hitMode`),
`CurveOptions` (trajectory / control points, end points, GAS build, `RefitPolicy`, `autoRefitMaxBatches`
(-1 = 2 for linear, 1 otherwise), `geometryFlags` (`std::optional`, nullopt = "auto" = single any-hit for
quadratic, none for linear and cubic, as the old `curveDefaultGeometryFlags` since `4ee064e`), spline weights dir). Top level: mode, `numStreams = 4`,
`hostDispatch = ThreadPerStream` (one worker thread per stream; since 2026-11-24 a check with a single batch is
issued on the calling thread, which saves the worker wake-up and join, 6-18 us per check on Windows), `maxBatchSize = 1024`, `maxItemsPerCheck = 8192`, `obbCulling = Auto`
(On for Mesh / Ray, Off for curves), `collapseRootLinkObb = true` (old `SKIP_BASE`), `sceneEdges = auto`
(Unique for Mesh, Loop for curves), verbosity, `graphPrepare = true` (added 2026-11-24: replay the capturable part of
a batch -- zeroing, staging copy, the backend's prepare -- as one CUDA graph launch per batch, only for batches whose
backend reports `prepareCapturable`; never changes results; no validate / resolved rule).

`validate()` rules (config-only; all violations reported at once):
- `numStreams >= 1`; `maxBatchSize >= 1`; `maxItemsPerCheck >= 1`.
- `optix.maxTraceDepth >= 1`.
- `mesh.compactionMinPoses` any value (0 = always compact, > maxBatchSize = never).
- curves: `2 <= trajectoryPoints <= kMaxTrajectoryPoints`; for degree 2/3: `degree < controlPoints < trajectoryPoints`
  and `controlPoints * trajectoryPoints <= kMaxSplineWeights`; `splineWeightsDir` non-empty for degree 2/3;
  `refit.kind == Every` needs `every >= 2`; `obbCulling == On` with `CubicCurve` is rejected (unsupported, as in
  the old stack); `curve.geometryFlags` only nullopt (auto), None or RequireSingleAnyHitCall.
- `sceneEdges` is ignored by Ray (no error).
`resolved()` replaces Auto values; `curve.geometryFlags` always holds a value afterwards (RequireSingleAnyHitCall
for `QuadraticCurve`, None for `LinearCurve`, `CubicCurve` and the non-curve modes; an explicit value is kept);
with OBB culling on, the effective curve refit policy is Never and
`ALLOW_UPDATE` is not set (old `curveUseRefit = !useOBB`).
Refit policy and results (Wave-2 close-out, measured): the policy is mostly a build vs. trace cost trade-off, but a
refitted curve GAS can flip grazing hits. On the quadratic regression inputs at N = 4096, `{"every": 3}` reports two
extra collisions (items 4065 and 3528), in the old and the new stack alike; `auto`, `always` and `never` showed no
flip in 324 compared checks. The goldens are recorded with `auto`. Construction-time checks against robot / scene / files
are in the checker (and backends): Ray `maxBatchSize * raysPerRobot < 2^30`, spline weight file exists and holds
exactly `controlPoints' * trajectoryPoints` floats, curves need `RobotModel::hasSpheres()`, `robot.dof() <= kMaxDof`.
Ownership: value. Threading: value. Errors: `ConfigError`.

### 3.8 `scene.h` -- `SceneDesc`, `Scene`

Purpose: `SceneDesc` = obstacles (name, OBJ path, pose, scale; world = pose * (scale * v), old
`meshModel(path, transform, scale)`). `Scene` = GPU scene: one compacted triangle GAS per obstacle
(PREFER_FAST_TRACE | ALLOW_COMPACTION, geometry DISABLE_ANYHIT), one IAS (flags NONE, instance o -> SBT record o),
both edge-ray sets (Loop only if every obstacle is watertight), obstacle OBBs on the device, host meshes.
Ownership: immutable after construction; shared read-only by checkers of the same Context; keeps its Context.
Threading: const methods thread-safe, no CUDA calls. Errors: `IoError`, `InvalidArgument`, `CudaError`,
`OptixError`; `edgeRays(Loop)` throws `UnsupportedError` naming the first non-watertight obstacle.

### 3.9 `robot_model.h` -- `RobotDesc`, `RobotModel`

Purpose: runtime robot from URDF. `RobotDesc` = URDF path, mesh source (dir, format "obj", per-link file
overrides, visual vs collision element), optional sphere source (dir, format "xyzr", overrides), base pose.
`RobotModel` (host only, immutable, `shared_ptr<const RobotModel>`): name, DOF, joint names, limits,
`KinematicsTable`, per-link `LinkGeometry` (mesh, spheres, OBB, files), flat sphere arrays (old
`sphereCentersTemplate / sphereRadiusTemplate / sphereIdxToLinkIdx` order), host reference FK.

Library `rtcd_robot` (Wave 1 d); link meshes, spheres and OBBs through the `rtcd_geometry` functions of 3.6.
Loading rules (full text in the header): revolute + continuous actuated, fixed folded / merged, prismatic /
floating / planar rejected; serial chain after merging; table derived as `T_j(q) = A_j Rot(k_j, q)` split into
cos / sin / one masks in double, entries `|v| < 1e-9` snapped to 0, rounded to float -- must equal
`RTCD/Robot/models/pandas.h` bit for bit for Panda. Mesh file mapping: override, else `mesh.dir / basename(URDF
mesh filename)` (old rule; `link7_nohand.obj` comes from the URDF visual element). Sphere file mapping: override,
else `spheres.dir / <linkName>.txt`. Sphere format "xyzr": one `x y z r` per line, `#` comments and blank lines
skipped, anything else is an `IoError` with line number (the current Panda files match:
`models/franka_description/spheres/panda_link0.txt` ... `panda_link7.txt`, 62 spheres in total).
Merged fixed links are appended in "URDF order", which is the order of urdfdom's child-joint container, i.e.
sorted by joint name, not file order.
Link OBB: old `computeBestFittingOBB`, not collapsed (collapse is a checker option applied by backends).
Ownership: host only, no device memory: each backend uploads what it needs. Threading: const thread-safe.
Errors: `IoError`, `UnsupportedError`, `InvalidArgument`.

### 3.10 `backend.h` -- `ProgramSet`, `StreamContext`, `BatchView`, `LaunchPlan`, `Backend`

Purpose: the seam between the mode-agnostic detector core and the backends.

- `ProgramSet` (constant after backend construction): embedded OptiX-IR module; raygen entries (a plan picks
  one by index; one program group and one raygen record each); optional miss; one hit group (any-hit, closest-hit,
  built-in IS with the exact GAS build flags); traversable graph, primitive types, payload / attribute counts,
  max traversable depth; hit-group record count. This is everything `createModule / createRaygenPrograms /
  createMissPrograms / createHitgroupPrograms / createPipeline / buildSBT` of the old detector decided with
  `if constexpr (TYPE == ...)`.
- `StreamContext {index, stream}`, `CheckInfo {itemCount, batchCount}`, `BatchView {batchIndex, firstItem,
  itemCount, items, hitCounts}`, `LaunchPlan {launch, traversable, width/height/depth, params, paramsBytes,
  raygen}`.
- Result location: `BatchView::hitCounts` (zeroed by the core) is the only result location; programs `atomicAdd`
  per batch-local item. It is either the stream's own counter buffer (the core then copies it into the check
  result) or, since 2026-11-24, the check's result range at `firstItem` itself: the core counts there directly when
  the stream's previous batch had the same target address (repeated checks into one result buffer), so params
  holding the pointer stay byte-identical; host checks (results in a pinned host buffer) never count directly.
- Items: `BatchView::items` is either the stream's staging buffer (graph-replayed batches and strided input) or,
  since 2026-11-24, the check's packed input itself at the batch's first item (no staging copy); for host checks
  that input is a pinned host buffer the GPU reads through its mapping. Both addresses may change from batch to
  batch: backends read them from every `BatchView` and never cache them (none of the three does).
- `Backend`: `mode()`, `programs()`, `beginCheck()` (host only, per check), `prepare()` (per batch),
  `synchronizesHost(itemCount)` (exact predicate: true iff that prepare blocks the host),
  `prepareCapturable(itemCount)` (added 2026-11-24; virtual with default false, so a backend that does not
  override it is never captured): true iff that prepare (a) enqueues only plain CUDA work on its stream (no OptiX
  API call of any kind, no host synchronization, no device allocation) and (b) returns a plan and params that
  depend only on (stream index, itemCount, `batch.hitCounts.ptr`; the core keys its graphs by all three), so replaying the captured work equals calling prepare again on new
  contents of the same buffers. Capturable implies not synchronizing (the core treats both true as an internal
  error). Hard rule, measured on OptiX 9.1 / driver 610.88 (`tests/experiments/graph_capture`): an OptiX call
  inside a stream capture is silently dropped from the graph (`optixLaunch`: only a 64-byte param copy recorded,
  the replay traces nothing; `optixAccelBuild`: nothing recorded, or for quadratic curves part of the build, whose
  replay hits an illegal address) or fails with `OPTIX_ERROR_CUDA_ERROR`, and it leaves the OptiX device context
  with an invalid internal event (later launches fail with `OPTIX_ERROR_INVALID_VALUE`). Ray: true for every item
  count it accepts (any hit mode, OBB culling on or off); mesh and curves: false (per-batch AS build).
- `prepare` contract: GPU work only on its stream; no allocation; no device sync; host block only when
  `synchronizesHost` says so; per-stream state only, so the core may call it concurrently for different stream
  indices; params valid until the next prepare of that stream index.
- Current device: the core makes the Context's device current on every thread that calls `beginCheck` /
  `prepare` (the calling thread, and each worker thread with `ThreadPerStream`, section 4 steps 2 and 6), so
  backends do not call `makeCurrent()` there (they do in their constructors, which the core also calls with the
  device current).
- Loop order per batch, as in section 4 step 7 and the `backend.h` header comment: zero hit counts, staging copy
  (unless the items are read in place), `prepare`, params upload + launch, result copy (unless counted directly).
- Factories `createMeshBackend / createRayBackend / createCurveBackend(BackendCreateInfo)`; the core switches on the
  mode once, at construction.
Ownership: the checker owns the backend (`unique_ptr`) and destroys it before Scene / RobotModel / Context
references it holds (`BackendCreateInfo` passes shared_ptrs; the config pointer is the checker's resolved copy).
Errors: construction throws for robot / scene / file dependent problems; `prepare` throws `rtcd::Error`, the core
drains all streams and rethrows on the calling thread.

### 3.11 `checker.h` -- `CollisionChecker`

Purpose: facade. Constructor `(Context, CheckerConfig, RobotModel, Scene)`. `check(HostItems, ResultDetail)` ->
`CheckResult {collides, hitCounts}`; `check(HostItems, span flags, span counts)` (no allocation);
`checkDevice(DeviceItems, DeviceSpan<uint32_t> hitCounts)` (device in / out, the benchmark path matching the old
timed region); `warmup()`; `lastStats()` (batches, launches, host syncs, param uploads); accessors (resolved config,
item layout, robot, scene).
Ownership: owns backend, pipeline, SBT, staging / result buffers, stream lease, worker threads; shares Context /
RobotModel / Scene. Move-only.
Threading: one call at a time per checker (not thread-safe); different checkers concurrently are fine.
Errors: `ConfigError / IoError / CapacityError / UnsupportedError / CudaError / OptixError` at construction;
`InvalidArgument / CapacityError` before GPU work in `check`; CUDA / OptiX errors after draining.
Result semantics: `collides[i] = hitCounts[i] > 0`; hit counts are diagnostic only.

### 3.12 `json_config.h` -- JSON module

Purpose: `loadCheckerConfig / parseCheckerConfig`, `loadSceneDesc / parseSceneDesc`,
`loadRobotDesc / parseRobotDesc`, `loadSetup` (checker + scene + robot, each inline or a path),
`applyOverrides("path=value,...")` (replaces `OptixCompileConfig::setFields` and the env parsing of
`RTCD/Benchmark/optixConfigEnv.h`; apps read env vars, the library never does), `dump*` (effective config, all
fields, absolute paths, round-trips).
Rules: defaults come only from the C++ default member initializers; unknown keys, wrong types and bad enum strings
are errors; relative paths are resolved against the directory of the file that contains them; error messages
"<file>: <json.path>: <message>"; JSON syntax errors as `IoError(path, line, column)`.
Ownership: functions, values. Threading: reentrant. Errors: `IoError`, `ConfigError`.

### 3.13 Internal contracts (`include/rtcd/internal/`)

- `algorithms.h` (core, 1a): `obbOverlapCounts` (= old `BBCDRobotMask`, bit-identical SAT), `exclusiveScanU32`,
  `flagsToBinaryU32` (= old `transformIsValid<uint,true>`), `compactByFlags` (stable multi-array compaction with a
  device count, = old `cub::DeviceSelect::FlaggedIf` use), scratch-size functions. No allocation, no sync.
- `accel.h` (core, 1a): construction-time `buildTriangleGas` (with compaction, = old `meshModel::buildGAS`) and
  `buildInstanceAs`; synchronous.
- `kinematics.h` (robot, 1d): `uploadKinematics`, `linkPoses` (= `batchFkineAdj`), `linkPosesAndObbs`
  (= fused `batchFkineAdjOBB`), `transformObbs` (= `batchOBBTransform`), `trajectoryLinkPoses`
  (= `batchFkineTraj`); bit-identical outputs; take `KinematicsRef {host table, device table}`.
- `reference_executor.h` (test support, 1e): `ReferenceExecutor` (pipeline + SBT from a `ProgramSet`, launches a
  `LaunchPlan`) and `ReferenceBatchRunner` (single-threaded batch loop following `backend.h`), so the Wave 2 backend
  agents can test without the detector core and the core can be cross-checked. Threading (Wave-2 close-out):
  `launch()` may be called concurrently for different streams, first launches included (the per-stream param
  blocks are kept in a `std::deque`, so creating one never moves another; before, a `std::vector` could reallocate
  under a concurrent first launch, found in Wave 2 h); launches on one stream come from one thread at a time.
  Tested by `ConcurrentFirstLaunchesOnDifferentStreams` (`tests/testsupport`).

## 4. The call sequence of one `check()`

`CollisionChecker::check(items)` with `n` items, batch size `B`, `S` streams (old code in brackets; file
`RTCD/CollisionDetector/CollisionDetector.h` unless named):

1. Validate the view: `n <= maxItemsPerCheck` (`CapacityError`), span size and stride (`InvalidArgument`);
   `n == 0` returns empty results. [new; old silently grew buffers]
2. Make the Context's device current. [new]
3. Host input only: one synchronous copy of the `n` items into the checker's input buffer (strided input with
   `cudaMemcpy2D`). [`CCWrappedRobot::prepareUpdateData` / `CCRayRobot::prepareUpdateData`, `CCRobotWrapper.h`]
4. Batches: `nb = ceil(n / B)`; batch `b` = items `[bB, min(n, (b+1)B))`, stream `b % S`. [`setNCheck`,
   `batchSizeInfo` tuples (frameId, size, start)]
5. `backend.beginCheck({n, nb})` on the calling thread. [`setNCheck` -> `setBatchCountHint` -> curve
   `applyRefitPolicy`]
6. Dispatch: `SingleThread`: the calling thread runs step 7 for all batches in order; `ThreadPerStream`: worker
   `s` first makes the Context's device current, then runs step 7 for batches `s, s+S, s+2S, ...` in order (so
   every thread that calls `prepare` has the device current, as step 2 does for the calling thread and
   `beginCheck`); no cross-stream barrier is needed because all mutable
   state is per stream. [`check()`: loop, or `ThreadPool<NSTREAM>` with a barrier after every round of NSTREAM
   batches, `threadPool.h`, `USE_MULTI_THREADING`]
7. Per batch `b` on stream `s`:
   1. `cudaMemsetAsync` of `hitCounts[0, itemCount)` of stream `s`. [`detectWithOBB` / `detect`: memset
      `hitBuffer`]
   2. D2D copy of the batch's items into stream `s`'s staging buffer (`BatchView::items`). [`movePosesToStream`
      -> `copyBatchToStream`, `CCRobotWrapper.h`]
   With `graphPrepare` and `backend.prepareCapturable(itemCount)` (and packed items), steps 1 - 3 are one
   `cudaGraphLaunch` instead: the core captures them once per (stream, itemCount) (`cudaStreamCaptureModeThreadLocal`,
   on the thread that runs the batch), keeps the plan and params `prepare` returned at capture, re-points the
   staging-copy node when the source address changes, and keeps at most 8 graphs per stream (oldest evicted; a miss
   costs ~30 us). `prepare` is then not called for that batch. Steps 4 and 5 stay direct calls.
   3. `plan = backend.prepare({s, stream}, batch)` (mode-specific, section 5). [`fkine` / `update`, `detectOBB`
      (`BBCDRobotMask`), `updateWithMask`, `getHandle`, `getRayInfo`, `getMapIndex` -- spread over
      `detect`, `detectWithOBB`, `detectOBB` and the robot classes]
   4. If `plan.launch`: copy `plan.params` into stream `s`'s parameter buffer unless byte-identical to its last
      upload; `optixLaunch(pipeline, stream, paramsBuf, plan.paramsBytes, sbt(raygen = plan.raygen), w, h, d)`.
      [`uploadLaunchParams` (memcmp cache), `optixLaunch` in `detect` / `detectWithOBB`]
   5. D2D copy `hitCounts[0, itemCount)` -> result buffer at `firstItem`. [`check()`: `cudaMemcpyAsync` into
      `resultBuffer + start`]
8. Join workers (rethrow the first exception after draining), then synchronize the checker's `S` streams.
   [`CUDA_SYNC_CHECK()` = `cudaDeviceSynchronize`]
9. Host results: download `n` counters (pinned buffer), `collides[i] = count > 0`. [`downloadResult`;
   regression computes `raw[i] > 0`]

`checkDevice` skips 3 and 9 (step 7.2 copies from the caller's buffer, 7.5 writes the caller's buffer).

## 5. What each backend does in `prepare()`

All three apply `collapseRootLinkObb` to their link-OBB template at construction (old `SKIP_BASE`:
`linkOBBs[0].halfSize = 0` in `robot<DOF>::loadURDF`), use `obbOverlapCounts` against `Scene::obbs()` for culling,
and write `EdgeRayParams` / `RobotRayParams`.

**Mesh (f)** [old `batchIASRobot.h`, `iasRobotKernels.cu`, programs `__raygen__oneside` + `__anyhit__IAS` of
`CCCuda.cu`]. Construction: per link a triangle GAS (`buildTriangleGas`, flags `toBuildFlags(linkGasBuild)`,
geometry REQUIRE_SINGLE_ANYHIT_CALL); per stream: `maxBatchSize * numLinks` instances in template order
(`pose * numLinks + link`, sbtOffset = that index, mask 255), compact instance buffer, instance -> pose template
map and compacted map, link poses, link OBBs, OBB counts, compaction scratch, pinned count, IAS buffer (initial
build over all instances, options FAST_BUILD | ALLOW_UPDATE, 1.5x size). Programs: one raygen over scene edges,
any-hit adds to `hitCounts[primToItem[optixGetInstanceIndex()]]`, no CH / miss, `SingleLevelInstancing`, depth 2,
`maxBatchSize * numLinks` hit records, 0 payloads.
prepare, OBB on: `linkPosesAndObbs`; `obbOverlapCounts`; if `itemCount >= compactionMinPoses`: write instance
transforms, `compactByFlags` (instances + map), D2H count, stream sync (`synchronizesHost` = true), launch = false if
0 survive, IAS BUILD over survivors, `primToItem` = compacted map; else: write transforms with visibility mask 0
for culled instances (no host round trip), IAS BUILD over `itemCount * numLinks`, template map. OBB off:
`linkPosesAndObbs` (the old `update()` also used the fused kernel), write transforms (park transform for unused
slots), IAS UPDATE (refit) over all `maxBatchSize * numLinks`. Params: traversable = IAS, edges =
`scene.edgeRays(sceneEdges)`, width = edge count.

**Ray (g)** [old `batchRayRobot.h`, `CCRay.cu`]. Construction: robot rays = every link's `uniqueEdges` in link
order (origin, target), ray -> link map, link-OBB template; per stream: link poses, OBBs, link mask; an all-ones
mask if OBB culling is off; check `maxBatchSize * raysPerRobot < 2^30`. Programs: the raygen of `ray.hitMode`
(grid-stride over ids, skip by link mask and, if `skipHitItems`, by a volatile hit-count read), CH for
ClosestHit / FirstHitCH, miss for FirstHitMiss, 1 payload, `SingleLevelInstancing`, depth 2, `numObstacles` hit
records. prepare: link poses + OBBs (old: `batchFkineAdj` + `batchOBBTransform`; the fused kernel gives identical
values), `obbOverlapCounts` into the mask, params (`totalRayCount = itemCount * raysPerRobot`,
`width = min(total, cap)` or total if cap 0, traversable = `scene.traversable()`). Never syncs.

**Curves (h)** [old `batchCurveRobot.h`, `sweptVolume.cu/.inl`, fused `fkineSphrTrajKern` in `robot.cu`, programs
`__raygen__oneside` + `__anyhit__GAS` of `CCCuda.cu`]. Construction: spline weights from
`splineWeightsDir/<quad|cubic>/<normal|repeated|phantom>/<nCtrl>_<nTraj>.txt` (one float per line, row-major
`[controlPoint][trajectoryPoint]`) into a per-checker device buffer (no `__constant__`); one curve per (item,
sphere) with radius = sphere radius; segments per curve: linear `trajectoryPoints - 1`, else
`controlPointsPerCurve - degree` (`controlPointsPerCurve` = controlPoints, +2 for Phantom); index buffer, segment
-> item map, radii; per stream: control points, GAS + temp (sized for the full batch, temp >= max(build, update)),
refit state. Build flags `toBuildFlags(gasBuild) | (refit possible ? ALLOW_UPDATE : 0)`, geometry flags from config,
end caps on; ProgramSet: built-in IS RoundLinear / RoundQuadratic / RoundCubic with exactly these flags,
`SingleGas`, depth 1, 1 hit record, primitive type of the degree. prepare: fused fkine + sphere centers + control
points when supported (else `trajectoryLinkPoses` + centers + control points), GAS BUILD or UPDATE per the refit
policy (refit only if the item count equals the last full build's), params with `primToItem` = segment -> item
map, edges = loop edges. OBB on (linear / quadratic only): segment OBBs, `obbOverlapCounts`, compaction with host
sync, see open question Q5.

## 6. Old-stack concept -> new-stack home

| old (file)                                                                 | new home |
|----------------------------------------------------------------------------|----------|
| template `NSTREAM` (detector, robots, scenes, wrappers)                    | `CheckerConfig::numStreams` [runtime]; streams leased from `Context` |
| `BATCHSIZE` (`batchRobotConfig`, `batchSceneConfig`)                       | `CheckerConfig::maxBatchSize` [capacity] |
| `Dim` / `DOF` (`pandas.h`, `robot<DOF>`, `robotConfig<DOF>`)               | `RobotModel::dof()` from the URDF; capacity `kMaxDof` |
| `LinkType` MESH / RAY_ONLINE / LINEAR / QUADRATIC / CUBIC_CURVE           | `Mode` |
| `LinkType` SPHERES, SPHERE_GAS, RAY, RAY_STATIC                            | dropped (Q9) |
| `BuildType` COMPACT / FAST_TRACE / FAST_BUILD / NONE                       | `BuildPreference` in `MeshOptions::linkGasBuild`, `CurveOptions::gasBuild` (Ray needs none) |
| `batchRobotConfig::nTrajPts`, `nCtrlPts`                                   | `CurveOptions::trajectoryPoints`, `controlPoints` [fixed] |
| `batchRobotConfig::useOBB`, `CCWrappedRobot::useOBB`, `baseBatchRobot::useOBB` | `CheckerConfig::obbCulling` (one field) |
| `batchRobotConfig::endPointType` (`curveEndPointType`)                     | `CurveOptions::endPoints` |
| `batchRobotConfig::curveDegree`                                            | `curveDegree(Mode)` |
| `batchSceneConfig::compactable` (scene-side compaction, unused)            | dropped (dead path) |
| `streamUniqueRobot::compactMinPoses`, `setCompactMinPoses`                 | `MeshOptions::compactionMinPoses` |
| `safeTransform` (`robotConfig`, `__constant__` in `robot.cu`)              | `MeshOptions::parkTransform` |
| `batchRayRobot::setLaunchWidthCap`, `setSkipHitPoses`                      | `RayOptions::launchWidthCap`, `skipHitItems` |
| `OptixCompileConfig::rayHitMode`                                           | `RayOptions::hitMode` |
| `OptixCompileConfig` opt / debug / maxreg / exceptions / tracedepth / pipelinelog | `OptixOptions` |
| `OptixCompileConfig::contextValidation`, `contextLogLevel`                 | `ContextOptions::optixValidation`, `optixLogLevel` |
| `OptixCompileConfig::setField(s)`, env `RTCD_OPTIX_CFG`                    | `applyOverrides` (json_config.h); apps read env vars |
| env `RTCD_CURVE_GEOM_FLAG`, `batchOptixCurveRobot::setGeometryFlags`       | `CurveOptions::geometryFlags` (override `curve.geometry_flags=none\|single_anyhit`) |
| `curveDefaultGeometryFlags` (`batchCurveRobot.h`)                           | `resolved()` of `curve.geometry_flags: "auto"` (default) |
| env `RTCD_BM_REFIT_EVERY`, `setRefitPolicy`, `REFIT_AUTO`, `refitMaxBatches` | `CurveOptions::refit`, `autoRefitMaxBatches` |
| `setBatchCountHint`                                                        | `Backend::beginCheck(CheckInfo)` |
| `curveUseRefit<CFG> = !useOBB`                                             | `resolved()`: OBB on -> refit Never, no ALLOW_UPDATE |
| `getBuildFlags`, `rebuildOptixIfFlagsChanged`, `moduleBuildFlags`          | `ProgramSet::builtinIsBuildFlags`, fixed at backend construction (no rebuild path needed) |
| `setCompileConfig` re-creating pipelines                                   | config is construction-time: new checker |
| `__constant__ cMask / sMask / oMask` (`robot.cu`), `setFkineMasks`        | `KinematicsTable` cos / sin / one masks, device buffer per robot owner |
| `__constant__ baseAffine`, `setBaseAffine`, `robotConfig::baseTransform`   | `KinematicsTable::base` = `RobotDesc::basePose` * root fixed chain |
| `__constant__ m1Indices / m2Indices`                                       | private to the kinematics `.cu` (constant index tables, not robot data) |
| `__constant__ splineWeights`, `curveSplineWeights`, `fillWeight`, `setCurveSplineWeights` | curve backend device buffer; capacity `kMaxSplineWeights` |
| `loadCurveWeight` path `PROJECT_BASE_DIR/RTCD/data/weights`                 | `CurveOptions::splineWeightsDir` (resolved from JSON) |
| `pandas.h` masks, `Config`, paths                                          | URDF + `RobotDesc` (JSON robot block) |
| `pandas.h` `lowerBound / upperBound`                                       | `RobotModel` limits from URDF; old bounds kept by the harness for input generation (Q7) |
| `SKIP_BASE` define (`robot.h`)                                             | `CheckerConfig::collapseRootLinkObb` |
| `robot<DOF>`, `meshModel`, `robotConfig`                                   | `RobotModel`, `LinkGeometry`, `TriangleMesh`, `RobotDesc` |
| `scene`, `obstacle`, `cube`, `batchScene`, `baseBatchScene`                | `Scene`, `SceneDesc`, `ObstacleDesc` |
| `Scene/shelf.h`, `denseShelf.h`, `shelfSimple.h`, `sceneBuilder.h`, defines `SHELF / DENSESHELF / SHELFSIMPLE` | `configs/scenes/shelf.json`, `shelf_dense.json`, `shelf_simple.json` |
| `Scene/debugBox.h` (`models/debugBox.obj`)                                  | not ported (no `configs/scenes/debug_box.json`; used by no test, benchmark or golden) |
| `buildSharedScene(loop)`                                                   | `CheckerConfig::sceneEdges` (+ `Scene` holds both sets) |
| `CUDABuffer`, `CUDAPitchBuffer`                                            | `DeviceBuffer`, `PinnedHostBuffer` (pitch buffer unused, dropped) |
| `CUDA_CHECK` (no-op in Release), `CUDA_CHECK_LAST`, `CUDA_SYNC_CHECK` (exit), `OPTIX_CHECK` (exit) | `RTCD_CUDA_CHECK`, `RTCD_CU_CHECK`, `RTCD_OPTIX_CHECK(_LOG)` (throw, all builds) |
| `createContextStream<NSTREAM>`, `defaultInitOptix`, `context_log_cb`       | `Context::create`, `acquireStreams`, Logger |
| `USE_MULTI_THREADING`, `ThreadPool<NSTREAM>`                                | `CheckerConfig::hostDispatch` |
| `CollisionDetector::setNCheck` + `check`                                   | `CollisionChecker::check` |
| `detect`, `detectWithOBB`, `detectOBB`, `BBCD`, `BBCDRobotMask`            | backend `prepare`; `internal::obbOverlapCounts` |
| `createModule / createRaygenPrograms / createMissPrograms / createHitgroupPrograms / createPipeline / buildSBT` | detector core from `ProgramSet` |
| `LaunchParams` union, `rayType`, `rayRobotCfg`, `rayMeshCfg`, `rayLinkCfg`  | `EdgeRayParams`, `RobotRayParams` (limits.h) |
| `meshRayInfo`, `rayRobotInfo`                                              | `LaunchPlan` + params |
| `RaygenRecord / MissRecord / HitgroupRecord`                               | `SbtRecord` |
| `optixLaunchParams` variable name                                          | `rtcdLaunchParams` (`kLaunchParamsName`) |
| `getMapIndex`, `primIdxToPoseIdx`                                          | `EdgeRayParams::primToItem` (backend-owned) |
| `hitBuffer` per stream, `resultBuffer`, `downloadResult`, `clearResult`    | `BatchView::hitCounts`, checker result buffer, `CheckResult` |
| `uploadLaunchParams` memcmp cache                                          | detector core, step 7.4 |
| `warmup(nPoses)`, `prepareRandomData`                                      | `CollisionChecker::warmup()` |
| `resetGraph`, graph members (never captured)                               | dropped; graph capture of steps 7.1 - 7.3 via `CheckerConfig::graphPrepare` + `Backend::prepareCapturable` (2026-11-24) |
| `resizeHitBuffer`, `getHitBuffer`, `updateASHandle`, `RAYDATAMODE`, `__anyhit__RayData`, `PEROBS` | dropped (Q9) |
| `PRINTIT`                                                                  | `OptixOptions::printPipelineLog`, `ContextOptions::optixLogLevel`, Logger level |
| `_DEBUG / DEBUG` (`buildDefault`)                                          | `OptixOptions::buildDefault()`, `ContextOptions::buildDefault()` (inline, per including TU) |
| `PROJECT_BASE_DIR`, `CONCAT_PATHS` (`config.h.in`)                         | JSON paths resolved relative to the JSON file; nothing baked in |
| `REG_IAS / REG_RAY / REG_LINEAR / REG_QUAD`, `REG_BATCHSIZE`, `REG_SINGLE_THREAD` | regression app arguments / JSON setups (harness, 1e) |

## 7. JSON schemas and examples

Conventions: keys snake_case; unknown keys are errors; every key optional unless marked required; defaults as in
the C++ structs; enums as strings; relative paths are relative to the file that contains them.

### 7.1 Checker config

| key | type | default |
|-----|------|---------|
| `mode` | `"mesh" \| "ray" \| "linear_curve" \| "quadratic_curve" \| "cubic_curve"` | `"mesh"` |
| `streams` | int >= 1 | 4 |
| `host_dispatch` | `"single_thread" \| "thread_per_stream"` | `"thread_per_stream"` |
| `max_batch_size` | int >= 1 | 1024 |
| `max_items_per_check` | int >= 1 | 8192 |
| `obb_culling` | `"auto" \| true \| false` | `"auto"` |
| `collapse_root_link_obb` | bool | true |
| `scene_edges` | `"auto" \| "unique" \| "loop"` | `"auto"` |
| `verbosity` | `"error" \| "warning" \| "info" \| "debug"` | `"warning"` |
| `graph_prepare` | bool | true |
| `optix.opt_level` | `"default" \| 0 \| 1 \| 2 \| 3` | `"default"` (Debug build of the app: 0) |
| `optix.debug_level` | `"default" \| "none" \| "minimal" \| "moderate" \| "full"` | `"default"` (Debug: `"full"`) |
| `optix.max_register_count` | int >= 0 | 50 |
| `optix.exceptions` | list of `"stack_overflow"`, `"trace_depth"`, `"user"` (or `["none"]`) | `[]` (Debug: trace_depth, stack_overflow) |
| `optix.max_trace_depth` | int >= 1 | 2 |
| `optix.print_pipeline_log` | bool | false |
| `mesh.link_gas_build` | `"fast_build" \| "fast_trace" \| "fast_trace_compact"` | `"fast_trace_compact"` |
| `mesh.compaction_min_poses` | int >= 0 | 128 |
| `mesh.park_transform` | 12 numbers (3x4 row-major) | scale 0.001, z -0.1 |
| `ray.launch_width_cap` | int >= 0 | 262144 |
| `ray.skip_hit_items` | bool | true |
| `ray.hit_mode` | `"closest_hit" \| "first_hit_ch" \| "first_hit_miss" \| "traverse"` | `"closest_hit"` |
| `curve.trajectory_points` | int | 16 |
| `curve.control_points` | int | 8 |
| `curve.end_points` | `"normal" \| "repeated" \| "phantom"` | `"normal"` |
| `curve.gas_build` | as `mesh.link_gas_build` | `"fast_build"` |
| `curve.refit` | `"auto" \| "always" \| "never" \| {"every": K}` | `"auto"` |
| `curve.auto_refit_max_batches` | `"auto" \| int >= 0` | `"auto"` (linear 2, else 1) |
| `curve.geometry_flags` | `"auto" \| "none" \| "single_anyhit"` | `"auto"` (quadratic: single_anyhit; linear, cubic: none) |
| `curve.spline_weights_dir` | path | none (required for quadratic / cubic) |

Example (`configs/checker_quad.json`, the regression's QUAD setup):

```json
{
  "mode": "quadratic_curve",
  "streams": 4,
  "host_dispatch": "thread_per_stream",
  "max_batch_size": 128,
  "max_items_per_check": 4096,
  "obb_culling": "auto",
  "collapse_root_link_obb": true,
  "scene_edges": "auto",
  "verbosity": "warning",
  "optix": {
    "opt_level": "default",
    "debug_level": "default",
    "max_register_count": 50,
    "exceptions": [],
    "max_trace_depth": 2,
    "print_pipeline_log": false
  },
  "curve": {
    "trajectory_points": 16,
    "control_points": 8,
    "end_points": "normal",
    "gas_build": "fast_build",
    "refit": "auto",
    "auto_refit_max_batches": "auto",
    "geometry_flags": "auto",
    "spline_weights_dir": "spline_weights"
  }
}
```

Here `"geometry_flags": "auto"` resolves to `single_anyhit` (quadratic), the old regression's default since
`4ee064e`; the curve backend's gate (section 8, h) also runs `"none"` against the same golden.
Other regression setups: IAS = `{"mode": "mesh", "max_batch_size": 128, "max_items_per_check": 4096}` (link GAS
`fast_trace_compact`, OBB on); RAY = `{"mode": "ray", "max_batch_size": 128, "max_items_per_check": 4096}`; LINEAR =
`{"mode": "linear_curve", "max_batch_size": 128, "max_items_per_check": 4096, "curve": {"trajectory_points": 16}}`.

### 7.2 Scene description

```
{ "name": string,
  "obstacles": [ { "name": string (default: mesh file stem),
                   "mesh": path (required, OBJ),
                   "scale": number > 0 (default 1),
                   "pose": { "translation": [x, y, z] (default 0),
                             exactly one of (default identity rotation):
                               "rpy": [roll, pitch, yaw]            (radians, URDF convention)
                               "quaternion": [x, y, z, w]
                               "matrix": [12 numbers, 3x4 row-major] (then "translation" is not allowed) } } ] }
```

Example (`configs/scenes/shelf.json`, the old `Scene/shelf.h` "medium shelf" used by the goldens):

```json
{
  "name": "shelf",
  "obstacles": [
    { "name": "shelf", "mesh": "../../models/curobo/shelves.obj", "scale": 1.0, "pose": { "translation": [0.45, 0.0, 0.4] } },
    { "name": "bin1", "mesh": "../../models/curobo/bin.obj", "pose": { "translation": [0.0, -0.6, 0.0] } },
    { "name": "bin2", "mesh": "../../models/curobo/bin.obj", "pose": { "translation": [0.0, 0.6, 0.0] } },
    { "name": "bob", "mesh": "../../models/curobo/Bob.obj", "pose": { "translation": [-0.4, 0.0, 0.0] } },
    { "name": "cow", "mesh": "../../models/curobo/Cow.obj", "pose": { "translation": [-0.4, 0.0, 0.0] } },
    { "name": "fish", "mesh": "../../models/curobo/Fish.obj", "pose": { "translation": [-0.4, 0.0, 0.0] } },
    { "name": "sheep", "mesh": "../../models/curobo/Sheep.obj", "pose": { "translation": [-0.4, 0.0, 0.0] } },
    { "name": "snakeboard", "mesh": "../../models/curobo/Snakeboard.obj", "pose": { "translation": [-0.4, 0.0, 0.0] } }
  ]
}
```

### 7.3 Robot reference block

```
"robot": { "description": path (required, URDF),
           "geometry": {
             "mesh":    { "dir": path (required), "format": "obj", "source": "visual" | "collision" (default visual),
                          "files": { "<link name>": path, ... } (optional overrides) },
             "spheres": { "dir": path, "format": "xyzr", "files": { ... } } (optional; required by curve modes) },
           "base_pose": pose object as in the scene (optional, default identity) }
```

Example (`configs/robots/panda.json`):

```json
{
  "robot": {
    "description": "../../models/franka_description/panda.urdf",
    "geometry": {
      "mesh": { "dir": "../../models/franka_description/meshes/Simplified/watertight", "format": "obj", "source": "visual" },
      "spheres": { "dir": "../../models/franka_description/spheres", "format": "xyzr" }
    }
  }
}
```

### 7.4 Setup file

```json
{ "checker": "../checker_quad.json", "scene": "../scenes/shelf.json", "robot": "../robots/panda.json" }
```

Each value may also be the object itself (inline). Paths inside a referenced file are resolved against that file.

## 8. Wave assignments

The headers are frozen (change procedure at the top of this document); each agent implements only the
declarations listed, in the directories it owns. Cross-module needs go through the declared contracts only.
An owned directory includes its `CMakeLists.txt` (keep the target name, its PUBLIC links as in section 2, and
`rtcd_register_new_target`; adding PRIVATE dependencies and sources is the owner's call). Main owns everything
else: `include/rtcd/`, `docs/design/`, `cmake/`, the top-level `CMakeLists.txt`, `src/CMakeLists.txt`,
`tests/CMakeLists.txt`, `apps/CMakeLists.txt`, `tests/interfaces_compile.cpp`.

> **Historical (Waves 1 and 2 are merged).** This section records the assignments as they were given. Waves 1
> and 2 are complete (state at `a605b47`): every library has real sources, the placeholder sources and main's
> `tests/skeleton` (with `test_skeleton_unit`) no longer exist, test-local stubs were deleted at the merges, and
> the backends' goldens run through `CollisionChecker`. The placeholder and skeleton rules below applied only
> while Wave 1 was in progress.

**Wave 1 (parallel)**

Five agents, all running at the same time. Rules common to all of them:

- Every cross-agent dependency is a function or type DECLARED in `include/rtcd/` (public or `internal/`). Nobody
  includes another agent's private header (`src/<x>/...`) or relies on a symbol not declared there.
- (Historical, Wave 1 only.) Each agent replaced the `placeholder.cpp` / `placeholder.cu` of its own libraries and
  deleted it with its first real source; main's `tests/skeleton` called the placeholders until main deleted it at
  the Wave-1 merge.
- Before another Wave-1 agent's implementation has landed, a test may link a clearly marked test-local stub or
  fake of a declared function, inside the test's own `tests/<x>/` folder (never under `src/`), deleted at merge.
  Code that needs the real thing is written now and integration-tested after the merge.
- The old stack is the reference: tests may include old-stack headers and link old-stack targets read-only (the
  namespaces `rtcd` and `RTCD` coexist). Nothing under `RTCD/` is edited.
- New test folders are picked up automatically (`rtcd_add_new_test`, labels `new_unit` / `new_gpu`).

- **(1a) core**
  (i) Owns `src/core`, `tests/core`.
  (ii) `errors.h`: `detail::throwCudaError / throwCuError / throwOptixError` (error names, sticky
  classification); `types.h`: `Logger::defaultSink`; `device_buffer.h`: `DeviceBuffer`, `PinnedHostBuffer`;
  `context.h`: `Context`, `ContextOptions`, `StreamLease`; `internal/algorithms.h`: `obbOverlapCounts`,
  `exclusiveScanU32`, `flagsToBinaryU32`, `compactByFlags`, scratch-size functions; `internal/accel.h`:
  `buildTriangleGas`, `buildInstanceAs`. Adopts `tests/core/abi_check.cpp` + `limits_device.cu` (already built
  as `rtcd_core_abi_check`, section 9).
  (iii) Depends on no other agent. Tests: buffers, context, leases, error classification, algorithms vs host
  references (compaction order, scan, OBB counts vs the old `BBCDRobotMask` on random OBBs), GAS / IAS builds.
- **(1b) config + JSON + configs/**
  (i) Owns `src/config`, `src/json`, `tests/config`, `tests/json`, `configs/**`.
  (ii) `config.h`: `CheckerConfig::validate`, `CheckerConfig::resolved` (including the `curve.geometryFlags`
  "auto" rule, 3.7); `json_config.h`: every function. `configs/`: checker per regression mode, scenes `shelf`,
  `shelf_dense`, `shelf_simple` ported from `RTCD/CollisionScenes/Scene/*.h` (`debug_box` was also listed here but
  was not ported, see section 6), `robots/panda.json`, setups, the spline-weights copy (section 11).
  (iii) Needs no other agent: `rtcd_config` has no dependencies and `rtcd_json` only `rtcd_config`; the structs it
  fills (`SceneDesc`, `RobotDesc`, `Transform`) and the exception classes are header-only. Tests: defaults, every
  validation rule, `resolved()` per mode, unknown keys, path resolution, overrides, dump round trip. No GPU.
- **(1c) geometry + scene**
  (i) Owns `src/geometry`, `src/scene`, `tests/geometry`, `tests/scene`.
  (ii) `geometry.h`: `loadObjMesh`, `computeEdges`, `fitObb`, `loadSpheres`; `scene.h`: `Scene` (all members).
  (iii) Host parts need nobody: unique edges, loop edges and OBBs identical to the old loader (`RTCD/Meshes/mesh.h`,
  `Obb.h`) for every Panda link mesh and every shelf obstacle; vertex positions identical to the old
  `meshModel(path, transform, scale)`; sphere-file rules. `Scene` needs 1a (`Context`, `DeviceBuffer`,
  `buildTriangleGas`, `buildInstanceAs`, the throw helpers): write it now, test it (IAS / edge-ray / OBB sanity
  against the old `scene`) after 1a lands or against test-local stubs.
- **(1d) robot model + kinematics**
  (i) Owns `src/robot`, `tests/robot`.
  (ii) `robot_model.h`: `RobotModel::load` and every `RobotModel` member (`linkPoses` included);
  `internal/kinematics.h`: `uploadKinematics`, `linkPoses`, `linkPosesAndObbs`, `transformObbs`,
  `trajectoryLinkPoses`.
  (iii) Assumes from 1c only the four `geometry.h` functions (link meshes, merged-mesh edges, link OBBs, sphere
  files) and from 1a `DeviceBuffer` (returned by `uploadKinematics`) and the throw helpers behind
  `RTCD_CUDA_CHECK`. Tests before they land: the URDF -> chain -> `KinematicsTable` derivation for `panda.urdf`
  equals the `pandas.h` masks bit for bit (keep the derivation callable without geometry, e.g. a private
  function in `src/robot` used by `load`); the FK kernels are bit-identical to the old `batchFkineAdj /
  batchFkineAdjOBB / batchOBBTransform / batchFkineTraj` on the cached poses (test buffers via plain
  `cudaMalloc` + `DeviceSpan`, reference via the old stack). The link-geometry part of `RobotModel::load`
  (meshes, spheres, OBBs, fixed-link merging) is integration-tested after the merge.
- **(1e) test support + apps**
  (i) Owns `tests/support` (library `rtcd_testsupport`), `tests/testsupport` (its tests),
  `apps/**` (new folders, e.g. `apps/regression`, `apps/bench`; `apps/CMakeLists.txt` stays as is).
  (ii) `internal/reference_executor.h`: `ReferenceExecutor`, `ReferenceBatchRunner`; utilities under
  `tests/support` (headers included as `<support/...>`): cached inputs from `data/motions` exactly as
  `ccRegression.cpp` loads them, golden read / compare, flag-diff printing; the new-stack regression app (same
  checks as `ccRegression.cpp`: determinism, prefix consistency with N = 1, 37, B, B*S + 5, 1000, 4095, golden
  diff; driven by setup JSONs); benchmark apps equivalent to `BM_IAS`, `BM_RAY`, `BM_CURVE_*` timing
  `checkDevice`; a side-by-side runner that times old and new in the same process for main's RUN REQUESTs.
  (iii) Uses everything, through declarations only: `Context` (1a), `loadSetup` and friends (1b), `Scene` (1c),
  `RobotModel` (1d), `CollisionChecker` (Wave 2 e). Tests before they land: the input / golden utilities
  against the old harness (byte-identical inputs, same golden decoding), host only; `ReferenceExecutor` with a
  trivial test program (its own `rtcd_add_optix_ir` target) once 1a lands. Apps are written now; while a symbol
  they call is not implemented yet (the `CollisionChecker` until Wave 2), build them compile-only (an OBJECT
  library, as `rtcd_interfaces_compile`) and switch to `rtcd_add_app` when they link; they run after Wave 2.

**Wave 2 (parallel, after Wave 1 is merged)** -- each agent owns its library directory and the test folder of
the same name: (e) `src/detector` + `tests/detector`, (f) `src/backend_mesh` + `tests/backend_mesh`, (g)
`src/backend_ray` + `tests/backend_ray`, (h) `src/backend_curve` + `tests/backend_curve`.

- **(e) detector core** -- `checker.h` (`CollisionChecker`), pipeline / module / program-group / SBT building from
  `ProgramSet`, batch loop and dispatch, param-upload cache, stats. Tests with a fake backend plus the reference
  executor as cross-check.
- **(f) mesh backend** -- `createMeshBackend` + its programs `.cu` (section 5). Gate: `golden_ias.bin`.
- **(g) ray backend** -- `createRayBackend` + programs for all four `RayHitMode`s. Gate: `golden_ray.bin` (all hit
  modes).
- **(h) curve backend** -- `createCurveBackend` + programs + curve kernels (fused and unfused point generation,
  control points, refit policy, curve OBB). Gates: `golden_linear.bin`, `golden_quad.bin` (both geometry flags).

Until (e) lands, (f)-(h) run their goldens through `ReferenceBatchRunner`; after, through `CollisionChecker`.

## 9. Compile check

Stage 0-B ran these by hand; since Stage 0-C they are CMake targets built by `AllNew` (compile only: OBJECT
libraries, never linked or run), so every agent's build enforces them:

1. `rtcd_headers_selfcontained` (`tests/CMakeLists.txt`): one generated TU per header under `include/rtcd/`
   (globbed at configure time, 17 headers) containing only `#include <that header>` twice (include guard).
2. `rtcd_interfaces_compile` (`tests/interfaces_compile.cpp`): all headers together, an implemented `Backend`,
   constexpr helpers, aggregate initialization, the `geometry.h` function signatures, the optional curve
   geometry flags, static_asserts on move-only / trivially-copyable properties.
   Both see ONLY `include/` (no CUDA, no OptiX, no library target linked) and compile with
   `/W4 /WX /permissive-` (C++20). Verified in Stage 0-C that a CUDA include in a public header fails the build
   (`C1083: Cannot open include file: 'cuda_runtime.h'`).
3. `rtcd_core_abi_check` (`tests/core/abi_check.cpp`, owned by 1a): with rtcd_core's CUDA 13.2 / OptiX 9.1
   include paths, static_asserts that `StreamHandle == cudaStream_t == CUstream`, `CudaContextHandle ==
   CUcontext`, `OptixContextHandle == OptixDeviceContext`, `DevicePtr == CUdeviceptr`, `TraversableHandle ==
   OptixTraversableHandle`, that every mirrored flag value equals its OptiX enumerator, `sizeof(OptixInstance) ==
   80`.
4. Same target, `tests/core/limits_device.cu` (nvcc): `limits.h` in device code, kernels reading
   `KinematicsTable`, `Obb`, `EdgeRayParams`, `RobotRayParams`, static_asserts of the SBT constants against
   `optix_types.h`.

## 10. Open questions

Each with the choice made in the headers and the recommendation.

**Settled by measurement (Wave-2 close-out, state at `a605b47`).** With the real `CollisionChecker` and backends
the new stack reproduces all four goldens (ias 1365, ray 1383, linear 2895, quad 2847 collisions of 4096; 0 diffs;
ray in all four hit modes, quad with both geometry flags, curves also with OBB culling on). Same-process timing old
vs new (`rtcd_sbs_shelf`, medium shelf, measured by main), new/old median: 0.97-1.01 (mesh), 0.87-0.97 (ray),
0.98-1.08 (linear), 0.98-1.04 (quad).
- Q1 (device-buffer kinematics table): kept; no measurable cost (the ratios above; no switch to by-value needed).
- Q2 (one pipeline per checker): kept; parity (the ratios above).
- Q5 (old curve OBB path): confirmed wrong, fixed in the new stack (two defects: compacted primitive indices were
  mapped through the uncompacted segment -> pose map, and the old quadratic `OBB_3p` boxes did not contain their
  segments, which dropped 32 quad regression hits with OBB culling on); OBB on == OBB off flags is tested.
- Q20 (error check after every kernel launch): kept; no measurable cost (the ratios above).
The entries below keep their original text; the "Settled" lines refer to this paragraph.

- **Q1 Kinematics table access in kernels.** Chosen: the table lives in a per-owner device buffer (section 3 of the
  brief); kernel wrappers take `KinematicsRef {host, device}` so the implementation can either read the device
  buffer (staging it in shared memory per block) or pass the host table by value as a `__grid_constant__` kernel
  parameter (2.4 KB, below the 4 KB parameter limit; constant-bank access like the old `__constant__`, still
  per-robot, no globals). Recommendation: implement the device-buffer read first; if BM_IAS / BM_RAY at small N
  regress, switch to by-value without any header change. **Settled:** device buffer, no measurable cost.
- **Q2 One pipeline per checker** (chosen) vs one identical pipeline per stream (old `createPipeline`). OptiX allows
  concurrent launches of one pipeline on several streams. Recommendation: one; measure once. **Settled:** one
  pipeline, parity.
- **Q3 Per-stream hit buffer + D2D copy into the result** (chosen, old behaviour, stable addresses for later graph
  capture) vs writing directly into the result buffer at `firstItem` (saves one copy per batch, but the params then
  change every batch, defeating the upload cache). Recommendation: keep; revisit with CUDA graphs.
- **Q4 Per-batch staging copy of the items** (chosen, old `movePosesToStream`) vs pointing `BatchView::items` into
  the input buffer. Same trade-off as Q3. Recommendation: keep for parity.
- **Q5 Curve OBB culling.** The old curve OBB path looks inconsistent: after compaction the GAS primitive indices
  are 0..maskCnt-1 but the any-hit still maps them through the uncompacted segment -> pose map
  (`batchOptixCurveRobot::getMapIndex` returns the template); cubic OBB throws "not implemented". It is not covered
  by any golden (curve goldens run with OBB off). Chosen: OBB culling allowed for linear / quadratic, rejected for
  cubic by `validate()`. Recommendation: (h) builds a compacted segment -> item map with `compactByFlags` and adds a
  test "OBB on == OBB off flags" on the regression inputs; report if the old path really is wrong. **Settled:**
  the old path is wrong; fixed (compacted map, corrected `OBB_3p`), OBB on == OBB off tested.
- **Q6 `collapseRootLinkObb`** reproduces `SKIP_BASE` (root OBB shrunk to a point -- not removed: a point inside an
  obstacle OBB still overlaps). Recommendation: keep default true until parity; then consider excluding fixed
  links from culling and tracing altogether.
- **Q7 Joint limits.** `pandas.h` bounds (e.g. +-2.8973, joint 4 upper 0.0698) differ from `panda.urdf`
  (+-2.9671, joint 4 upper 0). They only generated the random inputs, which are cached in `data/motions`.
  Recommendation: the harness keeps reading the cached files and, if it must generate, uses the old bounds
  (harness constant), never the URDF limits, so inputs stay identical.
- **Q8 Robot features beyond the old stack** (fixed-joint merging, URDF visual origin / scale, continuous joints,
  arbitrary joint axes) are specified but only Panda is tested; prismatic joints and geometry-less chain links are
  unsupported. Recommendation: keep; add a second robot later.
- **Q9 Dropped capabilities.** `LinkType::SPHERES` (its `optixSphereLink` builds a triangle GAS anyway),
  `SPHERE_GAS` (`batchGASRobot` does not match its base interface), `RAY`, `RAY_STATIC`, `RAYDATAMODE`, the
  headline-image helpers (`resizeHitBuffer`, `getHitBuffer`, `updateASHandle`), never-captured CUDA graph members.
  None is exercised by tests or benchmarks. Recommendation: drop; demos use `checkDevice`.
- **Q10 Ray mode with OBB culling off.** Old: crashes (`check()` -> `detect()` dereferences the null mesh robot).
  Chosen: allowed, the backend uses a constant all-ones link mask. Alternative: reject in `validate()`.
- **Q11 Trajectory / control point counts are fixed per checker** (chosen; topology and weight table depend on them;
  the old stack also needed one instantiation per pair). Alternative: per-call value <= capacity with several
  weight tables. Recommendation: fixed.
- **Q12 Scene edge sets.** Chosen: `Scene` builds both sets (Loop only if all obstacles are watertight; the old
  loader threw on any non-watertight mesh even when loop edges were unused). Memory cost: one more ray buffer.
- **Q13 Stream flags.** Pool streams use default flags as before; `cudaStreamNonBlocking` would be safe (the library
  never uses the legacy default stream) but changes behaviour for users mixing default-stream work. Recommendation:
  keep default until measured.
- **Q14 Final synchronization** is per checker stream instead of `cudaDeviceSynchronize`, so independent checkers do
  not wait for each other. Same semantics for one checker.
- **Q15 Spline weights home.** `RTCD/data/weights` disappears with the old stack. Chosen default in the JSON
  examples: `configs/spline_weights/` (same subtree). No "phantom" weight files exist, so `end_points: "phantom"`
  fails at construction with `IoError` until they are generated.
- **Q16 More items than `maxItemsPerCheck`**: `CapacityError` (chosen) vs automatic multiple rounds.
  Recommendation: error; callers size the capacity.
- **Q17 Mesh hit-group records**: `maxBatchSize * numLinks` identical records (old) vs one record and sbtOffset 0.
  Chosen: old layout for parity. Recommendation: simplify after parity (fewer bytes, identical results).
- **Q18 HostDispatch default** `ThreadPerStream` (what the regression and benchmarks compiled with) vs
  `SingleThread` (old library default without the macro). Recommendation: keep `ThreadPerStream`.
- **Q19 `base_pose` location**: robot block (chosen) vs scene file. Recommendation: robot block (it is the robot's
  placement, and the scene stays robot-independent).
- **Q20 Error check after every kernel launch** (`cudaGetLastError`, chosen) costs a few host calls per batch that
  the old Release build did not make. Recommendation: keep (correctness), include in the first parity
  measurement. **Settled:** kept, no measurable cost.
- **Q21 Snapping tolerance 1e-9** for the derived kinematics masks: chosen so that `cos(+-pi/2)` (about 5e-12 from
  the URDF's 1.57079632679) becomes exactly 0 as in `pandas.h`. If any Panda entry still differs, (1d) reports
  and the rule is adjusted, not the goldens.
- **Q22 kMaxTrajectoryPoints = 512** derives from the control-point reduction (`powOf2(n) / 2 <= 256`); for linear
  curves no such limit exists. Recommendation: keep one limit for simplicity.

## 11. Requests outside this track's ownership

- **Stage 0-A (CMake):** done (Stage 0-A + 0-C): (1) compile-only targets, section 9; (2) `rtcd_add_optix_ir`
  (`cmake/RTCDTargets.cmake`) compiles a backend's program `.cu` to OptiX IR and embeds it (replaces
  `getInputData("CCCuda.cu")` / `PROJECT_PTX_DIR`); (3) library targets as in section 2.
- **Wave 1 b (configs/):** copy `RTCD/data/weights/` to `configs/spline_weights/` (identical files) so the new stack
  does not reference the old tree.
- **Old stack:** no change requested.
