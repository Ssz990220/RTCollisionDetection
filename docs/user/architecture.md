# Architecture in one page (for contributors)

The library is plain C++20 libraries plus one compiled OptiX program module per backend. The full contract
(every header, the call sequence of a check, what each backend does, open questions) is
[docs/design/interfaces.md](../design/interfaces.md); the headers under `include/rtcd/` are frozen and changed only
through the procedure at the top of that document. The original header-only code under `RTCD/` (the "old stack")
was removed after this library reached parity with it; its last commit is tag `old-stack-final`, and its outputs
are kept as stored references under `tests/data/`.

## Modules
A library may depend only on libraries above it (`src/CMakeLists.txt` repeats this list).

| library | directory | what it does |
|---|---|---|
| `rtcd_core` | `src/core` | exceptions and error checks, `Logger`, `DeviceBuffer` / `PinnedHostBuffer`, `Context` (CUDA device, OptiX context, stream pool), internal algorithms (OBB overlap counts, scan, compaction) and acceleration-structure builders |
| `rtcd_config` | `src/config` | `CheckerConfig::validate()` / `resolved()`; no dependencies at all |
| `rtcd_json` | `src/json` | JSON files <-> `CheckerConfig`, `SceneDesc`, `RobotDesc`, setups, overrides, dumps; the only library that links nlohmann-json |
| `rtcd_geometry` | `src/geometry` | OBJ loading (tinyobjloader), unique / loop edges, OBB fit, sphere files; host only |
| `rtcd_scene` | `src/scene` | `Scene`: one compacted triangle GAS per obstacle, one IAS over them, both edge-ray sets, obstacle OBBs |
| `rtcd_robot` | `src/robot` | `RobotModel` from URDF (urdfdom): serial chain, kinematics table, link meshes / spheres / OBBs (host only); forward-kinematics kernels |
| `rtcd_backend_mesh`, `_ray`, `_curve` | `src/backend_<x>` | one backend per mode family, each with its own OptiX programs compiled to OptiX IR and embedded in the library |
| `rtcd_detector` | `src/detector` | `CollisionChecker`: pipeline / SBT from the backend's `ProgramSet`, the batch loop, host dispatch, results |
| `rtcd` | (INTERFACE) | what apps and tests link: detector + backends + config + JSON |
| `rtcd_testsupport` | `tests/support` | cached inputs and goldens, a reference executor that runs a backend without the detector core |

Public headers include no CUDA or OptiX header (checked by the compile-only targets `rtcd_headers_selfcontained`
and `rtcd_interfaces_compile`, part of `AllNew`). True compile-time constants and every struct shared between host
and device (launch parameters, SBT record, `Obb`, `KinematicsTable`) are in `include/rtcd/limits.h` only.
Everything else is a runtime option (`config.h`) or a capacity given at construction.

## One check, and the backend seam
`CollisionChecker::check()` splits the items into batches of `max_batch_size`; batch b runs on stream
b % `streams`. The detector core runs one loop with no mode branch in it (`include/rtcd/backend.h`):

```
backend.beginCheck(info)                                   once per check, host only
for each batch b on stream s (in batch order per stream; one worker thread per stream with thread_per_stream,
                              the calling thread for a single-batch check):
    core: zero the hit counters (s's own, or the result range itself when s's previous batch had the same target)
    core: items are read in place when packed (strided input: copy into s's staging buffer)
    plan = backend.prepare(streams[s], batch)              kinematics, OBB culling, AS build / refit, params
    core: upload plan.params if they changed, optixLaunch(...) unless plan.launch == false
    core: copy the hit counters into the result at batch.firstItem, unless counted there directly
core: synchronize the streams that ran batches; flag = hit count > 0
```
Host checks (`check(HostItems)`, `mapped_host_transfer` default on) pack the items into a pinned host buffer that
the GPU reads through its mapping and get the counters back through a second pinned buffer, written by a small
copy kernel: no copy-engine transfer, which costs 20-60 us of latency per small copy on Windows.
Each check first waits for pending work on the legacy default stream (query, then synchronize if not ready); with
blocking streams that work would be waited for on the device anyway; measured on Windows: ~15 us per check saved
when a pageable upload precedes the check. This is also why the Context's pool streams must stay blocking: a
caller's pageable `cudaMemcpy` of the items on the default stream must be ordered before the check reads them.

Seams sit only where calls are rare: the virtual `Backend` calls are once per checker, per check or per batch.
Kernel bodies and OptiX programs stay inside their own `.cu` translation unit; where a kernel needs a compile-time
value (e.g. the curve degree) it is a template explicitly instantiated in that `.cu`.

The `Backend` contract, in short (full text in `backend.h`):
- construction allocates all device memory for every stream index (sized by the capacities), builds static
  acceleration structures, uploads static data and fixes `programs()`;
- `prepare()` enqueues work only on its stream, never allocates, never synchronizes the device, touches only the
  state of its stream index (so different streams may call it concurrently), and blocks the host only when
  `synchronizesHost(itemCount)` says so (mesh compaction path, curve OBB path); the programs add hits of
  batch-local item i to `hitCounts[i]` with `atomicAdd`, the only result location;
- failures throw `rtcd::Error`; the core drains all streams and rethrows on the calling thread.

CUDA graph replay (`graph_prepare`, default on, since 2026-11-24): a backend may report
`prepareCapturable(itemCount)` -- its `prepare()` issues only plain CUDA work and its plan / params depend only on
(stream, item count). For such batches the core captures "zero hit counts, staging copy, `prepare()`" once per
(stream, item count, hit-count address) into a CUDA graph and afterwards issues it with one `cudaGraphLaunch`
(the staging copy is the core's copy kernel, re-pointed per batch; at most 8 graphs per stream); `optixLaunch` and the result copy stay direct calls. Only the
ray backend reports it: OptiX calls cannot be captured (an `optixLaunch` / `optixAccelBuild` inside a capture is
dropped from the graph and corrupts the OptiX context, measured with `tests/experiments/graph_capture`), and mesh /
curves build an acceleration structure in every `prepare()`.

## Where to add a mode
1. Contract (through main, since the headers are frozen): a `Mode` enumerator in `include/rtcd/types.h`, its
   options in `include/rtcd/config.h`, a factory declaration in `include/rtcd/backend.h`, and, if host and device
   share a new launch-parameter struct, its layout in `include/rtcd/limits.h` (size static_asserted).
2. `src/config`: validation and `resolved()` rules for the new options; `src/json`: the JSON spellings.
3. A library `src/backend_<x>`: a class implementing `Backend`, the factory, and the OptiX programs `.cu` embedded
   with `rtcd_add_optix_ir(<target> SOURCES ...)` (`cmake/RTCDTargets.cmake`) and returned in
   `ProgramSet::moduleCode`. Built-in intersection modules must use exactly the build flags of the GAS they
   intersect (`ProgramSet::builtinIsBuildFlags`).
4. `src/detector/backend_factory.cpp`: one `case` in `createBackendForMode`, the only mode switch of the detector;
   link the library into `rtcd_detector` and the `rtcd` umbrella (`src/CMakeLists.txt`).
5. Tests in `tests/backend_<x>` (`rtcd_add_new_test(... LABEL new_unit|new_gpu ...)`). A backend can be tested
   before or without the detector core through `internal::ReferenceBatchRunner`
   (`include/rtcd/internal/reference_executor.h`), as the existing backends were; collision flags are checked
   end to end with `rtcd_regression --setup <setup.json>`.

## Tests, apps, data
- ctest labels: `new_unit` (no GPU), `new_gpu` (GPU, includes the golden regression `rtcd_regression_*`, the
  stored-reference tests and the app smoke tests), `regression` (only the golden regression entries
  `rtcd_regression_*`, a subset of `new_gpu`). New test folders under `tests/` and new app folders under `apps/`
  are picked up at configure time.
- Correctness gate: per-item flags must equal `tests/data/CCTest/golden_{ias,ray,linear,quad}.bin` (4096 items
  each); hit counts may differ, flags may not. Goldens are never re-recorded.
- `tests/data/` (committed): goldens, the cached regression / benchmark inputs and the stored outputs of the old
  stack (`tests/data/reference`; format and generators in `tests/data/README.md`). The local, untracked `data/`
  folder only receives benchmark results and plots.
