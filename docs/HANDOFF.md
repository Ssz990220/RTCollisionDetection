# Handoff: RTCD on `optix9-upgrade` (written 2026-11-24)

For the agent continuing this work on a Linux machine. Goal there, as stated by the project owner: **reproduce the
results on Linux and find the performance issues.** Everything below was done and measured on one Windows machine;
nothing in the new code has been built or run on Linux yet.

Read this first, then `README.md` (build, usage), `docs/user/architecture.md` (one page), and
`docs/design/interfaces.md` (the module contract, long; sections 3.10, 4 and 5 matter most).

## 1. Where things are

| Ref | Commit | What it is |
|---|---|---|
| `optix9-upgrade` (pushed) | `4cc6346` | Current state: the new stack only |
| in this branch's history | `3daac7a` | **Best baseline commit**: old stack (`RTCD/`) and new stack side by side, reference data committed, and the old-vs-new timing runner `rtcd_sbs_*` still present |
| in this branch's history | `632b6a5` | Last commit with the old stack before reference data was committed (local tag `old-stack-final`, not pushed) |
| in this branch's history | `9e35db8` | Old stack only, ported to OptiX 9.1 / CUDA 13, before any restructuring ("Level 1") |
| `master` / `origin/master` | `bacbf77` | Upstream; already merged into `optix9-upgrade` |
| local only on the Windows box | `experiment/lss-hw-vs-sw` | Hardware vs software LSS measurement branch (not pushed; results are in section 5) |

`git checkout 3daac7a` works from a clone of the pushed branch; no tag is needed.

## 2. What the project is now

RTCD checks batches of robot poses or trajectories against mesh obstacles on the GPU with NVIDIA OptiX. Four modes:

| Mode | What is traced | Config `mode` |
|---|---|---|
| ObsToRobot ("IAS", "mesh") | obstacle edges against instanced robot link meshes | `mesh` |
| RobotToObs ("RAY") | robot link edges against the scene | `ray` |
| LINEAR | obstacle loop edges against swept robot spheres as linear curves (hardware LSS on RTX 50) | `linear_curve` |
| QUAD | same, quadratic B-spline curves | `quadratic_curve` (`cubic_curve` exists, no reference results) |

The original code was header-only and templated on stream count, batch size, DOF and mode, with the robot hard-coded.
It was replaced by a "new stack": compiled libraries, runtime configuration from JSON, the robot loaded from a URDF
at run time, one backend per mode behind `rtcd::Backend::prepare()`, and a facade `rtcd::CollisionChecker`.

```
include/rtcd/      public headers (frozen contract; no CUDA/OptiX device code)
src/core           errors, DeviceBuffer, Context + stream leases, GPU algorithms, accel builders
src/config, json   CheckerConfig validate/resolved; JSON loaders (nlohmann-json)
src/geometry       OBJ loading, edges, OBB fit, sphere files
src/scene          Scene: obstacle GASes + IAS, edge rays, obstacle OBBs
src/robot          RobotModel from URDF (urdfdom), forward-kinematics kernels
src/detector       CollisionChecker: pipeline/SBT from a backend's ProgramSet, batch loop, worker thread per stream
src/backend_mesh | backend_ray | backend_curve   the three backends and their OptiX programs (embedded OptiX IR)
apps/              rtcd_example, rtcd_regression, rtcd_bench_discrete, rtcd_bench_curve, rtcd_demo_*
tests/             unit + GPU tests; tests/support = test utilities; tests/data = reference inputs, goldens, references
configs/           checker / scene / robot / setup JSON files, spline weights
```

## 3. What is verified, and where

All on Windows 11, RTX 5080, driver 610.88, CUDA 13.2, OptiX 9.1.0, MSVC 14.52, CMake 4.3 + Ninja, Release.

- From-scratch build of `AllNew` at `928818a` (one commit before the current `.gitignore`-only commit), with no
  local `data/` folder: `ctest -L new_unit` 8/8, `-L new_gpu` 48/48, `-L regression` 8/8.
- Goldens (collisions out of 4096 items, medium shelf scene, Panda): ias 1365, ray 1383, linear 2895, quad 2847.
  Reproduced by the new stack with 0 differing items; ray in all four hit modes; quad with both curve geometry
  flags; curves also with OBB culling on.
- At `632b6a5` (both stacks): new_unit 10/10, new_gpu 49/49, old-stack regression 4/4.

**Not verified anywhere:** Linux; GCC/Clang; Debug builds; the Visual Studio generator; robots other than Panda;
cubic curves (no golden); the Python plotting tool `scripts/Benchmark/plots/plot_benchmarks.py` on new output (its parser accepts benchmark names with and without the `/min_warmup_time:` part); the Python tools merged from master
(`scripts/URDFParser`); `benchmark.sh` end to end.

## 4. Building on Linux: what to expect

The README's configure line is the Windows one. Things that will need attention (all untested guesses about
where it breaks, from reading the build files):

- `cmake/FindOptiX.cmake`: on UNIX it does not search default locations; set `OptiX_INSTALL_DIR` (CMake variable or
  environment) to the OptiX 9.1 SDK root (the folder containing `include/optix.h`).
- Requirements in `CMakeLists.txt`: CMake >= 3.25, CUDAToolkit >= 12.6, OptiX, urdfdom, Eigen3, Google Benchmark,
  nlohmann_json, GTest. vcpkg was used on Windows (`eigen3 urdfdom urdfdom-headers benchmark nlohmann-json gtest`);
  system packages may work.
- MSVC-only flags are guarded by `if(MSVC)` in `cmake/configCompilerFlags.cmake`; the
  `-DCMAKE_CUDA_FLAGS=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` configure argument is a Windows workaround and
  should not be needed.
- `CMAKE_CUDA_ARCHITECTURES` is `native`.
- Files that contain Windows-specific code or definitions (`_WIN32`, `NOMINMAX`, `windows.h`): `src/geometry/obj_mesh.cpp`,
  `tests/support/paths.cpp`, `tests/support/reference_executor.cpp`, `tests/scene/test_scene_gpu.cpp`,
  `tests/backend_ray/test_backend_ray.cpp`, and the CMake files of `src/backend_curve`, `src/detector`,
  `tests/backend_curve`, `tests/detector`. Check these first if something does not compile.
- `cmake/configure_tiny_obj_loader.cmake` downloads a pinned `tiny_obj_loader.h` at configure time (needs network).
- Tests run with the repository root as working directory and read only `tests/data/` (override: `RTCD_DATA_DIR`).
- At `3daac7a` / `632b6a5` the old stack is also built by the same CMake project. Build only its targets
  `AllRegressionTests`, `AllBenchmarks`, `AllDemos`; several of its other unit-test targets did not compile even
  on Windows and are not needed. Its dependency list on Linux has not been checked.

First milestones on Linux, in order: (1) configure + build `AllNew`; (2) `ctest -L new_unit`, then `-L new_gpu`,
then `-L regression`; (3) only then timing. If a stored reference under `tests/data/reference/` differs on Linux
(different compiler / libm / driver can change float results bit-wise), do NOT regenerate it: report which test
and how large the difference is, and judge by the goldens (per-item collision flags), which are the real gate.

## 5. Performance: what was measured on Windows

**Rules that made the numbers trustworthy** (please keep them):
- Timings on that machine drifted 5-10% over a day. Only numbers taken in the same run are comparable; always
  measure the baseline in the same run, alternate old/new, report medians with spread.
- One GPU job at a time for timing. Correctness tests can run in parallel.
- Google Benchmark 1.10 ignores per-benchmark `MinWarmUpTime()` unless `MinTime()` is set, so the benchmarks have
  no warm-up unless `--benchmark_min_warmup_time=1` is passed.

**New stack vs old stack, same process** (`rtcd_sbs_shelf` at the pre-deletion commits; ratio = new / old median):

| Mode | Ratio range over N | Notes |
|---|---|---|
| mesh (IAS) | 0.97 - 1.01 | parity |
| ray | 0.86 - 0.99 | new is faster (no per-round barrier in the detector) |
| linear | 0.98 - 1.09 | new is 6-8% slower at N = 128-256 (8-16 batches of 16), reproduced 4 times; ~1.0 at N=16 and N=1024 |
| quad | 0.98 - 1.05 | new is 4-5% slower at N = 128-256; noisy at N=1 |

Suspected cause of the curve gap (unconfirmed): the fused point-generation kernel reads the kinematics table and
spline weights from device buffers (global memory) instead of `__constant__` memory, roughly +10 us per batch.
The contract allows passing the table by value as a kernel parameter without a header change
(`docs/design/interfaces.md`, open question Q1).

**Absolute times now on the 5080 / Windows vs the originally reported results** (stored in the local
`data/Benchmark/result`, Feb 2025, Linux, GPU believed to be an RTX 4070; medium shelf; us):

| Case | Old report | 5080 Windows now | Now vs old |
|---|---|---|---|
| IAS N=1 / 1024 / 4096 | 90 / 595 / 1451 | 113 / 637 / 1039 | 0.80x / 0.93x / 1.40x |
| RAY N=1 / 1024 / 4096 | 54 / 557 / 1949 | 81 / 335 / 1096 | 0.67x / 1.66x / 1.78x |
| LINEAR<16,16,16> N=1 / 256 / 1024 | 225 / 2975 / 11997 | 94 / 2134 / 7383 | 2.40x / 1.39x / 1.63x |
| QUAD<8,16,16> N=1 / 256 / 1024 | 417 / 6990 / 27490 | 205 / 6097 / 21550 | faster after the single any-hit change below |

**The main open question for Linux:** small-batch discrete checks are slower on the 5080 under Windows than the
old report on a smaller GPU. A profile showed these checks are bound by host-side CUDA call overhead (each
launch / memset / async copy costs 5-30 us under the Windows WDDM driver model; GPU kernels take 0.5-2.5 us), not
by GPU work. The owner's working hypothesis is "it is Windows". Linux should confirm or refute this: measure
`BM_IAS` / `BM_RAY` at N = 1, 16 on the same code.

**Optimizations already in the code** (all kept in the new stack, all with unchanged collision flags):
- RAY: no compaction and no host sync; the raygen tests the OBB mask itself; grid-stride launch capped at 262144
  wide (`ray.launch_width_cap`); skip rays of items already hit (`ray.skip_hit_items`). 1.4-2.7x vs before.
- IAS: culled instances are disabled with the visibility mask (no host sync) below `mesh.compaction_min_poses`
  = 128 poses per batch; compaction with one host sync at or above it (measured crossover on Windows).
- Curves: one fused kernel for FK + sphere centers (+ control points); curve GAS built with `ALLOW_UPDATE`
  (this alone made multi-batch QUAD ~1.95x faster); refit instead of rebuild when a check has <= 2 batches (linear)
  or <= 1 (quad) (`curve.refit: "auto"`); quadratic curves built with the single-any-hit geometry flag
  (`curve.geometry_flags: "auto"`, 1.3-2x for QUAD).
- These thresholds were tuned on Windows / RTX 5080 only. **Re-measure them on Linux**: `mesh.compaction_min_poses`,
  `ray.launch_width_cap`, the refit batch thresholds, and the curve benchmark's batch size (batch 128 was much more
  efficient than the benchmark default of 16 with hardware LSS: 4942 vs 7668 us at N=1024).

**Measured and found to make no difference** (do not repeat): OptiX debug level none vs default; max register
count 0 / 32 / 64 vs 50; max trace depth 1 vs 2; first-hit ray flags with closest-hit or miss programs for RAY.
A RAY variant using `optixTraverse` without hit programs (`ray.hit_mode: "traverse"`) was 3-5% faster at
N >= 256; it is available but not the default.

**Hardware vs software LSS** (LINEAR mode, software path forced through OptiX's motion-blur fallback, so an upper
bound on the hardware advantage): end to end 1.9x (batch 16) to 2.2x (batch 128); per batch, ray tracing ~3x
faster, GAS build 1.35-1.5x faster, GAS memory 7.2 vs 13.4 MB (batch 128). With hardware LSS the GAS build
dominates at small batches (381 of ~520 us per batch of 16).

**CUDA Graph capture (Track G0, 2026-11-24; experiment `tests/experiments/graph_capture`, feature `graph_prepare`).**
- **OptiX calls cannot be captured** (OptiX 9.1, driver 610.88, CUDA runtime/driver API reported as 13030). In
  every capture mode (thread-local, global, relaxed): `optixLaunch` alone "captures" without error but the graph
  holds only a 64-byte D2D param memcpy; the replay (and the capture itself) traces nothing (0 hits vs 16374 direct).
  After other captured work it fails with `OPTIX_ERROR_CUDA_ERROR` ("Failed to query command list event",
  `cuEventQuery` -> `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, then "Error launching work to RTX"); in relaxed mode a
  whole 32-batch check captures (225 nodes, no trace work) and replays with 0 of 1383 collisions.
  `optixAccelBuild` adds no node for IAS BUILD / UPDATE and linear-curve GAS BUILD / UPDATE (the build is silently
  dropped); for quadratic curves part of the build is captured (4 kernels + 2 memcpy) and the replay fails with
  `CUDA_ERROR_ILLEGAL_ADDRESS` (sticky). Every capture containing an OptiX call leaves an OptiX internal event invalid:
  later direct `optixLaunch` calls may fail with `OPTIX_ERROR_INVALID_VALUE` and `optixDeviceContextDestroy` fails.
  `OPTIX_FORCE_DEPRECATED_LAUNCHER=1` changes nothing. Host-syncing paths (mesh compaction, curve OBB) fail cleanly
  (`cudaErrorStreamCaptureUnsupported` at the sync; the context stays usable).
- **Host cost per call** (direct, one stream, calling thread, medians [p90] us): memset 2.3 [13], D2D copy 3.1,
  kernel ~2.9, `optixLaunch` 8.1-9.6 [21-40]; `prepare()`: ray 8.7, mesh masked 15.8, mesh OBB off 11.8, linear
  8.3-9.0, quad 33-39 (IAS / GAS build ~5-7 us host for IAS / linear). A graph replacing memset + copy + 3 kernels
  costs ~4.6 us; capture ~10 us + instantiate ~22 us per new (stream, item count).
- **What is used:** for RAY the CUDA part (zero, staging copy, `prepare()` kernels) is replayed as one graph per
  batch (`graph_prepare`, default on; `Backend::prepareCapturable`); `optixLaunch` and the result copy stay direct
  (7 -> 3 host calls per batch). `rtcd_bench_discrete --rtcd_graph_bench` (4 streams, batch 1024), same process,
  10 interleaved repetitions, two runs, median us (stddev <= 4.6):

  | N | BM_RAY_NOGRAPH (off) | BM_RAY (on) | on / off |
  |---|---|---|---|
  | 1 | 50.65 / 50.68 | 38.23 / 38.96 | 0.75 / 0.77 |
  | 16 | 57.95 / 58.53 | 45.38 / 45.24 | 0.78 / 0.77 |
  | 256 | 118.96 / 119.77 | 107.67 / 108.48 | 0.91 / 0.91 |
  | 1024 | 311.15 / 313.13 | 299.72 / 300.57 | 0.96 / 0.96 |
  | 4096 | 1001.81 / 1001.47 | 936.57 / 936.12 | 0.93 / 0.93 |

  BM_IAS is unaffected (its `prepare()` builds the IAS, not capturable): N = 1 / 16 / 1024 at 98.5-99.6 /
  148.6-150.3 / 612.7-614.5 us on and off (separate processes, alternated).
- **Next (not done):** mesh / curves would need `prepare()` split into a capturable CUDA part and a direct OptiX part
  (estimated ~10 us per batch for mesh masked, ~4 us for curves); the result copy is gone for repeated
  targets since G2 (below).

**Host-time breakdown and call reduction (Track G2, 2026-11-24).** Tool: `tests/experiments/host_breakdown`
(`rtcd_exp_host_breakdown`: a `DetectorCore` of each benchmark configuration with a `PhaseTrace`
(`src/detector/phase_trace.h`, off by default: one null-pointer test per mark point) that timestamps every phase of
`checkDevice` / `checkHost`; traced and untraced checks alternate, so the tracing cost is visible; `--variants`
alternates several checker configs in one process and compares their flags every repetition;
`rtcd_exp_transfer_probe`: small transfers by method). Configs: mesh / ray 4 streams, batch 1024; linear<16,16,16>,
quad<8,16,16> 16 streams, batch 16; N = 1 and 16 are single-batch checks in all four.

Breakdown BEFORE (base `00c8591`), `checkDevice` N = 1, median us over 1500 traced reps (p10 / p90 in
`bd_base` runs of the G2 report):

| phase | ray | mesh | linear | quad |
|---|---|---|---|---|
| entry + argument checks + makeCurrent | 0.2 | 0.2 | 0.2 | 0.2 |
| wake worker (dispatch -> batch start) | 4.7 | 4.3 | 4.7 | 4.9 |
| zero hit counters | (graph) | 3.0 | 3.1 | 3.6 |
| staging copy | (graph) | 3.6 | 3.5 | 3.7 |
| prepare() / graph launch | 5.8 | 17.7 | 9.4 | 37.3 |
| params compare / upload | 0.0 | 0.0 | 0.1 | 0.1 |
| optixLaunch | 9.4 | 8.6 | 8.4 | 8.7 |
| result copy | 3.3 | 3.2 | 3.2 | 3.2 |
| join (batch end -> caller resumes) | 4.4 | 4.6 | 4.7 | 4.7 |
| stream sync (GPU still running; idle-stream sync costs 0.1-0.3) | 4.6 | 42 | 30 | 94 |
| sync of the unused streams (3 resp. 15) | 0.3 | 0.3 | 1.1 | 1.1 |
| total | 39.1 | 94.5 | 78.6 | 172.6 |

Host `check()` added: upload 3.2 us, download 24-40 us (!), so N = 1 host checks took 76 (ray) / 122 (mesh) /
111 (linear) / 205 (quad) us. Mesh and curves at N = 1 are already partly GPU-bound (the sync waits for the GPU);
at N = 16 the curve checks are GPU-bound (sync 270 / 800 us).

Changes kept (all defaults; flags compared every repetition, 0 mismatches; tests below):
1. **Single-batch checks on the calling thread** (no worker wake + join): -6 to -18 us per check.
2. **Packed items read in place** (no staging copy for non-graph batches): mesh N = 1 -5 us, linear N = 256 -7.7%.
3. **Counting straight into the result range** when a stream's previous batch had the same target (repeated
   checks into one buffer; params stay identical, ray graphs keyed by the hit-count address): ray N = 1 -7 us.
4. **Synchronize only the streams that ran batches**: -1 us at 16 streams.
5. **Host checks without copy-engine transfers** (`mapped_host_transfer`, default on): items packed into a pinned
   buffer the GPU reads through its mapping, counters written into a pinned buffer by a copy kernel; the ray graph's
   staging copy is a kernel node. Root cause, `rtcd_exp_transfer_probe`, 64 bytes: H2D copy + kernel 25-61 us vs
   mapped read 7-8 us; D2H right after GPU work ~20 us more than idle (4.5). Host check, same process: ray N = 1
   32 vs 68, mesh 87 vs 116, linear 67 vs 108, quad 137 vs 179 us (copy path = `mapped_host_transfer: false`);
   N = 1024-4096 within 1.5% either way (mesh N = 4096 +1.3%, inside the spread). Large end, mapped vs copy,
   median us: ray N=1024 289 vs 331, N=4096 934 vs 970; mesh N=1024 585 vs 620, N=4096 912 vs 900; linear
   N=1024 7003 vs 7044, N=4096 26659 vs 27252; quad N=1024 20511 vs 20461, N=4096 77694 vs 78171. No size threshold.
6. **Wait on the host for pending legacy-stream work** before issuing a check. A pageable `cudaMemcpy` (as the
   benchmarks do in their paused region, and as most callers upload device items) returns before its DMA ends;
   work submitted on the blocking pool streams meanwhile pays a cross-queue dependency of ~15 us on Windows. This
   is what made the worker hand-off look cheap before change 1: it hid the DMA. One `cudaStreamQuery` +
   `cudaStreamSynchronize(cudaStreamLegacy)` if not ready: BM_RAY N = 1 45 -> 27 us, BM_IAS N = 1 95 -> 78 us
   (same build, A/B). **Benchmark-harness note:** without this wait the paused pageable upload spills ~11-14 us
   into the timed `checkDevice` at N = 1-16 (host_breakdown `--settle 1`, which syncs the device after the upload:
   ray N = 1 34-38 -> 23.5, mesh N = 1 87-89 -> 75-83 us); the harness is unchanged.

**Re-measure both on Linux** (they are Windows / WDDM findings): `mapped_host_transfer` true vs false
(`rtcd_exp_host_breakdown --path host --variants "mapped_host_transfer=true/mapped_host_transfer=false"`, N = 1 ..
4096; on Linux small copies cost ~1 us and mapped PCIe reads may lose at large N), and the legacy-stream wait (it
has no switch; compare BM_RAY / BM_IAS N = 1 at `a786ba8` and its parent `f165115`). The pool streams must stay
blocking either way (design doc Q13).

Tried and dropped: caller spin-wait for the workers (join) and worker spin before sleeping (wake): no gain for
multi-batch checks, +1-3% at some sizes, burns cores; per-batch async D2H of host results into pinned memory
(copy engine): 3-8 us slower than one download.

Breakdown AFTER (`a786ba8` + tests), same tool, N = 1, median us (device check / host check):

| phase | ray | mesh | linear | quad |
|---|---|---|---|---|
| wake / join | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| zero hit counters | (graph) | 2.4 | 2.4 | 2.4 |
| staging copy | (graph) | 0 | 0 | 0 |
| prepare() / graph launch | 4.7 | 16.3 | 8.4 | 34.7 |
| optixLaunch | 8.4 | 8.3 | 8.3 | 8.8 |
| result copy (device: direct; host: copy kernel) | 0 / 2.9 | 0 / 2.9 | 0 / 2.9 | 0 / 2.9 |
| stream sync (now mostly GPU time) | 10.1 | 46.9 | 37.1 | 101 |
| upload / download (host) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| total device / host | 25.3 / 33.5 | 76.0 / 88.1 | 59.6 / 64.5 | 151.9 / 150.5 |

What remains on the host for a single batch: ray 2 calls (graph launch, `optixLaunch`) + sync; mesh 6 (memset,
3 kernels, IAS build, launch); curves 3-4. Next candidates, not done: a mesh `prepare()` split so memset + its
3 kernels become one graph launch (contract change, not approved; estimate from the measured costs: memset 2.4 +
3 kernels ~7.5 us -> graph ~4.7 us, i.e. ~5 us per mesh batch); fusing the mesh backend's transform/mask kernel with
the OBB overlap kernel (-1 launch, ~2.5 us; needs a bit-identical copy of the SAT test from `src/core`).

Final benchmarks, `rtcd_bench_discrete` / `rtcd_bench_curve`, base `00c8591` vs `a786ba8`, same session, base and
new alternated, 2 rounds, `--benchmark_min_warmup_time=1 --benchmark_repetitions=3`, median of each, us (ratio new /
base per round):

| benchmark | N=1 | N=16 | N=256 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| BM_IAS base / new | 99.0 / 78.2 (0.79) | 150.2 / 128.1 (0.85) | 310.5 / 289.0 (0.93) | 614.3 / 588.7 (0.96) | 948.4 / 899.3 (0.95) |
| BM_RAY base / new | 39.6 / 27.2 (0.69) | 47.1 / 34.8 (0.74-0.77) | 107.6 / 95.4 (0.88) | 303.3 / 286.3 (0.94-0.95) | 942.6 / 940.6 (1.00) |
| BM_LINEAR<16,16,16> | 86.1 / 62.3 (0.72-0.78) | 332.8 / 291.7 (0.88-0.95) | 2266 / 2095 (0.93-0.95) | 7197 / 6977 (0.97) | - |
| BM_QUAD<8,16,16> | 185.6 / 158.9 (0.86-0.87) | 952.5 / 891.4 (0.94-0.95) | 6133 / 6020 (0.98) | 20371 / 20298 (1.00) | - |

Nothing got slower at any size. With the paused pageable upload's DMA taken out of the timed region entirely
(`--settle 1`), the remaining difference is ~1.5-2 us (the legacy-stream wait itself): BM_RAY / BM_IAS N = 1 would
read about 25 / 76 us.

**Ideas not yet tried:** Rocaps intersector for quadratic
curves (`OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS`; it intersects a true swept sphere, so results
would change and need new references);
one hit-group record instead of `maxBatchSize * numLinks` identical ones in mesh mode; the fused FK+OBB
kernel in the ray backend (saves one launch per batch).

## 6. How to get an old-vs-new comparison on Linux

The cleanest tool is the side-by-side runner, which exists only at commits `0c8c213` .. `3daac7a`:

```
git worktree add ../rtcd-baseline 3daac7a      # or a second clone
# configure + build there: targets AllNew AllRegressionTests AllBenchmarks
build/bin/apps/rtcd_sbs_shelf --profile reg_ias --golden --reps 0      # correctness of both stacks first
build/bin/apps/rtcd_sbs_shelf --profile bm_ias --n 1,16,128,256,1024,4096
build/bin/apps/rtcd_sbs_shelf --profile bm_ray --n 1,16,256,1024,4096
build/bin/apps/rtcd_sbs_shelf --profile bm_linear_16 --n 1,16,32,128,256,1024
build/bin/apps/rtcd_sbs_shelf --profile bm_quad_8_16 --n 1,16,32,128,256,1024
```
It times old `CollisionDetector::check()` and new `CollisionChecker::checkDevice()` in one process on identical
inputs, alternating, and compares the flags every repetition (options: header of `apps/side_by_side/sbs_main.cpp`
at that commit; `--list` shows the profiles). The old-stack half of that commit has never been compiled on Linux
either; the original code was developed on Linux, but it has since been edited on Windows only.

At the current head the equivalents are the Google Benchmark apps (`rtcd_bench_discrete`, `rtcd_bench_curve`,
`--scene shelf|shelf_simple|shelf_dense`; same benchmark names as the original `BM_IAS`, `BM_RAY`,
`BM_LINEAR<..>`, `BM_QUAD<..>`), see `docs/Benchmark.md`.

For the comparison with the originally reported numbers, the stored result files are NOT in the repository (they
are in the Windows machine's local `data/Benchmark/result`); the table in section 5 has the key values.

## 7. Rules of the road

- **Goldens and stored references are never regenerated to make a test pass.** `tests/data/README.md` explains
  what each file is. A real, intended behaviour change needs the owner's decision.
- **`include/rtcd/` is a frozen contract.** Implementation changes are free; declaration changes should be
  deliberate, made together with `docs/design/interfaces.md`.
- Parity first: a change that alters per-item collision flags is a behaviour change, not an optimization. Hit
  counts may change; flags may not.
- No `exit()` in library code; errors are exceptions, checked in all build types.

## 8. Known issues and findings to carry forward

- **Refit policies can change grazing results.** With a forced `curve.refit: {"every": 3}`, two quad items
  (4065, 3528 at N=4096) are reported as colliding that the golden reports free. No change under the default
  `auto`, `always` or `never` in 324 compared checks. Cause inside OptiX unknown.
- **The old stack's quadratic OBB culling was wrong** (91.8% of its segment boxes did not contain their segment;
  with culling on it lost collisions). Fixed in `src/backend_curve`; the old code at the baseline commits still has
  the bug, so never compare QUAD with OBB culling on against the old stack. Benchmarks and goldens use culling off.
- **QUAD on this GPU / OptiX 9.1 was ~2x slower than the old report before `ALLOW_UPDATE` was added**; the reason
  why that flag helps is not understood (OptiX does not document it).
- **Nsight Systems hung** on the curve benchmark on the Windows machine (cause unknown), so the curve path has no
  per-call profile after the restructuring. It may work on Linux; a profile of one batch of LINEAR and QUAD would
  answer several open questions (curve gap, `ALLOW_UPDATE`).
- The dense shelf scene uses `shelves.obj`, not `shelves_dense.obj`, as the original code did (possibly an
  original oversight; kept for parity).
- The mesh "watertight" check inherited from the original code can never fail; loop edges are always built.
- `scripts/URDFParser` (from master): its sphere files match the loader's format, but the spheres are generated
  for links with fixed children merged, so the robot must then be loaded from `modified_<robot>.urdf` with the
  merged meshes; the script's last step writes into a folder that no longer exists (`RTCD/Robot/models`). Read
  only, never run by the previous agent. See `docs/How-to-New-Robot.md`.
- No install / package export exists for library users.
