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
cubic curves (no golden); the Julia plotting scripts on new output; the Python tools merged from master
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

**Ideas not yet tried:** CUDA Graph capture of the per-batch sequence (the paths without host sync are ready for
it: RAY always, IAS below the compaction threshold, curves without OBB culling); Rocaps intersector for quadratic
curves (`OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS`; it intersects a true swept sphere, so results
would change and need new references); running a single-batch check on the calling thread instead of waking a
worker; one hit-group record instead of `maxBatchSize * numLinks` identical ones in mesh mode; the fused FK+OBB
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
