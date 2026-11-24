# tests/data — committed reference data

The new stack's tests and apps read their fixed inputs and goldens from this folder (found by walking up from the
working directory or the executable; `RTCD_DATA_DIR`, or `--data <dir>` for the apps, points elsewhere, e.g. at a
local folder with the same layout). Everything here is binary and little-endian. A fresh clone runs every test
without any other data.

These files were produced by the OLD header-only stack (removed later; its last commit is tag `old-stack-final`,
commit `632b6a5`) and copied byte for byte from its local, gitignored `data/` folder.

| File | Content | Produced by (old stack) |
|---|---|---|
| `motions/Panda4096.bin` | 4096 Panda poses, 7 `float32` each: the discrete (IAS / RAY) regression input and the start / end samples of the curve regression trajectories | `randomMotions(4096, ...)` in `RTCD/Utils/Test/poseUtils.h` (Halton samples within the Panda joint limits) |
| `motions/Panda8192.bin` | 8192 poses, the benchmark pose pool | `randomMotions(8192, ...)` |
| `motions/Panda/1111111/4096_knn4_idx.bin` | 4096 x 4 `int32`: the 4 nearest neighbours (unit joint weights, hence `1111111`) of every 4096 sample; trajectory `t` runs from sample `t / 4` to sample `idx[t]` | `KNearestNeighbor` / `genCSpaceKNNTraj<7, 4096, 4>` |
| `motions/Panda/1111111/8192_knn4_idx.bin` | 8192 x 4 `int32`, the same for the benchmark pool | `genCSpaceKNNTraj<7, 8192, 4>` |
| `CCTest/golden_{ias,ray,linear,quad}.bin` | 4096 bytes each, 0 / 1: the collision flag of every regression item on the medium shelf scene (IAS = poses vs obstacle meshes, RAY = obstacle edge rays vs robot, LINEAR / QUAD = 16-point KNN trajectories as linear / quadratic swept-sphere curves) | `ccRegression --write-golden` (`RTCD/CollisionDetector/Test/ccRegression.cpp`) |

The old cache also held the `*_knn4_dis.bin` distance matrices (67 MB and 268 MB); nothing reads them, and they
are not committed. How the inputs are rebuilt from these files (trajectory arithmetic included) is
`tests/support/inputs.h`; how the goldens are read and compared is `tests/support/golden.h`.

**The goldens must not be regenerated casually.** They are the correctness gate of every mode: a change that
alters a collision flag is a bug until proven otherwise, and the old stack that produced them is gone. Per-item
hit counts may differ between implementations; flags may not. Replacing a golden needs an explicit decision by
the owner, with the reason and the differing items recorded in the commit message.

## reference/ — stored outputs of the old stack

Before the old stack was deleted, the new-stack tests that compared with it (by calling old code) were replaced by
comparisons with stored values. The files below were written FROM THE OLD STACK by those parity tests, run with
`RTCD_WRITE_REFERENCE=tests/data/reference` (each parity test has a `WriteStoredReferences` test or writes while it
compares; the generators were deleted with the old stack and live on in the history: tag `old-stack-final` plus the
commit that added this section). At that commit the new stack reproduced every one of them exactly.

Text files: `<key> <values...>`, `#` comments; float values as 8-digit hex bit patterns, hashes as 16-digit hex.
**Hash = 64-bit FNV-1a** (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`) over the raw little-endian bytes
of the array in element order (`tests/support/reference.h`). A key `<k> <n> <hash>` gives the element / byte count
first.

| File | Content (old code that produced it) | Compared by |
|---|---|---|
| `panda_kinematics.txt` | the hand-written Panda masks / base of `RTCD/Robot/models/pandas.h`, exact floats | `tests/robot/test_kinematics_table.cpp`, `test_robot_model.cpp` |
| `kinematics_fk.txt`, `kinematics_fk_poses.bin`, `kinematics_fk_traj.bin` | old FK kernels (`robot.cu`: `batchFkineAdj`, `batchFkineAdjOBB`, `batchOBBTransform`, `batchFkineTraj`) on the 4096 regression poses / trajectories: hashes of the complete outputs; raw link poses (float32) of every 64th pose and every 512th trajectory (`tests/robot/fk_reference_cases.h`) | `tests/robot/test_kinematics_kernels.cpp` |
| `meshes.txt` | old loader `RTCD::meshModel`: per mesh vertex / triangle / unique-edge / loop-edge counts and hashes, OBB, and for the 8 Panda links the sphere hashes (`tests/support/mesh_reference.h`) | `tests/geometry/test_geometry_reference.cpp`, `tests/robot/test_robot_model.cpp` |
| `scenes.txt` | old `RTCD::scene` on the three shelf scenes: edge-ray buffer hashes (unique and loop edges), per-obstacle ray counts / offsets, obstacle OBB hash | `tests/scene/test_scene_reference.cpp` |
| `obb_overlap.txt` | old `RTCD::BBCDRobotMask` counts on the deterministic boxes of `tests/core/obb_reference_cases.h` | `tests/core/test_core_obb_reference.cpp` |
| `curve_kernels.txt` | old curve kernels (`batchFkineSphrTraj`, `batchFkineTraj` + `batchSphrCntrTraj` + `batchGenControlPoints`, `genCurveOBB<1>`) on the regression trajectories | `tests/backend_curve/test_curve_kernels.cpp` |
| `refit_flags.txt` | old `CollisionDetector` (profiles `reg_linear` / `reg_quad`) under every refit policy: per check, the items whose flag differs from the golden | `tests/backend_curve/test_refit_reference.cpp` |
| `ias_hit_counts.txt`, `ias_hit_counts.bin` | old `CollisionDetector<MESH>` (profile `reg_ias`) per-item hit counts (uint32) on the first 4096 / 517 / 37 / 1 regression poses | `tests/backend_mesh/test_backend_mesh_reference.cpp` |
| `inputs.txt` | old harness `randomMotions` / `genCSpaceKNNTraj` outputs (regression inputs with 1, 8, 16, 32 points; benchmark pool) | `tests/testsupport/test_testsupport_unit.cpp` |
| `spline_weights.txt` | manifest of the old `RTCD/data/weights` tables (now `configs/spline_weights`), content hashed with CR removed | `tests/json/test_json.cpp` |

The hashes are bit-exact: a compiler, CUDA or driver change that alters the last bit of a kernel output makes the
corresponding test fail. That is intended (it is how the bit-identity with the old stack is kept), but such a
failure is not automatically a bug: check the four goldens (`rtcd_regression_*`, label `regression`) first. As with
the goldens, references are never re-recorded to make a test pass; replacing one is an explicit owner decision with
the reason in the commit message.