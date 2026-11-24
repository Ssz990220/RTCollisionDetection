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
