# Benchmarking

## Scene structure

<div align="center">

<img src="../assets/collisionScene.png" width="96%" />
<p><em>All benchmarks run in a scene like this. Scene complexity (triangles / edges):<br>
Simple: 15k / 22k &nbsp;&nbsp;|&nbsp;&nbsp; Medium: 71k / 107k &nbsp;&nbsp;|&nbsp;&nbsp; Dense: 191k / 321k</em></p>

</div>

The three scenes are `configs/scenes/shelf_simple.json`, `shelf.json` (medium, the scene of the goldens) and
`shelf_dense.json`.

## RTCD benchmark

The Google Benchmark apps are `rtcd_bench_discrete` (poses, modes `mesh` and `ray`: `BM_IAS`, `BM_RAY`) and
`rtcd_bench_curve` (trajectories: `BM_LINEAR`, `BM_QUAD`, `BM_MESH`, `BM_RAY` over discretized trajectories), built
by the target `AllNew` into `build/bin/apps`. They time `CollisionChecker::checkDevice` on poses or KNN trajectories
sampled from the committed inputs under `tests/data/motions` (8192 poses and their 4 nearest neighbours; nothing
is generated). The scene is a runtime option, so one executable serves all three scenes:
```bash
build/bin/apps/rtcd_bench_discrete --scene configs/scenes/shelf_dense.json --benchmark_out=<file>.json
```
App options: `--scene <scene.json | shelf | shelf_simple | shelf_dense>` (default `configs/scenes/shelf.json`),
`--robot <robot.json>` (default `configs/robots/panda.json`), `--data <dir>` (default: env `RTCD_DATA_DIR`, else
`tests/data`); every other argument goes to Google Benchmark. The environment variable `RTCD_OVERRIDES`
(`"path=value,..."`) changes the checker configs they build; `RTCD_OPTIX_CFG`, `RTCD_CURVE_GEOM_FLAG` and
`RTCD_BM_REFIT_EVERY` are the old benchmarks' knobs (`tests/support/options.h`).

The scripts in the repository root run both apps over the three scenes and write the JSON where the plotting
tool (`scripts/Benchmark/plots/plot_benchmarks.py`, see [below](#plotting-benchmark-results)) reads it:
```bash
./benchmark.sh  [--only discrete|curve] [--build <dir>] [--out <dir>] [-- <Google Benchmark args>]   # Linux
```
```bat
.\benchmark.bat [--only discrete|curve] [--build <dir>] [--out <dir>] [--no-pause] ["<GB arg>" ...]
```
- On Windows invoke the batch file **with a path** (`.\benchmark.bat`): PowerShell does not run a script from the
  current directory by its bare name. Quote Google Benchmark arguments that contain `=` (`"--benchmark_filter=BM_IAS"`):
  `cmd` splits on it.
- `--build` (default `./build`): executables are looked up in `<build>/bin/apps`, or `<build>/bin/apps/Release`
  for the Visual Studio generator.
- `--out` (default `./data/Benchmark/result`, in the local, gitignored `data/` folder):
  `<out>/discrete/{denseShelf,shelf,shelfSimple}.json` and `<out>/curve/{cur_denseShelf,cur_shelf,cur_shelfSimple}.json`.
- The benchmark names and argument ranges are those of the old stack's executables (`benchmarkShelf` ...,
  `BM_CURVE_Shelf` ..., removed), so results stay comparable with earlier ones.
- Running the whole suite takes a while (a few minutes per executable and scene). Time on an otherwise idle GPU.

## cuRobo benchmark

We use a modified version of [cuRobo](https://github.com/Ssz990220/curobo) with self-collision and other
constraints disabled for a fair comparison. The scripts are in `scripts/Benchmark/curobo/`; they read the same
cached poses / KNN indices (`Panda<N>.bin`, `Panda/1111111/<N>_knn4_idx.bin`). Note: they still look for them
under `data/motions/` (the old location; the files are now committed under `tests/data/motions/`), so copy or
link that folder, or adjust the path in the scripts (not verified here).

## FCL benchmark

Benchmarks are run through MoveIt's FCL interface. Source and setup: [rtcc_benchmark](https://github.com/Ssz990220/rtcc_benchmark).

## Plotting benchmark results

Run the RTCD and cuRobo benchmarks first. The plotting tool is `scripts/Benchmark/plots/plot_benchmarks.py`
(Python 3.9+, matplotlib >= 3.7, numpy, PyYAML). From the repository root:
```bash
pip install -r scripts/Benchmark/plots/requirements.txt
python scripts/Benchmark/plots/plot_benchmarks.py --results data/Benchmark/result --out data/Benchmark/Plots
```
(or, without installing anything, `uv run --with matplotlib --with numpy --with pyyaml python scripts/Benchmark/plots/plot_benchmarks.py ...`).
It reads `data/Benchmark/result/{discrete,curve}/` (the layout `benchmark.sh` / `.bat` write, plus the cuRobo
YAML files `discrete/kinematics_benchmark_shelf{_dense,,_simple}.yml` and
`curve/curobo_swept_benchmark_shelf{_dense,,_simple}.yml`) and writes, as PNG, PDF and SVG:
- `discrete/{DenseShelf,Shelf,ShelfSimple}`, `curve/cur_{denseShelf,shelf,shelfSimple}`: one figure per scene, time
  per pose / trajectory against batch size with cuRobo and FCL, as in the README (the former Julia figures);
- `overview_{discrete,curve}`: the three scenes side by side, time per item and the speedup over cuRobo at each
  batch size (`--reference fcl|none|<series>` picks another reference, e.g. `RobotToObs` or `Linear-16`);
- `throughput_{discrete,curve}`: checked items per second;
- `summary_{discrete,curve}.csv`: every plotted number.

The headline ratios (Two-way vs cuRobo / FCL, Linear vs cuRobo) are printed. With `--benchmark_repetitions`, the
median is plotted (`--stat mean` for the mean) and +-1 standard deviation is drawn as a band. A missing cuRobo
file, a truncated JSON or failed benchmark rows are reported and skipped. The FCL lines are constants from the
earlier FCL runs (`FCL_*_US_PER_POSE` in `bench_results.py`). `--note "RTX 4070, Linux, original code"` adds a
remark to the subtitles, which otherwise show device, stack and date from the JSON `context`; `--methods REGEX`
and `--all-series` choose the series.

To compare result sets (e.g. two machines, or the code before and after a change), give two or more
`LABEL=PATH` runs; the first is the reference. A path is a result directory (the layout above, or any directory
of Google Benchmark JSON) or a single JSON file:
```bash
python scripts/Benchmark/plots/plot_benchmarks.py compare --out data/Benchmark/Plots/compare \
    --run "4070 Linux, original code=data/Benchmark/result" --run "5080 Windows=path/to/result"
```
It writes `compare_<kind>_time` (time per item of each series, the runs overlaid), `compare_<kind>_speedup`
(t(reference) / t(run) at each batch size; above 1 = faster than the reference) and `compare_<kind>.csv`. Series
are matched by name and scenes by the file layout or the JSON `context` (`--scene` when a JSON file without a
recorded scene meets a directory of several scenes).

`python scripts/Benchmark/plots/plot_benchmarks.py --self-test` checks the tool on the small synthetic inputs in
`scripts/Benchmark/plots/testdata/` (no `data/` folder needed).
