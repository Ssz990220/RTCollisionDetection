# 📊 Benchmarking

## Scene Structure

<div align="center">

<img src="../assets/collisionScene.png" width="96%" />
<p><em>Scene complexity (triangles/edges):<br>
Simple: 15k / 22k &nbsp;&nbsp;|&nbsp;&nbsp; Medium: 71k / 107k &nbsp;&nbsp;|&nbsp;&nbsp; Dense: 191k / 321k</em></p>

</div>

## RTCD Benchmark

```bash
cd build
cmake --build . --target AllBenchmarks --config Release
cd ..
./benchmark.sh    # Linux
./benchmark.bat   # Windows
```

> RTCD generates benchmark poses. Run this before benchmarking cuRobo.

---

## cuRobo Benchmark

We use a modified version of [cuRobo](https://github.com/Ssz990220/curobo) with self-collision and constraints disabled for fair comparison.  
Scripts are available in `scripts/Benchmark/curobo/`.

---

## FCL Benchmark

Benchmarks are run through MoveIt’s FCL interface. Source and setup here: [rtcc_benchmark](https://github.com/Ssz990220/rtcc_benchmark).

---

## 📈 Plotting Benchmark Results

Ensure RTCD and cuRobo benchmarks are complete. Then:

```bash
cd scripts
```

In Julia (>= 1.10):

```julia
using Pkg
Pkg.activate(".")
] instantiate
```

Back in shell:

```bash
julia -O3 ./scripts/Benchmark/plots/discrete/resultAnalysis.jl   # Discrete results
julia -O3 ./scripts/Benchmark/plots/curve/resultAnalysisEff.jl   # Continuous results
```

Plots will be saved in `./data/Plots/`.

---