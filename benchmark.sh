#!/usr/bin/env bash
# Runs the collision-checking benchmarks for the three shelf scenes and writes Google Benchmark JSON where the
# plotting tool reads it (scripts/Benchmark/plots/plot_benchmarks.py, default --results data/Benchmark/result):
#   <out>/discrete/{denseShelf,shelf,shelfSimple}.json
#   <out>/curve/{cur_denseShelf,cur_shelf,cur_shelfSimple}.json
# Takes a few minutes per executable. Run from the repository root after building.
#
# usage: ./benchmark.sh [--only discrete|curve] [--build <dir>] [--out <dir>] [-- <Google Benchmark args>]
#   runs build/bin/apps/rtcd_bench_discrete and rtcd_bench_curve, the scene chosen with --scene (build targets
#   rtcd_bench_discrete rtcd_bench_curve, part of AllNew)
#   --only      run only the discrete or only the curve benchmarks
#   --build     build directory (default ./build); executables are looked up in <build>/bin/apps and, for
#               multi-config generators, <build>/bin/apps/Release
#   --out       result root (default ./data/Benchmark/result)
#   anything after -- (or any unknown argument) is passed to every executable, e.g. --benchmark_filter=BM_IAS
# The benchmark names are those of the old stack's executables, so results stay comparable with earlier ones.
set -euo pipefail

only=all
build=./build
out=./data/Benchmark/result
extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --only) only="$2"; shift ;;
        --build) build="$2"; shift ;;
        --out) out="$2"; shift ;;
        --) shift; extra+=("$@"); break ;;
        *) extra+=("$1") ;;
    esac
    shift
done
case "$only" in all|discrete|curve) ;; *) echo "--only must be discrete or curve" >&2; exit 2 ;; esac

sub=apps; probe=rtcd_bench_discrete
bin=""
for d in "$build/bin/$sub" "$build/bin/$sub/Release"; do
    if [ -e "$d/$probe" ] || [ -e "$d/$probe.exe" ]; then bin="$d"; break; fi
done
if [ -z "$bin" ]; then
    echo "no $probe in $build/bin/$sub[/Release]; build it first" >&2
    exit 1
fi

run() { # <exe> <output json> [args...]
    local exe="$1" json="$2"; shift 2
    echo "== $bin/$exe $* --benchmark_out=$json"
    "$bin/$exe" "$@" --benchmark_out="$json" --benchmark_out_format=json ${extra[@]+"${extra[@]}"}
}

mkdir -p "$out/discrete" "$out/curve"
if [ "$only" != curve ]; then
    run rtcd_bench_discrete "$out/discrete/denseShelf.json" --scene shelf_dense
    run rtcd_bench_discrete "$out/discrete/shelf.json" --scene shelf
    run rtcd_bench_discrete "$out/discrete/shelfSimple.json" --scene shelf_simple
fi
if [ "$only" != discrete ]; then
    run rtcd_bench_curve "$out/curve/cur_denseShelf.json" --scene shelf_dense
    run rtcd_bench_curve "$out/curve/cur_shelf.json" --scene shelf
    run rtcd_bench_curve "$out/curve/cur_shelfSimple.json" --scene shelf_simple
fi
echo "All benchmarks executed; results in $out/{discrete,curve}."
