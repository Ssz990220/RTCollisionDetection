"""Tests of the benchmark-result parser and a smoke test of the plotting tool, on the synthetic files in testdata/.

Run with plain Python from anywhere (needs numpy, matplotlib, pyyaml; the parser tests need only pyyaml):
    python scripts/Benchmark/plots/test_bench_results.py
or  python scripts/Benchmark/plots/plot_benchmarks.py --self-test

testdata/results  benchmark.sh layout, scene 'shelf' only:
    discrete/shelf.json   Google Benchmark 1.6.1 name form (no min_warmup_time), plain rows in ns, one zero-time
                          row and one error row, no rtcd_* context
    curve/cur_shelf.json  current form: min_warmup_time, aggregate rows only (mean/median/stddev/cv), rtcd_* context
    + cuRobo YAML for both (the curve one has a repeated (batch size, sweep steps) row)
testdata/run_b/disc_shelf.json  a second discrete run, one file: per-repetition rows plus aggregates
"""

import io
import math
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_results as br  # noqa: E402

DATA = os.path.join(HERE, "testdata")
RESULTS = os.path.join(DATA, "results")
RUN_B = os.path.join(DATA, "run_b", "disc_shelf.json")


def quiet():
    msgs = []
    return msgs, msgs.append


class BenchmarkNames(unittest.TestCase):
    def test_current_form(self):
        b = br.parse_benchmark_name("BM_IAS/64/min_warmup_time:4.000/real_time")
        self.assertEqual((b.family, b.targs, b.n, b.aggregate, b.key), ("IAS", (), 64, None, "IAS"))

    def test_old_google_benchmark_form(self):
        b = br.parse_benchmark_name("BM_RAY/4096/real_time")
        self.assertEqual((b.family, b.n, b.key), ("RAY", 4096, "RAY"))
        self.assertEqual(br.parse_benchmark_name("BM_RAY/8").n, 8)

    def test_template_arguments(self):
        b = br.parse_benchmark_name("BM_LINEAR<8, 8, 16>/256/min_warmup_time:1.000/real_time")
        self.assertEqual((b.family, b.targs, b.n, b.key), ("LINEAR", (8, 8, 16), 256, "LINEAR<8, 8, 16>"))
        self.assertEqual(br.parse_benchmark_name("BM_MESH<16>/1/real_time").targs, (16,))

    def test_aggregate_suffix_and_field(self):
        b = br.parse_benchmark_name("BM_QUAD<4, 8, 16>/16/min_warmup_time:1.000/real_time_median")
        self.assertEqual((b.n, b.aggregate, b.key), (16, "median", "QUAD<4, 8, 16>"))
        self.assertEqual(br.parse_benchmark_name("BM_IAS/2/real_time_stddev").aggregate, "stddev")
        self.assertEqual(br.parse_benchmark_name("BM_IAS/2_cv").aggregate, "cv")
        b = br.parse_benchmark_name("BM_IAS/2/real_time", aggregate_name="mean")
        self.assertEqual((b.n, b.aggregate), (2, "mean"))

    def test_other_families_and_extra_arguments(self):
        b = br.parse_benchmark_name("BM_RAY_NOGRAPH/16/min_warmup_time:4.000/real_time")
        self.assertEqual((b.family, b.key), ("RAY_NOGRAPH", "RAY_NOGRAPH"))
        b = br.parse_benchmark_name("BM_X/16/3/repeats:5/real_time")
        self.assertEqual((b.n, b.extra_args, b.key), (16, (3,), "X/3"))

    def test_rejects_names_without_a_batch_size(self):
        for bad in ("BM_IAS", "BM_IAS_BigO", "BM_IAS/abc/real_time", ""):
            with self.assertRaises(ValueError):
                br.parse_benchmark_name(bad)


class Units(unittest.TestCase):
    def test_conversion(self):
        self.assertAlmostEqual(br.time_to_us(1500, "ns"), 1.5)
        self.assertAlmostEqual(br.time_to_us(2, "us"), 2.0)
        self.assertAlmostEqual(br.time_to_us(3, "ms"), 3000.0)
        self.assertAlmostEqual(br.time_to_us(0.5, "s"), 5e5)
        with self.assertRaises(ValueError):
            br.time_to_us(1, "min")


def gb(rows, context=None):
    return {"context": context or {}, "benchmarks": rows}


def it(name, t, unit="us"):
    return {"name": name, "run_type": "iteration", "real_time": t, "time_unit": unit}


def agg(base, kind, t, unit="us", reps=3):
    return {"name": base + "_" + kind, "run_name": base, "run_type": "aggregate", "aggregate_name": kind,
            "aggregate_unit": "percentage" if kind == "cv" else "time", "real_time": t, "time_unit": unit,
            "repetitions": reps}


class GoogleBenchmarkJson(unittest.TestCase):
    def test_repetitions_without_aggregates(self):
        rs = br.parse_gbench(gb([it("BM_IAS/4/real_time", t) for t in (10.0, 12.0, 20.0)]))
        p = rs.series["IAS"].points[4]
        self.assertEqual((p.time_us, p.reps), (12.0, 3))
        self.assertAlmostEqual(p.std_us, math.sqrt(28.0))
        self.assertAlmostEqual(p.per_item_us, 3.0)
        rs = br.parse_gbench(gb([it("BM_IAS/4/real_time", t) for t in (10.0, 12.0, 20.0)]), stat="mean")
        self.assertAlmostEqual(rs.series["IAS"].points[4].time_us, 14.0)
        self.assertTrue(rs.has_spread)

    def test_aggregates_win_over_rows_and_cv_is_ignored(self):
        base = "BM_RAY/16/min_warmup_time:4.000/real_time"
        rows = [it(base, 100.0), it(base, 101.0), it(base, 130.0),
                agg(base, "mean", 0.110, "ms"), agg(base, "median", 0.101, "ms"), agg(base, "stddev", 0.0165, "ms"),
                agg(base, "cv", 0.15, "ms")]
        p = br.parse_gbench(gb(rows)).series["RAY"].points[16]
        self.assertAlmostEqual(p.time_us, 101.0)
        self.assertAlmostEqual(p.std_us, 16.5)
        self.assertEqual(p.reps, 3)
        self.assertAlmostEqual(br.parse_gbench(gb(rows), stat="mean").series["RAY"].points[16].time_us, 110.0)

    def test_skips_errors_zero_times_and_unparseable_rows(self):
        rows = [it("BM_IAS/1/real_time", 5.0), it("BM_IAS/2/real_time", 0.0),
                dict(it("BM_IAS/4/real_time", 1.0), error_occurred=True, error_message="boom"),
                it("BM_IAS_BigO", 1.0)]
        rs = br.parse_gbench(gb(rows))
        self.assertEqual(rs.series["IAS"].ns(), [1])
        self.assertEqual(len(rs.notes), 3)
        self.assertTrue(any("boom" in n for n in rs.notes))

    def test_kind_scene_and_errors(self):
        rs = br.parse_gbench(gb([it("BM_LINEAR<8, 8, 16>/1/real_time", 1.0)], {"rtcd_scene": "shelf_dense"}))
        self.assertEqual((rs.kind, rs.scene), ("curve", "shelf_dense"))
        rs = br.parse_gbench(gb([it("BM_IAS/1/real_time", 1.0)]))
        self.assertEqual((rs.kind, rs.scene), ("discrete", br.UNSPECIFIED_SCENE))
        with self.assertRaises(br.ResultError):
            br.parse_gbench(gb([]))
        with self.assertRaises(br.ResultError):
            br.parse_gbench({"no": "benchmarks"})
        with self.assertRaises(br.ResultError):
            br.parse_gbench(gb([it("BM_IAS/1/real_time", 1.0, unit="fortnights")]))

    def test_files(self):
        rs = br.load_gbench(os.path.join(RESULTS, "discrete", "shelf.json"))
        self.assertEqual(rs.kind, "discrete")
        self.assertAlmostEqual(rs.series["IAS"].points[4].time_us, 160.0)  # 160000 ns
        self.assertEqual(rs.series["RAY"].ns(), [1, 4, 16, 64])
        self.assertFalse(rs.has_spread)
        rs = br.load_gbench(os.path.join(RESULTS, "curve", "cur_shelf.json"))
        self.assertEqual((rs.kind, rs.scene), ("curve", "shelf"))
        self.assertEqual(sorted(rs.series), ["LINEAR<8, 8, 16>", "MESH<16>", "QUAD<4, 8, 16>", "RAY<16>"])
        self.assertTrue(rs.has_spread)
        self.assertAlmostEqual(rs.series["MESH<16>"].points[1].time_us, 150.0)
        self.assertEqual(rs.series["LINEAR<8, 8, 16>"].variant("curve"), 8)
        rs = br.load_gbench(RUN_B)
        self.assertEqual(rs.series["IAS"].points[16].reps, 3)
        tmp = tempfile.mkdtemp()
        try:
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as f:
                f.write('{"benchmarks": [ {"name": "BM_IAS/1/real_time", ')  # an interrupted run
            with self.assertRaises(br.ResultError):
                br.load_gbench(bad)
        finally:
            shutil.rmtree(tmp)


class CuroboAndFcl(unittest.TestCase):
    def test_discrete(self):
        (s,) = br.parse_curobo({"Batch Size": [2, 1], "Collision Checking": [2e-4, 1e-4], "robot": ["a", "a"]},
                               "discrete")
        self.assertEqual((s.key, s.ns(), s.external), ("curobo", [1, 2], True))
        self.assertAlmostEqual(s.points[2].time_us, 200.0)

    def test_curve_groups_and_averages(self):
        out = br.parse_curobo({"Batch Size": [1, 1, 1, 2], "Collision Checking": [1e-3, 2e-3, 4e-3, 5e-3],
                               "sweep_steps": [8, 16, 16, 8]}, "curve")
        self.assertEqual([s.key for s in out], ["curobo/8", "curobo/16"])
        self.assertAlmostEqual(out[1].points[1].time_us, 3000.0)
        self.assertEqual(out[1].points[1].reps, 2)
        self.assertEqual(out[0].variant("curve"), 8)

    def test_bad_files(self):
        with self.assertRaises(br.ResultError):
            br.parse_curobo({"Batch Size": [1, 2], "Collision Checking": [1.0]}, "discrete")
        with self.assertRaises(br.ResultError):
            br.parse_curobo({"Batch Size": [1], "Collision Checking": [1.0]}, "curve")
        with self.assertRaises(br.ResultError):
            br.parse_curobo(["not", "a", "dict"], "discrete")

    def test_fcl(self):
        (s,) = br.fcl_series("discrete", "shelf", [1, 4])
        self.assertAlmostEqual(s.points[4].per_item_us, 95000 / 4096)
        out = br.fcl_series("curve", "shelf_simple", [2])
        self.assertEqual([s.key for s in out], ["fcl/16", "fcl/32"])
        self.assertAlmostEqual(out[1].points[2].per_item_us, 2938 * 1000 / 4096)
        self.assertEqual(br.fcl_series("discrete", br.UNSPECIFIED_SCENE, [1]), [])


class TwoWay(unittest.TestCase):
    def test_discrete_sum(self):
        rows = [agg("BM_IAS/1/real_time", "median", 3.0), agg("BM_IAS/1/real_time", "stddev", 3.0),
                agg("BM_RAY/1/real_time", "median", 4.0), agg("BM_RAY/1/real_time", "stddev", 4.0),
                it("BM_RAY/2/real_time", 1.0)]
        rs = br.parse_gbench(gb(rows))
        self.assertEqual(br.add_two_way(rs), ["two_way"])
        tw = rs.series["two_way"]
        self.assertEqual(tw.ns(), [1])  # only batch sizes both directions have
        self.assertAlmostEqual(tw.points[1].time_us, 7.0)
        self.assertAlmostEqual(tw.points[1].std_us, 5.0)

    def test_curve_pairs_by_points(self):
        rows = [it("BM_MESH<16>/1/real_time", 1.0), it("BM_RAY<16>/1/real_time", 2.0),
                it("BM_MESH<32>/1/real_time", 5.0)]
        rs = br.parse_gbench(gb(rows))
        self.assertEqual(br.add_two_way(rs), ["two_way/16"])
        self.assertAlmostEqual(rs.series["two_way/16"].points[1].time_us, 3.0)
        self.assertEqual(rs.series["two_way/16"].variant("curve"), 16)


class Discover(unittest.TestCase):
    def test_standard_layout(self):
        msgs, log = quiet()
        sets = br.discover(RESULTS, log=log)
        self.assertEqual([(r.kind, r.scene) for r in sets], [("discrete", "shelf"), ("curve", "shelf")])
        d, c = sets
        self.assertEqual(sorted(d.series), ["IAS", "RAY", "curobo", "fcl", "two_way"])
        self.assertIn("curobo/16", c.series)
        self.assertIn("two_way/16", c.series)
        self.assertEqual(c.series["curobo/16"].points[1].reps, 2)
        self.assertTrue(any("synthetic failure" in m for m in msgs))
        sets = br.discover(RESULTS, externals=False, log=log)
        self.assertFalse(any(s.external for r in sets for s in r.series.values()))

    def test_missing_curobo_and_broken_files_degrade(self):
        tmp = tempfile.mkdtemp()
        try:
            shutil.copytree(RESULTS, os.path.join(tmp, "r"))
            os.remove(os.path.join(tmp, "r", "curve", "curobo_swept_benchmark_shelf.yml"))
            with open(os.path.join(tmp, "r", "discrete", "shelfSimple.json"), "w") as f:
                f.write("{ truncated")
            msgs, log = quiet()
            sets = br.discover(os.path.join(tmp, "r"), log=log)
            self.assertEqual(len(sets), 2)
            self.assertTrue(any("no cuRobo result" in m for m in msgs))
            self.assertTrue(any("not valid JSON" in m for m in msgs))
            # a directory that is not in the standard layout is searched for JSON
            sets = br.discover(os.path.join(DATA, "run_b"), log=log)
            self.assertEqual([(r.kind, r.scene) for r in sets], [("discrete", "shelf")])
            with self.assertRaises(br.ResultError):
                br.discover(os.path.join(tmp, "does-not-exist"), log=log)
        finally:
            shutil.rmtree(tmp)


class PlottingSmoke(unittest.TestCase):
    """Renders every figure from testdata/ (skipped without matplotlib)."""

    @classmethod
    def setUpClass(cls):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("matplotlib not installed")

    def run_tool(self, argv):
        import contextlib

        import plot_benchmarks

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = plot_benchmarks.main(argv)
        return code, out.getvalue()

    def test_figures_and_compare(self):
        tmp = tempfile.mkdtemp()
        try:
            code, out = self.run_tool(["--results", RESULTS, "--out", tmp, "--formats", "png,svg,pdf",
                                       "--dpi", "60"])
            self.assertEqual(code, 0, out)
            for rel in ("discrete/Shelf.png", "discrete/Shelf.svg", "discrete/Shelf.pdf", "curve/cur_shelf.png",
                        "overview_discrete.png", "overview_curve.png", "throughput_curve.png",
                        "summary_discrete.csv", "summary_curve.csv"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, rel)), rel)
            self.assertIn("Two-way vs cuRobo", out)
            code, out = self.run_tool(["compare", "--run", "A=" + RESULTS, "--run", "B=" + RUN_B, "--out",
                                       os.path.join(tmp, "cmp"), "--formats", "png", "--dpi", "60"])
            self.assertEqual(code, 0, out)
            for rel in ("compare_discrete_time.png", "compare_discrete_speedup.png", "compare_discrete.csv"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, "cmp", rel)), rel)
            with open(os.path.join(tmp, "cmp", "compare_discrete.csv")) as f:
                rows = [r for r in f.read().splitlines()[1:] if ",B," in r and "RobotToObs" in r]
            self.assertEqual(len(rows), 4)
            # B is RAY 45 us * sqrt(n) vs A 50 us * sqrt(n): 50/45 faster at every n
            self.assertTrue(all(abs(float(r.split(",")[-1]) - 50 / 45) < 1e-3 for r in rows), rows)
            code, out = self.run_tool(["compare", "--run", "A=" + RESULTS, "--out", tmp])
            self.assertEqual(code, 2)
            code, out = self.run_tool(["--results", os.path.join(tmp, "nothing-here"), "--out", tmp])
            self.assertEqual(code, 2)
            self.assertIn("error", out)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
