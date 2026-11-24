"""Reading RTCD benchmark results: Google Benchmark JSON, cuRobo YAML, and the FCL reference constants.

Everything that interprets result files lives here, so it can be tested without matplotlib
(`test_bench_results.py`). The plotting front end is `plot_benchmarks.py`.

Inputs
------
* Google Benchmark JSON written by `rtcd_bench_discrete` / `rtcd_bench_curve` (and by the old stack's
  executables). Benchmark names look like
      BM_IAS/64/min_warmup_time:4.000/real_time            discrete, 64 poses per batch
      BM_LINEAR<8, 8, 16>/256/min_warmup_time:1.000/real_time   curve, 256 trajectories per batch
  Older Google Benchmark versions (1.6.x) omit the `/min_warmup_time:...` part; both forms are accepted.
  Runs with `--benchmark_repetitions` produce aggregate rows (`..._mean`, `_median`, `_stddev`, `_cv`), with or
  without the per-repetition rows. Times are converted to microseconds from each row's `time_unit`.
* cuRobo results (`scripts/Benchmark/curobo/*.py`): YAML with the lists `Batch Size`, `Collision Checking`
  (seconds per batch) and, for the swept (curve) benchmark, `sweep_steps`.
* FCL (CPU, through MoveIt): no files; the per-pose times hard-coded in the former Julia scripts are kept in
  `FCL_DISCRETE_US_PER_POSE` / `FCL_CURVE_US_PER_POSE`.

The standard directory layout (what `benchmark.sh` / `benchmark.bat` write, plus the cuRobo files) is
    <results>/discrete/{denseShelf,shelf,shelfSimple}.json   <results>/discrete/kinematics_benchmark_shelf{_dense,,_simple}.yml
    <results>/curve/cur_{denseShelf,shelf,shelfSimple}.json  <results>/curve/curobo_swept_benchmark_shelf{_dense,,_simple}.yml
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------------------------
# Scenes and reference constants
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    id: str  # rtcd_scene in the JSON context, and the configs/scenes/<id>.json stem
    stem: str  # result file stem used by benchmark.sh / .bat
    title: str  # display name
    curobo_suffix: str  # suffix of the cuRobo YAML file name


SCENES = (
    Scene("shelf_dense", "denseShelf", "Dense shelf", "_dense"),
    Scene("shelf", "shelf", "Medium shelf", ""),
    Scene("shelf_simple", "shelfSimple", "Simple shelf", "_simple"),
)
SCENE_BY_ID = {s.id: s for s in SCENES}
UNSPECIFIED_SCENE = "unspecified"

KINDS = ("discrete", "curve")
ITEM_NOUN = {"discrete": "pose", "curve": "trajectory"}
ITEM_PLURAL = {"discrete": "poses", "curve": "trajectories"}

# FCL reference (CPU), copied from the Julia scripts: total time for 4096 poses (discrete), and for 4096
# trajectories of 32 poses (curve). A curve trajectory with k discretization points costs k poses.
FCL_DISCRETE_US_PER_POSE = {
    "shelf_dense": 105000 / 4096,
    "shelf": 95000 / 4096,
    "shelf_simple": 70000 / 4096,
}
FCL_CURVE_US_PER_POSE = {
    "shelf_dense": 4060 * 1000 / 4096 / 32,
    "shelf": 4065 * 1000 / 4096 / 32,
    "shelf_simple": 2938 * 1000 / 4096 / 32,
}
FCL_CURVE_POSES_PER_TRAJECTORY = (16, 32)


def scene_title(scene_id):
    s = SCENE_BY_ID.get(scene_id)
    if s:
        return s.title
    return "Scene not recorded" if scene_id == UNSPECIFIED_SCENE else scene_id


def scene_order(scene_id):
    ids = [s.id for s in SCENES]
    return (ids.index(scene_id), "") if scene_id in ids else (len(ids), scene_id)


class ResultError(Exception):
    """A result file that cannot be used; the message says which file and why."""


# --------------------------------------------------------------------------------------------------------------
# Benchmark names and units
# --------------------------------------------------------------------------------------------------------------

AGGREGATE_NAMES = ("mean", "median", "stddev", "cv", "min", "max")
_AGG_SUFFIX = re.compile(r"_(%s)$" % "|".join(AGGREGATE_NAMES))
_NAME = re.compile(r"^(?P<family>[^/<>]+?)(?:<(?P<targs>[^<>]*)>)?/(?P<n>\d+)(?:/(?P<rest>.*))?$")

_TIME_UNIT_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}


@dataclass(frozen=True)
class BenchName:
    family: str  # "IAS", "LINEAR", "RAY_NOGRAPH" (the "BM_" prefix removed)
    targs: tuple  # template arguments, ints where numeric: (8, 8, 16)
    n: int  # first benchmark argument: poses (discrete) or trajectories (curve) per batch
    extra_args: tuple  # further numeric arguments, if any (kept so different runs never merge)
    aggregate: str | None  # "median", "mean", ... for aggregate rows, else None

    @property
    def key(self):
        """Identity of the series this row belongs to (everything but n and the aggregate)."""
        s = self.family
        if self.targs:
            s += "<" + ", ".join(str(a) for a in self.targs) + ">"
        if self.extra_args:
            s += "/" + "/".join(str(a) for a in self.extra_args)
        return s


def _maybe_int(s):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return s


def parse_benchmark_name(name, aggregate_name=None):
    """Parse a Google Benchmark row name.

    Accepts `BM_X/<n>/...`, `BM_X<a, b, c>/<n>/...`, with or without `min_warmup_time:...`, `real_time`,
    `repeats:...` and with an aggregate suffix (`_median`, ...). `aggregate_name` (the row's field of that name)
    takes precedence over the suffix. Raises ValueError for names without a numeric first argument.
    """
    s = name.strip()
    agg = aggregate_name or None
    m = _AGG_SUFFIX.search(s)
    if m:
        agg = agg or m.group(1)
        s = s[: m.start()]
    elif agg and s.endswith("_" + agg):
        s = s[: -len(agg) - 1]
    m = _NAME.match(s)
    if not m:
        raise ValueError("benchmark name %r has no '<family>/<n>' form" % name)
    family = m.group("family")
    if family.startswith("BM_"):
        family = family[3:]
    targs = tuple(_maybe_int(a) for a in m.group("targs").split(",")) if m.group("targs") else ()
    extra = []
    for part in (m.group("rest") or "").split("/"):
        if part.isdigit():
            extra.append(int(part))
        # everything else (min_warmup_time:4.000, real_time, repeats:5, threads:1, ...) does not identify a series
    return BenchName(family, targs, int(m.group("n")), tuple(extra), agg)


def time_to_us(value, unit):
    """Convert a Google Benchmark time in `unit` (ns, us, ms, s) to microseconds."""
    try:
        return float(value) * _TIME_UNIT_TO_US[unit]
    except KeyError:
        raise ValueError("unknown time_unit %r" % (unit,)) from None


# --------------------------------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------------------------------


@dataclass
class Point:
    n: int  # items per batch
    time_us: float  # central value (median or mean, see Series.stat) per batch, in microseconds
    std_us: float | None = None  # standard deviation over repetitions, if known
    reps: int = 1

    @property
    def per_item_us(self):
        return self.time_us / self.n

    @property
    def std_per_item_us(self):
        return None if self.std_us is None else self.std_us / self.n


@dataclass
class Series:
    key: str  # "LINEAR<8, 8, 16>", "IAS", "curobo/8", "fcl/16", "two_way/16"
    family: str  # "LINEAR", "IAS", "CUROBO", "FCL", "TWO_WAY", ...
    targs: tuple = ()
    points: dict = field(default_factory=dict)  # n -> Point
    external: bool = False  # cuRobo / FCL (not measured by the RTCD benchmark apps)

    def ns(self):
        return sorted(self.points)

    def variant(self, kind):
        """Number of control / discretization points of a curve series (its first template argument), else None."""
        if kind != "curve":
            return None
        if self.targs and isinstance(self.targs[0], int):
            return self.targs[0]
        return None


@dataclass
class ResultSet:
    kind: str  # "discrete" | "curve"
    scene: str  # scene id, or UNSPECIFIED_SCENE
    source: str  # file it came from
    context: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)  # key -> Series
    notes: list = field(default_factory=list)  # warnings collected while reading
    stat: str = "median"
    has_spread: bool = False

    def item_noun(self):
        return ITEM_NOUN[self.kind]


# --------------------------------------------------------------------------------------------------------------
# Google Benchmark JSON
# --------------------------------------------------------------------------------------------------------------


def infer_kind(series_keys):
    """'curve' if any series has template arguments (BM_LINEAR<...>, BM_MESH<16>), else 'discrete'."""
    return "curve" if any("<" in k for k in series_keys) else "discrete"


def parse_gbench(data, source="<memory>", stat="median", kind=None, scene=None):
    """Build a ResultSet from parsed Google Benchmark JSON (a dict with 'benchmarks').

    stat: 'median' or 'mean', the central value used when aggregates or several repetitions exist.
    kind / scene: override what is inferred from the names / the context (rtcd_scene) / nothing.
    """
    if stat not in ("median", "mean"):
        raise ValueError("stat must be 'median' or 'mean'")
    if not isinstance(data, dict) or not isinstance(data.get("benchmarks"), list):
        raise ResultError("%s: not Google Benchmark JSON (no 'benchmarks' list)" % source)
    context = data.get("context") or {}
    notes = []
    runs = {}  # (key, n) -> list of per-repetition times (us)
    aggs = {}  # (key, n) -> {aggregate name: value}
    reps = {}
    names = {}  # key -> BenchName
    for row in data["benchmarks"]:
        rname = row.get("name", "?")
        if row.get("error_occurred"):
            notes.append("skipped %s: %s" % (rname, row.get("error_message", "error")))
            continue
        run_type = row.get("run_type", "iteration")
        agg_name = row.get("aggregate_name") if run_type == "aggregate" else None
        try:
            bn = parse_benchmark_name(row.get("run_name") or rname, agg_name) if agg_name else parse_benchmark_name(rname)
        except ValueError as e:
            notes.append("skipped row: %s" % e)
            continue
        if run_type == "aggregate" and bn.aggregate is None:
            notes.append("skipped aggregate row without a name: %s" % rname)
            continue
        names.setdefault(bn.key, bn)
        k = (bn.key, bn.n)
        if bn.aggregate is not None:
            if bn.aggregate == "cv" or row.get("aggregate_unit") == "percentage":
                continue  # coefficient of variation: a ratio, not a time; stddev carries the same information
            if "real_time" not in row:
                continue
            try:
                aggs.setdefault(k, {})[bn.aggregate] = time_to_us(row["real_time"], row.get("time_unit", "ns"))
            except ValueError as e:
                raise ResultError("%s: %s in row %s" % (source, e, rname)) from None
            reps[k] = max(reps.get(k, 1), int(row.get("repetitions", 1) or 1))
        else:
            if "real_time" not in row:
                notes.append("skipped %s: no real_time" % rname)
                continue
            try:
                t = time_to_us(row["real_time"], row.get("time_unit", "ns"))
            except ValueError as e:
                raise ResultError("%s: %s in row %s" % (source, e, rname)) from None
            if t == 0 or not math.isfinite(t):  # the Julia scripts treated 0 as missing too
                notes.append("skipped %s: real_time %r" % (rname, row["real_time"]))
                continue
            runs.setdefault(k, []).append(t)

    series = {}
    has_spread = False
    for k in sorted(set(runs) | set(aggs), key=lambda x: (x[0], x[1])):
        key, n = k
        a = aggs.get(k, {})
        r = runs.get(k, [])
        center = std = None
        nrep = 1
        if stat in a or ("median" in a or "mean" in a):
            center = a.get(stat, a.get("median", a.get("mean")))
            std = a.get("stddev")
            nrep = reps.get(k, max(len(r), 1))
        elif r:
            center = statistics.median(r) if stat == "median" else statistics.fmean(r)
            std = statistics.stdev(r) if len(r) >= 2 else None
            nrep = len(r)
        if center is None or center == 0 or not math.isfinite(center):
            notes.append("no usable time for %s/%d" % (key, n))
            continue
        bn = names[key]
        s = series.setdefault(key, Series(key, bn.family, bn.targs))
        s.points[n] = Point(n, center, std, nrep)
        has_spread = has_spread or std is not None

    if not series:
        raise ResultError("%s: no usable benchmark rows" % source)
    kind = kind or infer_kind(series)
    scene = scene or context.get("rtcd_scene") or UNSPECIFIED_SCENE
    return ResultSet(kind, scene, source, context, series, notes, stat, has_spread)


def load_gbench(path, stat="median", kind=None, scene=None):
    """Read a Google Benchmark JSON file into a ResultSet (ResultError on unreadable / unusable files)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise ResultError("%s: cannot read (%s)" % (path, e.strerror or e)) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ResultError("%s: not valid JSON (%s); was the run interrupted?" % (path, e)) from None
    return parse_gbench(data, source=path, stat=stat, kind=kind, scene=scene)


# --------------------------------------------------------------------------------------------------------------
# cuRobo YAML and FCL
# --------------------------------------------------------------------------------------------------------------


def parse_curobo(data, kind, source="<memory>"):
    """cuRobo results -> list of Series (seconds per batch -> us). Curve results are grouped by sweep_steps and
    repeated (batch size, sweep_steps) rows are averaged, as the Julia script did."""
    if not isinstance(data, dict) or "Batch Size" not in data or "Collision Checking" not in data:
        raise ResultError("%s: expected the lists 'Batch Size' and 'Collision Checking'" % source)
    sizes, times = data["Batch Size"], data["Collision Checking"]
    steps = data.get("sweep_steps") if kind == "curve" else None
    if kind == "curve" and steps is None:
        raise ResultError("%s: a swept (curve) cuRobo result needs 'sweep_steps'" % source)
    if len(sizes) != len(times) or (steps is not None and len(steps) != len(sizes)):
        raise ResultError("%s: the result lists have different lengths" % source)
    groups = {}
    for i, (n, t) in enumerate(zip(sizes, times)):
        v = int(steps[i]) if steps is not None else None
        groups.setdefault(v, {}).setdefault(int(n), []).append(float(t) * 1e6)
    out = []
    for v in sorted(groups, key=lambda x: (x is None, x)):
        key = "curobo" if v is None else "curobo/%d" % v
        s = Series(key, "CUROBO", (v,) if v is not None else (), external=True)
        for n, ts in sorted(groups[v].items()):
            s.points[n] = Point(n, statistics.fmean(ts), statistics.stdev(ts) if len(ts) > 1 else None, len(ts))
        out.append(s)
    return out


def load_curobo(path, kind):
    try:
        import yaml
    except ImportError:
        raise ResultError("%s: PyYAML is not installed (pip install pyyaml); cuRobo results skipped" % path) from None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        raise ResultError("%s: cannot read (%s)" % (path, e.strerror or e)) from None
    except yaml.YAMLError as e:
        raise ResultError("%s: not valid YAML (%s)" % (path, e)) from None
    return parse_curobo(data, kind, source=path)


def fcl_series(kind, scene, ns):
    """FCL reference series (constant per item) at the batch sizes `ns`, or [] for an unknown scene."""
    if kind == "discrete":
        per_pose = FCL_DISCRETE_US_PER_POSE.get(scene)
        if per_pose is None:
            return []
        s = Series("fcl", "FCL", (), external=True)
        s.points = {n: Point(n, per_pose * n) for n in ns}
        return [s]
    per_pose = FCL_CURVE_US_PER_POSE.get(scene)
    if per_pose is None:
        return []
    out = []
    for k in FCL_CURVE_POSES_PER_TRAJECTORY:
        s = Series("fcl/%d" % k, "FCL", (k,), external=True)
        s.points = {n: Point(n, per_pose * k * n) for n in ns}
        out.append(s)
    return out


# --------------------------------------------------------------------------------------------------------------
# Derived series
# --------------------------------------------------------------------------------------------------------------


def _sum_points(a, b):
    out = {}
    for n in sorted(set(a.points) & set(b.points)):
        pa, pb = a.points[n], b.points[n]
        std = math.hypot(pa.std_us, pb.std_us) if pa.std_us is not None and pb.std_us is not None else None
        out[n] = Point(n, pa.time_us + pb.time_us, std, min(pa.reps, pb.reps))
    return out


def add_two_way(rs):
    """Add the 'Two-way' series of the Julia scripts: discrete BM_IAS + BM_RAY (ObsToRobot + RobotToObs), curve
    BM_MESH<k> + BM_RAY<k> (both directions on k discretized poses). Std devs add in quadrature. Returns the
    keys added."""
    added = []
    if rs.kind == "discrete":
        pairs = [("IAS", "RAY", "two_way", ())] if "IAS" in rs.series and "RAY" in rs.series else []
    else:
        pairs = []
        for key, s in rs.series.items():
            if s.family == "MESH" and len(s.targs) == 1:
                ray = "RAY<%s>" % s.targs[0]
                if ray in rs.series:
                    pairs.append((key, ray, "two_way/%s" % s.targs[0], s.targs))
    for ka, kb, key, targs in pairs:
        pts = _sum_points(rs.series[ka], rs.series[kb])
        if pts:
            rs.series[key] = Series(key, "TWO_WAY", targs, pts)
            added.append(key)
    return added


# --------------------------------------------------------------------------------------------------------------
# Discovering result files
# --------------------------------------------------------------------------------------------------------------


def standard_paths(root, kind, scene):
    """(benchmark JSON, cuRobo YAML) paths of the standard layout for a kind and a Scene."""
    if kind == "discrete":
        return (os.path.join(root, "discrete", scene.stem + ".json"),
                os.path.join(root, "discrete", "kinematics_benchmark_shelf%s.yml" % scene.curobo_suffix))
    return (os.path.join(root, "curve", "cur_%s.json" % scene.stem),
            os.path.join(root, "curve", "curobo_swept_benchmark_shelf%s.yml" % scene.curobo_suffix))


def _scene_from_filename(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem[4:] if stem.startswith("cur_") else stem
    for s in SCENES:
        if stem in (s.stem, s.id):
            return s.id
    return None


def discover(path, stat="median", externals=True, log=print):
    """Load every result under `path` (a result directory or one JSON file).

    A directory in the standard layout gives up to 3 discrete and 3 curve ResultSets, each with its cuRobo
    and FCL reference series attached when available (externals=True). Any other directory is searched
    recursively for Google Benchmark JSON. Problems are reported through `log` and skipped; returns a list of
    ResultSets, sorted by kind and scene."""
    sets = []

    def note(msg):
        log("  note: " + msg)

    if os.path.isfile(path):
        files = [(path, None, None, None)]
    elif os.path.isdir(path):
        files = []
        for kind in KINDS:
            for sc in SCENES:
                js, yml = standard_paths(path, kind, sc)
                if os.path.isfile(js):
                    files.append((js, kind, sc.id, yml))
        if not files:
            for js in sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)):
                files.append((js, None, None, None))
    else:
        raise ResultError("%s: no such file or directory" % path)

    for js, kind, scene, yml in files:
        try:
            rs = load_gbench(js, stat=stat, kind=kind)
        except ResultError as e:
            note(str(e))
            continue
        if scene and rs.scene == UNSPECIFIED_SCENE:
            rs.scene = scene
        elif rs.scene == UNSPECIFIED_SCENE:
            rs.scene = _scene_from_filename(js) or UNSPECIFIED_SCENE
        if scene and rs.scene != scene:
            note("%s: context says scene %r, the file name says %r; using %r" % (js, rs.scene, scene, rs.scene))
        for msg in rs.notes:
            note("%s: %s" % (os.path.basename(js), msg))
        add_two_way(rs)
        if externals:
            if yml:
                if os.path.isfile(yml):
                    try:
                        for s in load_curobo(yml, rs.kind):
                            rs.series[s.key] = s
                    except ResultError as e:
                        note(str(e))
                else:
                    note("no cuRobo result %s; plotting without cuRobo" % yml)
            ns = sorted({n for s in rs.series.values() for n in s.points})
            for s in fcl_series(rs.kind, rs.scene, ns):
                rs.series[s.key] = s
        sets.append(rs)
    sets.sort(key=lambda r: (KINDS.index(r.kind), scene_order(r.scene), r.source))
    return sets
