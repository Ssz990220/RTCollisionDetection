#!/usr/bin/env python3
"""Plot RTCD benchmark results (Google Benchmark JSON) against cuRobo and FCL, and compare result sets.

Run from the repository root (needs matplotlib, numpy, pyyaml: scripts/Benchmark/plots/requirements.txt):

  # the figures of the README (one per scene, as the former Julia scripts made them) plus the overview figures
  python scripts/Benchmark/plots/plot_benchmarks.py --results data/Benchmark/result --out data/Benchmark/Plots

  # compare result sets (directories in the standard layout, or single JSON files), first one is the reference
  python scripts/Benchmark/plots/plot_benchmarks.py compare --run "4070 Linux, old code=data/Benchmark/result" \
      --run "5080 Windows, new code=path/to/result" --out data/Benchmark/Plots/compare

  # check the tool itself on the small synthetic inputs in testdata/ (no data/ folder needed)
  python scripts/Benchmark/plots/plot_benchmarks.py --self-test

Outputs (figures in every --formats format, png/pdf/svg by default):
  <out>/discrete/{DenseShelf,Shelf,ShelfSimple}  <out>/curve/cur_{denseShelf,shelf,shelfSimple}
      the per-scene figures of the former Julia scripts: time per item vs batch size, FCL and cuRobo included
  <out>/overview_<kind>     all methods, the three scenes side by side: time per item, and speedup over a
                            reference method (cuRobo by default) at each batch size; bands = +-1 std dev when
                            the JSON has repetitions
  <out>/throughput_<kind>   checked items per second vs batch size
  <out>/summary_<kind>.csv  every plotted number
  compare: <out>/compare_<kind>_time, compare_<kind>_speedup, compare_<kind>.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bench_results as br  # noqa: E402

# --------------------------------------------------------------------------------------------------------------
# Styling: one colour / marker per method in every figure (Okabe-Ito, colour-blind safe), one line style per
# number of control / discretization points (curve methods), markers so lines stay apart in grayscale.
# --------------------------------------------------------------------------------------------------------------

INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"

# method id -> (label, colour, marker, fixed line style or None)
METHODS = {
    "robot_to_obs": ("RobotToObs", "#0072B2", "o", None),
    "obs_to_robot": ("ObsToRobot", "#009E73", "s", None),
    "linear": ("Linear", "#0072B2", "o", None),
    "quadratic": ("Quadratic", "#009E73", "s", None),
    "cubic": ("Cubic", "#56B4E9", "v", None),
    "mesh": ("Discretized ObsToRobot", "#E69F00", "P", None),
    "ray": ("Discretized RobotToObs", "#56B4E9", "X", None),
    "two_way": ("Two-way", "#CC79A7", "D", None),
    "curobo": ("cuRobo", "#D55E00", "^", "--"),
    "fcl": ("FCL (CPU)", "#000000", None, "--"),
}
DISCRETE_FAMILY = {"RAY": "robot_to_obs", "IAS": "obs_to_robot"}
CURVE_FAMILY = {"LINEAR": "linear", "QUAD": "quadratic", "CUBIC": "cubic", "MESH": "mesh", "RAY": "ray"}
COMMON_FAMILY = {"TWO_WAY": "two_way", "CUROBO": "curobo", "FCL": "fcl"}
# methods of the Julia figures (others, e.g. the curve MESH / RAY halves of Two-way, only with --all-series)
DEFAULT_METHODS = {
    "discrete": ["fcl", "curobo", "obs_to_robot", "robot_to_obs", "two_way"],
    "curve": ["fcl", "curobo", "two_way", "linear", "quadratic", "cubic"],
}
# Families not known above get these, in order of first appearance (sorted by name): Paul Tol's muted set.
EXTRA_COLORS = ["#882255", "#44AA99", "#999933", "#332288", "#AA4499", "#117733", "#88CCEE", "#DDCC77"]
EXTRA_MARKERS = ["h", "p", "*", "<", ">", "d", "8", "H"]
VARIANT_LS = {4: "-.", 8: "--", 16: "-", 32: ":"}
RUN_COLORS = ["#000000", "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
RUN_MARKERS = ["o", "s", "^", "D", "v", "P"]
RUN_LS = ["-", "--", "-.", ":", "-", "--"]

_extra_assigned = {}


def method_of(kind, s):
    fam = s.family
    if fam in COMMON_FAMILY:
        return COMMON_FAMILY[fam]
    table = DISCRETE_FAMILY if kind == "discrete" else CURVE_FAMILY
    if fam in table and (kind == "discrete") == (not s.targs):
        return table[fam]
    return "x:" + fam  # unknown family (e.g. RAY_NOGRAPH, LINEAR_SW): styled from EXTRA_*


def method_style(mid):
    """(label, colour, marker, fixed line style or None) of a method id."""
    if mid in METHODS:
        return METHODS[mid]
    if mid not in _extra_assigned:
        i = len(_extra_assigned)
        fam = mid[2:]
        _extra_assigned[mid] = (fam, EXTRA_COLORS[i % len(EXTRA_COLORS)], EXTRA_MARKERS[i % len(EXTRA_MARKERS)], None)
    return _extra_assigned[mid]


def variant_ls(v):
    return VARIANT_LS.get(v, (0, (5, 1, 1, 1, 1, 1)))


def series_label(kind, s, ambiguous=False):
    mid = method_of(kind, s)
    label = method_style(mid)[0]
    if kind == "curve":
        v = s.variant(kind)
        if ambiguous and s.targs and not s.external and s.family != "TWO_WAY":
            label += "<" + ", ".join(str(a) for a in s.targs) + ">"
        elif v is not None:
            label += "-%d" % v
    elif s.targs and mid.startswith("x:"):
        label += "<" + ", ".join(str(a) for a in s.targs) + ">"
    return label


ALT_LS = [(0, (1, 1.2)), (0, (4, 1.5, 1, 1.5, 1, 1.5)), (0, (7, 2))]


def series_style(kind, s, peers=()):
    """matplotlib keyword arguments for a series. Series of the same method and variant among `peers` (e.g.
    BM_LINEAR<16, 16, 16> and BM_LINEAR<16, 16, 128>) get a different line style and hollow markers."""
    mid = method_of(kind, s)
    _, color, marker, fixed_ls = method_style(mid)
    ls = variant_ls(s.variant(kind)) if kind == "curve" and s.variant(kind) is not None else (fixed_ls or "-")
    st = dict(color=color, marker=marker, linestyle=ls)
    twins = sorted(p.key for p in peers if method_of(kind, p) == mid and p.variant(kind) == s.variant(kind))
    if s.key in twins and twins.index(s.key) > 0:
        st["linestyle"] = ALT_LS[(twins.index(s.key) - 1) % len(ALT_LS)]
        st["markerfacecolor"] = "white"
    return st


def label_series(rs):
    """{key: label}; labels made unique (full template arguments) where two series would share one."""
    plain = {k: series_label(rs.kind, s) for k, s in rs.series.items()}
    counts = {}
    for lab in plain.values():
        counts[lab] = counts.get(lab, 0) + 1
    return {k: (series_label(rs.kind, rs.series[k], True) if counts[plain[k]] > 1 else plain[k]) for k in plain}


def series_order(kind, s):
    mid = method_of(kind, s)
    order = list(METHODS)
    return (order.index(mid) if mid in order else len(order), mid, s.variant(kind) or 0, s.key)


def select_series(rs, all_series=False, pattern=None):
    """Series to plot, in a stable order: the Julia figures' methods unless all_series; `pattern` filters by
    label (regular expression, case-insensitive)."""
    import re

    labels = label_series(rs)
    keep = []
    for k, s in rs.series.items():
        mid = method_of(rs.kind, s)
        if not all_series and not mid.startswith("x:") and mid not in DEFAULT_METHODS[rs.kind]:
            continue
        # the curve figure skipped cuRobo with 4 sweep steps
        if rs.kind == "curve" and mid == "curobo" and s.variant("curve") == 4 and not all_series:
            continue
        if pattern and not re.search(pattern, labels[k], re.I):
            continue
        keep.append(s)
    keep.sort(key=lambda s: series_order(rs.kind, s))
    return keep, labels


# --------------------------------------------------------------------------------------------------------------
# matplotlib helpers
# --------------------------------------------------------------------------------------------------------------


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return plt


def set_batch_axis(ax, ns, label=None, max_labels=8):
    """log2 x axis with ticks at the batch sizes; every other label hidden when crowded."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_xscale("log", base=2)
    ns = sorted(ns)
    if not ns:
        return
    step = 1
    while len(ns) / step > max_labels:
        step *= 2
    labelled = set(ns[::-1][::step])  # always label the largest batch
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%d" % round(v)) if round(v) in labelled else ""))
    ax.set_xlim(ns[0] / 1.25, ns[-1] * 1.25)
    if label:
        ax.set_xlabel(label)


def set_log_y(ax):
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))


def fmt_factor(v, _=None):
    return ("%.3g" % v) + "×"


def speedup_axis(ax, ratios, ref_label):
    """log y axis around 1 with 'faster' / 'slower' regions shaded and named."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    finite = [r for r in ratios if r and r > 0 and math.isfinite(r)]
    lo = min(finite + [0.8]) / 1.5
    hi = max(finite + [1.25]) * 1.5
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.axhspan(1, hi, color="#0072B2", alpha=0.06, lw=0, zorder=0)
    ax.axhspan(lo, 1, color="#D55E00", alpha=0.06, lw=0, zorder=0)
    ax.axhline(1, color=MUTED, lw=1.0, zorder=1)
    ax.text(0.02, 0.97, "faster than %s" % ref_label, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            color=MUTED, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.text(0.02, 0.03, "slower than %s" % ref_label, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8, color=MUTED, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    decades = math.log10(hi / lo)
    if decades < 0.8:  # a narrow range: explicit factors so more than the 1x tick shows
        from matplotlib.ticker import FixedLocator

        fine = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        ax.yaxis.set_major_locator(FixedLocator([v for v in fine if lo <= v <= hi][::1 if decades < 0.5 else 2]
                                                + ([1.0] if decades >= 0.5 else [])))
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if decades < 1.6 else
                                              (1.0, 3.0) if decades < 3 else (1.0,)))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_factor))
    ax.yaxis.set_minor_formatter(NullFormatter())


def short_path(p):
    try:
        r = os.path.relpath(p)
    except ValueError:  # another drive on Windows
        return p
    return p if r.startswith("..") else r


def save(fig, base, formats, dpi, log=print):
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    for fmt in formats:
        fig.savefig("%s.%s" % (base, fmt), dpi=dpi if fmt == "png" else None)
    log("  wrote %s.{%s}" % (short_path(base), ",".join(formats)))


def context_line(sets, extra=None):
    """'NVIDIA GeForce RTX 5080 · stack new · 2026-11-23' from the JSON context of the result sets."""
    parts = []
    ctx = sets[0].context if sets else {}

    def same(key):
        vals = {r.context.get(key) for r in sets}
        return ctx.get(key) if len(vals) == 1 else None

    dev = same("rtcd_device")
    parts.append(dev if dev else ("host %s" % same("host_name") if same("host_name") else None))
    if same("rtcd_stack"):
        parts.append("%s stack" % same("rtcd_stack"))
    if same("rtcd_robot"):
        parts.append("robot %s" % same("rtcd_robot"))
    dates = sorted({(r.context.get("date") or "")[:10] for r in sets} - {""})
    if dates:
        parts.append(dates[0] if len(dates) == 1 else "%s to %s" % (dates[0], dates[-1]))
    reps = {p.reps for r in sets for s in r.series.values() if not s.external for p in s.points.values()}
    if reps and max(reps) > 1:
        parts.append("%s of %d repetitions" % (sets[0].stat, max(reps)))
    if extra:
        parts.append(extra)
    return " · ".join(p for p in parts if p)


def draw_series(ax, xs, ys, std, style, label, lw=1.6, ms=4.0, band=True):
    ax.plot(xs, ys, label=label, lw=lw, ms=ms, markeredgewidth=1.0 if "markerfacecolor" in style else 0, **style)
    if band and std is not None and any(v is not None for v in std):
        lo = [y - (e or 0) for y, e in zip(ys, std)]
        hi = [y + (e or 0) for y, e in zip(ys, std)]
        lo = [max(v, y * 0.05) for v, y in zip(lo, ys)]
        ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.18, lw=0)


def per_item(s, kind_noun=None):
    xs = s.ns()
    ys = [s.points[n].per_item_us for n in xs]
    std = [s.points[n].std_per_item_us for n in xs]
    return xs, ys, (std if any(v is not None for v in std) else None)


# --------------------------------------------------------------------------------------------------------------
# Figures of the former Julia scripts (one per scene)
# --------------------------------------------------------------------------------------------------------------

PARITY_NAMES = {
    "discrete": {"shelf_dense": "DenseShelf", "shelf": "Shelf", "shelf_simple": "ShelfSimple"},
    "curve": {"shelf_dense": "cur_denseShelf", "shelf": "cur_shelf", "shelf_simple": "cur_shelfSimple"},
}


def parity_figure(plt, rs, args):
    from matplotlib.lines import Line2D

    series, labels = select_series(rs, args.all_series, args.methods)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ns = sorted({n for s in series for n in s.points})
    for s in series:
        xs, ys, std = per_item(s)
        st = series_style(rs.kind, s, series)
        st["marker"] = None  # the Julia figures had no markers
        draw_series(ax, xs, ys, std, st, labels[s.key], lw=2.2)
    set_batch_axis(ax, ns, "Number of poses in batch" if rs.kind == "discrete" else
                   "Number of trajectories per batch", max_labels=16)
    for t in ax.get_xticklabels():
        t.set_rotation(45)
    set_log_y(ax)
    ax.set_ylabel("Time [µs / %s]" % rs.item_noun())
    ax.set_title("%s · %s" % (br.scene_title(rs.scene), context_line([rs], args.note)), fontsize=8,
                 color=MUTED, loc="left")
    if rs.kind == "discrete":
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=9)
    else:
        # two-part legend as in the Julia figure: colour = method, line style = number of control points
        mids = []
        for s in series:
            m = method_of(rs.kind, s)
            if m not in mids:
                mids.append(m)
        variants = sorted({s.variant(rs.kind) for s in series if s.variant(rs.kind) is not None})
        h = [Line2D([], [], color=method_style(m)[1], lw=6) for m in mids]
        h += [Line2D([], [], color=INK, lw=1.8, linestyle=variant_ls(v)) for v in variants]
        lab = [method_style(m)[0] for m in mids] + ["%d points" % v for v in variants]
        ax.legend(h, lab, loc="upper right", ncol=2, frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=8,
                  handlelength=2.2, title="method (colour) / control points (line)", title_fontsize=7)
    return fig


# --------------------------------------------------------------------------------------------------------------
# Speedup over a reference method
# --------------------------------------------------------------------------------------------------------------


def find_reference(rs, ref):
    """Reference series of a result set for `ref` ('curobo', 'fcl', a label such as 'RobotToObs' or
    'Linear-16', or a series key). Returns the list of candidate series (several when they differ by variant)."""
    labels = label_series(rs)
    r = ref.lower()
    out = [s for k, s in rs.series.items()
           if method_of(rs.kind, s) == r or labels[k].lower() == r or k.lower() == r
           or method_style(method_of(rs.kind, s))[0].lower() == r]
    return out


def match_reference(kind, s, refs):
    """The reference series for s: the only one, or the one with the same number of control points."""
    if not refs:
        return None
    if len(refs) == 1:
        return refs[0]
    v = s.variant(kind)
    for r in refs:
        if r.variant(kind) == v:
            return r
    return None


def ratio_points(ref, s):
    """[(n, t_ref / t_s, relative std of the ratio or None)] at the common batch sizes."""
    out = []
    for n in sorted(set(ref.points) & set(s.points)):
        a, b = ref.points[n], s.points[n]
        r = a.time_us / b.time_us
        rel = None
        if a.std_us is not None or b.std_us is not None:
            rel = math.hypot((a.std_us or 0) / a.time_us, (b.std_us or 0) / b.time_us)
        out.append((n, r, rel))
    return out


# --------------------------------------------------------------------------------------------------------------
# Overview: scenes side by side
# --------------------------------------------------------------------------------------------------------------


def new_figure(plt, nrows, ncols, figsize, **kw):
    """Figure with constrained layout (title, panels and the shared legend never overlap)."""
    figsize = (max(figsize[0], 5.5), figsize[1])  # room for the legend and the subtitle with one column
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, layout="constrained", **kw)
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.06, hspace=0.04, wspace=0.03)
    return fig, axes


def finish_figure(fig, axes, title, subtitle=None):
    """Shared legend below the panels; title (bold) and subtitle (context, muted) above, left aligned."""
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    if seen:
        ncol = min(len(seen), 6)
        fig.legend(list(seen.values()), list(seen.keys()), loc="outside lower center", ncol=ncol, fontsize=9,
                   handlelength=2.6)
    # one suptitle keeps constrained layout in charge of the space; the subtitle is its second line
    fig.suptitle(title + ("\n" + subtitle if subtitle else ""), x=0.01, ha="left", fontsize=11,
                 linespacing=1.5)


def overview_figure(plt, sets, args, kind):
    ncols = len(sets)
    refs_by_set = [find_reference(rs, args.reference) if args.reference != "none" else [] for rs in sets]
    ref_name = next((method_style(method_of(kind, r[0]))[0] for r in refs_by_set if r), None)
    if ref_name is None:  # no reference anywhere: time row only
        fig, axes = new_figure(plt, 1, ncols, (3.6 * ncols + 0.6, 4.2), sharex="col")
        if args.reference != "none":
            print("  %s: no %s results; overview without the speedup row" % (kind, args.reference))
    else:
        fig, axes = new_figure(plt, 2, ncols, (3.6 * ncols + 0.6, 6.6), sharex="col",
                               gridspec_kw=dict(height_ratios=[1.45, 1]))
    noun = br.ITEM_NOUN[kind]
    all_t = []
    for j, rs in enumerate(sets):
        series, labels = select_series(rs, args.all_series, args.methods)
        ax = axes[0][j]
        ns = sorted({n for s in series for n in s.points})
        if ref_name is None:
            for s in series:
                xs, ys, std = per_item(s)
                draw_series(ax, xs, ys, std, series_style(kind, s, series), labels[s.key])
                all_t += ys
            set_log_y(ax)
            ax.set_title(br.scene_title(rs.scene))
            if j == 0:
                ax.set_ylabel("Time per %s [\u00b5s]" % noun)
            set_batch_axis(ax, ns, "%s per batch" % br.ITEM_PLURAL[kind].capitalize())
            continue
        axs = axes[1][j]
        ns = sorted({n for s in series for n in s.points})
        for s in series:
            xs, ys, std = per_item(s)
            draw_series(ax, xs, ys, std, series_style(kind, s, series), labels[s.key])
            all_t += ys
        set_log_y(ax)
        ax.set_title(br.scene_title(rs.scene))
        if j == 0:
            ax.set_ylabel("Time per %s [µs]" % noun)
        # speedup row
        refs = refs_by_set[j]
        ratios = []
        if refs:
            for s in series:
                if any(s is r for r in refs):
                    continue
                r = match_reference(kind, s, refs)
                if r is None:
                    continue
                pts = ratio_points(r, s)
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                std = [p[1] * p[2] if p[2] is not None else None for p in pts]
                draw_series(axs, xs, ys, std if any(v is not None for v in std) else None,
                            series_style(kind, s, series), labels[s.key])
                ratios += ys
            speedup_axis(axs, ratios, ref_name)
        else:
            axs.text(0.5, 0.5, "no %s results\nfor this scene" % ref_name, transform=axs.transAxes,
                     ha="center", va="center", color=MUTED)
            axs.set_yticks([])
        if j == 0:
            axs.set_ylabel("Speedup over %s" % ref_name)
        set_batch_axis(axs, ns, "%s per batch" % br.ITEM_PLURAL[kind].capitalize())
    # one y range for the time row, so the scenes can be compared
    if all_t:
        lo, hi = min(all_t), max(all_t)
        for ax in axes[0]:
            ax.set_ylim(lo / 1.6, hi * 1.6)
    title = "%s collision checking" % ("Discrete (per pose)" if kind == "discrete" else "Continuous (per trajectory)")
    finish_figure(fig, [a for row in axes for a in row], title, context_line(sets, args.note))
    return fig


def throughput_figure(plt, sets, args, kind):
    ncols = len(sets)
    fig, axes = new_figure(plt, 1, ncols, (3.6 * ncols + 0.6, 3.9), sharey=True)
    noun = br.ITEM_NOUN[kind]
    for j, rs in enumerate(sets):
        series, labels = select_series(rs, args.all_series, args.methods)
        ax = axes[0][j]
        ns = sorted({n for s in series for n in s.points})
        for s in series:
            xs = s.ns()
            ys = [1e6 / s.points[n].per_item_us for n in xs]
            std = [None if s.points[n].std_us is None else ys[i] * s.points[n].std_us / s.points[n].time_us
                   for i, n in enumerate(xs)]
            draw_series(ax, xs, ys, std if any(v is not None for v in std) else None, series_style(kind, s, series),
                        labels[s.key])
        set_log_y(ax)
        set_batch_axis(ax, ns, "%s per batch" % br.ITEM_PLURAL[kind].capitalize())
        ax.set_title(br.scene_title(rs.scene))
        if j == 0:
            ax.set_ylabel("Throughput [%s / s]" % br.ITEM_PLURAL[kind])
    finish_figure(fig, list(axes[0]), "Throughput, %s checks" % ("discrete" if kind == "discrete" else "continuous"),
                  context_line(sets, args.note))
    return fig


def write_summary(sets, path, args, log=print):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "scene", "series", "key", "n", "time_per_batch_us", "std_per_batch_us", "repetitions",
                    "time_per_item_us", "items_per_s", "source"])
        for rs in sets:
            series, labels = select_series(rs, True, None)
            for s in series:
                for n in s.ns():
                    p = s.points[n]
                    w.writerow([rs.kind, rs.scene, labels[s.key], s.key, n, "%.6g" % p.time_us,
                                "" if p.std_us is None else "%.6g" % p.std_us, p.reps, "%.6g" % p.per_item_us,
                                "%.6g" % (1e6 / p.per_item_us), "external" if s.external else rs.source])
    log("  wrote %s" % short_path(path))


def print_headlines(sets, args, log=print):
    """The numbers the Julia scripts printed, in words."""
    for rs in sets:
        labels = label_series(rs)
        by_label = {labels[k]: s for k, s in rs.series.items()}
        log("  %s / %s:" % (rs.kind, br.scene_title(rs.scene)))
        if rs.kind == "discrete":
            tw, cu, fcl = by_label.get("Two-way"), by_label.get("cuRobo"), by_label.get("FCL (CPU)")
            for ref, name in ((cu, "cuRobo"), (fcl, "FCL")):
                if tw and ref:
                    pts = ratio_points(ref, tw)
                    if pts:
                        best = max(pts, key=lambda p: p[1])
                        log("    Two-way vs %s: best speedup %.2fx (at %d poses; > 1 = Two-way faster)" % (name, best[1], best[0]))
        else:
            for v in (8, 16):
                lin, cu = by_label.get("Linear-%d" % v), by_label.get("cuRobo-%d" % v)
                if lin and cu:
                    rr = [p[1] for p in ratio_points(cu, lin)]
                    if rr:
                        log("    Linear-%d vs cuRobo-%d: max speedup %.2fx, mean %.2fx" % (v, v, max(rr),
                                                                                         sum(rr) / len(rr)))
            lin, fcl = by_label.get("Linear-16"), by_label.get("FCL (CPU)-16")
            if lin and fcl:
                rr = [p[1] for p in ratio_points(fcl, lin)]
                if rr:
                    log("    Linear-16 vs FCL-16: best speedup %.1fx" % max(rr))


def cmd_figures(args, log=print):
    plt = setup_matplotlib()
    log("reading %s" % args.results)
    try:
        sets = br.discover(args.results, stat=args.stat, externals=not args.no_external, log=log)
    except br.ResultError as e:
        log("error: %s" % e)
        return 2
    if not sets:
        log("error: no usable benchmark results under %s" % args.results)
        return 2
    os.makedirs(args.out, exist_ok=True)
    print_headlines(sets, args, log)
    for kind in br.KINDS:
        ks = [r for r in sets if r.kind == kind]
        if not ks:
            log("  no %s results; skipping the %s figures" % (kind, kind))
            continue
        if "parity" in args.figures:
            for rs in ks:
                name = PARITY_NAMES[kind].get(rs.scene) or "%s_%s" % (kind, os.path.splitext(
                    os.path.basename(rs.source))[0])
                fig = parity_figure(plt, rs, args)
                save(fig, os.path.join(args.out, kind, name), args.formats, args.dpi, log)
                plt.close(fig)
        if "overview" in args.figures:
            fig = overview_figure(plt, ks, args, kind)
            save(fig, os.path.join(args.out, "overview_%s" % kind), args.formats, args.dpi, log)
            plt.close(fig)
        if "throughput" in args.figures:
            fig = throughput_figure(plt, ks, args, kind)
            save(fig, os.path.join(args.out, "throughput_%s" % kind), args.formats, args.dpi, log)
            plt.close(fig)
        write_summary(ks, os.path.join(args.out, "summary_%s.csv" % kind), args, log)
    return 0


# --------------------------------------------------------------------------------------------------------------
# Comparison of result sets
# --------------------------------------------------------------------------------------------------------------


def parse_run_arg(text):
    if "=" in text:
        label, path = text.split("=", 1)
        if label.strip() and path.strip():
            return label.strip(), path.strip()
    return os.path.basename(os.path.normpath(text)) or text, text


def match_runs(runs, kind, scene=None, log=print):
    """[(scene title, [ResultSet per run])] for one kind. Scenes are matched by id; when every run has exactly
    one result set of this kind (single JSON files, scene possibly not recorded), those are matched as is."""
    per_run = [[rs for rs in sets if rs.kind == kind] for _, sets in runs]
    if any(not p for p in per_run):
        return []
    if all(len(p) == 1 for p in per_run):
        scenes = {p[0].scene for p in per_run} - {br.UNSPECIFIED_SCENE}
        title = br.scene_title(scenes.pop()) if len(scenes) == 1 else ("Scene not recorded" if not scenes else
                                                                       " vs ".join(sorted(scenes)))
        return [(title, [p[0] for p in per_run])]
    # a single result set without a recorded scene (an old JSON file) stands in for whichever scene is compared
    wild = [len(p) == 1 and p[0].scene == br.UNSPECIFIED_SCENE for p in per_run]
    common = set.intersection(*[{rs.scene for rs in p} for p, w in zip(per_run, wild) if not w])
    common -= {br.UNSPECIFIED_SCENE}
    if scene:
        common &= {scene}
    if any(wild) and len(common) > 1:
        log("  %s: %s has no scene in its JSON and could stand for %s; choose one with --scene" % (
            kind, ", ".join(runs[i][0] for i, w in enumerate(wild) if w), ", ".join(sorted(common))))
        return []
    out = []
    for sc in sorted(common, key=br.scene_order):
        out.append((br.scene_title(sc), [p[0] if w else next(rs for rs in p if rs.scene == sc)
                                         for p, w in zip(per_run, wild)]))
    return out


def compare_series(sets, kind, args):
    """Labels present in at least two runs (and in the reference run), in plotting order."""
    present = []
    for rs in sets:
        series, labels = select_series(rs, args.all_series, args.methods)
        present.append({labels[s.key]: s for s in series})
    order = []
    for d in present:
        for lab, s in d.items():
            if lab not in [o[0] for o in order]:
                order.append((lab, s))
    order.sort(key=lambda o: series_order(kind, o[1]))
    keep = [lab for lab, _ in order if lab in present[0] and sum(lab in d for d in present) >= 2]
    return keep, present


def cmd_compare(args, log=print):
    plt = setup_matplotlib()
    if len(args.run) < 2:
        log("error: compare needs at least two --run LABEL=PATH")
        return 2
    runs = []
    for text in args.run:
        label, path = parse_run_arg(text)
        log("reading %s (%s)" % (path, label))
        try:
            sets = br.discover(path, stat=args.stat, externals=args.include_external, log=log)
        except br.ResultError as e:
            log("error: %s" % e)
            return 2
        if not sets:
            log("error: no usable benchmark results in %s" % path)
            return 2
        runs.append((label, sets))
    os.makedirs(args.out, exist_ok=True)
    ref_label = runs[0][0]
    made = 0
    for kind in br.KINDS:
        groups = match_runs(runs, kind, args.scene, log)
        if not groups:
            if any(rs.kind == kind for _, sets in runs for rs in sets):
                log("  %s: not every run has %s results of a common scene; no %s comparison" % (kind, kind, kind))
            continue
        groups = [(t, sets, *compare_series(sets, kind, args)) for t, sets in groups]
        groups = [g for g in groups if g[2]]
        if not groups:
            log("  %s: no series in common between the runs; skipped" % kind)
            continue
        made += 1
        compare_time_figure(plt, runs, groups, kind, args, log)
        compare_speedup_figure(plt, runs, groups, kind, args, ref_label, log)
        write_compare_csv(runs, groups, kind, os.path.join(args.out, "compare_%s.csv" % kind), log)
    if not made:
        log("error: nothing to compare (the runs share no kind / scene / series)")
        return 2
    return 0


def runs_subtitle(runs, groups):
    """One line per run: 'reference  old code: host xingru · 2025-02-15'."""
    parts = []
    for r, (label, _) in enumerate(runs):
        ctx = context_line([g[1][r] for g in groups])
        parts.append("%s%s%s" % ("reference = " if r == 0 else "", label, (": " + ctx) if ctx else ""))
    return "\n".join(parts)


def compare_time_figure(plt, runs, groups, kind, args, log):
    labels_all = []
    for g in groups:
        for lab in g[2]:
            if lab not in labels_all:
                labels_all.append(lab)
    nrows, ncols = len(labels_all), len(groups)
    fig, axes = new_figure(plt, nrows, ncols, (3.4 * ncols + 0.9, 1.9 * nrows + 1.4), sharex="col")
    noun = br.ITEM_NOUN[kind]
    for j, (title, sets, keep, present) in enumerate(groups):
        ns = sorted({n for d in present for lab in keep for n in (d[lab].points if lab in d else [])})
        for i, lab in enumerate(labels_all):
            ax = axes[i][j]
            vals = []
            for r, d in enumerate(present):
                if lab not in d or lab not in keep:
                    continue
                xs, ys, std = per_item(d[lab])
                st = dict(color=RUN_COLORS[r % len(RUN_COLORS)], marker=RUN_MARKERS[r % len(RUN_MARKERS)],
                          linestyle=RUN_LS[r % len(RUN_LS)])
                draw_series(ax, xs, ys, std, st, runs[r][0])
                vals += ys
            set_log_y(ax)
            if not vals:
                ax.text(0.5, 0.5, "not in this scene", transform=ax.transAxes, ha="center", va="center",
                        color=MUTED, fontsize=8)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel("%s\n[µs / %s]" % (lab, noun), fontsize=9)
            set_batch_axis(ax, ns, ("%s per batch" % br.ITEM_PLURAL[kind].capitalize()) if i == nrows - 1 else None)
    finish_figure(fig, [a for row in axes for a in row], "Time per %s, %s checks" % (noun, kind),
                  runs_subtitle(runs, groups))
    save(fig, os.path.join(args.out, "compare_%s_time" % kind), args.formats, args.dpi, log)
    plt.close(fig)


def compare_speedup_figure(plt, runs, groups, kind, args, ref_label, log):
    nrows, ncols = len(runs) - 1, len(groups)
    fig, axes = new_figure(plt, nrows, ncols, (3.6 * ncols + 0.9, 2.9 * nrows + 1.6), sharex="col")
    log("  speedup of each run over '%s' (t_%s / t_run; > 1 = faster):" % (ref_label, ref_label))
    for j, (title, sets, keep, present) in enumerate(groups):
        ns = sorted({n for d in present for lab in keep for n in (d[lab].points if lab in d else [])})
        for i in range(nrows):
            r = i + 1
            ax = axes[i][j]
            ratios = []
            for lab in keep:
                if lab not in present[r]:
                    continue
                ref, s = present[0][lab], present[r][lab]
                pts = ratio_points(ref, s)
                if not pts:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                std = [p[1] * p[2] if p[2] is not None else None for p in pts]
                draw_series(ax, xs, ys, std if any(v is not None for v in std) else None, series_style(kind, s, list(present[r].values())), lab)
                ratios += ys
                log("    %-14s %-26s %s" % (title, "%s, %s" % (runs[r][0], lab) if nrows > 1 else lab,
                                            "  ".join("%d:%.2f" % (n, v) for n, v in zip(xs, ys))))
            speedup_axis(ax, ratios, "reference")
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel("Speedup of %s" % runs[r][0], fontsize=9)
            set_batch_axis(ax, ns, ("%s per batch" % br.ITEM_PLURAL[kind].capitalize()) if i == nrows - 1 else None)
    finish_figure(fig, [a for row in axes for a in row],
                  "Speedup over “%s”, %s checks: t(%s) / t(run) at each batch size" % (
                      ref_label, kind, ref_label), runs_subtitle(runs, groups))
    save(fig, os.path.join(args.out, "compare_%s_speedup" % kind), args.formats, args.dpi, log)
    plt.close(fig)


def write_compare_csv(runs, groups, kind, path, log):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "scene", "series", "run", "n", "time_per_item_us", "std_per_item_us",
                    "speedup_over_reference"])
        for title, sets, keep, present in groups:
            for lab in keep:
                ref = present[0][lab]
                for r, d in enumerate(present):
                    if lab not in d:
                        continue
                    for n in d[lab].ns():
                        p = d[lab].points[n]
                        sp = ref.points[n].time_us / p.time_us if n in ref.points else None
                        w.writerow([kind, title, lab, runs[r][0], n, "%.6g" % p.per_item_us,
                                    "" if p.std_us is None else "%.6g" % p.std_per_item_us,
                                    "" if sp is None else "%.4f" % sp])
    log("  wrote %s" % short_path(path))


# --------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------


def add_common(p):
    p.add_argument("--out", default=os.path.join("data", "Benchmark", "Plots"),
                   help="output directory (default: data/Benchmark/Plots)")
    p.add_argument("--formats", default="png,pdf,svg",
                   help="comma-separated figure formats (default: png,pdf,svg)")
    p.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    p.add_argument("--stat", choices=("median", "mean"), default="median",
                   help="central value when the JSON has repetitions (default: median)")
    p.add_argument("--methods", metavar="REGEX",
                   help="only series whose label matches (case-insensitive), e.g. 'Linear|Quadratic'")
    p.add_argument("--all-series", action="store_true",
                   help="also plot series the Julia figures left out (curve BM_MESH / BM_RAY halves of Two-way, "
                        "cuRobo with 4 sweep steps)")
    p.add_argument("--note", help="text appended to the figure subtitles, e.g. 'RTX 4070, Linux, original code'")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="plot_benchmarks.py",
        description="Plot RTCD benchmark results (Google Benchmark JSON) with cuRobo / FCL references, or "
                    "compare result sets. Without a command, 'figures' is run.",
        epilog="Examples:\n"
               "  python scripts/Benchmark/plots/plot_benchmarks.py --results data/Benchmark/result "
               "--out data/Benchmark/Plots\n"
               "  python scripts/Benchmark/plots/plot_benchmarks.py compare --run \"old=dirA\" "
               "--run \"new=dirB/shelf.json\" --out cmp\n"
               "  python scripts/Benchmark/plots/plot_benchmarks.py --self-test\n"
               "Options of each command: plot_benchmarks.py figures --help, plot_benchmarks.py compare --help",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the parser tests and render every figure from testdata/ into a temporary folder")
    sub = ap.add_subparsers(dest="command")

    f = sub.add_parser("figures", help="figures of one result directory (default command)",
                       description="Per-scene figures (as the former Julia scripts), overview and throughput "
                                   "figures, and a CSV summary of one result directory or JSON file.")
    f.add_argument("--results", default=os.path.join("data", "Benchmark", "result"),
                   help="result directory in the benchmark.sh layout, any directory with Google Benchmark JSON, "
                        "or one JSON file (default: data/Benchmark/result)")
    add_common(f)
    f.add_argument("--figures", default="parity,overview,throughput",
                   help="which figures: comma-separated subset of parity,overview,throughput (default: all)")
    f.add_argument("--reference", default="curobo",
                   help="reference of the speedup row: curobo (default), fcl, none, or a series label such as "
                        "RobotToObs or Linear-16")
    f.add_argument("--no-external", action="store_true", help="leave out cuRobo and FCL")

    c = sub.add_parser("compare", help="compare two or more result sets",
                       description="Overlay and speedup figures of two or more result sets; the first --run is "
                                   "the reference. Series are matched by label (e.g. RobotToObs, Linear-16) and "
                                   "scene (from the file layout or the JSON context).")
    c.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                   help="a result set: directory (benchmark.sh layout or any JSON inside) or one JSON file; "
                        "repeat, first = reference")
    add_common(c)
    c.add_argument("--scene", choices=[s.id for s in br.SCENES],
                   help="compare only this scene (needed when a JSON file without a recorded scene is compared "
                        "with a directory of several scenes)")
    c.add_argument("--include-external", action="store_true",
                   help="also compare cuRobo / FCL series (normally left out: they are not RTCD measurements)")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        import unittest

        suite = unittest.defaultTestLoader.discover(HERE, pattern="test_bench_results.py", top_level_dir=HERE)
        ok = unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()
        return 0 if ok else 1
    commands = ("figures", "compare")
    if not argv or (argv[0] not in commands and argv[0] not in ("-h", "--help")):
        argv = ["figures"] + argv
    args = build_parser().parse_args(argv)
    args.formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    if args.command == "compare":
        return cmd_compare(args)
    args.figures = {x.strip() for x in args.figures.split(",")}
    unknown = args.figures - {"parity", "overview", "throughput"}
    if unknown:
        print("error: unknown --figures %s" % ",".join(sorted(unknown)))
        return 2
    return cmd_figures(args)


if __name__ == "__main__":
    sys.exit(main())
