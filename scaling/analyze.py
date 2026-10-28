"""Tables and figures for the scaling study (step 12 of the plan).

Reads what the runs and evaluations leave on the shared volume:

* ``<data>/logs/<run>.out``                       training log (iteration, elapsed hours, it/s, train return, online WM error)
* ``<data>/results/<run>/ckpt_XXXXXXX/summary.json`` held-out returns per set (``eval_heldout.py``)
* ``<data>/results/wm_error_ckpt_XXXXXXX.json``   pooled 8-step WM error of every run's WM on the same windows (``wm_error.py``)
* ``<data>/subsets/subsets_report.json``          hours and clips per training subset

and writes to ``--out-dir``: ``results.json``, ``results.md`` (the README tables) and four figures.

Conventions (``docs/CONTRACTS.md``, results reporting): runs are compared at the same iteration; the headline
number of a run is the mean over its last ``--last-k`` checkpoints (min and max shown as the error bar), which
smooths the checkpoint-to-checkpoint swings seen in held-out returns. Compute to a target is the first crossing,
linearly interpolated between checkpoints, in iterations and in GPU-hours (elapsed training time from the log, one
H100 SXM per run, compile time included). Missing runs or sets are skipped, so it also works on partial results.

  uv run --with matplotlib python scaling/analyze.py --data $DATA --out-dir results/figures
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import tyro

RUNS = ["D1", "D7", "D40", "Dall"]
SETS = ["motion", "performer", "lafan1", "train"]
SET_LABELS = {"motion": "Unseen motions", "performer": "Unseen performers", "lafan1": "LAFAN1", "train": "Own training motions"}
DEFAULT_HOURS = {"D1": 1.0, "D7": 7.0, "D40": 40.0, "Dall": 254.36}

LOG_RE = re.compile(
    r"^it (\d+) \| sim steps \d+ \| ([\d.]+) h \| ([\d.]+) it/s \| train_ret ([-\d.na]+) .*?\| wm nse1 ([-\d.na]+) nse8 ([-\d.na]+)"
)


# ---------------------------------------------------------------- loading


def parse_log(path: Path) -> list[dict]:
    """One dict per progress line of a training log."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        m = LOG_RE.match(line.strip())
        if m:
            it, h, its, tr, n1, n8 = m.groups()
            rows.append({"it": int(it), "hours": float(h), "it_s": float(its), "train_ret": float(tr), "nse1": float(n1), "nse8": float(n8)})
    return rows


def load_hours(data: Path) -> dict[str, float]:
    p = data / "subsets" / "subsets_report.json"
    if p.exists():
        rep = json.loads(p.read_text())
        return {r: float(rep[r]["hours"]) for r in RUNS if r in rep}
    return dict(DEFAULT_HOURS)


def load_evals(data: Path, run: str) -> dict[int, dict[str, dict]]:
    """Iteration -> set -> summary (only checkpoints named ckpt_<iteration>)."""
    out = {}
    for f in sorted((data / "results" / run).glob("ckpt_*/summary.json")):
        m = re.fullmatch(r"ckpt_(\d+)", f.parent.name)
        if m:
            s = json.loads(f.read_text())
            out[int(m.group(1))] = {k: v for k, v in s.items() if isinstance(v, dict)}
    return out


def load_pooled_wm(data: Path) -> dict[int, dict[str, float]]:
    """Iteration -> run -> pooled 8-step error."""
    out: dict[int, dict[str, float]] = {}
    for f in sorted((data / "results").glob("wm_error_ckpt_*.json")):
        m = re.search(r"ckpt_(\d+)", f.name)
        if not m:
            continue
        rep = json.loads(f.read_text())
        for ck, v in rep.get("runs", {}).items():
            run = Path(ck).parent.name
            out.setdefault(int(m.group(1)), {})[run] = float(v["nse_8step"])
    return out


# ---------------------------------------------------------------- numbers


def last_k(evals: dict[int, dict[str, dict]], s: str, k: int) -> dict | None:
    its = [it for it in sorted(evals) if s in evals[it]]
    if not its:
        return None
    sel = its[-k:]
    vals = [evals[it][s]["mean_norm_return"] for it in sel]
    fails = [evals[it][s]["failed"] / max(evals[it][s]["clips"], 1) for it in sel]
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "fail_rate": sum(fails) / len(fails), "iterations": sel}


def crossing(its: list[int], vals: list[float], target: float) -> float | None:
    """First iteration at which ``vals`` reaches ``target``, linear between checkpoints; None if never."""
    prev_it, prev_v = 0, None
    for it, v in zip(its, vals):
        if v >= target:
            if prev_v is None or prev_v >= target:
                return float(it)
            return prev_it + (target - prev_v) / (v - prev_v) * (it - prev_it)
        prev_it, prev_v = it, v
    return None


def hours_at(log: list[dict], it: float) -> float | None:
    """Elapsed training hours at iteration ``it``, interpolated from the log."""
    if not log:
        return None
    pts = [(0, 0.0)] + [(r["it"], r["hours"]) for r in log]
    for (i0, h0), (i1, h1) in zip(pts, pts[1:]):
        if i0 <= it <= i1:
            return h0 + (h1 - h0) * (it - i0) / max(i1 - i0, 1)
    # beyond the last log line: extrapolate at the mean speed of the last lines
    tail = log[-10:]
    rate = sum(r["it_s"] for r in tail) / len(tail)
    return pts[-1][1] + (it - pts[-1][0]) / max(rate, 1e-9) / 3600


def summarize(data: Path, last_k_ckpts: int = 3, targets: tuple[float, ...] = (0.7, 0.8, 0.85, 0.9)) -> dict:
    hours = load_hours(data)
    pooled = load_pooled_wm(data)
    res = {"last_k": last_k_ckpts, "targets": list(targets), "runs": {}}
    for run in RUNS:
        log = parse_log(data / "logs" / f"{run}.out")
        evals = load_evals(data, run)
        if not log and not evals:
            continue
        r: dict = {"hours_data": hours.get(run), "eval_iterations": sorted(evals)}
        if log:
            tail = log[-20:]
            r["final_iteration"] = log[-1]["it"]
            r["train_gpu_hours"] = log[-1]["hours"]
            r["it_per_s"] = sum(x["it_s"] for x in tail) / len(tail)
            r["online_nse8"] = sum(x["nse8"] for x in tail) / len(tail)
            r["train_ret_online"] = sum(x["train_ret"] for x in tail) / len(tail)
        r["sets"] = {}
        for s in SETS:
            lk = last_k(evals, s, last_k_ckpts)
            if lk is None:
                continue
            its = [it for it in sorted(evals) if s in evals[it]]
            vals = [evals[it][s]["mean_norm_return"] for it in its]
            cross = {}
            for t in targets:
                c = crossing(its, vals, t)
                cross[f"{t:g}"] = None if c is None else {"iterations": round(c), "gpu_hours": round(hours_at(log, c), 3) if log else None}
            r["sets"][s] = {**lk, "curve": {"iterations": its, "mean_norm_return": vals}, "to_target": cross}
        wm_its = sorted(it for it in pooled if run in pooled[it])
        if wm_its:
            sel = wm_its[-last_k_ckpts:]
            r["pooled_nse8"] = sum(pooled[it][run] for it in sel) / len(sel)
            r["pooled_nse8_curve"] = {"iterations": wm_its, "nse8": [pooled[it][run] for it in wm_its]}
        res["runs"][run] = r
    return res


# ---------------------------------------------------------------- tables


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    runs = res["runs"]
    k = res["last_k"]
    lines = [f"Held-out mean return (mean of the last {k} checkpoints; min–max in brackets) and failure rate.", ""]
    sets = [s for s in SETS if any(s in r["sets"] for r in runs.values())]
    lines.append("| Run | Data (h) | " + " | ".join(SET_LABELS[s] for s in sets) + " |")
    lines.append("|---|---:|" + "---:|" * len(sets))
    for name, r in runs.items():
        cells = []
        for s in sets:
            v = r["sets"].get(s)
            cells.append("–" if v is None else f"{v['mean']:.3f} [{v['min']:.3f}–{v['max']:.3f}], fail {100 * v['fail_rate']:.1f}%")
        lines.append(f"| {name} | {fmt(r['hours_data'], 1)} | " + " | ".join(cells) + " |")
    lines += ["", "World-model 8-step error (normalised, x1e-3): pooled held-out windows vs the run's own training rollouts.", ""]
    lines.append("| Run | Pooled held-out | Own training rollouts (online) | it/s | GPU-h trained |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, r in runs.items():
        p = r.get("pooled_nse8")
        o = r.get("online_nse8")
        lines.append(
            f"| {name} | {fmt(None if p is None else 1e3 * p, 1)} | {fmt(None if o is None else 1e3 * o, 1)} | {fmt(r.get('it_per_s'), 2)} | {fmt(r.get('train_gpu_hours'), 2)} |"
        )
    for t in res["targets"]:
        lines += ["", f"Compute to a mean return of {t:g} (GPU-hours; iterations in brackets; – = not reached).", ""]
        lines.append("| Run | " + " | ".join(SET_LABELS[s] for s in sets) + " |")
        lines.append("|---|" + "---:|" * len(sets))
        for name, r in runs.items():
            cells = []
            for s in sets:
                c = r["sets"].get(s, {}).get("to_target", {}).get(f"{t:g}")
                cells.append("–" if not c else f"{fmt(c['gpu_hours'], 2)} ({c['iterations']})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures

# Validated with the dataviz skill's palette checker (light surface): categorical slots in fixed order for the
# evaluation sets, one blue ramp (ordinal, light -> dark = little -> much data) for the runs.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SET_COLORS = {"motion": "#2a78d6", "performer": "#eb6834", "lafan1": "#1baf7a", "train": "#eda100"}
SET_MARKERS = {"motion": "o", "performer": "s", "lafan1": "D", "train": "^"}
RUN_COLORS = {"D1": "#86b6ef", "D7": "#3987e5", "D40": "#1c5cab", "Dall": "#0d366b"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _end_labels(ax, items, min_gap: float = 0.06):
    """Direct labels at the right end of each line, pushed apart vertically so they never overlap
    (``min_gap`` is a fraction of the y-range)."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    items = sorted(items, key=lambda t: t[1])
    ys = [t[1] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - (hi - 0.3 * gap)  # keep the stack inside the axes: shift down, then re-space downwards
    if over > 0:
        ys[-1] -= over
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    for (x, y, text), ly in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(x * 1.25, ly), textcoords="data", va="center", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8) if abs(ly - y) > 1e-9 else None)


def _hours_axis(ax, hours):
    ax.set_xscale("log")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:.0f} h" if h >= 10 else f"{h:g} h" for h in hours])
    ax.minorticks_off()
    ax.set_xlabel("Training data (hours of motion)", color=INK2, fontsize=10)


def figures(res: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = res["runs"]
    names = [n for n in RUNS if n in runs and runs[n].get("hours_data")]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if not names:
        return written
    hours = [runs[n]["hours_data"] for n in names]
    sets = [s for s in SETS if any(s in runs[n]["sets"] for n in names)]

    # 1. headline: held-out return vs data size
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    _style(ax)
    ends = []
    for s in sets:
        xs, ys, lo, hi = [], [], [], []
        for n in names:
            v = runs[n]["sets"].get(s)
            if v:
                xs.append(runs[n]["hours_data"]), ys.append(v["mean"]), lo.append(max(v["mean"] - v["min"], 0.0)), hi.append(max(v["max"] - v["mean"], 0.0))
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SET_COLORS[s], marker=SET_MARKERS[s], markersize=7, linewidth=2, capsize=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=SET_LABELS[s])
        ends.append((xs[-1], ys[-1], SET_LABELS[s]))
    _hours_axis(ax, hours)
    ax.set_xlim(min(hours) / 1.6, max(hours) * 12)
    _end_labels(ax, ends)
    ax.set_ylabel(f"Mean return (last {res['last_k']} checkpoints)", color=INK2, fontsize=10)
    ax.set_title("Held-out tracking return vs training data", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    written.append(out_dir / "fig1_heldout_vs_data.png")
    fig.savefig(written[-1], dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2. learning curves, one panel per set
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.6), facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, s in zip(axes[0], sets):
        _style(ax)
        for n in names:
            v = runs[n]["sets"].get(s)
            if v:
                its = [i / 1000 for i in v["curve"]["iterations"]]
                ax.plot(its, v["curve"]["mean_norm_return"], color=RUN_COLORS[n], linewidth=2, marker="o", markersize=4, label=n)
        ax.set_title(SET_LABELS[s], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Iteration (thousands)", color=INK2, fontsize=9)
    axes[0][0].set_ylabel("Mean return", color=INK2, fontsize=10)
    axes[0][-1].legend(frameon=False, fontsize=9, title="Run", title_fontsize=9, loc="lower right")
    fig.tight_layout()
    written.append(out_dir / "fig2_learning_curves.png")
    fig.savefig(written[-1], dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 3. world-model error vs data size (both are the same normalised error, so one axis)
    series = [("pooled_nse8", "Held-out windows (pooled)", "#2a78d6", "o"), ("online_nse8", "Own training rollouts", "#eb6834", "s")]
    if any(runs[n].get(k) is not None for n in names for k, *_ in series):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
        _style(ax)
        ends = []
        for key, label, color, mk in series:
            pts = [(runs[n]["hours_data"], 1e3 * runs[n][key]) for n in names if runs[n].get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, marker=mk, markersize=7, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
                ends.append((*pts[-1], label))
        _hours_axis(ax, hours)
        ax.set_xlim(min(hours) / 1.6, max(hours) * 12)
        ax.set_ylim(bottom=0)
        _end_labels(ax, ends)
        ax.set_ylabel("8-step error (x1e-3, normalised)", color=INK2, fontsize=10)
        ax.set_title("World-model error vs training data", loc="left", color=INK, fontsize=12)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        fig.tight_layout()
        written.append(out_dir / "fig3_wm_error.png")
        fig.savefig(written[-1], dpi=160, facecolor=SURFACE)
        plt.close(fig)

    # 4. compute to target, one panel per target
    targets = [t for t in res["targets"] if t >= 0.85]
    if targets:
        fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.8), facecolor=SURFACE, sharey=True, squeeze=False)
        for ax, t in zip(axes[0], targets):
            _style(ax)
            for s in sets:
                pts = []
                for n in names:
                    c = runs[n]["sets"].get(s, {}).get("to_target", {}).get(f"{t:g}")
                    if c and c.get("gpu_hours") is not None:
                        pts.append((runs[n]["hours_data"], c["gpu_hours"]))
                if pts:
                    ax.plot(*zip(*pts), color=SET_COLORS[s], marker=SET_MARKERS[s], markersize=7, linewidth=2,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, label=SET_LABELS[s])
            _hours_axis(ax, hours)
            ax.set_title(f"GPU-hours to a return of {t:g}", loc="left", color=INK, fontsize=10)
        axes[0][0].set_ylabel("GPU-hours (H100 SXM)", color=INK2, fontsize=10)
        handles, labels = {}, {}
        for ax in axes[0]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        if handles:
            fig.legend(list(handles.values()), list(handles), frameon=False, fontsize=9, loc="lower center", ncol=len(handles))
        written.append(out_dir / "fig4_compute_to_target.png")
        fig.savefig(written[-1], dpi=160, facecolor=SURFACE)
        plt.close(fig)
    return written


def main(data: Path, out_dir: Path, last_k: int = 3, targets: list[float] = [0.7, 0.8, 0.85, 0.9], no_figures: bool = False) -> None:
    res = summarize(data, last_k, tuple(targets))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    md = markdown(res)
    (out_dir / "results.md").write_text(md)
    print(md)
    if not no_figures:
        for p in figures(res, out_dir):
            print(f"wrote {p}")


if __name__ == "__main__":
    tyro.cli(main)
