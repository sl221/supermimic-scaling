"""analyze.py on a synthetic results tree (CPU only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import crossing, hours_at, markdown, parse_log, summarize  # noqa: E402

HOURS = {"D1": 1.0, "D7": 7.0, "D40": 40.0, "Dall": 254.36}


def fake_tree(root: Path, runs=("D1", "D7", "D40", "Dall")) -> None:
    (root / "subsets").mkdir(parents=True)
    (root / "subsets" / "subsets_report.json").write_text(json.dumps({r: {"clips": 1, "hours": HOURS[r]} for r in HOURS}))
    (root / "logs").mkdir()
    for j, run in enumerate(runs):
        lines = ["[INFO] params: world model 8.82M"]
        for it in range(50, 40001, 50):
            h = 0.2 + it / 2.3 / 3600
            lines.append(
                f"it {it} | sim steps {it * 2} | {h:.2f} h | 2.3 it/s | train_ret 0.900 (n=10) | wm nse1 0.0050 nse8 {0.010 + 0.001 * j:.4f} | "
                "imag_ret 2.0 surv 0.99 | t sim/wm/pi 1/1/1s"
            )
        (root / "logs" / f"{run}.out").write_text("\n".join(lines))
        for it in range(4000, 40001, 4000):
            d = root / "results" / run / f"ckpt_{it:07d}"
            d.mkdir(parents=True)
            gain = 0.05 * j  # more data -> higher held-out return
            s = {"ckpt": "x"}
            for k, base in (("motion", 0.6), ("performer", 0.62), ("lafan1", 0.5), ("train", 0.7)):
                v = min(0.99, base + gain + 0.3 * (1 - math.exp(-it / 12000)))
                s[k] = {"clips": 100, "failed": 5, "mean_norm_return": v}
            (d / "summary.json").write_text(json.dumps(s))
    for it in range(4000, 40001, 4000):
        rep = {"runs": {f"/w/logs/{run}/ckpt_{it:07d}.pt": {"nse_8step": 0.02 - 0.002 * j} for j, run in enumerate(runs)}}
        (root / "results" / f"wm_error_ckpt_{it:07d}.json").write_text(json.dumps(rep))


def test_crossing_interpolates_and_reports_never():
    assert crossing([4000, 8000], [0.80, 0.90], 0.85) == pytest.approx(6000)
    assert crossing([4000, 8000], [0.90, 0.95], 0.85) == 4000
    assert crossing([4000, 8000], [0.70, 0.80], 0.85) is None


def test_hours_at_interpolates_and_extrapolates():
    log = [{"it": 100, "hours": 0.5, "it_s": 2.0}, {"it": 200, "hours": 1.0, "it_s": 2.0}]
    assert hours_at(log, 150) == pytest.approx(0.75)
    assert hours_at(log, 200 + 7200) == pytest.approx(2.0)


def test_parse_log_reads_progress_lines(tmp_path):
    fake_tree(tmp_path, runs=("D1",))
    rows = parse_log(tmp_path / "logs" / "D1.out")
    assert len(rows) == 800 and rows[-1]["it"] == 40000 and rows[0]["nse8"] == pytest.approx(0.010)


def test_summary_tables_and_figures(tmp_path):
    fake_tree(tmp_path)
    res = summarize(tmp_path, last_k_ckpts=3)
    assert list(res["runs"]) == ["D1", "D7", "D40", "Dall"]
    m = [res["runs"][r]["sets"]["motion"]["mean"] for r in res["runs"]]
    assert m == sorted(m)  # the synthetic gain shows up in order
    assert res["runs"]["Dall"]["sets"]["motion"]["iterations"] == [32000, 36000, 40000]
    assert res["runs"]["D1"]["pooled_nse8"] == pytest.approx(0.02)
    c = res["runs"]["Dall"]["sets"]["train"]["to_target"]["0.85"]
    assert c is not None and c["gpu_hours"] > 0
    md = markdown(res)
    assert "| Dall | 254.4 |" in md and "Compute to a mean return of 0.9" in md
    pytest.importorskip("matplotlib")
    from analyze import figures

    written = figures(res, tmp_path / "fig")
    assert len(written) == 4 and all(p.stat().st_size > 10_000 for p in written)


def test_partial_results_are_fine(tmp_path):
    fake_tree(tmp_path, runs=("D40",))
    res = summarize(tmp_path)
    assert list(res["runs"]) == ["D40"]


def test_resumed_log_is_stitched(tmp_path):
    lines = [f"it {it} | sim steps 1 | {it / 1000:.2f} h | 2.3 it/s | train_ret 0.5 (n=1) | wm nse1 0.01 nse8 0.02 | x" for it in range(1000, 6001, 1000)]
    lines.append("[INFO] resumed from ckpt_0004000.pt at iteration 4000 (1 sim steps)")
    lines += [f"it {it} | sim steps 1 | {0.2 + (it - 4000) / 1000:.2f} h | 2.3 it/s | train_ret 0.6 (n=1) | wm nse1 0.01 nse8 0.02 | x" for it in range(5000, 8001, 1000)]
    (tmp_path / "r.out").write_text("\n".join(lines))
    rows = parse_log(tmp_path / "r.out")
    assert [r["it"] for r in rows] == [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    assert rows[4]["hours"] == pytest.approx(4.0 + 1.2) and rows[4]["train_ret"] == 0.6


def test_per_set_world_model_figures(tmp_path):
    fake_tree(tmp_path)
    rep = {}
    for it in range(4000, 40001, 4000):
        rep[f"ckpt_{it:07d}"] = {s: {r: {"nse_1step": 0.01, "nse_8step": 0.02 + 0.01 * j + 0.001 * k}
                                     for j, r in enumerate(("D1", "D7", "D40", "Dall"))}
                                 for k, s in enumerate(("motion", "performer", "lafan1"))}
    (tmp_path / "results" / "wm_error_sets.json").write_text(json.dumps(rep))
    res = summarize(tmp_path)
    assert res["runs"]["D1"]["wm_sets"]["lafan1"]["mean"] == pytest.approx(0.022)
    assert len(res["runs"]["D1"]["online_nse8_curve"]["nse8"]) == 10
    pytest.importorskip("matplotlib")
    from analyze import figures

    names = [p.name for p in figures(res, tmp_path / "fig")]
    assert "fig3_wm_vs_data.png" in names and "fig3b_wm_curves.png" in names and "fig3_wm_error.png" not in names
