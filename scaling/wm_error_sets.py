"""World-model error per held-out set, at every checkpoint step (complements the pooled ``wm_error.py``).

For each checkpoint step and each held-out set, every run's world model is scored on the same windows: that set's
windows recorded by all runs' policies at that step (``eval_heldout.py`` output). One process, so the simulator used to
build the world model is created once.

  python scaling/wm_error_sets.py --data $DATA --out $DATA/results/wm_error_sets.json

Writes ``{step: {set: {run: {"nse_1step", "nse_8step"}}}}``.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(
    data: Path,
    out: Path,
    runs: list[str] = ["D1", "D7", "D40", "Dall"],
    sets: list[str] = ["motion", "performer", "lafan1"],
    device: str = "cuda:0",
) -> None:
    from wm_error import build_world_model, load_windows, score

    steps = sorted({Path(p).name for r in runs for p in glob.glob(str(data / "results" / r / "ckpt_00*"))})
    report = json.loads(out.read_text()) if out.exists() else {}
    wm = None
    for step in steps:
        if step in report and all(s in report[step] for s in sets):
            continue
        windows = {}
        for s in sets:
            pattern = str(data / "results" / "*" / step / f"{s}_windows.pt")
            if glob.glob(pattern):
                windows[s], _ = load_windows(pattern)
        report[step] = {s: {} for s in windows}
        for run in runs:
            ck_path = data / "logs" / run / f"{step}.pt"
            if not ck_path.exists():
                continue
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            if wm is None:
                wm = build_world_model(ck["config"], device)
            wm.load_state_dict(ck["wm"])
            wm.eval()
            for s, w in windows.items():
                err = score(wm, w, device)
                report[step][s][run] = {"nse_1step": err[0], "nse_8step": err[-1]}
        line = " | ".join(f"{s} " + " ".join(f"{r}={1e3 * v['nse_8step']:.1f}" for r, v in report[step][s].items()) for s in windows)
        print(f"{step}: {line}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))  # after every step, so an interruption loses little


if __name__ == "__main__":
    tyro.cli(main)
