"""Held-out evaluation for the scaling study (step 7 of the plan).

For one checkpoint and each held-out library (10 s segments), this:

1. plays every segment once from its first frame (paper Appendix F protocol) and records mean return;
2. records every simulator transition and saves ``--windows`` random 8-step windows (states, executed
   actions, sim context). ``wm_error.py`` pools these windows across all runs and scores every world model on
   the same transitions.

  python scaling/eval_heldout.py --ckpt logs/D40/ckpt_0040000.pt \
      --library motion=$DATA/lib/heldout_motion_seg10.pt performer=$DATA/lib/heldout_performer_seg10.pt \
                lafan1=$DATA/lib/lafan1_seg10.pt --out-dir results/D40/ckpt_0040000

Writes ``<out-dir>/<set>.json`` (per-segment results + summary) and ``<out-dir>/<set>_windows.pt``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))



def replay_rows_needed(lengths: list[int], num_envs: int) -> int:
    """Rows per env so the replay buffer never wraps: env e plays clips e, e+N, ...; each clip writes its
    steps plus one terminal row."""
    n = min(num_envs, len(lengths))
    return max(sum(lengths[e::n]) + len(lengths[e::n]) for e in range(n)) + 8


def main(
    ckpt: Path,
    library: list[str],
    out_dir: Path,
    num_envs: int = 1024,
    windows: int = 8192,
    k: int = 8,
    device: str = "cuda:0",
    seed: int = 0,
) -> None:
    """``--library`` takes ``name=path`` pairs after ONE flag (``--library a=x b=y``); repeating the flag keeps
    only the last pair."""
    from supermimic.config import obs_mode_from_config
    from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize

    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(device=device, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], obs_mode=obs_mode_from_config(cfg))
    # merge into an existing summary, so sets evaluated in separate passes (held-out now, training sample later)
    # end up in one file
    prev = out_dir / "summary.json"
    summary_all = json.loads(prev.read_text()) if prev.exists() else {}
    summary_all["ckpt"] = str(ckpt)
    print(f"held-out sets: {[item.split('=', 1)[0] for item in library]}", flush=True)
    for item in library:
        name, path = item.split("=", 1)
        pack = torch.load(path, map_location="cpu", weights_only=False)
        lengths = [int(l) for l in pack["lengths"]]
        rows = replay_rows_needed(lengths, num_envs)
        print(f"[{name}] {len(lengths)} segments, replay rows per env {rows}", flush=True)
        saved = {}

        def grab(env, name=name):
            buf = env.recorder_manager.get_term("replay").buffer
            g = torch.Generator(device="cpu").manual_seed(seed)
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            w = buf.sample_windows(windows, k)
            saved["n_rows"] = int(buf.valid.sum())
            torch.save({"states": w.states.cpu(), "a_exec": w.a_exec.cpu(), "xi": w.xi.cpu(), "k": k, "set": name, "ckpt": str(ckpt)}, out_dir / f"{name}_windows.pt")

        t0 = time.time()
        results = run_benchmark(policy, [path], compact_library=True, num_clips=len(lengths), record_transitions=True, replay_capacity=rows, on_finish=grab, verbose=False, **common)
        s = summarize(results)
        s["seconds"] = round(time.time() - t0, 1)
        s["replay_rows"] = saved.get("n_rows")
        save_results(results, out_dir / f"{name}.json")
        summary_all[name] = s
        print(f"[{name}] mean return {s['mean_norm_return']:.4f} | failed {s['failed']}/{s['clips']} | {s['seconds']} s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary_all, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
