"""Pooled 8-step world-model error for the scaling study (step 8 of the plan).

Every run's world model is scored on the *same* held-out transitions: the windows that ``eval_heldout.py``
saved for all runs' policies, pooled. This separates world-model quality from which states a given policy
happens to visit. The error is the paper's: open-loop from the window's first logged state on logged executed
actions, state-only, normalised by the fixed nominal scales (``StateSpec``); the 8-step value is the error at
step k, as the training loop logs ``wm/online_nse_8step``.

  python scaling/wm_error.py --ckpt logs/D1/ckpt_0040000.pt logs/D7/ckpt_0040000.pt ... \
      --windows "results/*/ckpt_0040000/*_windows.pt" --out results/wm_error_ckpt_0040000.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_world_model(cfg: dict, device: str):
    """The world model as the training loop builds it (its analytic heads need the G1 joint limits and default
    pose, read from a one-env simulator)."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    from supermimic.world_model.state_spec import StateSpecCfg

    tup = lambda d: {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}  # noqa: E731
    raw = dict(cfg["world_model"])
    raw["state_spec"] = StateSpecCfg(**tup(raw.get("state_spec", {})))
    wcfg = WorldModelCfg(**tup(raw))
    lib = cfg["data"].get("library")
    args = dict(num_envs=1, record_transitions=False, domain_randomization=False, obs_noise=False, reset_noise=False)
    if lib:
        env = make_env(SuperMimicEnvCfgArgs(npz_files=[lib], compact_library=True, **args), device=device)
    else:
        from supermimic.envs.mdp.motion_library import discover_npz

        env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(discover_npz(Path(cfg["data"]["npz_dir"]))[0])], **args), device=device)
    try:
        robot = env.scene["robot"]
        fk = G1Kinematics(load_g1_model(), device=device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        return WorldModel(wcfg, heads, device)
    finally:
        env.close()


def load_windows(pattern: str) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no window files match {pattern}")
    parts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    ks = {p["k"] for p in parts}
    if len(ks) != 1:
        raise SystemExit(f"window files mix k values {ks}")
    pooled = {
        "states": torch.cat([p["states"] for p in parts], 1),  # (k+1, n, S)
        "a_exec": torch.cat([p["a_exec"] for p in parts], 1),  # (k, n, A)
        "xi": torch.cat([p["xi"] for p in parts], 0),  # (n, X)
    }
    counts = {}
    for f, p in zip(files, parts):
        counts[f] = int(p["xi"].shape[0])
    return pooled, counts


@torch.no_grad()
def score(wm, w: dict[str, torch.Tensor], device: str, batch: int = 4096) -> list[float]:
    n = w["xi"].shape[0]
    tot = None
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        e = wm.rollout_error(w["states"][:, sl].to(device), w["a_exec"][:, sl].to(device), w["xi"][sl].to(device))
        e = e * (min(i + batch, n) - i)
        tot = e if tot is None else tot + e
    return (tot / n).cpu().tolist()


def main(ckpt: list[Path], windows: str, out: Path, device: str = "cuda:0") -> None:
    pooled, counts = load_windows(windows)
    print(f"{pooled['xi'].shape[0]} pooled windows from {len(counts)} files")
    report = {"windows": counts, "runs": {}}
    wm = None
    for c in ckpt:
        ck = torch.load(c, map_location=device, weights_only=False)
        if wm is None:
            wm = build_world_model(ck["config"], device)
        if tuple(ck["config"]["world_model"].get("hidden", ())) != tuple(wm.cfg.hidden):
            wm = build_world_model(ck["config"], device)  # a different width needs its own network
        wm.load_state_dict(ck["wm"])
        wm.eval()
        err = score(wm, pooled, device)
        report["runs"][str(c)] = {"per_step": err, "nse_1step": err[0], "nse_8step": err[-1]}
        print(f"{c}: 1-step {err[0]:.5f}  8-step {err[-1]:.5f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
