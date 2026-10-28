"""CPU checks for the evaluation helpers (the GPU paths are exercised by the smoke test)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_heldout import replay_rows_needed  # noqa: E402
from wm_error import load_windows, score  # noqa: E402


def test_replay_rows_cover_every_env():
    lengths = [100, 250, 300, 50, 500]
    # 2 envs: env0 plays clips 0,2,4 (900 steps + 3 terminal rows); env1 plays 1,3 (300 + 2)
    assert replay_rows_needed(lengths, 2) == 903 + 8
    assert replay_rows_needed(lengths, 10) == 501 + 8  # more envs than clips: one clip each


def _fake_windows(path, n, k=8, s=5, a=3, x=2, fill=0.0):
    torch.save({"states": torch.full((k + 1, n, s), fill), "a_exec": torch.zeros(k, n, a), "xi": torch.zeros(n, x), "k": k}, path)


def test_windows_pool_across_files(tmp_path):
    _fake_windows(tmp_path / "r1_windows.pt", 10)
    _fake_windows(tmp_path / "r2_windows.pt", 6)
    pooled, counts = load_windows(str(tmp_path / "*_windows.pt"))
    assert pooled["states"].shape == (9, 16, 5) and pooled["xi"].shape == (16, 2)
    assert sorted(counts.values()) == [6, 10]


class _MeanWM:
    """rollout_error = per-step batch mean of the first state feature, like the real one averages over the batch."""

    def rollout_error(self, states, a_exec, xi):
        return states[1:, :, 0].mean(dim=1)


def test_score_is_a_true_mean_across_uneven_batches():
    k, n = 8, 10
    states = torch.zeros(k + 1, n, 5)
    states[:, :, 0] = torch.arange(n, dtype=torch.float32)  # sample i has error i
    w = {"states": states, "a_exec": torch.zeros(k, n, 3), "xi": torch.zeros(n, 2)}
    err = score(_MeanWM(), w, "cpu", batch=4)  # batches of 4, 4, 2
    assert all(abs(e - 4.5) < 1e-6 for e in err)
