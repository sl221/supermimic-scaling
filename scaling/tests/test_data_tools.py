"""Checks for the scaling study's data tools, on a synthetic library (CPU only)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_splits import motion_group, split  # noqa: E402
from make_subsets import nested_prefixes  # noqa: E402
from segment_library import segment, segment_bounds  # noqa: E402


def synthetic(n_clips=3000, n_perf=40, n_motion=300, seed=1):
    rng = random.Random(seed)
    lengths, meta = {}, {}
    for i in range(n_clips):
        name = f"clip{i:05d}"
        lengths[name] = rng.randint(100, 1500)  # 2-30 s at 50 Hz
        meta[name] = (f"p{rng.randrange(n_perf)}", f"m{rng.randrange(n_motion)}")
    return lengths, meta


def test_splits_are_disjoint_and_hold_out_whole_groups():
    lengths, meta = synthetic()
    s = split(lengths, meta, target_clips=300, seed=0)
    perf, mot, train = map(set, (s["heldout_performer"], s["heldout_motion"], s["train_pool"]))
    assert not (perf & mot) and not (perf & train) and not (mot & train)
    assert perf | mot | train == set(lengths)
    held_p, held_m = set(s["_held_performers"]), set(s["_held_motions"])
    assert all(meta[n][0] not in held_p for n in train | mot)
    assert all(meta[n][1] not in held_m for n in train)
    assert len(perf) >= 300 and len(mot) >= 300


def test_splits_are_reproducible():
    lengths, meta = synthetic()
    assert split(lengths, meta, 300, seed=0) == split(lengths, meta, 300, seed=0)
    assert split(lengths, meta, 300, seed=0) != split(lengths, meta, 300, seed=1)


def test_subsets_are_nested_and_reach_their_hours():
    lengths, _ = synthetic()
    pool = sorted(lengths)
    subs = nested_prefixes(pool, lengths, [1.0, 2.0, 5.0], seed=0)
    keys = ["D1", "D2", "D5", "Dall"]
    for a, b in zip(keys, keys[1:]):
        assert set(subs[a]) < set(subs[b])
    for key, h in (("D1", 1.0), ("D2", 2.0), ("D5", 5.0)):
        got = sum(lengths[n] for n in subs[key]) / 50 / 3600
        assert h <= got < h * 1.01 + 30 / 3600  # overshoot at most one clip
    assert set(subs["Dall"]) == set(pool)


def test_segment_bounds():
    assert segment_bounds(400, 500, 100) == [(0, 400)]
    assert segment_bounds(1000, 500, 100) == [(0, 500), (500, 1000)]
    assert segment_bounds(1050, 500, 100) == [(0, 500), (500, 1050)]  # 1 s tail merged
    assert segment_bounds(1200, 500, 100) == [(0, 500), (500, 1000), (1000, 1200)]


def test_segment_library_keeps_every_frame_in_order():
    lens = [400, 1050, 1200]
    ref = torch.arange(sum(lens) * 2, dtype=torch.float32).reshape(-1, 2)
    starts = [0, 400, 1450]
    out = segment({"ref": ref, "names": ["a", "b", "c"], "starts": starts, "lengths": lens}, 10.0, 2.0)
    assert out["names"] == ["a__seg00", "b__seg00", "b__seg01", "c__seg00", "c__seg01", "c__seg02"]
    assert sum(out["lengths"]) == ref.shape[0]
    assert torch.equal(out["ref"], ref)
    for s, l, n in zip(out["starts"], out["lengths"], out["names"]):
        assert out["ref"][s : s + l].shape[0] == l, n


def test_mirrors_share_a_motion_group_and_never_leak():
    assert motion_group("body_check_M", "_M") == "body_check" and motion_group("body_check", "_M") == "body_check"
    lengths, meta = synthetic()
    # give every clip a mirror with the same performer, as BONES-SEED does
    for n in list(lengths):
        lengths[n + "_M"] = lengths[n]
        meta[n + "_M"] = meta[n]
    s = split(lengths, meta, target_clips=300, seed=0)
    train = set(s["train_pool"])
    for held in ("heldout_motion", "heldout_performer"):
        for n in s[held]:
            twin = n[:-2] if n.endswith("_M") else n + "_M"
            assert twin not in train, (held, n)


def test_large_motion_groups_are_never_held_out():
    lengths, meta = synthetic()
    for i in range(500):  # one very common motion
        lengths[f"common{i:03d}"] = 300
        meta[f"common{i:03d}"] = (f"p{i % 40}", "common")
    for seed in range(5):
        s = split(lengths, meta, target_clips=300, seed=seed, max_motion_clips=100)
        assert "common" not in s["_held_motions"]
