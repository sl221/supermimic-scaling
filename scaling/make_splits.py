"""Held-out splits for the scaling study (step 4 of the plan).

Carves two held-out sets out of the filtered BONES-SEED clips *before* any training subset is drawn:

* unseen performers: every clip of a random set of performers (about ``--target-clips`` clips);
* unseen motions:   every clip, by any performer, of a random set of motion groups (about ``--target-clips``).

Everything else is the training pool. A clip that belongs to a held-out performer *and* a held-out motion
goes to the performer set, so no clip is in two sets and no held-out motion or performer appears in training.

BONES-SEED mirrors every take (``*_M`` clips) and gives the mirror its own ``content_name`` (``body_check`` /
``body_check_M``). The motion group strips that suffix, so a held-out motion takes its mirrors with it; otherwise
the mirrored copy would leak the held-out motion into training. Very common motions (more than
``--max-motion-clips`` clips, e.g. ``body_check`` with 400 performers) are never held out, so one motion cannot
dominate the held-out set; they stay in training.

  python scaling/make_splits.py --library $DATA/bones_seed_compact.pt --metadata seed_metadata_v004.csv \
      --out-dir $DATA/splits

The defaults match ``seed_metadata_v004.csv`` (``filename``, ``actor_uid``, ``content_name``); ``--show-columns``
lists the columns of another metadata file.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import tyro

FPS = 50


def read_library_index(library: Path) -> dict[str, int]:
    """Clip name -> length in frames (50 Hz), from a compact library."""
    import torch

    pack = torch.load(library, map_location="cpu", weights_only=False)
    return {str(n): int(l) for n, l in zip(pack["names"], pack["lengths"])}


def motion_group(content: str, mirror_suffix: str) -> str:
    """A take and its mirror share one motion group."""
    return content[: -len(mirror_suffix)] if mirror_suffix and content.endswith(mirror_suffix) else content


def read_metadata(path: Path, name_col: str, performer_col: str, motion_col: str, mirror_suffix: str = "_M") -> dict[str, tuple[str, str]]:
    """Clip name -> (performer, motion group). The name column holds the CSV/npz stem."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = Path(row[name_col]).stem
            out[name] = (row[performer_col].strip(), motion_group(row[motion_col].strip(), mirror_suffix))
    return out


def pick_groups(groups: dict[str, list[str]], target: int, rng: random.Random, max_size: int | None = None) -> set[str]:
    """Random whole groups until their clips reach ``target`` (the last group may overshoot). Groups larger
    than ``max_size`` are never picked."""
    keys = sorted(k for k in groups if max_size is None or len(groups[k]) <= max_size)
    rng.shuffle(keys)
    chosen, n = set(), 0
    for k in keys:
        if n >= target:
            break
        chosen.add(k)
        n += len(groups[k])
    return chosen


def split(
    lengths: dict[str, int], meta: dict[str, tuple[str, str]], target_clips: int, seed: int, max_motion_clips: int | None = None
) -> dict[str, list[str]]:
    names = sorted(n for n in lengths if n in meta)
    rng = random.Random(seed)
    by_performer: dict[str, list[str]] = {}
    for n in names:
        by_performer.setdefault(meta[n][0], []).append(n)
    held_perf = pick_groups(by_performer, target_clips, rng)
    perf_set = {n for n in names if meta[n][0] in held_perf}

    rest = [n for n in names if n not in perf_set]
    by_motion: dict[str, list[str]] = {}
    for n in rest:
        by_motion.setdefault(meta[n][1], []).append(n)
    held_motion = pick_groups(by_motion, target_clips, rng, max_motion_clips)
    # every clip of a held-out motion leaves training, whoever performed it
    motion_set = {n for n in rest if meta[n][1] in held_motion}
    train = [n for n in rest if n not in motion_set]
    return {
        "heldout_performer": sorted(perf_set),
        "heldout_motion": sorted(motion_set),
        "train_pool": sorted(train),
        "_held_performers": sorted(held_perf),
        "_held_motions": sorted(held_motion),
    }


def hours(names: list[str], lengths: dict[str, int]) -> float:
    return sum(lengths[n] for n in names) / FPS / 3600


def main(
    library: Path,
    metadata: Path,
    out_dir: Path,
    name_col: str = "filename",
    performer_col: str = "actor_uid",
    motion_col: str = "content_name",
    mirror_suffix: str = "_M",
    max_motion_clips: int = 100,
    target_clips: int = 2000,
    seed: int = 0,
    show_columns: bool = False,
) -> None:
    if show_columns:
        with open(metadata, newline="") as f:
            row = next(csv.DictReader(f))
        for k, v in row.items():
            print(f"{k:32s} e.g. {v}")
        return
    lengths = read_library_index(library)
    meta = read_metadata(metadata, name_col, performer_col, motion_col, mirror_suffix)
    missing = [n for n in lengths if n not in meta]
    if missing:
        print(f"[WARN] {len(missing)} library clips have no metadata row and are left out, e.g. {missing[:3]}")
    s = split(lengths, meta, target_clips, seed, max_motion_clips)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": seed,
        "target_clips": target_clips,
        "columns": [name_col, performer_col, motion_col],
        "mirror_suffix": mirror_suffix,
        "max_motion_clips": max_motion_clips,
    }
    for key in ("heldout_performer", "heldout_motion", "train_pool"):
        (out_dir / f"{key}.txt").write_text("".join(n + "\n" for n in s[key]))
        report[key] = {"clips": len(s[key]), "hours": round(hours(s[key], lengths), 3)}
    report["held_performers"] = s["_held_performers"]
    report["held_motion_groups"] = len(s["_held_motions"])
    (out_dir / "splits_report.json").write_text(json.dumps(report, indent=1))
    for key in ("heldout_performer", "heldout_motion", "train_pool"):
        print(f"{key:18s} {report[key]['clips']:7d} clips {report[key]['hours']:8.2f} h")


if __name__ == "__main__":
    tyro.cli(main)
