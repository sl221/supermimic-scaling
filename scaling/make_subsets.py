"""Nested training subsets for the scaling study (step 5 of the plan).

One random order of the training pool; each subset is the shortest prefix of that order reaching its target
hours, so every subset contains all the smaller ones. The last subset is the whole pool.

  python scaling/make_subsets.py --library $DATA/bones_seed_compact.pt --train-pool $DATA/splits/train_pool.txt \
      --out-dir $DATA/subsets --hours 1 7 40
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import tyro

from make_splits import FPS, read_library_index


def nested_prefixes(names: list[str], lengths: dict[str, int], targets_h: list[float], seed: int) -> dict[str, list[str]]:
    order = sorted(names)
    random.Random(seed).shuffle(order)
    out, frames, i = {}, 0, 0
    for h in sorted(targets_h):
        goal = h * 3600 * FPS
        while i < len(order) and frames < goal:
            frames += lengths[order[i]]
            i += 1
        if frames < goal:
            raise SystemExit(f"training pool has only {frames / FPS / 3600:.2f} h, less than {h} h")
        out[f"D{h:g}"] = order[:i]
    out["Dall"] = order
    return out


def main(library: Path, train_pool: Path, out_dir: Path, hours: list[float] = [1.0, 7.0, 40.0], seed: int = 0) -> None:
    lengths = read_library_index(library)
    pool = [l.strip() for l in open(train_pool) if l.strip()]
    subsets = nested_prefixes(pool, lengths, hours, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": seed}
    for key, names in subsets.items():
        (out_dir / f"{key}.txt").write_text("".join(n + "\n" for n in sorted(names)))
        h = sum(lengths[n] for n in names) / FPS / 3600
        report[key] = {"clips": len(names), "hours": round(h, 3)}
        print(f"{key:6s} {len(names):7d} clips {h:8.2f} h")
    (out_dir / "subsets_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
