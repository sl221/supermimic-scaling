"""Cut every clip of a compact library into segments of at most ``--seconds`` (step 6 of the plan).

Held-out evaluation plays each segment once from its first frame, so long clips do not dominate the mean.
A tail shorter than ``--min-seconds`` is merged into the previous segment; a clip shorter than that stays whole.

  python scaling/segment_library.py --library heldout_motion.pt --out heldout_motion_seg10.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
import tyro

FPS = 50


def segment_bounds(length: int, seg: int, min_len: int) -> list[tuple[int, int]]:
    if length <= seg:
        return [(0, length)]
    bounds = [(s, min(s + seg, length)) for s in range(0, length, seg)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_len:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    return bounds


def segment(pack: dict, seconds: float, min_seconds: float) -> dict:
    seg, min_len = int(round(seconds * FPS)), int(round(min_seconds * FPS))
    refs, names, lengths = [], [], []
    for name, start, length in zip(pack["names"], pack["starts"], pack["lengths"]):
        clip = pack["ref"][int(start) : int(start) + int(length)]
        for k, (a, b) in enumerate(segment_bounds(int(length), seg, min_len)):
            refs.append(clip[a:b])
            names.append(f"{name}__seg{k:02d}")
            lengths.append(b - a)
    ref = torch.cat(refs, 0)
    starts = torch.cumsum(torch.tensor([0] + lengths[:-1]), 0).tolist()
    return {"ref": ref, "names": names, "starts": starts, "lengths": lengths}


def main(library: Path, out: Path, seconds: float = 10.0, min_seconds: float = 2.0) -> None:
    pack = torch.load(library, map_location="cpu", weights_only=False)
    out_pack = segment(pack, seconds, min_seconds)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_pack, out)
    print(f"saved {out}: {len(pack['names'])} clips -> {len(out_pack['names'])} segments, {out_pack['ref'].shape[0]} frames")


if __name__ == "__main__":
    tyro.cli(main)
