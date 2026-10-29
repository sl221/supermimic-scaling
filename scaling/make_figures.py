"""Regenerate every figure in the README from the data stored in the repository (no GPU, no dataset needed).

  python scaling/make_figures.py

Reads ``scaling/results/results.json`` (written by ``analyze.py`` from the run logs and evaluation summaries) and
writes ``scaling/figures/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))

from analyze import figures  # noqa: E402
from timeline import main as timeline  # noqa: E402

if __name__ == "__main__":
    res = json.loads((here / "results" / "results.json").read_text())
    for p in figures(res, here / "figures"):
        print(f"wrote {p}")
    timeline(here / "figures" / "fig5_timeline.png")
