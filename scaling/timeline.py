"""The project timeline as a chart (README, "How this was built"). Times are Pacific."""

from __future__ import annotations

from pathlib import Path

import tyro

# (phase, start, end) in hours from Wednesday 00:00 Pacific
PHASES = [
    ("Idea and literature check", 8.5, 9.5),
    ("Experiment design and plan", 9.5, 10.5),
    ("Code and tests", 10.5, 11 + 40 / 60),
    ("Accounts, GPUs, data pipeline", 11 + 40 / 60, 15.0),
    ("Smoke test and launch", 15.0, 16.0),
    ("Training: 4 runs on 2 H100s", 16.0, 24 + 3.0),
    ("Evaluation and analysis", 24 + 4.0, 24 + 4.75),
]


def main(out: Path = Path(__file__).resolve().parent / "figures" / "fig5_timeline.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    work, train = "#2a78d6", "#86b6ef"
    fig, ax = plt.subplots(figsize=(9.6, 3.2), facecolor=surface)
    ax.set_facecolor(surface)
    for i, (name, a, b) in enumerate(PHASES):
        y = len(PHASES) - 1 - i
        ax.barh(y, b - a, left=a, height=0.62, color=train if name.startswith("Training") else work, edgecolor=surface, linewidth=2)
        ax.text(a - 0.25, y, name, ha="right", va="center", fontsize=9, color=ink)
    ticks = [8, 12, 16, 20, 24, 28]
    ax.set_xticks(ticks)
    ax.set_xticklabels([("Wed " if t < 24 else "Thu ") + f"{t % 24:02d}:00" for t in ticks], fontsize=9, color=ink2)
    ax.set_yticks([])
    ax.set_xlim(1.0, 29.5)
    ax.set_ylim(-0.7, len(PHASES) - 0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ink2)
    ax.grid(True, axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("20 hours from idea to results (Pacific time)", loc="left", color=ink, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)
