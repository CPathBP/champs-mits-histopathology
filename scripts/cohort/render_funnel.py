"""Render the cohort funnel from the cohort artifacts.

Reads ``funnel.json`` and writes ``funnel_cases.csv`` (the report arm),
``funnel_slides.csv`` (the slide arm, with lung, liver and case counts)
and ``fig_funnel.png``. No count is computed here.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def draw(boxes, ax, title):
    """One column of boxes, top down, with the drop reason beside each step."""
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(boxes) + 0.5)
    ax.axis("off")
    for i, box in enumerate(boxes):
        y = len(boxes) - i
        ax.add_patch(Rectangle((0.2, y - 0.35), 4.6, 0.7, fill=False, linewidth=0.8))
        ax.text(2.5, y, f"{box['stage']}\nn = {box['n']:,}", ha="center", va="center",
                fontsize=7)
        if box["n_dropped"]:
            ax.text(5.1, y, f"- {box['n_dropped']:,}: {box['drop_reason']}", va="center",
                    fontsize=6.5)
        if i:
            ax.annotate("", xy=(2.5, y + 0.35), xytext=(2.5, y + 0.65),
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    funnel = json.loads((args.cohort_dir / "funnel.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(funnel["report_arm"])
    slides = pd.DataFrame(funnel["slide_arm"])
    cases.to_csv(args.out_dir / "funnel_cases.csv", index=False)
    slides.to_csv(args.out_dir / "funnel_slides.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1.6]})
    draw(funnel["report_arm"], axes[0], "Cases")
    draw(funnel["slide_arm"], axes[1], "Slides")
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_funnel.png", dpi=200)
    print(cases.to_string(index=False))
    print(slides.to_string(index=False))


if __name__ == "__main__":
    main()
