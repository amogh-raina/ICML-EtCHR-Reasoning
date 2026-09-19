#!/usr/bin/env python3
"""
Build the paper-style four-panel heatmap figure comparing human annotations
with pooled LLM-judge scores.

Inputs:
    data/annotation/human_annotations.json
    data/judge_eval_summary/long_rows.csv

Output:
    data/annotation/plots/human_model_step_heatmaps.png
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).parent.parent
HUMAN_JSON = BASE_DIR / "data" / "annotation" / "human_annotations.json"
JUDGE_CSV = BASE_DIR / "data" / "judge_eval_summary" / "long_rows.csv"
PLOT_DIR = BASE_DIR / "data" / "annotation" / "plots"

CATEGORIES = [
    ("non_curated", "A: Non-Curated"),
    ("instruction", "B: Curated\n(Expert)"),
    ("gpt_assisted", "C: Curated\n(Guide)"),
]
STEPS = ["step_1", "step_2", "step_3", "step_4"]
STEP_LABELS = ["Step 1", "Step 2", "Step 3", "Step 4"]


def human_matrix(metric_key: str) -> np.ndarray:
    data = json.loads(HUMAN_JSON.read_text(encoding="utf-8"))
    matrix = np.full((len(STEPS), len(CATEGORIES)), np.nan)

    for s_idx, step in enumerate(STEPS):
        for c_idx, (cat_key, _) in enumerate(CATEGORIES):
            vals = data.get(cat_key, {}).get(step, {}).get(metric_key, [])
            matrix[s_idx, c_idx] = np.mean(vals) if vals else np.nan

    return matrix


def judge_matrix(metric_suffix: str) -> np.ndarray:
    csv.field_size_limit(sys.maxsize)
    with JUDGE_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    matrix = np.full((len(STEPS), len(CATEGORIES)), np.nan)
    for s_idx, step in enumerate(STEPS):
        col = f"{step}.{metric_suffix}"
        for c_idx, (cat_key, _) in enumerate(CATEGORIES):
            vals = [float(r[col]) for r in rows if r["category"] == cat_key]
            matrix[s_idx, c_idx] = np.mean(vals) if vals else np.nan

    return matrix


def draw_heatmap(
    ax,
    matrix: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    show_ylabels: bool,
):
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([label for _, label in CATEGORIES], fontsize=10)
    ax.set_yticks(range(len(STEPS)))
    if show_ylabels:
        ax.set_yticklabels(STEP_LABELS, fontsize=10)
    else:
        ax.set_yticklabels([])

    threshold = vmin + 0.6 * (vmax - vmin)
    for s_idx in range(matrix.shape[0]):
        for c_idx in range(matrix.shape[1]):
            val = matrix[s_idx, c_idx]
            text_color = "white" if val < threshold else "black"
            ax.text(
                c_idx,
                s_idx,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    return im


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    human_occ = human_matrix("occurrence")
    judge_occ = judge_matrix("occurrence")
    human_comp = human_matrix("comprehensiveness")
    judge_comp = judge_matrix("comprehensiveness")

    fig = plt.figure(figsize=(20.48, 4.55), dpi=100)
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=[1, 1, 0.04, 0.18, 1, 1, 0.04],
        left=0.045,
        right=0.965,
        bottom=0.14,
        top=0.81,
        wspace=0.12,
    )

    ax_occ_h = fig.add_subplot(grid[0, 0])
    ax_occ_m = fig.add_subplot(grid[0, 1])
    cax_occ = fig.add_subplot(grid[0, 2])
    ax_comp_h = fig.add_subplot(grid[0, 4])
    ax_comp_m = fig.add_subplot(grid[0, 5])
    cax_comp = fig.add_subplot(grid[0, 6])

    im_occ = draw_heatmap(
        ax_occ_h, human_occ, "Human (Student)", 0.0, 1.0, show_ylabels=True
    )
    draw_heatmap(ax_occ_m, judge_occ, "Model Judges", 0.0, 1.0, show_ylabels=False)
    im_comp = draw_heatmap(
        ax_comp_h, human_comp, "Human (Student)", 0.0, 5.0, show_ylabels=True
    )
    draw_heatmap(ax_comp_m, judge_comp, "Model Judges", 0.0, 5.0, show_ylabels=False)

    ax_occ_h.set_ylabel("Reasoning Step", fontsize=11)

    cbar_occ = fig.colorbar(im_occ, cax=cax_occ)
    cbar_occ.set_label("Mean Occurrence (0-1)", fontsize=10, labelpad=8)
    cbar_occ.ax.tick_params(labelsize=10)

    cbar_comp = fig.colorbar(im_comp, cax=cax_comp)
    cbar_comp.set_label("Mean Comprehensiveness (0-5)", fontsize=10, labelpad=8)
    cbar_comp.ax.tick_params(labelsize=10)

    occ_center = (ax_occ_h.get_position().x0 + ax_occ_m.get_position().x1) / 2
    comp_center = (ax_comp_h.get_position().x0 + ax_comp_m.get_position().x1) / 2

    fig.text(
        occ_center,
        0.985,
        "Step Occurrence",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        comp_center,
        0.985,
        "Step Comprehensiveness",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
    )

    out_png = PLOT_DIR / "human_model_step_heatmaps.png"
    out_pdf = PLOT_DIR / "human_model_step_heatmaps.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
