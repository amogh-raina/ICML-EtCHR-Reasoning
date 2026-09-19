#!/usr/bin/env python3
"""
Render figures summarising LLM-judge evaluation scores per prompt category.

Reads `data/judge_eval_summary/long_rows.csv` (produced by
`Scripts/compute_eval_scores.py`) and generates three figures, pooled
across the three judge models:

    1. step_occurrence_heatmap.png
       4 steps × 3 prompt categories. Cell = mean occurrence (0–1).
    2. step_comprehensiveness_heatmap.png
       4 steps × 3 prompt categories. Cell = mean comprehensiveness (0–5).
    3. overall_conciseness_bar.png
       Mean ± std overall conciseness, one bar per category.

Output: data/judge_eval_summary/plots/
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR  = Path(__file__).parent.parent
LONG_CSV  = BASE_DIR / "data" / "judge_eval_summary" / "long_rows.csv"
PLOT_DIR  = BASE_DIR / "data" / "judge_eval_summary" / "plots"

# Display order + labels matching the paper's LaTeX table
CATEGORIES = [
    ("non_curated",  "A: Non-Curated"),
    ("instruction",  "B: Curated\n(Expert)"),
    ("gpt_assisted", "C: Curated\n(Guide)"),
]
STEPS = ["step_1", "step_2", "step_3", "step_4"]
STEP_LABELS = ["Step 1", "Step 2", "Step 3", "Step 4"]

# Across-judges palette — fixed so the same colour means the same judge
# in every figure.
JUDGES = [
    ("claude",   "Claude",   "#5b8def"),
    ("deepseek", "DeepSeek", "#e0883d"),
    ("gpt5.5",   "GPT-5.5",  "#3aa17e"),
]

OCC_KEYS  = [f"{s}.occurrence"        for s in STEPS]
COMP_KEYS = [f"{s}.comprehensiveness" for s in STEPS]


def load_rows() -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with open(LONG_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def heatmap(rows: list[dict], metric_suffix: str, vmin: float, vmax: float,
            title: str, cbar_label: str, fname: str, fmt: str = ".2f") -> None:
    """Render a steps × categories heatmap with mean values."""
    # data[step_idx, cat_idx] = mean
    data     = np.zeros((len(STEPS), len(CATEGORIES)))
    counts   = np.zeros_like(data, dtype=int)
    for s_idx, step in enumerate(STEPS):
        col = f"{step}.{metric_suffix}"
        for c_idx, (cat_key, _) in enumerate(CATEGORIES):
            vals = [float(r[col]) for r in rows if r["category"] == cat_key]
            data[s_idx, c_idx]   = np.mean(vals) if vals else np.nan
            counts[s_idx, c_idx] = len(vals)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([label for _, label in CATEGORIES])
    ax.set_yticks(range(len(STEPS)))
    ax.set_yticklabels(STEP_LABELS)
    ax.set_xlabel("Prompt Setting")
    ax.set_ylabel("Reasoning Step")
    ax.set_title(title)

    # Annotate each cell with mean value, auto-contrast text
    threshold = vmin + 0.6 * (vmax - vmin)
    for s_idx in range(len(STEPS)):
        for c_idx in range(len(CATEGORIES)):
            v = data[s_idx, c_idx]
            color = "white" if v < threshold else "black"
            ax.text(c_idx, s_idx, format(v, fmt),
                    ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    out = PLOT_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def conciseness_bar(rows: list[dict]) -> None:
    """Mean ± std bar chart for overall_conciseness."""
    means, stds, ns = [], [], []
    for cat_key, _ in CATEGORIES:
        vals = [float(r["overall_conciseness"]) for r in rows
                if r["category"] == cat_key]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
        ns.append(len(vals))

    labels = [label for _, label in CATEGORIES]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(x, means, yerr=stds, capsize=8,
                  color=["#3b7dd8", "#3aa17e", "#d97f3a"],
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Overall Conciseness (1–5)")
    ax.set_title("Overall Conciseness by Prompt Setting\n(mean ± std, pooled across judge models)")
    ax.set_ylim(0, 5.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotate above each bar
    for bar, m, s, n in zip(bars, means, stds, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.10,
                f"{m:.2f} ± {s:.2f}\nn={n}",
                ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out = PLOT_DIR / "overall_conciseness_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Across-judge grouped bar charts
# ---------------------------------------------------------------------------

def _pooled_values(rows: list[dict], cat: str, judge: str,
                   metric_keys: list[str]) -> list[float]:
    """All metric values for a given (category, judge), pooled across rows
    and across the listed metric keys (e.g. all 4 step occurrences)."""
    vals: list[float] = []
    for r in rows:
        if r["category"] != cat or r["model"] != judge:
            continue
        for k in metric_keys:
            vals.append(float(r[k]))
    return vals


def grouped_bar_on_ax(ax, rows: list[dict], metric_keys: list[str],
                      ylabel: str, ymax: float, title: str,
                      annotate_value: bool = True) -> None:
    """Draw a grouped bar chart on the given Axes.
    Groups (x-axis) = prompt categories; bars within group = judge models."""
    n_groups = len(CATEGORIES)
    n_bars   = len(JUDGES)
    width    = 0.8 / n_bars
    x_centres = np.arange(n_groups)

    for j, (judge_key, judge_label, color) in enumerate(JUDGES):
        means, stds = [], []
        for cat_key, _ in CATEGORIES:
            vals = _pooled_values(rows, cat_key, judge_key, metric_keys)
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals, ddof=1) if len(vals) >= 2 else 0.0)
        offsets = x_centres + (j - (n_bars - 1) / 2) * width
        bars = ax.bar(offsets, means, width, yerr=stds, capsize=3,
                      color=color, edgecolor="black", linewidth=0.6,
                      label=judge_label)
        if annotate_value:
            # Threshold: bars near the y-axis ceiling get labels INSIDE so
            # they don't get clipped when ymax = metric ceiling (1.0 / 5.0).
            inside_threshold = 0.90 * ymax
            for bar, m in zip(bars, means):
                h = bar.get_height()
                if h > inside_threshold:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h - 0.03 * ymax,
                            f"{m:.2f}", ha="center", va="top",
                            color="white", fontsize=8, fontweight="bold")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + 0.02 * ymax,
                            f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_centres)
    ax.set_xticklabels([label for _, label in CATEGORIES])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Judge", fontsize=9, frameon=False)


def grouped_bar_standalone(rows: list[dict], metric_keys: list[str],
                           ylabel: str, ymax: float, title: str,
                           fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    grouped_bar_on_ax(ax, rows, metric_keys, ylabel, ymax, title)
    fig.tight_layout()
    out = PLOT_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def composite_across_judges(rows: list[dict]) -> None:
    """Single figure with 3 side-by-side grouped bar panels.

    Y-axes are capped at each metric's natural ceiling (1.0 for occurrence,
    5.0 for the two 1–5 Likert metrics). Error bars and bar-value labels are
    visually clipped at the ceiling — which is honest, since the underlying
    metrics can't exceed it.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    grouped_bar_on_ax(axes[0], rows, OCC_KEYS,
                      ylabel="Step Occurrence (0–1)",
                      ymax=1.0,
                      title="Step Occurrence (avg over Steps 1–4)")
    grouped_bar_on_ax(axes[1], rows, COMP_KEYS,
                      ylabel="Step Comprehensiveness (0–5)",
                      ymax=5.0,
                      title="Step Comprehensiveness (avg over Steps 1–4)")
    grouped_bar_on_ax(axes[2], rows, ["overall_conciseness"],
                      ylabel="Overall Conciseness (1–5)",
                      ymax=5.0,
                      title="Overall Conciseness")

    fig.suptitle("Judge-model comparison across prompt settings  (mean ± std)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = PLOT_DIR / "across_judges_composite.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {LONG_CSV}")

    # Across-prompt heatmaps + conciseness bar (pooled across judges)
    heatmap(rows,
            metric_suffix="occurrence",
            vmin=0.0, vmax=1.0,
            title="Step Occurrence by Prompt Setting\n(mean pooled across judge models)",
            cbar_label="Mean Occurrence (0–1)",
            fname="step_occurrence_heatmap.png",
            fmt=".2f")
    heatmap(rows,
            metric_suffix="comprehensiveness",
            vmin=0.0, vmax=5.0,
            title="Step Comprehensiveness by Prompt Setting\n(mean pooled across judge models)",
            cbar_label="Mean Comprehensiveness (0–5)",
            fname="step_comprehensiveness_heatmap.png",
            fmt=".2f")
    conciseness_bar(rows)

    # Across-judge grouped bar charts (3 standalone + 1 composite)
    grouped_bar_standalone(rows, OCC_KEYS,
                           ylabel="Step Occurrence (0–1)",
                           ymax=1.20,
                           title="Step Occurrence by Prompt Setting and Judge\n"
                                 "(mean ± std, averaged over Steps 1–4)",
                           fname="across_judges_occurrence.png")
    grouped_bar_standalone(rows, COMP_KEYS,
                           ylabel="Step Comprehensiveness (0–5)",
                           ymax=6.0,
                           title="Step Comprehensiveness by Prompt Setting and Judge\n"
                                 "(mean ± std, averaged over Steps 1–4)",
                           fname="across_judges_comprehensiveness.png")
    grouped_bar_standalone(rows, ["overall_conciseness"],
                           ylabel="Overall Conciseness (1–5)",
                           ymax=6.0,
                           title="Overall Conciseness by Prompt Setting and Judge\n"
                                 "(mean ± std)",
                           fname="across_judges_conciseness.png")
    composite_across_judges(rows)


if __name__ == "__main__":
    main()
