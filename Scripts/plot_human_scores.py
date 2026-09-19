#!/usr/bin/env python3
"""
Render the same two heatmaps we produced for the LLM-judge scores
(step occurrence and step comprehensiveness × prompt category), but for
the human-annotation scores in `data/annotation/human_annotations.json`.

Source structure:
    {
      "<prompt_category>": {
        "step_1": {"occurrence": [...], "comprehensiveness": [...]},
        ...
        "step_4": {"occurrence": [...], "comprehensiveness": [...]},
        "overall_conciseness": [...]
      },
      ...
    }

Cell value = mean of whichever scores the annotators provided for that
(step, category) pair. Array lengths vary slightly per cell (10–15
annotations); we just take the mean of what's there.

Output:
    data/annotation/plots/human_step_occurrence_heatmap.png
    data/annotation/plots/human_step_comprehensiveness_heatmap.png
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR  = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "annotation" / "human_annotations.json"
LLM_CSV   = BASE_DIR / "data" / "judge_eval_summary" / "long_rows.csv"
PLOT_DIR  = BASE_DIR / "data" / "annotation" / "plots"

# Display order + labels matching the paper's LaTeX table
CATEGORIES = [
    ("non_curated",  "A: Non-Curated"),
    ("instruction",  "B: Curated\n(Expert)"),
    ("gpt_assisted", "C: Curated\n(Guide)"),
]
STEPS       = ["step_1", "step_2", "step_3", "step_4"]
STEP_LABELS = ["Step 1", "Step 2", "Step 3", "Step 4"]


def load_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def heatmap(data: dict, metric_key: str, vmin: float, vmax: float,
            title: str, cbar_label: str, fname: str,
            fmt: str = ".2f") -> None:
    """Render a steps × categories heatmap. Cell = mean of the array values."""
    means  = np.zeros((len(STEPS), len(CATEGORIES)))
    counts = np.zeros_like(means, dtype=int)

    for s_idx, step in enumerate(STEPS):
        for c_idx, (cat_key, _) in enumerate(CATEGORIES):
            arr = data.get(cat_key, {}).get(step, {}).get(metric_key, [])
            means[s_idx, c_idx]  = np.mean(arr) if arr else np.nan
            counts[s_idx, c_idx] = len(arr)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(means, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

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
            v = means[s_idx, c_idx]
            color = "white" if v < threshold else "black"
            ax.text(c_idx, s_idx, format(v, fmt),
                    ha="center", va="center", color=color,
                    fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    out = PLOT_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Delta heatmaps: human - LLM-judge (pooled across all 3 judges)
# ---------------------------------------------------------------------------

def llm_judge_means() -> dict[str, np.ndarray]:
    """Return {'occurrence': arr, 'comprehensiveness': arr} where each array
    is shape (n_steps, n_categories) of pooled-across-judges means."""
    csv.field_size_limit(sys.maxsize)
    with open(LLM_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out: dict[str, np.ndarray] = {}
    for metric_suffix in ("occurrence", "comprehensiveness"):
        m = np.zeros((len(STEPS), len(CATEGORIES)))
        for s_idx, step in enumerate(STEPS):
            col = f"{step}.{metric_suffix}"
            for c_idx, (cat_key, _) in enumerate(CATEGORIES):
                vals = [float(r[col]) for r in rows if r["category"] == cat_key]
                m[s_idx, c_idx] = np.mean(vals) if vals else np.nan
        out[metric_suffix] = m
    return out


def human_means() -> dict[str, np.ndarray]:
    data = load_data()
    out: dict[str, np.ndarray] = {}
    for metric_key in ("occurrence", "comprehensiveness"):
        m = np.zeros((len(STEPS), len(CATEGORIES)))
        for s_idx, step in enumerate(STEPS):
            for c_idx, (cat_key, _) in enumerate(CATEGORIES):
                arr = data.get(cat_key, {}).get(step, {}).get(metric_key, [])
                m[s_idx, c_idx] = np.mean(arr) if arr else np.nan
        out[metric_key] = m
    return out


def delta_heatmap(delta: np.ndarray, vlim: float,
                  title: str, cbar_label: str, fname: str,
                  fmt: str = "+.2f") -> None:
    """Render a human - LLM delta heatmap with divergent colormap centered at 0."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([label for _, label in CATEGORIES])
    ax.set_yticks(range(len(STEPS)))
    ax.set_yticklabels(STEP_LABELS)
    ax.set_xlabel("Prompt Setting")
    ax.set_ylabel("Reasoning Step")
    ax.set_title(title)

    # Annotations always black for divergent map (good contrast on light center)
    for s_idx in range(len(STEPS)):
        for c_idx in range(len(CATEGORIES)):
            v = delta[s_idx, c_idx]
            ax.text(c_idx, s_idx, format(v, fmt),
                    ha="center", va="center", color="black",
                    fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    cbar.ax.text(0.5, 1.02, "humans rate higher",
                 transform=cbar.ax.transAxes, ha="center", va="bottom",
                 fontsize=8, color="#7d2a2a")
    cbar.ax.text(0.5, -0.02, "LLM judges rate higher",
                 transform=cbar.ax.transAxes, ha="center", va="top",
                 fontsize=8, color="#2a4d7d")

    fig.tight_layout()
    out = PLOT_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    print(f"Loaded {DATA_PATH}")
    print(f"Categories: {list(data.keys())}")

    # Human-only heatmaps
    heatmap(data,
            metric_key="occurrence",
            vmin=0.0, vmax=1.0,
            title="Step Occurrence by Prompt Setting\n(human annotations)",
            cbar_label="Mean Occurrence (0–1)",
            fname="human_step_occurrence_heatmap.png",
            fmt=".2f")
    heatmap(data,
            metric_key="comprehensiveness",
            vmin=0.0, vmax=5.0,
            title="Step Comprehensiveness by Prompt Setting\n(human annotations)",
            cbar_label="Mean Comprehensiveness (0–5)",
            fname="human_step_comprehensiveness_heatmap.png",
            fmt=".2f")

    # Delta heatmaps (human - LLM-judge pooled)
    print(f"Loading LLM-judge data from {LLM_CSV}")
    h_means   = human_means()
    llm_means = llm_judge_means()

    delta_occ  = h_means["occurrence"]        - llm_means["occurrence"]
    delta_comp = h_means["comprehensiveness"] - llm_means["comprehensiveness"]

    delta_heatmap(delta_occ,
                  vlim=0.20,
                  title="Step Occurrence: Human − LLM Judges\n"
                        "(divergent, centered at 0)",
                  cbar_label="Δ Mean Occurrence",
                  fname="delta_step_occurrence_heatmap.png")
    delta_heatmap(delta_comp,
                  vlim=1.5,
                  title="Step Comprehensiveness: Human − LLM Judges\n"
                        "(divergent, centered at 0)",
                  cbar_label="Δ Mean Comprehensiveness",
                  fname="delta_step_comprehensiveness_heatmap.png")


if __name__ == "__main__":
    main()
