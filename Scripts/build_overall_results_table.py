#!/usr/bin/env python3
"""
Build the overall LaTeX results table from current human annotations,
LLM-judge scores, paragraph coverage, and violation prediction accuracy.

Inputs:
    data/annotation/human_annotations.json
    data/judge_eval_summary/long_rows.csv
    data/refparagraph_eval/refparagraph_eval.csv
    data/violation_eval/violation_eval.csv

Outputs:
    data/results_overall_table/overall_results.csv
    data/results_overall_table/overall_results_table.tex
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
HUMAN_JSON = BASE_DIR / "data" / "annotation" / "human_annotations.json"
JUDGE_CSV = BASE_DIR / "data" / "judge_eval_summary" / "long_rows.csv"
REF_CSV = BASE_DIR / "data" / "refparagraph_eval" / "refparagraph_eval.csv"
VIOLATION_CSV = BASE_DIR / "data" / "violation_eval" / "violation_eval.csv"
OUT_DIR = BASE_DIR / "data" / "results_overall_table"

CATEGORIES = [
    ("non_curated", "A: Non-Curated"),
    ("instruction", "B: Curated (Expert)"),
    ("gpt_assisted", "C: Curated (Guide)"),
]
STEPS = ["step_1", "step_2", "step_3", "step_4"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> tuple[float, float, int]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std, len(values)


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def fmt_mean(mean: float) -> str:
    return f"{mean:.2f}"


def human_values(human: dict, category: str) -> dict[str, tuple[float, float, int]]:
    occurrence: list[float] = []
    comprehensiveness: list[float] = []
    for step in STEPS:
        occurrence.extend(float(v) for v in human[category][step]["occurrence"])
        comprehensiveness.extend(
            float(v) for v in human[category][step]["comprehensiveness"]
        )
    conciseness = [float(v) for v in human[category]["overall_conciseness"]]
    return {
        "step_occurrence_human": mean_std(occurrence),
        "step_comprehensiveness_human": mean_std(comprehensiveness),
        "overall_conciseness_human": mean_std(conciseness),
    }


def judge_values(rows: list[dict[str, str]], category: str) -> dict[str, tuple[float, float, int]]:
    cat_rows = [row for row in rows if row["category"] == category]
    occurrence: list[float] = []
    comprehensiveness: list[float] = []
    for row in cat_rows:
        for step in STEPS:
            occurrence.append(float(row[f"{step}.occurrence"]))
            comprehensiveness.append(float(row[f"{step}.comprehensiveness"]))
    conciseness = [float(row["overall_conciseness"]) for row in cat_rows]
    return {
        "step_occurrence_models": mean_std(occurrence),
        "step_comprehensiveness_models": mean_std(comprehensiveness),
        "overall_conciseness_models": mean_std(conciseness),
    }


def ref_values(rows: list[dict[str, str]], category: str) -> tuple[float, float, int]:
    vals = [
        float(row["f1_in_facts"])
        for row in rows
        if row["prompt_name"] == category and row.get("f1_in_facts")
    ]
    return mean_std(vals)


def accuracy_values(rows: list[dict[str, str]], category: str) -> tuple[float, float, int]:
    vals = [
        float(row["correct"])
        for row in rows
        if row["prompt_name"] == category and row.get("correct") not in {"", None}
    ]
    return mean_std(vals)


def best_keys(table_rows: list[dict]) -> dict[str, set[str]]:
    """Return metric column names whose mean is maximal, for underline styling."""
    metric_groups = [
        ("step_occurrence_human", "step_occurrence_human_fmt"),
        ("step_occurrence_models", "step_occurrence_models_fmt"),
        ("step_comprehensiveness_human", "step_comprehensiveness_human_fmt"),
        ("step_comprehensiveness_models", "step_comprehensiveness_models_fmt"),
        ("overall_conciseness_human", "overall_conciseness_human_fmt"),
        ("overall_conciseness_models", "overall_conciseness_models_fmt"),
        ("paragraph_coverage", "paragraph_coverage_fmt"),
        ("accuracy", "accuracy_fmt"),
    ]
    out: dict[str, set[str]] = {}
    for raw_key, fmt_key in metric_groups:
        best = max(row[raw_key][0] for row in table_rows)
        out[fmt_key] = {row["category"] for row in table_rows if row[raw_key][0] == best}
    return out


def maybe_underline(value: str, category: str, fmt_key: str, winners: dict[str, set[str]]) -> str:
    if category in winners.get(fmt_key, set()):
        return f"\\underline{{{value}}}"
    return value


def build_table() -> tuple[list[dict], str]:
    human = json.loads(HUMAN_JSON.read_text(encoding="utf-8"))
    judge_rows = read_csv(JUDGE_CSV)
    ref_rows = read_csv(REF_CSV)
    violation_rows = read_csv(VIOLATION_CSV)

    rows: list[dict] = []
    for category, label in CATEGORIES:
        row: dict = {"category": category, "label": label}
        row.update(human_values(human, category))
        row.update(judge_values(judge_rows, category))
        row["paragraph_coverage"] = ref_values(ref_rows, category)
        row["accuracy"] = accuracy_values(violation_rows, category)
        rows.append(row)

    for row in rows:
        row["step_occurrence_human_fmt"] = fmt_mean_std(*row["step_occurrence_human"][:2])
        row["step_occurrence_models_fmt"] = fmt_mean_std(*row["step_occurrence_models"][:2])
        row["step_comprehensiveness_human_fmt"] = fmt_mean_std(
            *row["step_comprehensiveness_human"][:2]
        )
        row["step_comprehensiveness_models_fmt"] = fmt_mean_std(
            *row["step_comprehensiveness_models"][:2]
        )
        row["overall_conciseness_human_fmt"] = fmt_mean_std(
            *row["overall_conciseness_human"][:2]
        )
        row["overall_conciseness_models_fmt"] = fmt_mean_std(
            *row["overall_conciseness_models"][:2]
        )
        row["paragraph_coverage_fmt"] = fmt_mean_std(*row["paragraph_coverage"][:2])
        row["accuracy_fmt"] = fmt_mean(row["accuracy"][0])

    winners = best_keys(rows)
    latex_rows = []
    for row in rows:
        cells = [
            row["label"],
            maybe_underline(
                row["step_occurrence_human_fmt"],
                row["category"],
                "step_occurrence_human_fmt",
                winners,
            ),
            maybe_underline(
                row["step_occurrence_models_fmt"],
                row["category"],
                "step_occurrence_models_fmt",
                winners,
            ),
            maybe_underline(
                row["step_comprehensiveness_human_fmt"],
                row["category"],
                "step_comprehensiveness_human_fmt",
                winners,
            ),
            maybe_underline(
                row["step_comprehensiveness_models_fmt"],
                row["category"],
                "step_comprehensiveness_models_fmt",
                winners,
            ),
            maybe_underline(
                row["overall_conciseness_human_fmt"],
                row["category"],
                "overall_conciseness_human_fmt",
                winners,
            ),
            maybe_underline(
                row["overall_conciseness_models_fmt"],
                row["category"],
                "overall_conciseness_models_fmt",
                winners,
            ),
            maybe_underline(
                row["paragraph_coverage_fmt"],
                row["category"],
                "paragraph_coverage_fmt",
                winners,
            ),
            maybe_underline(row["accuracy_fmt"], row["category"], "accuracy_fmt", winners),
        ]
        latex_rows.append("         " + " & ".join(cells) + r" \\")

    body = "\n".join(latex_rows)
    latex = rf"""\begin{{table*}}[]
    \centering
    \resizebox{{\textwidth}}{{!}}{{
    \begin{{tabular}}{{l|c|c|c|c|c|c|c|c|}}
         \multirow{{2}}{{*}}{{\textbf{{Prompt Setting}}}} &  \multicolumn{{2}}{{c|}}{{\textbf{{Step Occurrence}}}} & \multicolumn{{2}}{{c|}}{{\textbf{{Step Comprehensiveness}}}} & \multicolumn{{2}}{{c|}}{{\textbf{{Overall Conciseness}}}} & \multirow{{2}}{{*}}{{\textbf{{Par. Coverage}}}} & \multirow{{2}}{{*}}{{\textbf{{Accuracy}}}} \\
         & Human & Models  & Human & Models  & Human & Models & & \\
         \midrule
{body}
    \end{{tabular}}
    }}
    \caption{{Results for each examined setting (A-C) across all metrics, averaged over all steps and cases. For each setting, both Human ratings and Models (LLM-as-a-Judge) are reported.}}
    \vspace{{-3mm}}
    \label{{tab:overall}}
\end{{table*}}
"""
    return rows, latex


def main() -> None:
    rows, latex = build_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_cols = [
        "category",
        "label",
        "step_occurrence_human_fmt",
        "step_occurrence_models_fmt",
        "step_comprehensiveness_human_fmt",
        "step_comprehensiveness_models_fmt",
        "overall_conciseness_human_fmt",
        "overall_conciseness_models_fmt",
        "paragraph_coverage_fmt",
        "accuracy_fmt",
    ]
    with (OUT_DIR / "overall_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in csv_cols})

    (OUT_DIR / "overall_results_table.tex").write_text(latex, encoding="utf-8")
    print(latex)
    print(f"Wrote {OUT_DIR / 'overall_results.csv'}")
    print(f"Wrote {OUT_DIR / 'overall_results_table.tex'}")


if __name__ == "__main__":
    main()
