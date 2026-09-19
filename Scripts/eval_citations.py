#!/usr/bin/env python3
"""
Evaluate citation usage in model-generated assessments against ground truth.

Ground truth: sequential citations from the Merits subsection(s) of Article 10
court assessments in `data/parsed_cases_all(3,9,10,11).json`.

Generated: the `citations` field added to each
`Runs/outputs_new/{item_id}/{prompt_name}/response.json` by `add_citations.py`.

For each (case, prompt) pair we record:
    - ground_truth        : full GT citation list (sequential, deduped)
    - generated           : citations the model used (in its own order)
    - gt_subset_in_order  : intersection, in GT order
    - generated_matched   : intersection, in the order the model used them
    - extra_citations     : generated citations not present in GT
    - kendall_tau / spearman_rho on the matched subset (None if n < 2)

Outputs:
    data/citation_eval/citation_eval.csv   (one row per (item_id, prompt_name))
    data/citation_eval/citation_eval.json  (full nested data)
"""

import csv
import json
from pathlib import Path

BASE_DIR        = Path(__file__).parent.parent
PARSED_PATH     = BASE_DIR / "data" / "parsed_cases_all(3,9,10,11).json"
OUTPUTS_DIR     = BASE_DIR / "Runs" / "outputs_new"
EVAL_DIR        = BASE_DIR / "data" / "citation_eval"
PROMPT_NAMES    = ("instruction", "non_curated")
TARGET_ARTICLE  = 10


def gt_merits_citations(case: dict) -> list[str]:
    """Sequential Merits citations for Article 10, deduped (first-occurrence order)."""
    seen: set[str] = set()
    out: list[str] = []
    for ca in case.get("court_assessments", []):
        if TARGET_ARTICLE not in ca.get("articles", []):
            continue
        for sub in ca.get("subsections", []):
            if sub.get("subsection_title") != "Merits":
                continue
            for c in sub.get("citations", []):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


def kendall_tau(matched_in_gen_order: list[str], gt_order: list[str]) -> float | None:
    """Tau-a on the matched subset; both lists are unique permutations of the same set."""
    n = len(matched_in_gen_order)
    if n < 2:
        return None
    gt_rank = {c: i for i, c in enumerate(gt_order)}
    ranks = [gt_rank[c] for c in matched_in_gen_order]
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] < ranks[j]:
                concordant += 1
            elif ranks[i] > ranks[j]:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def spearman_rho(matched_in_gen_order: list[str], gt_order: list[str]) -> float | None:
    """Spearman's rho with no ties (matched subset has unique items)."""
    n = len(matched_in_gen_order)
    if n < 2:
        return None
    gt_rank = {c: i for i, c in enumerate(gt_order)}
    d_squared = sum((i - gt_rank[c]) ** 2 for i, c in enumerate(matched_in_gen_order))
    return 1 - 6 * d_squared / (n * (n ** 2 - 1))


def evaluate_pair(gt: list[str], generated: list[str]) -> dict:
    gt_set = set(gt)
    matched_in_gen = [c for c in generated if c in gt_set]
    extras = [c for c in generated if c not in gt_set]
    matched_set = set(matched_in_gen)
    gt_subset_in_order = [c for c in gt if c in matched_set]
    return {
        "ground_truth":        gt,
        "generated":           generated,
        "gt_subset_in_order":  gt_subset_in_order,
        "generated_matched":   matched_in_gen,
        "extra_citations":    extras,
        "n_gt":                len(gt),
        "n_generated":         len(generated),
        "n_matched":           len(matched_in_gen),
        "n_extra":             len(extras),
        "kendall_tau":         kendall_tau(matched_in_gen, gt_subset_in_order),
        "spearman_rho":        spearman_rho(matched_in_gen, gt_subset_in_order),
    }


def main() -> None:
    cases_by_id: dict[str, dict] = {}
    with open(PARSED_PATH, encoding="utf-8") as f:
        for c in json.load(f):
            cases_by_id[c["item_id"]] = c

    rows: list[dict] = []
    missing: list[str] = []

    for item_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not item_dir.is_dir():
            continue
        item_id = item_dir.name
        case = cases_by_id.get(item_id)
        if case is None:
            missing.append(item_id)
            continue
        gt = gt_merits_citations(case)

        for prompt_name in PROMPT_NAMES:
            rj = item_dir / prompt_name / "response.json"
            if not rj.exists():
                continue
            with open(rj, encoding="utf-8") as f:
                resp = json.load(f)
            generated = resp.get("citations") or []

            rec = {
                "item_id":     item_id,
                "prompt_name": prompt_name,
                "case_name":   case.get("case_name"),
                **evaluate_pair(gt, generated),
            }
            rows.append(rec)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_path = EVAL_DIR / "citation_eval.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = EVAL_DIR / "citation_eval.csv"
    csv_cols = [
        "item_id", "prompt_name", "case_name",
        "n_gt", "n_generated", "n_matched", "n_extra",
        "kendall_tau", "spearman_rho",
        "ground_truth", "generated", "gt_subset_in_order",
        "generated_matched", "extra_citations",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in csv_cols}
            for list_col in ("ground_truth", "generated", "gt_subset_in_order",
                             "generated_matched", "extra_citations"):
                row[list_col] = "|".join(row[list_col]) if row[list_col] else ""
            writer.writerow(row)

    n_with_tau = sum(1 for r in rows if r["kendall_tau"] is not None)
    mean_tau = (sum(r["kendall_tau"] for r in rows if r["kendall_tau"] is not None)
                / n_with_tau) if n_with_tau else None
    mean_rho = (sum(r["spearman_rho"] for r in rows if r["spearman_rho"] is not None)
                / n_with_tau) if n_with_tau else None

    print(f"Wrote {len(rows)} rows -> {csv_path}")
    print(f"Wrote {len(rows)} rows -> {json_path}")
    print(f"Pairs with computable tau/rho (n_matched >= 2): {n_with_tau}/{len(rows)}")
    if mean_tau is not None:
        print(f"Mean Kendall's tau: {mean_tau:+.3f}")
        print(f"Mean Spearman's rho: {mean_rho:+.3f}")
    if missing:
        print(f"WARN: {len(missing)} item(s) in outputs_new not found in parsed cases: {missing}")


if __name__ == "__main__":
    main()
