#!/usr/bin/env python3
"""
Build the violation-prediction dataset and compute accuracy per prompt.

For each (case, prompt) we record:
    - violation_ground_truth     raw list from the parsed cases JSON,
                                 e.g. ['10', '10-1'] or ['11', '11-1']
    - gt_art10                   "yes" iff any code has base 10, else "no"
                                 (this is the actual target for the Art-10
                                 prediction task)
    - prediction                 the model's "violation_prediction" (yes/no/unknown)
    - correct                    1 if prediction == gt_art10, else 0
                                 (None if prediction is missing/unparseable)

Accuracy per prompt = mean(correct) over cases where prediction is defined.
Std reported is the sample std of the 0/1 correctness vector — appropriate
for reporting in a LaTeX-table "Accuracy" column as mean ± std.

Outputs:
    data/violation_eval/violation_eval.csv
    data/violation_eval/violation_eval.json
"""

import csv
import sys
import json
from pathlib import Path

import numpy as np

BASE_DIR    = Path(__file__).parent.parent
PARSED_PATH = BASE_DIR / "data" / "parsed_cases_all(3,9,10,11).json"
NEW_DIR     = BASE_DIR / "Runs" / "outputs_new"
GPT_DIR     = BASE_DIR / "Runs" / "outputs_gpt_assisted"
OUT_DIR     = BASE_DIR / "data" / "violation_eval"

# The case set is defined by the annotation dataset (the same distinct cases
# the human/judge evaluation uses), NOT by the outputs_new directory listing.
# combined_30_seed42_og.csv has 30 (case x prompt-pair) rows over 22 distinct
# cases; we evaluate the model's per-prompt prediction on those 22 distinct
# cases, which INCLUDES the two Grand Chamber cases present in the dataset
# (001-244292, 001-247738). This keeps violation accuracy on the same cases as
# the reasoning-quality metrics.
ANNOTATION_CSV = BASE_DIR / "data" / "annotation" / "combined_30_seed42_og.csv"

PROMPT_SOURCES = {
    "instruction":  NEW_DIR,
    "non_curated":  NEW_DIR,
    "gpt_assisted": GPT_DIR,
}


def gt_art10(violation_codes: list[str]) -> str:
    """Return 'yes' iff any code in the list refers to Article 10."""
    for code in violation_codes or []:
        base = str(code).split("-")[0].split("+")[0].strip()
        if base == "10":
            return "yes"
    return "no"


def normalise_pred(pred) -> str | None:
    """Normalise the model's violation_prediction value to 'yes' / 'no' / None."""
    if pred is None:
        return None
    s = str(pred).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    return None


def main() -> None:
    # GT is read directly from the parsed-cases JSON so that any manual
    # corrections to the `violation` field take effect. Earlier versions
    # of this script read the `violation_ground_truth` snapshot inside each
    # response.json — that snapshot was captured at inference time and is
    # stale after edits to the parsed JSON.
    parsed = {c["item_id"]: c
              for c in json.loads(PARSED_PATH.read_text(encoding="utf-8"))}

    # Distinct cases defined by the annotation dataset.
    csv.field_size_limit(sys.maxsize)
    with ANNOTATION_CSV.open(newline="", encoding="utf-8-sig") as fh:
        item_ids = sorted({r["item_id"] for r in csv.DictReader(fh)})

    rows: list[dict] = []
    for cid in item_ids:
        case = parsed.get(cid)
        if case is None:
            continue
        gt_list  = case.get("violation") or []
        gt_label = gt_art10(gt_list)

        for prompt_name, source_dir in PROMPT_SOURCES.items():
            rj = source_dir / cid / prompt_name / "response.json"
            if not rj.exists():
                continue
            d = json.loads(rj.read_text(encoding="utf-8"))

            pred_raw = d.get("violation_prediction")
            pred     = normalise_pred(pred_raw)

            correct = (1 if pred == gt_label else 0) if pred is not None else None

            rows.append({
                "item_id":                cid,
                "prompt_name":            prompt_name,
                "case_name":              case.get("case_name") or d.get("case_name"),
                "violation_ground_truth": gt_list,
                "gt_art10":               gt_label,
                "prediction_raw":         pred_raw,
                "prediction":             pred,
                "correct":                correct,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "violation_eval.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_cols = ["item_id", "prompt_name", "case_name",
                "violation_ground_truth", "gt_art10",
                "prediction_raw", "prediction", "correct"]
    with open(OUT_DIR / "violation_eval.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row["violation_ground_truth"] = "|".join(row["violation_ground_truth"])
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows -> {OUT_DIR}/violation_eval.{{csv,json}}\n")

    # ---- accuracy per prompt ----
    n_cases = len({r["item_id"] for r in rows})
    yes_rate = (sum(1 for r in rows if r["gt_art10"] == "yes")
                / max(1, len(rows)))
    print(f"Dataset: {n_cases} distinct cases "
          f"(annotation set; incl. Grand Chamber cases in it)")
    print(f"Always-'yes' baseline accuracy = {yes_rate:.3f}\n")
    print(f"{'prompt':14} {'n':>4} {'n_defined':>10} {'accuracy':>22} {'yes_in_gt':>12} {'parse_err':>12}")
    for prompt_name in PROMPT_SOURCES:
        sub = [r for r in rows if r["prompt_name"] == prompt_name]
        defined = [r for r in sub if r["correct"] is not None]
        yes_gt  = sum(1 for r in sub if r["gt_art10"] == "yes")
        unparsed = sum(1 for r in sub if r["correct"] is None)
        vals = [r["correct"] for r in defined]
        if vals:
            mu = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) >= 2 else 0.0
            cell = f"{mu:.3f} ± {sd:.3f}"
        else:
            cell = "—"
        print(f"{prompt_name:14} {len(sub):>4} {len(defined):>10} {cell:>22} {yes_gt:>12} {unparsed:>12}")


if __name__ == "__main__":
    main()
