#!/usr/bin/env python3
"""
Build three CSV datasets for human annotation, one per pairwise comparison:

    A: Student A = instruction        vs Student B = non_curated
    B: Student A = instruction        vs Student B = gpt_assisted
    C: Student A = gpt_assisted       vs Student B = non_curated

Annotation columns (occurrence / comprehensiveness / overall conciseness)
are left empty and are filled in by the human annotator.

Source data:
    data/parsed_cases_all(3,9,10,11).json    — facts, court Merits assessment, citations
    Runs/outputs_new/{cid}/instruction/response.json
    Runs/outputs_new/{cid}/non_curated/response.json
    Runs/outputs_gpt_assisted/{cid}/gpt_assisted/response.json

Outputs:
    data/annotation/A_instruction_vs_non_curated.csv
    data/annotation/B_instruction_vs_gpt_assisted.csv
    data/annotation/C_gpt_assisted_vs_non_curated.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
DATA_PATH   = BASE_DIR / "data" / "parsed_cases_all(3,9,10,11).json"
NEW_DIR     = BASE_DIR / "Runs" / "outputs_new"
GPT_DIR     = BASE_DIR / "Runs" / "outputs_gpt_assisted"
OUT_DIR     = BASE_DIR / "data" / "annotation"
TARGET_ARTICLE = 10

# (run_tag, csv_filename, side_a_label, side_b_label, side_a_dir, side_b_dir)
RUNS = [
    ("A", "A_instruction_vs_non_curated.csv",   "instruction",  "non_curated",   NEW_DIR, NEW_DIR),
    ("B", "B_instruction_vs_gpt_assisted.csv",  "instruction",  "gpt_assisted",  NEW_DIR, GPT_DIR),
    ("C", "C_gpt_assisted_vs_non_curated.csv",  "gpt_assisted", "non_curated",   GPT_DIR, NEW_DIR),
]

# Map (category_label) -> directory it lives in
CATEGORY_DIR = {
    "instruction":   NEW_DIR,
    "non_curated":   NEW_DIR,
    "gpt_assisted":  GPT_DIR,
}

COLUMNS = [
    "item_id", "case_name", "date", "facts", "court_assessment", "citations",
    "student_a_response", "A_violation_prediction",
    "A_Step_1 Occurrence",  "A_Step_1 Comprehensiveness",
    "A_Step_2 Occurrence",  "A_Step_2 Comprehensiveness",
    "A_Step_3 Occurrence",  "A_Step_3 Comprehensiveness",
    "A_Step_4 Occurrence",  "A_Step_4 Comprehensiveness",
    "A_Overall Conciseness",
    "student_b_response", "B_violation_prediction",
    "B_Step_1 Occurrence",  "B_Step_1 Comprehensiveness",
    "B_Step_2 Occurrence",  "B_Step_2 Comprehensiveness",
    "B_Step_3 Occurrence",  "B_Step_3 Comprehensiveness",
    "B_Step_4 Occurrence",  "B_Step_4 Comprehensiveness",
    "B_Overall Conciseness",
    "A_category", "B_category",
]

ANNOTATION_FIELDS = [c for c in COLUMNS if c.startswith(("A_Step", "B_Step", "A_Overall", "B_Overall"))]


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

def get_merits_text(case: dict) -> str:
    """Concatenate paragraphs from EVERY Article-10 Merits subsection, each
    prefixed by its parent section_title. Some cases have two Article-10
    sections (e.g. substantive + procedural limbs); the early-return version
    of this function silently dropped the second one."""
    blocks: list[str] = []
    for ca in case.get("court_assessments", []):
        if TARGET_ARTICLE not in ca.get("articles", []):
            continue
        section_title = (ca.get("section_title") or "").strip()
        for sub in ca.get("subsections", []):
            if sub.get("subsection_title") != "Merits":
                continue
            paras = sub.get("paragraphs", [])
            if not paras:
                continue
            body = "\n\n".join(paras)
            blocks.append(f"{section_title}\n\n{body}" if section_title else body)
    if not blocks:
        return "[No Merits assessment found]"
    return "\n\n———\n\n".join(blocks)


def get_merits_citations(case: dict) -> list[str]:
    seen, out = set(), []
    for ca in case.get("court_assessments", []):
        if TARGET_ARTICLE not in ca.get("articles", []):
            continue
        for sub in ca.get("subsections", []):
            if sub.get("subsection_title") == "Merits":
                for c in sub.get("citations", []):
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
    return out


def format_assessment(assessment) -> str:
    """Same logic as run_eval.format_assessment — list of single-key dicts -> readable text."""
    if isinstance(assessment, list):
        parts = []
        for item in assessment:
            if isinstance(item, dict):
                for title, content in item.items():
                    parts.append(f"{title}\n\n{content}")
        return "\n\n".join(parts)
    return str(assessment) if assessment is not None else "[No assessment]"


def load_student_data(base_dir: Path, item_id: str, prompt_name: str) -> tuple[str, str]:
    """Returns (assessment_text, violation_prediction). Prediction is stored
    in its own CSV column rather than being appended to the assessment text."""
    path = base_dir / item_id / prompt_name / "response.json"
    if not path.exists():
        return f"[Missing response: {path}]", ""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    body = format_assessment(d.get("assessment"))
    pred = d.get("violation_prediction")
    pred_str = "unknown" if pred is None else str(pred)
    return body, pred_str


# ---------------------------------------------------------------------------
# Build a CSV
# ---------------------------------------------------------------------------

def build_csv(out_path: Path, a_label: str, b_label: str, a_dir: Path, b_dir: Path,
              cases_by_id: dict[str, dict], item_ids: list[str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for cid in item_ids:
            case = cases_by_id.get(cid)
            if case is None:
                continue
            facts_body = "\n\n".join(case.get("facts", [])) or "[No facts available]"

            a_body, a_pred = load_student_data(a_dir, cid, a_label)
            b_body, b_pred = load_student_data(b_dir, cid, b_label)
            row = {c: "" for c in COLUMNS}
            row.update({
                "item_id":           cid,
                "case_name":         case.get("case_name", "") or "",
                "date":              case.get("date", "") or "",
                "facts":             facts_body,
                "court_assessment":  get_merits_text(case),
                "citations":         ", ".join(get_merits_citations(case)),
                "student_a_response":     a_body,
                "A_violation_prediction": a_pred,
                "student_b_response":     b_body,
                "B_violation_prediction": b_pred,
                "A_category":        a_label,
                "B_category":        b_label,
            })
            writer.writerow(row)
    return len(item_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(DATA_PATH, encoding="utf-8") as f:
        parsed = json.load(f)
    cases_by_id = {c["item_id"]: c for c in parsed}

    item_ids = sorted(p.name for p in NEW_DIR.iterdir() if p.is_dir())
    print(f"Cases (Article 10): {len(item_ids)}")

    for _tag, filename, a_label, b_label, a_dir, b_dir in RUNS:
        out_path = OUT_DIR / filename
        n = build_csv(out_path, a_label, b_label, a_dir, b_dir, cases_by_id, item_ids)
        print(f"  {filename}: wrote {n} rows -> {out_path}")


# ---------------------------------------------------------------------------
# Combined annotation CSV: 10 rows from each run, randomized A/B side
# ---------------------------------------------------------------------------

COMBINED_COLUMNS = COLUMNS + ["source_run", "swapped"]


def _build_run_row(cid: str, a_label: str, b_label: str, a_dir: Path, b_dir: Path,
                   case: dict) -> dict:
    """Build one row in canonical (un-swapped) orientation."""
    facts_body = "\n\n".join(case.get("facts", [])) or "[No facts available]"
    a_body, a_pred = load_student_data(a_dir, cid, a_label)
    b_body, b_pred = load_student_data(b_dir, cid, b_label)
    row = {c: "" for c in COMBINED_COLUMNS}
    row.update({
        "item_id":           cid,
        "case_name":         case.get("case_name", "") or "",
        "date":              case.get("date", "") or "",
        "facts":             facts_body,
        "court_assessment":  get_merits_text(case),
        "citations":         ", ".join(get_merits_citations(case)),
        "student_a_response":     a_body,
        "A_violation_prediction": a_pred,
        "student_b_response":     b_body,
        "B_violation_prediction": b_pred,
        "A_category":        a_label,
        "B_category":        b_label,
    })
    return row


def _swap_sides(row: dict) -> None:
    """Swap A and B in place: responses, categories, and any annotation cells."""
    pairs = [
        ("student_a_response",        "student_b_response"),
        ("A_violation_prediction",    "B_violation_prediction"),
        ("A_category",                "B_category"),
        ("A_Step_1 Occurrence",       "B_Step_1 Occurrence"),
        ("A_Step_1 Comprehensiveness","B_Step_1 Comprehensiveness"),
        ("A_Step_2 Occurrence",       "B_Step_2 Occurrence"),
        ("A_Step_2 Comprehensiveness","B_Step_2 Comprehensiveness"),
        ("A_Step_3 Occurrence",       "B_Step_3 Occurrence"),
        ("A_Step_3 Comprehensiveness","B_Step_3 Comprehensiveness"),
        ("A_Step_4 Occurrence",       "B_Step_4 Occurrence"),
        ("A_Step_4 Comprehensiveness","B_Step_4 Comprehensiveness"),
        ("A_Overall Conciseness",     "B_Overall Conciseness"),
    ]
    for k1, k2 in pairs:
        row[k1], row[k2] = row[k2], row[k1]


def build_combined_csv(out_path: Path, cases_by_id: dict[str, dict],
                       item_ids: list[str], n_per_run: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    combined: list[dict] = []

    for tag, _filename, a_label, b_label, a_dir, b_dir in RUNS:
        sampled_ids = rng.sample(item_ids, n_per_run)
        for cid in sampled_ids:
            case = cases_by_id.get(cid)
            if case is None:
                continue
            row = _build_run_row(cid, a_label, b_label, a_dir, b_dir, case)
            row["source_run"] = tag
            swap = rng.random() < 0.5
            if swap:
                _swap_sides(row)
            row["swapped"] = "yes" if swap else "no"
            combined.append(row)

    rng.shuffle(combined)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in combined:
            writer.writerow(row)

    return combined


# ---------------------------------------------------------------------------
# Verification: every row's A_category/B_category must match the actual
# response text in Student A response / Student B response.
# ---------------------------------------------------------------------------

def verify_combined(rows: list[dict]) -> tuple[int, list[str]]:
    """Returns (n_ok, errors). Verifies both response text AND violation
    prediction match what the named category should have produced."""
    errors: list[str] = []
    for i, row in enumerate(rows):
        cid = row["item_id"]
        for side in ("A", "B"):
            cat = row[f"{side}_category"]
            actual_body = row[f"student_{side.lower()}_response"]
            actual_pred = row[f"{side}_violation_prediction"]
            expected_dir = CATEGORY_DIR.get(cat)
            if expected_dir is None:
                errors.append(f"row {i} cid={cid}: unknown category '{cat}'")
                continue
            expected_body, expected_pred = load_student_data(expected_dir, cid, cat)
            if expected_body != actual_body:
                errors.append(
                    f"row {i} cid={cid} side={side} swapped={row['swapped']} "
                    f"source_run={row['source_run']}: response body does not match category '{cat}'"
                )
            if expected_pred != actual_pred:
                errors.append(
                    f"row {i} cid={cid} side={side} swapped={row['swapped']} "
                    f"source_run={row['source_run']}: violation_prediction does not match category '{cat}'"
                )
    return len(rows) * 2 - len(errors), errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-run and combined annotation CSVs")
    parser.add_argument("--n-per-run", type=int, default=10,
                        help="Number of rows sampled per run for the combined CSV (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling and A/B swap (default: 42)")
    parser.add_argument("--combined-only", action="store_true",
                        help="Skip rebuilding the per-run CSVs; only build the combined CSV")
    args = parser.parse_args()

    with open(DATA_PATH, encoding="utf-8") as f:
        parsed = json.load(f)
    cases_by_id = {c["item_id"]: c for c in parsed}
    item_ids = sorted(p.name for p in NEW_DIR.iterdir() if p.is_dir())

    if not args.combined_only:
        main()

    combined_path = OUT_DIR / f"combined_30_seed{args.seed}.csv"
    print(f"\nBuilding combined CSV (n_per_run={args.n_per_run}, seed={args.seed}) ...")
    rows = build_combined_csv(combined_path, cases_by_id, item_ids,
                              args.n_per_run, args.seed)
    print(f"  wrote {len(rows)} rows -> {combined_path}")

    n_checks, errors = verify_combined(rows)
    expected_checks = len(rows) * 2
    if errors:
        print(f"\nVERIFICATION FAILED: {len(errors)} mismatch(es)")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print(f"VERIFICATION OK: all {expected_checks} (row × side) "
              f"category↔response pairings match.")

    # Stats
    swapped_count = sum(1 for r in rows if r["swapped"] == "yes")
    by_run = {tag: sum(1 for r in rows if r["source_run"] == tag) for tag in ("A","B","C")}
    print(f"  per-run counts : {by_run}")
    print(f"  swapped        : {swapped_count}/{len(rows)}")
