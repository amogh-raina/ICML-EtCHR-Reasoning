#!/usr/bin/env python3
"""
Aggregate LLM-judge evaluation scores per prompt category.

Reads every `Runs/outputs_eval_(<cat_a> vs <cat_b>)_<model>/<case>/response.json`
produced by `Scripts/run_eval.py`, attributes each student_a / student_b score
block to its prompt category (instruction / non_curated / gpt_assisted), then
reports mean ± std for every metric — both pooled across judge models and
broken down per model.

Metrics per category:
    step_1.occurrence ... step_4.occurrence       (binary, 0/1)
    step_1.comprehensiveness ... step_4.compr.    (0 if not occurred, else 1-5)
    overall_conciseness                            (1-5)

Outputs:
    data/judge_eval_summary/long_rows.csv   — one row per (case, model, pair, side)
    data/judge_eval_summary/aggregates.json — mean/std/n per (category, metric)
                                              and per (category, model, metric)
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
RUNS_DIR = BASE_DIR / "Runs"
OUT_DIR  = BASE_DIR / "data" / "judge_eval_summary"

# Grand Chamber cases excluded from all evaluation — paper argues we do
# not evaluate on them.
EXCLUDED_CASES = {"001-244292", "001-247738", "001-247839"}

# Folder name fragments → canonical category names (folder names use
# inconsistent separators and one folder has a typo "instructions")
CATEGORY_ALIAS = {
    "instruction":  "instruction",
    "instructions": "instruction",
    "non-curated":  "non_curated",
    "non_curated":  "non_curated",
    "gpt_assisted": "gpt_assisted",
    "gpt-assisted": "gpt_assisted",
}

FOLDER_RE      = re.compile(r"^outputs_eval_\((.+)\)_(.+)$")
PAIR_SPLIT_RE  = re.compile(r"\s+vs\s+|_vs_", re.IGNORECASE)

CATEGORIES  = ["instruction", "non_curated", "gpt_assisted"]
STEPS       = ["step_1", "step_2", "step_3", "step_4"]
METRIC_KEYS = (
    [f"{s}.occurrence"        for s in STEPS]
    + [f"{s}.comprehensiveness" for s in STEPS]
    + ["overall_conciseness"]
)


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def discover_folders() -> list[tuple[Path, str, str, str]]:
    """Return (folder, cat_a, cat_b, model) for every parseable eval folder."""
    found = []
    for p in sorted(RUNS_DIR.iterdir()):
        if not p.is_dir():
            continue
        m = FOLDER_RE.match(p.name)
        if not m:
            continue
        pair_raw, model_raw = m.group(1), m.group(2)
        parts = PAIR_SPLIT_RE.split(pair_raw)
        if len(parts) != 2:
            continue
        cat_a = CATEGORY_ALIAS.get(parts[0].strip().lower())
        cat_b = CATEGORY_ALIAS.get(parts[1].strip().lower())
        if not cat_a or not cat_b:
            continue
        found.append((p, cat_a, cat_b, model_raw.strip()))
    return found


# ---------------------------------------------------------------------------
# Robust JSON-shape parser
#
# The judge LLMs produce evaluation blocks in inconsistent shapes. We try
# every shape we've seen and return a flat {metric: int} dict, or None if
# we can't extract all 9 metrics for the given student.
# ---------------------------------------------------------------------------

def _find_student_block(evaluation, student_name: str):
    """Locate the data block for 'student_a' / 'student_b' in any nesting."""
    if isinstance(evaluation, list):
        for entry in evaluation:
            if isinstance(entry, dict) and student_name in entry:
                return entry[student_name]
    if isinstance(evaluation, dict) and student_name in evaluation:
        return evaluation[student_name]
    return None


def _read_step_dict(d: dict) -> tuple[int | None, int | None]:
    """Read (occurrence, comprehensiveness) from a single step dict."""
    if not isinstance(d, dict):
        return None, None
    occ  = d.get("occurrence")
    comp = d.get("comprehensiveness")
    return (int(occ)  if occ  is not None else None,
            int(comp) if comp is not None else None)


def extract_scores(block) -> dict[str, int] | None:
    """Pull the 9 metrics out of a student block, regardless of nesting style.

    Handles:
      - list of single-key dicts: [{'step_1': {...}}, ..., {'overall_conciseness': N}]
      - flat dict:                {'step_1': {...}, ..., 'overall_conciseness': N}
      - list with one wrapping dict: [{'step_1': {...}, 'step_2': {...}, ...}]
    """
    if block is None:
        return None

    scores: dict[str, int | None] = {k: None for k in METRIC_KEYS}

    # Normalise to an iterable of (key, value) pairs at the step level
    def absorb(key: str, val) -> None:
        if key in STEPS:
            occ, comp = _read_step_dict(val)
            if occ  is not None: scores[f"{key}.occurrence"]        = occ
            if comp is not None: scores[f"{key}.comprehensiveness"] = comp
        elif key == "overall_conciseness" and val is not None:
            scores["overall_conciseness"] = int(val)

    if isinstance(block, list):
        for entry in block:
            if not isinstance(entry, dict):
                continue
            for k, v in entry.items():
                absorb(k, v)
    elif isinstance(block, dict):
        for k, v in block.items():
            absorb(k, v)

    if any(scores[k] is None for k in METRIC_KEYS):
        return None
    return scores  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Per-folder extraction
# ---------------------------------------------------------------------------

def parse_folder(folder: Path, cat_a: str, cat_b: str, model: str
                 ) -> tuple[list[dict], list[str]]:
    """Yield one long-format row per (case × side). Returns (rows, skipped)."""
    rows: list[dict] = []
    skipped: list[str] = []

    for case_dir in sorted(folder.iterdir()):
        if not case_dir.is_dir():
            continue
        if case_dir.name in EXCLUDED_CASES:
            continue
        rj = case_dir / "response.json"
        if not rj.exists():
            continue
        try:
            d = json.loads(rj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append(case_dir.name + " (bad json)")
            continue

        parsed = d.get("parsed") or {}
        # Judge LLMs frame the wrapper inconsistently. We've seen all of:
        #   {"evaluation": [...]}
        #   {"assessment": [...]}
        #   {"evaluation": {"assessment": [...]}}      ← DeepSeek follows
        #                                                the prompt example
        #                                                literally
        #   {"evaluation": [{"student_a": ...}]}
        # Walk a couple of levels down to find the actual student blocks.
        evaluation = parsed.get("evaluation") or parsed.get("assessment") or parsed
        if isinstance(evaluation, dict):
            inner = evaluation.get("assessment") or evaluation.get("evaluation")
            if inner is not None:
                evaluation = inner

        for side_key, (side, category) in (("student_a", ("a", cat_a)),
                                            ("student_b", ("b", cat_b))):
            block  = _find_student_block(evaluation, side_key)
            scores = extract_scores(block)
            if scores is None:
                skipped.append(f"{case_dir.name}/{side_key}")
                continue
            rows.append({
                "item_id":  case_dir.name,
                "folder":   folder.name,
                "model":    model,
                "pair":     f"{cat_a}_vs_{cat_b}",
                "side":     side,
                "category": category,
                **scores,
            })
    return rows, skipped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_std_n(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values, ddof=1)) if n >= 2 else 0.0,
        "n":    n,
    }


def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        per_model: dict[str, list[dict]] = defaultdict(list)
        for r in cat_rows:
            per_model[r["model"]].append(r)

        out[cat] = {
            "n_observations": len(cat_rows),
            "models":         sorted(per_model.keys()),
            "pooled":         {m: _mean_std_n([r[m] for r in cat_rows])
                               for m in METRIC_KEYS},
            "per_model":      {
                model: {m: _mean_std_n([r[m] for r in rs]) for m in METRIC_KEYS}
                for model, rs in per_model.items()
            },
        }
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def write_long_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["item_id", "folder", "model", "pair", "side", "category"] + METRIC_KEYS
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in cols})


def print_summary(agg: dict) -> None:
    print("\n" + "=" * 100)
    print("POOLED across judge models  (mean ± std,  n)")
    print("=" * 100)
    print(f"{'metric':<32}" + "".join(f"{c:>22}" for c in CATEGORIES))
    for metric in METRIC_KEYS:
        line = f"{metric:<32}"
        for cat in CATEGORIES:
            cell = agg[cat]["pooled"][metric]
            line += f"   {_fmt(cell['mean']):>4} ± {_fmt(cell['std']):<4} n={cell['n']:>3}"
        print(line)

    print("\n" + "=" * 100)
    print("BREAKDOWN by judge model")
    print("=" * 100)
    for cat in CATEGORIES:
        models = agg[cat]["models"]
        if not models:
            continue
        print(f"\n[{cat}]  (total observations across models: {agg[cat]['n_observations']})")
        print(f"  {'metric':<32}" + "".join(f"{m:>22}" for m in models))
        for metric in METRIC_KEYS:
            line = f"  {metric:<32}"
            for m in models:
                cell = agg[cat]["per_model"][m][metric]
                line += f"   {_fmt(cell['mean']):>4} ± {_fmt(cell['std']):<4} n={cell['n']:>3}"
            print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    folders = discover_folders()
    print(f"Discovered {len(folders)} eval folders:")
    for path, cat_a, cat_b, model in folders:
        print(f"  {path.name}  →  A={cat_a}  B={cat_b}  model={model}")

    all_rows: list[dict] = []
    all_skipped: list[str] = []
    for path, cat_a, cat_b, model in folders:
        rows, skipped = parse_folder(path, cat_a, cat_b, model)
        all_rows.extend(rows)
        all_skipped.extend(f"{path.name}: {s}" for s in skipped)
        print(f"    parsed {len(rows)} score rows ({len(skipped)} skipped)")

    print(f"\nTotal score rows: {len(all_rows)}  Skipped: {len(all_skipped)}")
    if all_skipped:
        print("Skipped detail (first 20):")
        for s in all_skipped[:20]:
            print(f"  {s}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_csv = OUT_DIR / "long_rows.csv"
    write_long_csv(all_rows, long_csv)
    print(f"\nWrote long-format CSV -> {long_csv}")

    agg = aggregate(all_rows)
    (OUT_DIR / "aggregates.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote aggregates JSON -> {OUT_DIR / 'aggregates.json'}")

    print_summary(agg)


if __name__ == "__main__":
    main()
