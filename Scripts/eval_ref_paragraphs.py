#!/usr/bin/env python3
"""
Build the ref_paragraphs comparison dataset, three lists per (case, prompt):

    facts_paragraphs          numbers of paragraphs that actually appear in
                              the Facts section (extracted via the K=2
                              small-gap heuristic — see extract_facts_*).
    gt_ref_paragraphs         the union of `ref_paragraphs` across every
                              Article-10 Merits subsection in the parsed
                              case JSON. These are paragraphs the court
                              itself cited in its assessment.
    generated_ref_paragraphs  the `ref_paragraphs` field on each model
                              response (added by Scripts/add_citations.py).

Outputs:
    data/refparagraph_eval/refparagraph_eval.csv
    data/refparagraph_eval/refparagraph_eval.json
"""

import csv
import sys
import json
import re
from pathlib import Path

BASE_DIR        = Path(__file__).parent.parent
PARSED_PATH     = BASE_DIR / "data" / "parsed_cases_all(3,9,10,11).json"
NEW_DIR         = BASE_DIR / "Runs" / "outputs_new"
GPT_DIR         = BASE_DIR / "Runs" / "outputs_gpt_assisted"
EVAL_DIR        = BASE_DIR / "data" / "refparagraph_eval"
TARGET_ARTICLE  = 10
GAP_K           = 2   # facts-paragraph small-gap filter threshold

# The case set is the annotation dataset (same 22 distinct cases the
# human/judge and violation-prediction evaluations use), sourced from
# combined_30_seed42_og.csv. This INCLUDES the two Grand Chamber cases in the
# dataset (001-244292, 001-247738); the 3rd GC case 001-247839 is not in it.
ANNOTATION_CSV  = BASE_DIR / "data" / "annotation" / "combined_30_seed42_og.csv"

# Each prompt's response.json lives under a different parent directory.
PROMPT_SOURCES = {
    "instruction":  NEW_DIR,
    "non_curated":  NEW_DIR,
    "gpt_assisted": GPT_DIR,
}
PROMPT_NAMES = tuple(PROMPT_SOURCES.keys())


# ---------------------------------------------------------------------------
# Facts-paragraph extraction
# ---------------------------------------------------------------------------

_PARA_NUM_RE = re.compile(r'^(\d+)\.\s')


def extract_facts_paragraph_numbers(facts: list[str], max_gap: int = GAP_K) -> list[int]:
    """Return the real paragraph numbers in a facts section.

    Strategy: pull every '^N. ' candidate, then keep only those that form a
    strictly increasing sequence where each step is ≤ +max_gap. This drops
    sub-numbered list items in quoted decisions (which restart at 1, 2, 3)
    AND quoted-document paragraph numbers that jump far above the current
    real ECHR paragraph (e.g. quoted OSCE Guidelines paragraph 154)."""
    nums: list[int] = []
    last: int | None = None
    for text in facts:
        m = _PARA_NUM_RE.match(text)
        if not m:
            continue
        n = int(m.group(1))
        if last is None or last < n <= last + max_gap:
            nums.append(n)
            last = n
    return nums


# ---------------------------------------------------------------------------
# Ground-truth ref_paragraphs from Merits subsection(s)
# ---------------------------------------------------------------------------

# Subsection titles that hold the court's merits reasoning for Article 10.
# Chamber cases use "Merits"; Grand Chamber cases title the equivalent
# analysis "General". "Admissibility" is deliberately excluded. Case
# 001-242859 legitimately cites no facts paragraphs (true empty GT).
GT_SUBSECTION_TITLES = {"Merits", "General"}


def gt_merits_ref_paragraphs(case: dict) -> list[int]:
    """Sorted, deduped union of ref_paragraphs across every Article-10 merits
    subsection (title in GT_SUBSECTION_TITLES). Multi-section cases (e.g.
    001-243982, 001-247549) contribute paragraphs from all their merits blocks;
    Grand Chamber cases (001-244292, 001-247738) contribute via 'General'."""
    seen: set[int] = set()
    for ca in case.get("court_assessments", []):
        if TARGET_ARTICLE not in ca.get("articles", []):
            continue
        for sub in ca.get("subsections", []):
            if sub.get("subsection_title") not in GT_SUBSECTION_TITLES:
                continue
            for n in sub.get("ref_paragraphs", []):
                seen.add(int(n))
    return sorted(seen)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(a: int, b: int) -> float | None:
    return a / b if b else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p + r == 0:
        return None
    return 2 * p * r / (p + r)


def compute_metrics(gt: list[int], gen: list[int], facts: list[int]) -> dict:
    gt_set, gen_set, facts_set = set(gt), set(gen), set(facts)

    # Raw set comparison (Framing 1): unrestricted
    inter_raw = gt_set & gen_set
    raw_p = _safe_div(len(inter_raw), len(gen_set))
    raw_r = _safe_div(len(inter_raw), len(gt_set))
    raw_f1 = _f1(raw_p, raw_r)

    # Scope split
    gt_in_facts   = sorted(gt_set & facts_set)
    gt_outside    = sorted(gt_set - facts_set)
    gen_in_facts  = sorted(gen_set & facts_set)
    gen_outside   = sorted(gen_set - facts_set)

    # In-facts set comparison (Framing 2): both filtered to facts
    inter_if = set(gt_in_facts) & set(gen_in_facts)
    p_if = _safe_div(len(inter_if), len(gen_in_facts))
    r_if = _safe_div(len(inter_if), len(gt_in_facts))
    f1_if = _f1(p_if, r_if)

    # Groundedness: of everything the model emitted, what fraction was in
    # the facts paragraph set (not whether it matched GT).
    groundedness = _safe_div(len(gen_in_facts), len(gen_set))

    return {
        "n_gt_in_facts":      len(gt_in_facts),
        "n_gt_outside":       len(gt_outside),
        "n_gen_in_facts":     len(gen_in_facts),
        "n_gen_outside":      len(gen_outside),
        "n_intersection_raw":      len(inter_raw),
        "n_intersection_in_facts": len(inter_if),
        "raw_precision":      raw_p,
        "raw_recall":         raw_r,
        "raw_f1":             raw_f1,
        "precision_in_facts": p_if,
        "recall_in_facts":    r_if,
        "f1_in_facts":        f1_if,
        "groundedness":       groundedness,
        "gt_in_facts":        gt_in_facts,
        "gt_outside_facts":   gt_outside,
        "gen_in_facts":       gen_in_facts,
        "gen_outside_facts":  gen_outside,
        "intersection_in_facts": sorted(inter_if),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cases_by_id: dict[str, dict] = {}
    with open(PARSED_PATH, encoding="utf-8") as f:
        for c in json.load(f):
            cases_by_id[c["item_id"]] = c

    rows: list[dict] = []
    missing: list[str] = []

    csv.field_size_limit(sys.maxsize)
    with ANNOTATION_CSV.open(newline="", encoding="utf-8-sig") as fh:
        item_ids = sorted({r["item_id"] for r in csv.DictReader(fh)})

    for item_id in item_ids:
        case = cases_by_id.get(item_id)
        if case is None:
            missing.append(item_id)
            continue

        facts_nums = extract_facts_paragraph_numbers(case.get("facts", []))
        gt_refs    = gt_merits_ref_paragraphs(case)

        for prompt_name, source_dir in PROMPT_SOURCES.items():
            rj = source_dir / item_id / prompt_name / "response.json"
            if not rj.exists():
                continue
            with open(rj, encoding="utf-8") as f:
                resp = json.load(f)
            generated = sorted({int(n) for n in (resp.get("ref_paragraphs") or [])})

            metrics = compute_metrics(gt_refs, generated, facts_nums)
            rows.append({
                "item_id":                 item_id,
                "prompt_name":             prompt_name,
                "case_name":               case.get("case_name"),
                "facts_paragraphs":        facts_nums,
                "gt_ref_paragraphs":       gt_refs,
                "generated_ref_paragraphs": generated,
                "n_facts":                 len(facts_nums),
                "facts_min":               facts_nums[0]  if facts_nums else None,
                "facts_max":               facts_nums[-1] if facts_nums else None,
                "n_gt":                    len(gt_refs),
                "n_generated":             len(generated),
                **metrics,
            })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_path = EVAL_DIR / "refparagraph_eval.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = EVAL_DIR / "refparagraph_eval.csv"
    csv_cols = [
        "item_id", "prompt_name", "case_name",
        "n_facts", "facts_min", "facts_max",
        "n_gt", "n_generated",
        "n_gt_in_facts", "n_gt_outside", "n_gen_in_facts", "n_gen_outside",
        "n_intersection_raw", "n_intersection_in_facts",
        "raw_precision", "raw_recall", "raw_f1",
        "precision_in_facts", "recall_in_facts", "f1_in_facts",
        "groundedness",
        "facts_paragraphs", "gt_ref_paragraphs", "generated_ref_paragraphs",
        "gt_in_facts", "gt_outside_facts",
        "gen_in_facts", "gen_outside_facts",
        "intersection_in_facts",
    ]
    list_cols = {
        "facts_paragraphs", "gt_ref_paragraphs", "generated_ref_paragraphs",
        "gt_in_facts", "gt_outside_facts", "gen_in_facts", "gen_outside_facts",
        "intersection_in_facts",
    }
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in csv_cols}
            for c in list_cols:
                row[c] = "|".join(str(n) for n in row[c]) if row[c] else ""
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows -> {csv_path}")
    print(f"Wrote {len(rows)} rows -> {json_path}")
    if missing:
        print(f"WARN: {len(missing)} item(s) in outputs_new not in parsed cases: {missing}")

    # ---- aggregate stats ----
    def _mean(values: list[float]) -> float | None:
        defined = [v for v in values if v is not None]
        return sum(defined) / len(defined) if defined else None

    def _median(values: list[float]) -> float | None:
        defined = sorted(v for v in values if v is not None)
        if not defined:
            return None
        n = len(defined)
        mid = n // 2
        return defined[mid] if n % 2 else (defined[mid - 1] + defined[mid]) / 2

    by_prompt = {p: [r for r in rows if r["prompt_name"] == p] for p in PROMPT_NAMES}
    metric_keys = [
        "raw_precision", "raw_recall", "raw_f1",
        "precision_in_facts", "recall_in_facts", "f1_in_facts",
        "groundedness",
    ]

    print(f"\n{'metric':22} " + " ".join(f"{p:>20}" for p in PROMPT_NAMES))
    for key in metric_keys:
        line = f"{key:22} "
        for p in PROMPT_NAMES:
            vals = [r[key] for r in by_prompt[p]]
            m = _mean(vals); med = _median(vals)
            n_defined = sum(1 for v in vals if v is not None)
            cell = (f"mean={m:+.3f} med={med:+.3f} (n={n_defined})"
                    if m is not None else "—")
            line += f"{cell:>20} "
        print(line)

    # Counts
    print()
    for p, rs in by_prompt.items():
        avg_gt    = sum(r["n_gt"] for r in rs) / len(rs) if rs else 0
        avg_gen   = sum(r["n_generated"] for r in rs) / len(rs) if rs else 0
        avg_facts = sum(r["n_facts"] for r in rs) / len(rs) if rs else 0
        avg_gt_if = sum(r["n_gt_in_facts"] for r in rs) / len(rs) if rs else 0
        avg_gn_if = sum(r["n_gen_in_facts"] for r in rs) / len(rs) if rs else 0
        print(f"  {p:14} cases={len(rs):>3}  avg n_facts={avg_facts:.1f}  "
              f"n_gt={avg_gt:.1f} (in_facts={avg_gt_if:.1f})  "
              f"n_gen={avg_gen:.1f} (in_facts={avg_gn_if:.1f})")

    all_mins = [r["facts_min"] for r in rows if r["facts_min"] is not None]
    all_maxs = [r["facts_max"] for r in rows if r["facts_max"] is not None]
    print(f"\nfacts paragraph range across all cases: min={min(all_mins)}  max={max(all_maxs)}")


if __name__ == "__main__":
    main()
