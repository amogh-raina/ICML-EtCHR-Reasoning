#!/usr/bin/env python3
"""
Aggregate completion / reasoning / output token usage per prompt category.

For each (case, prompt) response.json we pull `usage`:
    completion_tokens          = visible output + reasoning tokens
    reasoning_tokens           = from usage.completion_tokens_details
    output_tokens (derived)    = completion_tokens - reasoning_tokens
    prompt_tokens, total_tokens are kept for context

Then we report mean ± std per prompt category for output_tokens and
reasoning_tokens (the two numbers the user wants on the LaTeX table).

Outputs:
    data/token_usage/token_usage.csv   — one row per (item_id, prompt_name)
    data/token_usage/token_usage.json
"""

import csv
import sys
import json
from pathlib import Path

import numpy as np

BASE_DIR  = Path(__file__).parent.parent
NEW_DIR   = BASE_DIR / "Runs" / "outputs_new"
GPT_DIR   = BASE_DIR / "Runs" / "outputs_gpt_assisted"
OUT_DIR   = BASE_DIR / "data" / "token_usage"

# Case set = the 22 distinct cases of the annotation dataset (same set the
# other evaluations use), sourced from combined_30_seed42_og.csv. Includes the
# two Grand Chamber cases present in the dataset (001-244292, 001-247738).
ANNOTATION_CSV = BASE_DIR / "data" / "annotation" / "combined_30_seed42_og.csv"

PROMPT_SOURCES = {
    "instruction":  NEW_DIR,
    "non_curated":  NEW_DIR,
    "gpt_assisted": GPT_DIR,
}


def extract_usage(d: dict) -> dict | None:
    """Return {completion, reasoning, output, prompt, total} or None if absent."""
    u = d.get("usage") or {}
    if not u:
        return None
    completion = u.get("completion_tokens")
    prompt     = u.get("prompt_tokens")
    total      = u.get("total_tokens")
    details    = u.get("completion_tokens_details") or {}
    reasoning  = details.get("reasoning_tokens") or 0
    if completion is None:
        return None
    output = int(completion) - int(reasoning)
    return {
        "completion_tokens": int(completion),
        "reasoning_tokens":  int(reasoning),
        "output_tokens":     int(output),
        "prompt_tokens":     int(prompt) if prompt is not None else None,
        "total_tokens":      int(total)  if total  is not None else None,
        "model":             d.get("model"),
    }


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    with ANNOTATION_CSV.open(newline="", encoding="utf-8-sig") as fh:
        item_ids = sorted({r["item_id"] for r in csv.DictReader(fh)})

    rows: list[dict] = []
    missing: list[str] = []

    for cid in item_ids:
        for prompt_name, source_dir in PROMPT_SOURCES.items():
            rj = source_dir / cid / prompt_name / "response.json"
            if not rj.exists():
                missing.append(f"{cid}/{prompt_name}: missing file")
                continue
            d = json.loads(rj.read_text(encoding="utf-8"))
            usage = extract_usage(d)
            if usage is None:
                missing.append(f"{cid}/{prompt_name}: no usage block")
                continue
            rows.append({
                "item_id":     cid,
                "prompt_name": prompt_name,
                "case_name":   d.get("case_name"),
                **usage,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "token_usage.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_cols = ["item_id", "prompt_name", "case_name", "model",
                "prompt_tokens", "completion_tokens", "reasoning_tokens",
                "output_tokens", "total_tokens"]
    with open(OUT_DIR / "token_usage.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in csv_cols})

    print(f"Wrote {len(rows)} rows -> {OUT_DIR}/token_usage.{{csv,json}}")
    if missing:
        print(f"\n{len(missing)} skipped:")
        for m in missing:
            print(f"  {m}")

    # ---- per-prompt aggregates ----
    print(f"\n{'prompt':14} {'n':>3}  {'output_tokens':>22}  {'reasoning_tokens':>22}  {'completion_tokens':>22}  {'prompt_tokens':>20}")
    for prompt_name in PROMPT_SOURCES:
        sub = [r for r in rows if r["prompt_name"] == prompt_name]
        def stats(key: str) -> str:
            vals = [r[key] for r in sub if r.get(key) is not None]
            if not vals:
                return "—"
            mu = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) >= 2 else 0.0
            return f"{mu:>8.1f} ± {sd:>6.1f}"
        print(f"{prompt_name:14} {len(sub):>3}  "
              f"{stats('output_tokens'):>22}  "
              f"{stats('reasoning_tokens'):>22}  "
              f"{stats('completion_tokens'):>22}  "
              f"{stats('prompt_tokens'):>20}")


if __name__ == "__main__":
    main()
