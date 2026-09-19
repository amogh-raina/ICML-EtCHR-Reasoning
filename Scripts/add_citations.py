#!/usr/bin/env python3
"""
Post-process response.json files in outputs_new to extract and add
'citations' and 'ref_paragraphs' from each case's 'assessment' field.

Reuses extract_citations() from scraping/parse_cases.py for citation
extraction.  Uses a broader paragraph-reference pattern than parse_cases.py
because the model responses use ECHR-proceedings format "(see paragraph N)"
without the "above"/"below" qualifier required by the original parser.

Usage:
    python3 Scripts/add_citations.py               # dry run (print only)
    python3 Scripts/add_citations.py --write        # update files in place

Processes: Runs/outputs_new/{item_id}/{instruction,non_curated}/response.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Reuse citation extraction from the scraping module
sys.path.insert(0, str(Path(__file__).parent.parent / "scraping"))
from parse_cases import extract_citations  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent

# (parent dir, prompt-subfolder names that live under each case dir)
SOURCES = [
    (BASE_DIR / "Runs" / "outputs_new",          {"instruction", "non_curated"}),
    (BASE_DIR / "Runs" / "outputs_gpt_assisted", {"gpt_assisted"}),
]

# ---------------------------------------------------------------------------
# Paragraph-reference extraction (broader than parse_cases.py)
#
# parse_cases.py requires "above"/"below" to avoid false positives from
# §-number citations inside court text.  Model responses use the ECHR format
# "(see paragraph N)" without that qualifier, but also don't use "§ N" for
# their own para refs, so the broader pattern is safe here.
# ---------------------------------------------------------------------------

_DASH_CLASS = r'[-‐‑‒–—―−]'
_DASH_CHARS = '‐‑‒–—―−'

_PARA_REF_RE = re.compile(
    r'\bparagraphs?\s+'
    r'((?:\d+(?:\s*(?:' + _DASH_CLASS + r'|\bto\b)\s*\d+)?'
    r'\s*(?:,\s*|\s+and\s+))*'
    r'\d+(?:\s*(?:' + _DASH_CLASS + r'|\bto\b)\s*\d+)?)',
    re.IGNORECASE,
)


def _expand_para_list(s: str) -> list[int]:
    """'5, 6 and 8' or '23-25' → sorted list of ints."""
    nums: set[int] = set()
    s = re.sub(r'\band\b', ',', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+to\s+', '-', s, flags=re.IGNORECASE)
    for ch in _DASH_CHARS:
        s = s.replace(ch, '-')
    s = s.replace('\xa0', ' ')
    s = re.sub(r'\s*-\s*', '-', s)
    for token in re.split(r'[,\s]+', s):
        token = token.strip()
        if not token:
            continue
        m = re.match(r'^(\d+)-(\d+)$', token)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a < b < a + 500:
                nums.update(range(a, b + 1))
        elif re.match(r'^\d+$', token):
            nums.add(int(token))
    return sorted(nums)


def extract_ref_paragraphs(texts: list[str]) -> list[int]:
    nums: set[int] = set()
    for text in texts:
        for m in _PARA_REF_RE.finditer(text.replace('\xa0', ' ')):
            nums.update(_expand_para_list(m.group(1)))
    return sorted(nums)


# ---------------------------------------------------------------------------
# Assessment text extraction
# ---------------------------------------------------------------------------

def assessment_to_texts(assessment) -> list[str]:
    """Convert stored assessment (list of {title: content} dicts) to a list of text strings."""
    if not isinstance(assessment, list):
        return []
    texts = []
    for item in assessment:
        if isinstance(item, dict):
            for content in item.values():
                if content:
                    texts.append(str(content))
    return texts


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(path: Path, write: bool) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    assessment = d.get("assessment")
    texts = assessment_to_texts(assessment)

    citations    = extract_citations(texts)
    ref_paragraphs = extract_ref_paragraphs(texts)

    result = {"citations": citations, "ref_paragraphs": ref_paragraphs}

    if write:
        d["citations"]     = citations
        d["ref_paragraphs"] = ref_paragraphs
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write extracted fields back to response.json (default: dry run)")
    args = parser.parse_args()

    if not args.write:
        print("DRY RUN — pass --write to update files\n")

    processed = skipped = 0

    for outputs_dir, prompt_names in SOURCES:
        if not outputs_dir.exists():
            continue
        for item_dir in sorted(outputs_dir.iterdir()):
            if not item_dir.is_dir():
                continue
            for prompt_name in sorted(prompt_names):
                rj = item_dir / prompt_name / "response.json"
                if not rj.exists():
                    continue

                result = process_file(rj, write=args.write)
                action = "updated" if args.write else "would update"
                print(
                    f"[{outputs_dir.name}/{item_dir.name}/{prompt_name}]  {action}  "
                    f"citations={result['citations']}  "
                    f"ref_paragraphs={result['ref_paragraphs']}"
                )
                processed += 1

    print(f"\n{'='*60}")
    print(f"{'Updated' if args.write else 'Would update'}: {processed} files")
