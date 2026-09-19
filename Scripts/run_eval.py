#!/usr/bin/env python3
"""
LLM judge evaluation pipeline for ECHR Article 10 cases.

For each of the 29 Art 10 cases it:
  - loads case facts and the court's Merits assessment from the parsed cases JSON
  - loads Response A and Response B from configurable output directories/prompt subfolders
  - fills the four {} placeholders in Prompts/judge_prompt.txt
  - calls the judge LLM and saves the evaluation

Usage:
    # Default: instruction vs non_curated (both from outputs_new)
    python3 Scripts/run_eval.py

    # instruction (outputs_new) vs gpt_assisted (outputs_gpt_assisted)
    python3 Scripts/run_eval.py \
        --dir-b Runs/outputs_gpt_assisted \
        --prompt-b gpt_assisted \
        --output-dir Runs/outputs_eval_gpt_assisted

Outputs per case:
    {output_dir}/{item_id}/response.json
    {output_dir}/{item_id}/reasoning.txt
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "deepseek/deepseek-v4-pro"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# On Anthropic 4.7+ this is sent as `verbosity` (Anthropic ignores reasoning.effort
# and uses adaptive-only thinking); on OpenAI/Deepseek it's sent as `reasoning.effort`.
REASONING_EFFORT = "high"


def build_extra_body(model: str, effort: str) -> dict:
    """OpenRouter request body shape depends on the model family.

    Anthropic Claude 4.7 Opus: reasoning is adaptive-only and `reasoning.effort`
    / `reasoning.max_tokens` are silently ignored. `verbosity` (→ Anthropic's
    output_config.effort) is the remaining lever for response effort and
    accepts: low / medium / high / xhigh / max.

    Everyone else (OpenAI, Deepseek, etc.): pass `reasoning.effort` directly.
    """
    if model.startswith("anthropic/"):
        return {"reasoning": {"enabled": True}, "verbosity": effort}
    return {"reasoning": {"effort": effort}}

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 5

TARGET_ARTICLE = 10

BASE_DIR = Path(__file__).parent.parent
DATA_PATH         = BASE_DIR / "data" / "parsed_cases_all(3,9,10,11).json"
JUDGE_PROMPT_FILE = BASE_DIR / "Prompts" / "judge_prompt.txt"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_art10_cases() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    seen_ids: set[str] = set()
    cases = []
    for c in data:
        if c.get("error") or c.get("empty_file"):
            continue
        cid = c["item_id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        if any(TARGET_ARTICLE in ca.get("articles", []) for ca in c.get("court_assessments", [])):
            cases.append(c)
    return cases


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


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def format_assessment(assessment) -> str:
    """Convert the stored assessment (list of single-key dicts) to readable text."""
    if isinstance(assessment, list):
        parts = []
        for item in assessment:
            if isinstance(item, dict):
                for title, content in item.items():
                    parts.append(f"{title}\n\n{content}")
        return "\n\n".join(parts)
    return str(assessment) if assessment is not None else "[No assessment]"


def load_response_assessment(base_dir: Path, item_id: str, prompt_name: str) -> str | None:
    path = base_dir / item_id / prompt_name / "response.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return format_assessment(d.get("assessment"))


def build_message(case: dict, dir_a: Path, prompt_a: str, dir_b: Path, prompt_b: str) -> str | None:
    template = JUDGE_PROMPT_FILE.read_text(encoding="utf-8").strip()

    # Case facts
    case_name  = case.get("case_name", "")
    facts_body = "\n\n".join(case.get("facts", [])) or "[No facts available]"
    case_facts = f"{case_name}\n\n{facts_body}" if case_name else facts_body

    # Court's Merits assessment
    courts_assessment = get_merits_text(case)

    # Response A and B from their respective directories/subfolders
    response_a = load_response_assessment(dir_a, case["item_id"], prompt_a)
    response_b = load_response_assessment(dir_b, case["item_id"], prompt_b)

    if response_a is None or response_b is None:
        return None

    # Replace the four {} placeholders one at a time.
    # Cannot use str.format() because the template contains other braces (in examples).
    filled = template
    for value in [case_facts, courts_assessment, response_a, response_b]:
        filled = filled.replace("{}", value, 1)
    return filled


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(client: OpenAI, user_content: str):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_content}],
                extra_body=build_extra_body(MODEL, REASONING_EFFORT),
            )
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    attempt {attempt + 1} failed ({exc}); retrying in {delay}s …")
                time.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def save_result(case: dict, response, parsed: dict | None, output_dir: Path) -> None:
    out_dir = output_dir / case["item_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    msg         = response.choices[0].message
    raw_content = msg.content or ""
    reasoning   = getattr(msg, "reasoning", None) or ""

    result = {
        "item_id":          case["item_id"],
        "case_name":        case.get("case_name"),
        "model":            MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "raw_content":      raw_content,
        "parsed":           parsed,
        "usage":            response.usage.model_dump() if response.usage else {},
    }

    (out_dir / "response.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"ECHR Art {TARGET_ARTICLE} LLM judge evaluation pipeline"
    )
    parser.add_argument("--dir-a",    default=str(BASE_DIR / "Runs" / "outputs_new"),
                        help="Base directory for Response A (default: Runs/outputs_new)")
    parser.add_argument("--prompt-a", default="instruction",
                        help="Prompt subfolder for Response A (default: instruction)")
    parser.add_argument("--dir-b",    default=str(BASE_DIR / "Runs" / "outputs_new"),
                        help="Base directory for Response B (default: Runs/outputs_new)")
    parser.add_argument("--prompt-b", default="non_curated",
                        help="Prompt subfolder for Response B (default: non_curated)")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "Runs" / "outputs_eval"),
                        help="Directory to save evaluation results (default: Runs/outputs_eval)")
    args = parser.parse_args()

    dir_a      = Path(args.dir_a)
    dir_b      = Path(args.dir_b)
    output_dir = Path(args.output_dir)

    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    cases = load_art10_cases()
    print(f"Article {TARGET_ARTICLE} cases loaded: {len(cases)}")
    print(f"Response A: {dir_a.name}/{args.prompt_a}")
    print(f"Response B: {dir_b.name}/{args.prompt_b}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    done   = 0
    errors: list[tuple] = []

    for case in cases:
        cid      = case["item_id"]
        out_path = output_dir / cid / "response.json"

        if out_path.exists():
            print(f"[{cid}] already done — skipping")
            done += 1
            continue

        print(f"[{cid}]  {case.get('case_name', '')} …", end=" ", flush=True)

        message = build_message(case, dir_a, args.prompt_a, dir_b, args.prompt_b)
        if message is None:
            print(f"SKIP (missing {args.prompt_a} or {args.prompt_b} response)")
            continue

        try:
            response    = call_api(client, message)
            raw_content = response.choices[0].message.content or ""
            parsed      = parse_json_response(raw_content)
            save_result(case, response, parsed, output_dir)

            total_t = response.usage.total_tokens if response.usage else "?"
            print(f"OK  total_tokens={total_t}")
            done += 1

        except Exception as exc:
            print(f"ERROR: {exc}")
            errors.append((cid, str(exc)))

    print(f"\n{'='*60}")
    print(f"Completed: {done}/{len(cases)}  |  Errors: {len(errors)}")
    if errors:
        for cid, err in errors:
            print(f"  {cid}: {err}")
