#!/usr/bin/env python3
"""
LLM prediction pipeline for ECHR Article 10 cases via OpenRouter.

Usage:
    python3 Runs/run_pipeline.py
    or: OPENROUTER_API_KEY=sk-... python3 Runs/run_pipeline.py

Outputs per case × prompt:
    Runs/outputs/{item_id}/{prompt_name}/response.json
    Runs/outputs/{item_id}/{prompt_name}/reasoning.txt
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load the .env file
load_dotenv()

# OpenRouter model ID — user specified "GPT 5.4". Verify at openrouter.ai/models
MODEL = "openai/gpt-5.4"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Reasoning effort for GPT-5 series: "xhigh", "high", "medium", "low", "minimal", "none"
REASONING_EFFORT = "medium"

TARGET_ARTICLE = 10

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 5  # seconds, doubles on each retry

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "parsed_cases_all(3,9,10,11).json"
PROMPT_FILES = {
    "instruction":  BASE_DIR / "Prompts" / "Instruction_prompt.txt",
    "non_curated":  BASE_DIR / "Prompts" / "Non-Curated_prompt.txt",
    # "gpt_assisted": BASE_DIR / "Runs" / "gpt_assisted_prompt.txt",
}
OUTPUT_DIR = BASE_DIR / "Runs" / "outputs_new_gpt-5.4"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_art10_cases():
    """Load all cases that have an Article 10 court assessment section."""
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    seen_ids = set()
    cases = []
    for c in data:
        if c.get("error") or c.get("empty_file"):
            continue
        cid = c["item_id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        has_art10 = any(
            TARGET_ARTICLE in ca.get("articles", [])
            for ca in c.get("court_assessments", [])
        )
        if has_art10:
            cases.append(c)
    return cases


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_and_split_prompt(path: Path) -> str:
    """
    Return the static system message: everything in the prompt file before the
    'Relevant Case Law' section header. This part is identical across all cases
    so OpenAI caches it automatically after the first call per prompt type.
    """
    raw = path.read_text(encoding="utf-8").strip()

    boundary = re.search(r"\n(Relevant Case Law[:\s]*)\n", raw, re.IGNORECASE)
    static_part = raw[: boundary.start()].strip() if boundary else raw

    # Remove stray [...] placeholder lines
    static_part = re.sub(r"\n\s*\[\.\.\.\]\s*\n", "\n", static_part)
    static_part = re.sub(r"\n\s*\[\.\.\.\]\s*$", "", static_part)

    return static_part


def get_merits_citations(case: dict) -> list[str]:
    """Return deduplicated citations from the Merits subsection of Art 10 assessments."""
    seen: set[str] = set()
    citations: list[str] = []
    for ca in case.get("court_assessments", []):
        if TARGET_ARTICLE not in ca.get("articles", []):
            continue
        for sub in ca.get("subsections", []):
            if sub.get("subsection_title") == "Merits":
                for c in sub.get("citations", []):
                    if c not in seen:
                        seen.add(c)
                        citations.append(c)
    return citations


def build_messages(system_text: str, case: dict) -> list[dict]:
    """
    System message = static instructions (cached prefix, identical for every case).
    User message   = shuffled Merits citations + case facts.
    """
    citations = get_merits_citations(case)
    random.shuffle(citations)

    case_name = case.get("case_name", "")
    facts = case.get("facts", [])
    facts_body = "\n\n".join(facts) if facts else "[No facts available]"
    facts_text = f"{case_name}\n\n{facts_body}" if case_name else facts_body

    if citations:
        citations_text = "\n\n".join(citations)
        user_content = (
            f"Relevant Case Law\n\n{citations_text}\n\n"
            f"Examined Case Facts\n\n{facts_text}"
        )
    else:
        user_content = f"Examined Case Facts\n\n{facts_text}"

    return [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(client: OpenAI, messages: list[dict]):
    """Call OpenRouter with retry and exponential back-off."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                # max_tokens=6000,
                extra_body={
                    "reasoning": {
                        "effort": REASONING_EFFORT,
                    }
                },
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
    """Extract the JSON payload from the model's raw text reply."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON object anywhere in the text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def cached_tokens(response) -> int:
    try:
        return response.usage.prompt_tokens_details.cached_tokens or 0
    except AttributeError:
        return 0


def save_result(
    case: dict,
    prompt_name: str,
    response,
    parsed: dict | None,
) -> Path:
    out_dir = OUTPUT_DIR / case["item_id"] / prompt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    msg = response.choices[0].message
    raw_content = msg.content or ""
    reasoning   = getattr(msg, "reasoning", None) or ""

    result = {
        "item_id":                case["item_id"],
        "case_name":              case.get("case_name"),
        "case_number":            case.get("case_number"),
        "date":                   case.get("date"),
        "prompt_type":            prompt_name,
        "model":                  MODEL,
        "reasoning_effort":       REASONING_EFFORT,
        "violation_ground_truth": case.get("violation", []),
        "violation_prediction":   parsed.get("violation")   if parsed else None,
        "assessment":             parsed.get("assessment")  if parsed else None,
        "raw_content":            raw_content,
        "usage": (
            response.usage.model_dump()
            if response.usage else {}
        ),
    }

    (out_dir / "response.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")

    return out_dir


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(api_key: str):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    cases = load_art10_cases()
    print(f"Article {TARGET_ARTICLE} violation cases: {len(cases)}")

    # Load prompts once — system messages are reused identically for every case
    prompt_configs: dict[str, str] = {}
    for name, path in PROMPT_FILES.items():
        system_text = load_and_split_prompt(path)
        prompt_configs[name] = system_text
        print(f"  Prompt '{name}': {len(system_text.split())} words in system message (cached prefix)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total  = len(cases) * len(prompt_configs)
    done   = 0
    errors: list[tuple] = []

    for case in cases:
        cid = case["item_id"]
        print(f"\n[{cid}]  {case.get('case_name', '')}")

        for prompt_name, system_text in prompt_configs.items():
            out_path = OUTPUT_DIR / cid / prompt_name / "response.json"

            if out_path.exists():
                print(f"  [{prompt_name}] already done — skipping")
                done += 1
                continue

            print(f"  [{prompt_name}] calling {MODEL} …", end=" ", flush=True)
            messages = build_messages(system_text, case)

            try:
                response = call_api(client, messages)
                raw_content = response.choices[0].message.content or ""
                parsed      = parse_json_response(raw_content)
                out_dir     = save_result(case, prompt_name, response, parsed)

                pred    = parsed.get("violation", "?") if parsed else "parse_error"
                cached  = cached_tokens(response)
                total_t = response.usage.total_tokens if response.usage else "?"
                print(
                    f"OK  prediction={pred}  "
                    f"total_tokens={total_t}  cached_tokens={cached}"
                )
                done += 1

            except Exception as exc:
                print(f"ERROR: {exc}")
                errors.append((cid, prompt_name, str(exc)))

    print(f"\n{'='*60}")
    print(f"Completed: {done}/{total}  |  Errors: {len(errors)}")
    if errors:
        for cid, pname, err in errors:
            print(f"  {cid} [{pname}]: {err}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"ECHR Art {TARGET_ARTICLE} LLM prediction pipeline via OpenRouter"
    )
    parser.add_argument(
        "--api-key",
        default=OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"OpenRouter model ID (default: {MODEL})",
    )
    parser.add_argument(
        "--effort",
        default=REASONING_EFFORT,
        choices=["xhigh", "high", "medium", "low", "minimal", "none"],
        help=f"Reasoning effort level (default: {REASONING_EFFORT})",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY not set. Use --api-key or export the environment variable."
        )

    MODEL            = args.model
    REASONING_EFFORT = args.effort

    run_pipeline(args.api_key)
