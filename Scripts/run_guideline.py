#!/usr/bin/env python3
"""
Send the Article 10 guide through the guideline prompt and save the output.

Usage:
    python3 Runs/run_guideline.py               # with guideline
    python3 Runs/run_guideline.py --no-guideline # prompt only, no guide text
    or: OPENROUTER_API_KEY=sk-... python3 Runs/run_guideline.py [--no-guideline]

Outputs (with guideline):
    Runs/outputs_guideline_v4/response.json
    Runs/outputs_guideline_v4/reasoning.txt

Outputs (no guideline):
    Runs/outputs_no_guideline/response.json
    Runs/outputs_no_guideline/reasoning.txt
"""

import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "openai/gpt-5.4"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
REASONING_EFFORT = "high"

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 5

BASE_DIR = Path(__file__).parent.parent
GUIDE_FILE  = BASE_DIR / "data" / "guide_10.txt"

GUIDELINE_PROMPT_FILE = BASE_DIR / "Runs" / "guideline_prompt.txt"
GUIDELINE_OUTPUT_DIR  = BASE_DIR / "Runs" / "outputs_guideline_v4"

NO_GUIDELINE_PROMPT_FILE = BASE_DIR / "Runs" / "no_guideline_prompt.txt"
NO_GUIDELINE_OUTPUT_DIR  = BASE_DIR / "Runs" / "outputs_no_guideline"

PLACEHOLDER = "{article 10 guide.txt}"

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_message(no_guideline: bool) -> str:
    if no_guideline:
        return NO_GUIDELINE_PROMPT_FILE.read_text(encoding="utf-8").strip()
    prompt = GUIDELINE_PROMPT_FILE.read_text(encoding="utf-8").strip()
    guide  = GUIDE_FILE.read_text(encoding="utf-8").strip()
    return prompt.replace(PLACEHOLDER, guide)


def call_api(client: OpenAI, user_content: str):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_content}],
                # max_tokens=10000,
                extra_body={"reasoning": {"effort": REASONING_EFFORT}},
            )
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  attempt {attempt + 1} failed ({exc}); retrying in {delay}s …")
                time.sleep(delay)
            else:
                raise


def save(response, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    msg         = response.choices[0].message
    raw_content = msg.content or ""
    reasoning   = getattr(msg, "reasoning", None) or ""

    parsed = None
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    result = {
        "model":            MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "raw_content":      raw_content,
        "parsed":           parsed,
        "usage":            response.usage.model_dump() if response.usage else {},
    }

    (output_dir / "response.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")
    print(f"Saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-guideline", action="store_true",
                        help="Use no_guideline_prompt.txt without injecting the guide text")
    args = parser.parse_args()

    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if args.no_guideline:
        prompt_name = NO_GUIDELINE_PROMPT_FILE.name
        output_dir  = NO_GUIDELINE_OUTPUT_DIR
        print(f"Building message from {prompt_name} (no guide) …")
    else:
        prompt_name = GUIDELINE_PROMPT_FILE.name
        output_dir  = GUIDELINE_OUTPUT_DIR
        print(f"Building message from {GUIDE_FILE.name} + {prompt_name} …")

    user_content = build_message(args.no_guideline)
    print(f"Message length: {len(user_content):,} chars — calling {MODEL} …")

    response = call_api(client, user_content)

    total_t = response.usage.total_tokens if response.usage else "?"
    print(f"Done. total_tokens={total_t}")

    save(response, output_dir)
