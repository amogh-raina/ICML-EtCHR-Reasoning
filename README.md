# Can LLMs Reason in a Legally Meaningful Manner?

Code, prompts, model outputs and human annotations for **"Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases"** (Raina, Chalkidis, Hershcovich, Olsen), presented at the **AI4LAW workshop, ICML 2026**.

Paper: <https://arxiv.org/abs/2608.17168>

## About

We use European Court of Human Rights (ECtHR) cases on **Article 10 (freedom of expression)** as a testbed for legal reasoning in LLMs. Given only a case's facts, GPT-5.4 writes a step-by-step legal assessment and predicts whether Article 10 was violated. We compare three prompting strategies that differ in how much they tell the model about what legally meaningful reasoning looks like, and score the assessments against the court's own reasoning with three human annotators and three LLM judges.

| Strategy (folder / JSON key) | Paper label | What the prompt gives the model |
|---|---|---|
| `non_curated` | A: Non-Curated | Free-form, no structure |
| `instruction` | B: Curated (Expert) | An expert-written 4-step scaffold: interference, prescribed by law, legitimate aim, necessary in a democratic society |
| `gpt_assisted` | C: Curated (Guide) | A strategy an LLM derived from the ECtHR Article 10 case-law guide |

**Main findings** (see the paper for numbers):

- The model produces structurally complete but substantively shallow legal analyses.
- LLM judges agree with each other but align only weakly with trained human annotators: reliable, not a valid substitute for human evaluation.
- The expert-curated prompt yields more comprehensive reasoning, but not more accurate predictions. Task accuracy is a poor proxy for reasoning quality.

## What is in this repository

| Path | Contents |
|---|---|
| `Prompts/` | All prompts: the three generator prompts, the judge rubric, and the two prompts used to generate the guide-based strategy |
| `Runs/` | Every raw model output. `outputs_new/` (strategies A and B) and `outputs_gpt_assisted/` (strategy C) are the GPT-5.4 predictions; `outputs_eval_*` are the LLM-judge scores (GPT-5.5, Claude Opus 4.7, DeepSeek V4 Pro); `outputs_mistral*` is a Mistral Medium 3.5 robustness run; `outputs_guideline*` are the guide-to-strategy generations |
| `data/annotation/` | The human annotation sheets, the three annotators' scores (`human_{1,2,3}_annotations.json`, pooled in `human_annotations.json`), the annotation manifest and the annotation guidelines |
| `data/` (other folders) | Computed results: judge scores, inter-annotator agreement, violation accuracy, token usage, citation and paragraph-reference evaluation, the overall results table |
| `Scripts/` | Generation, judging, evaluation, agreement and plotting scripts |
| `scraping/` | HUDOC scraper and the HTML-to-JSON case parser |

Experiment scope: **29 Article 10 cases** have model outputs. Human annotation covers a seeded sample of 30 (case, strategy-pair) items spanning 22 distinct cases, each rated by all three annotators on step occurrence, step comprehensiveness (1-5) and overall conciseness (1-5).

`CLAUDE.md` is a detailed map of the repo (which script reads and writes what, file formats, and known pitfalls). It is written for coding agents such as Claude Code, but it is a useful reference for people too.

## Getting started

```bash
pip install -r requirements.txt          # Python 3.9+
```

**Case data.** The parsed case file is too large for git. Download `parsed_cases_all(3,9,10,11).json` (and the original annotator `.xlsx` workbooks, if you want them) from **[GOOGLE DRIVE](https://drive.google.com/drive/folders/11Ug0WSMM433xYsdvCwwe9Mx-lLqY_Pir?usp=sharing)** and place the JSON at `data/parsed_cases_all(3,9,10,11).json`. Article 10 cases are the entries with a court assessment covering Article 10.

**API key** (only for scripts that call an LLM). Everything goes through [OpenRouter](https://openrouter.ai):

```bash
cp .env.example .env                     # then set OPENROUTER_API_KEY
```

### What runs without downloads or an API key

All the model outputs and human annotations are in the repo, so the analysis can be reproduced from a plain clone:

```bash
python3 Scripts/compute_eval_scores.py   # judge outputs -> data/judge_eval_summary/
python3 Scripts/compute_agreement.py     # human IAA and judge-vs-human alignment -> data/agreement/
python3 Scripts/eval_token_usage.py      # token cost per strategy
python3 Scripts/build_overall_results_table.py   # the paper's main results table
python3 Scripts/build_combined_heatmaps.py
```

### What needs the case data

`eval_violation_prediction.py`, `eval_citations.py` and `eval_ref_paragraphs.py` compare against the court's text, so they need the parsed JSON. Their outputs are already in `data/`, so you only need to re-run them if you change the model outputs.

### Re-running generation and judging (needs the case data and an API key)

```bash
python3 Scripts/run_pipeline.py          # predictions; model, effort and output dir are constants in the file
python3 Scripts/add_citations.py --write # extract citations / paragraph references from responses
python3 Scripts/run_eval.py --dir-a Runs/outputs_new --prompt-a instruction \
    --dir-b Runs/outputs_gpt_assisted --prompt-b gpt_assisted \
    --output-dir "Runs/outputs_eval_(instructions vs gpt_assisted)_gpt5.5"
```

Model IDs and reasoning effort are set as constants at the top of each script. Run all commands from the repository root. Some scripts reference paths that have moved since they were written; `CLAUDE.md` lists them.

## Data notes

- Case texts come from [HUDOC](https://hudoc.echr.coe.int/), the ECtHR's public case-law database. Please check HUDOC's terms before redistributing them.
- Human annotation is on a small sample and the annotators' agreement is low on the 1-5 scales (though high within one point), so treat the annotations as a study of evaluation reliability, not as a gold-standard benchmark.
- Strategy C was run at higher reasoning effort (`high`) than A and B (`medium`), so comparisons involving it are confounded by compute.

## Citation

```bibtex
@misc{raina2026legallymeaningful,
  title         = {Can {LLMs} Reason in a Legally Meaningful Manner? A Small-scale Study on {European Court of Human Rights} Cases},
  author        = {Raina, Amogh and Chalkidis, Ilias and Hershcovich, Daniel and Olsen, Henrik Palmer},
  year          = {2026},
  eprint        = {2608.17168},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.17168}
}
```
