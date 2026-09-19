#!/usr/bin/env python3
"""
Compute (1) inter-annotator agreement across the three human annotators and
(2) alignment between each LLM judge and the human consensus, for the ECtHR
Article-10 reasoning-quality annotations.

Design decisions (agreed with the authors):
  * Comprehensiveness agreement is computed on OCCURRED steps only: a rater's
    0 ("step did not occur") is treated as missing, so comprehensiveness
    reliability is not confounded with occurrence disagreement. Occurrence is
    its own criterion.
  * The human reference for judge alignment is the per-item CONSENSUS:
    mean for ordinal criteria (comprehensiveness, conciseness), majority for
    binary occurrence.

Item alignment
--------------
The human JSON arrays (human_{1,2,3}_annotations.json) are ordered exactly as
convert_human_annotations.py produced them: iterate combined_30_seed42_og.csv
in row order, side A then side B, appending to annotations[category]. We replay
that walk to recover, for every array position, the (item_id, setting, pair)
it belongs to. Judges are then matched on (item_id, setting) within the SAME
pair the human saw, using long_rows.csv.

Outputs (to data/agreement/):
  agreement.json         machine-readable results
  agreement_tables.md    human-readable tables
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
ANN = BASE / "data" / "annotation"
META_CSV = ANN / "combined_30_seed42_og.csv"
LONG_ROWS = BASE / "data" / "judge_eval_summary" / "long_rows.csv"
OUT_DIR = BASE / "data" / "agreement"

SETTINGS = ["non_curated", "instruction", "gpt_assisted"]
STEPS = ["step_1", "step_2", "step_3", "step_4"]
JUDGES = ["claude", "deepseek", "gpt5.5"]

SETTING_LABEL = {
    "non_curated": "A: Non-Curated",
    "instruction": "B: Curated (Expert)",
    "gpt_assisted": "C: Curated (Guide)",
}


# --------------------------------------------------------------------------- #
# Item-order reconstruction
# --------------------------------------------------------------------------- #
def pair_name(a: str, b: str) -> frozenset:
    return frozenset({a, b})


def reconstruct_order() -> dict:
    """Return {setting: [(item_id, pair_frozenset), ...]} in JSON array order."""
    csv.field_size_limit(sys.maxsize)
    with META_CSV.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    order = {s: [] for s in SETTINGS}
    for r in rows:
        pr = pair_name(r["A_category"], r["B_category"])
        for side in ("A", "B"):
            setting = r[f"{side}_category"]
            order[setting].append((r["item_id"], pr))
    return order


# --------------------------------------------------------------------------- #
# Krippendorff's alpha (nominal / ordinal), with missing values as None
# --------------------------------------------------------------------------- #
def krippendorff(raters: list[list], level: str) -> float:
    """raters: list of equal-length sequences; None = missing."""
    n_units = len(raters[0])
    from collections import Counter

    marg = Counter()
    for r in raters:
        for v in r:
            if v is not None:
                marg[v] += 1
    grades = sorted(marg)

    def delta(a, b):
        if a == b:
            return 0.0
        if level == "nominal":
            return 1.0
        if level == "interval":
            return float((a - b) ** 2)
        # ordinal
        lo, hi = sorted((a, b))
        s = sum(marg[g] for g in grades if lo <= g <= hi)
        s -= (marg[a] + marg[b]) / 2.0
        return s * s

    # observed disagreement
    Do = 0.0
    total = 0
    for u in range(n_units):
        col = [r[u] for r in raters if r[u] is not None]
        m = len(col)
        if m < 2:
            continue
        Do += sum(delta(a, b) for a, b in itertools.permutations(col, 2)) / (m - 1)
        total += m
    if total == 0:
        return float("nan")
    Do /= total

    allv = [v for r in raters for v in r if v is not None]
    De = sum(delta(a, b) for a, b in itertools.permutations(allv, 2)) / (
        len(allv) * (len(allv) - 1)
    )
    return 1 - Do / De if De > 0 else float("nan")


def pct_agreement(raters: list[list]) -> tuple[float, float, int]:
    """Pairwise exact and within-1 agreement over items where >=2 raters present."""
    n = len(raters[0])
    exact = within1 = tot = 0
    for u in range(n):
        col = [(i, r[u]) for i, r in enumerate(raters) if r[u] is not None]
        for (_, a), (_, b) in itertools.combinations(col, 2):
            tot += 1
            d = abs(a - b)
            exact += d == 0
            within1 += d <= 1
    if tot == 0:
        return float("nan"), float("nan"), 0
    return exact / tot, within1 / tot, tot


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx * dy else float("nan")


# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
def load_humans() -> list[dict]:
    return [json.loads((ANN / f"human_{i}_annotations.json").read_text()) for i in (1, 2, 3)]


def load_judges() -> dict:
    """{(item_id, pair_frozenset, setting, judge): {metric: value}}"""
    out = {}
    with LONG_ROWS.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pr = frozenset(row["pair"].split("_vs_"))
            key = (row["item_id"], pr, row["category"], row["model"])
            rec = {}
            for st in STEPS:
                rec[f"{st}.occurrence"] = _to_int(row[f"{st}.occurrence"])
                rec[f"{st}.comprehensiveness"] = _to_int(row[f"{st}.comprehensiveness"])
            rec["overall_conciseness"] = _to_int(row["overall_conciseness"])
            out[key] = rec
    return out


def _to_int(v):
    v = (v or "").strip()
    if v == "":
        return None
    try:
        return int(round(float(v)))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Build aligned rater vectors
# --------------------------------------------------------------------------- #
def build_vectors(humans, order):
    """
    Returns dict of flat aligned vectors across all settings/steps:
      occ[rater]  : occurrence, one entry per (setting, step, item)
      comp[rater] : comprehensiveness, 0 -> None (occurred-only)
      conc[rater] : conciseness, one entry per (setting, item)
    Plus item_keys aligned to occ/comp entries: (item_id, pair, setting, step)
    and conc_keys: (item_id, pair, setting).
    """
    occ = [[], [], []]
    comp = [[], [], []]
    conc = [[], [], []]
    item_keys = []
    conc_keys = []

    for setting in SETTINGS:
        ids = order[setting]  # list of (item_id, pair)
        for st in STEPS:
            for pos, (item_id, pr) in enumerate(ids):
                for ri in range(3):
                    o = humans[ri][setting][st]["occurrence"][pos]
                    c = humans[ri][setting][st]["comprehensiveness"][pos]
                    occ[ri].append(o)
                    comp[ri].append(None if c == 0 else c)
                item_keys.append((item_id, pr, setting, st))
        for pos, (item_id, pr) in enumerate(ids):
            for ri in range(3):
                conc[ri].append(humans[ri][setting]["overall_conciseness"][pos])
            conc_keys.append((item_id, pr, setting))
    return occ, comp, conc, item_keys, conc_keys


def consensus(vecs):
    """Per-position consensus: mean (ordinal) ignoring None; None if all None."""
    out = []
    for i in range(len(vecs[0])):
        vals = [v[i] for v in vecs if v[i] is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def majority(vecs):
    out = []
    for i in range(len(vecs[0])):
        vals = [v[i] for v in vecs if v[i] is not None]
        out.append(1 if vals and sum(vals) / len(vals) >= 0.5 else (0 if vals else None))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    order = reconstruct_order()
    humans = load_humans()
    judges = load_judges()
    occ, comp, conc, item_keys, conc_keys = build_vectors(humans, order)

    results = {"human_iaa": {}, "judge_alignment": {}, "judge_iaa": {}}

    # ---- Human inter-annotator agreement ----
    a_occ = krippendorff(occ, "nominal")
    ex_o, w1_o, n_o = pct_agreement(occ)
    a_comp = krippendorff(comp, "ordinal")
    ex_c, w1_c, n_c = pct_agreement(comp)
    a_conc = krippendorff(conc, "ordinal")
    ex_z, w1_z, n_z = pct_agreement(conc)
    results["human_iaa"] = {
        "occurrence": {"alpha_nominal": a_occ, "exact": ex_o, "within1": w1_o, "n_pairs": n_o},
        "comprehensiveness": {"alpha_ordinal": a_comp, "exact": ex_c, "within1": w1_c, "n_pairs": n_c},
        "conciseness": {"alpha_ordinal": a_conc, "exact": ex_z, "within1": w1_z, "n_pairs": n_z},
    }

    # ---- Human consensus vectors ----
    occ_cons = majority(occ)
    comp_cons = consensus(comp)
    conc_cons = consensus(conc)

    # ---- Judge alignment vs human consensus ----
    for judge in JUDGES:
        jc = {}
        # comprehensiveness (occurred-only) at (item, step) granularity
        hx, jx = [], []
        agree_exact = agree_within1 = n = 0
        sdiff = 0.0
        for idx, (item_id, pr, setting, st) in enumerate(item_keys):
            h = comp_cons[idx]
            if h is None:
                continue
            rec = judges.get((item_id, pr, setting, judge))
            if not rec:
                continue
            jv = rec[f"{st}.comprehensiveness"]
            if jv is None or jv == 0:
                continue
            hx.append(h)
            jx.append(jv)
            sdiff += jv - h
            agree_exact += abs(round(h) - jv) == 0
            agree_within1 += abs(h - jv) <= 1
            n += 1
        jc["comprehensiveness"] = {
            "spearman": spearman(hx, jx) if n > 1 else float("nan"),
            "mean_signed_diff_judge_minus_human": sdiff / n if n else float("nan"),
            "within1": agree_within1 / n if n else float("nan"),
            "n": n,
        }

        # conciseness at (item) granularity
        hx, jx = [], []
        n = 0
        sdiff = 0.0
        w1 = 0
        for idx, (item_id, pr, setting) in enumerate(conc_keys):
            h = conc_cons[idx]
            if h is None:
                continue
            rec = judges.get((item_id, pr, setting, judge))
            if not rec:
                continue
            jv = rec["overall_conciseness"]
            if jv is None:
                continue
            hx.append(h)
            jx.append(jv)
            sdiff += jv - h
            w1 += abs(h - jv) <= 1
            n += 1
        jc["conciseness"] = {
            "spearman": spearman(hx, jx) if n > 1 else float("nan"),
            "mean_signed_diff_judge_minus_human": sdiff / n if n else float("nan"),
            "within1": w1 / n if n else float("nan"),
            "n": n,
        }

        # occurrence: % agreement with human majority
        agree = n = 0
        for idx, (item_id, pr, setting, st) in enumerate(item_keys):
            h = occ_cons[idx]
            if h is None:
                continue
            rec = judges.get((item_id, pr, setting, judge))
            if not rec:
                continue
            jv = rec[f"{st}.occurrence"]
            if jv is None:
                continue
            agree += h == jv
            n += 1
        jc["occurrence"] = {"pct_agreement": agree / n if n else float("nan"), "n": n}

        results["judge_alignment"][judge] = jc

    # ---- Judge-judge agreement (ordinal alpha on comprehensiveness/conciseness) ----
    # Build per-judge aligned comprehensiveness vectors over item_keys
    jvecs_comp = {j: [] for j in JUDGES}
    for idx, (item_id, pr, setting, st) in enumerate(item_keys):
        for j in JUDGES:
            rec = judges.get((item_id, pr, setting, j))
            v = rec[f"{st}.comprehensiveness"] if rec else None
            jvecs_comp[j].append(None if (v is None or v == 0) else v)
    results["judge_iaa"]["comprehensiveness_alpha_ordinal"] = krippendorff(
        [jvecs_comp[j] for j in JUDGES], "ordinal"
    )
    jvecs_conc = {j: [] for j in JUDGES}
    for idx, (item_id, pr, setting) in enumerate(conc_keys):
        for j in JUDGES:
            rec = judges.get((item_id, pr, setting, j))
            v = rec["overall_conciseness"] if rec else None
            jvecs_conc[j].append(v)
    results["judge_iaa"]["conciseness_alpha_ordinal"] = krippendorff(
        [jvecs_conc[j] for j in JUDGES], "ordinal"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "agreement.json").write_text(json.dumps(results, indent=2))
    write_markdown(results)
    print((OUT_DIR / "agreement_tables.md").read_text())


def write_markdown(res):
    L = []
    h = res["human_iaa"]
    L.append("## Human inter-annotator agreement (3 annotators)\n")
    L.append("| Criterion | Krippendorff α | Exact % | Within-1 % | n pairs |")
    L.append("|---|---|---|---|---|")
    L.append(f"| Step occurrence (nominal) | {h['occurrence']['alpha_nominal']:.3f} | "
             f"{h['occurrence']['exact']*100:.1f} | {h['occurrence']['within1']*100:.1f} | {h['occurrence']['n_pairs']} |")
    L.append(f"| Step comprehensiveness (ordinal, occurred-only) | {h['comprehensiveness']['alpha_ordinal']:.3f} | "
             f"{h['comprehensiveness']['exact']*100:.1f} | {h['comprehensiveness']['within1']*100:.1f} | {h['comprehensiveness']['n_pairs']} |")
    L.append(f"| Overall conciseness (ordinal) | {h['conciseness']['alpha_ordinal']:.3f} | "
             f"{h['conciseness']['exact']*100:.1f} | {h['conciseness']['within1']*100:.1f} | {h['conciseness']['n_pairs']} |")

    L.append("\n## Judge ↔ human-consensus alignment\n")
    L.append("| Judge | Compr. ρ | Compr. bias (J−H) | Compr. within-1 | Concise ρ | Concise bias | Occurrence %agree |")
    L.append("|---|---|---|---|---|---|---|")
    for j, d in res["judge_alignment"].items():
        c, z, o = d["comprehensiveness"], d["conciseness"], d["occurrence"]
        L.append(f"| {j} | {c['spearman']:.3f} | {c['mean_signed_diff_judge_minus_human']:+.2f} | "
                 f"{c['within1']*100:.1f} | {z['spearman']:.3f} | {z['mean_signed_diff_judge_minus_human']:+.2f} | "
                 f"{o['pct_agreement']*100:.1f} |")
    ji = res["judge_iaa"]
    L.append(f"\nJudge–judge α: comprehensiveness={ji['comprehensiveness_alpha_ordinal']:.3f}, "
             f"conciseness={ji['conciseness_alpha_ordinal']:.3f}")
    L.append("\n(ρ = Spearman across matched items; bias = mean(judge − human consensus), "
             "positive = judge more lenient.)")
    OUT_DIR.joinpath("agreement_tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
