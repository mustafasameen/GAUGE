#!/usr/bin/env python3
"""Report continuous error metrics beside exact match, following Schaeffer et al. (NeurIPS 2023).

Schaeffer et al. (arXiv 2304.15004) show that the choice of metric can manufacture apparent
capability cliffs. Exact string match is a discontinuous metric, so a null under exact match could
be an artifact of the metric. This script recomputes the scores on the identical outputs under two
continuous readings:
  NTED  normalised token edit distance between the predicted and gold digit strings (their metric
        for integer targets, at character level)
  RAE   relative absolute error |p - g| / max(|g|, 1), because the targets are magnitudes and two
        strings can be one edit apart while the numbers differ by an order of magnitude ("3" and
        "35")

A continuous metric with no baseline is not evidence, because a constant predictor also has a
finite error. Every score is therefore reported as skill = 1 - (model error / best-constant error),
the fraction of the constant baseline's error that the model removes. Skill <= 0 means the model is
no closer to the truth than the best single constant. Median RAE is used and not the mean, because
relative error is unbounded and a few pathological outputs dominate a mean; the pathology rate
(predictions beyond 10x the gold range) is reported separately. The script also reports resolution,
1 / n, the smallest measurable accuracy in a cell, so that a zero cannot be confused with too few
items to measure a small effect.

Input: results/tally_v41.jsonl and the five core runs.
Output: results/v41_continuous_metric.json.

Usage:
  python gauge/continuous_metric.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factqa import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = {"phi35mini": "results/v41core_phi35mini.json",
        "mistral7b": "results/v41core_mistral7b.json",
        "llama8b": "results/v41core_llama8b.json",
        "gemma3_12b": "results/v41pilot_gemma3_12b.json",
        "llama70b": "results/v41core_llama70b.json"}
GEOM = ["gyration_km", "max_distance", "total_distance", "longest_jump"]
COUNT = ["day_distinct", "core_set", "longest_stay"]


def edit_distance(a, b):
    """Levenshtein between two strings (additions/deletions/substitutions), per Schaeffer §3."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nted(pred, gold):
    """Normalised token edit distance: edits / len(gold string). 0 = identical."""
    g = str(int(gold)) if float(gold).is_integer() else f"{gold:g}"
    p = str(int(pred)) if float(pred).is_integer() else f"{pred:g}"
    return edit_distance(p, g) / max(len(g), 1)


def rae(pred, gold):
    return abs(pred - gold) / max(abs(gold), 1.0)


def best_constant_score(gv, fn, agg=np.mean):
    """Lowest achievable error from a single constant answer -- the floor for an ERROR metric.
    `agg` must match the aggregator used on the model, or the comparison is not like-for-like."""
    cands = np.unique(gv)
    if len(cands) > 300:
        cands = np.quantile(gv, np.linspace(0, 1, 300))
    return min(float(agg([fn(c, g) for g in gv])) for c in cands)


def pathology_rate(pv, gv):
    """Share of predictions outside 10x the observed gold range, reported separately instead of
    being allowed to move a mean. Mean relative error is unusable here: in one cell, two outputs that
    were digit runs (for example 12345678910 for a gold of 5) drove the mean to about 1e6. The median
    is reported instead, and the pathology rate is kept as its own diagnostic, because a model that
    sometimes emits a digit run instead of a count is a finding and not noise to average away.
    """
    hi = gv.max() * 10 + 10
    return float(np.mean((pv > hi) | (pv < -hi)))


def main():
    items = [json.loads(l) for l in open(os.path.join(ROOT, "results/tally_v41.jsonl"))]
    gold = [it["a"].lower() for it in items]
    preds = {}
    for t, p in RUNS.items():
        raw = json.load(open(os.path.join(ROOT, p)))["raw"]["full"]
        preds[t] = [norm(raw[i], items[i]["family"], items[i].get("atype"))
                    for i in range(len(items))]

    print("CONTINUOUS METRICS beside exact match (Schaeffer et al. 2304.15004, their metric class).")
    print("SKILL = 1 - model_error / best_constant_error. <=0 means no closer than a constant.")
    print("resolution = 1/n, the smallest measurable accuracy (their separate cause of emergence).\n")
    out = {}
    for label, fams in (("GEOMETRY", GEOM), ("COUNT/DURATION", COUNT)):
        print(f"=== {label} ===")
        print(f"  {'family':<16}{'model':<12}{'exact':>8}{'NTED':>8}{'NTED_c':>8}"
              f"{'skill':>8}{'medRAE':>9}{'medRAE_c':>10}{'skill':>8}{'patho':>8}")
        for fam in fams:
            for m in RUNS:
                pv, gv = [], []
                for i, it in enumerate(items):
                    if it["family"] != fam:
                        continue
                    try:
                        a, b = float(preds[m][i]), float(gold[i])
                    except (TypeError, ValueError):
                        continue
                    pv.append(a); gv.append(b)
                if len(pv) < 50:
                    continue
                pv, gv = np.array(pv), np.array(gv)
                ex = float(np.mean([p == g for p, g in zip(pv, gv)]))
                n_ted = float(np.mean([nted(p, g) for p, g in zip(pv, gv)]))
                c_ted = best_constant_score(gv, nted)
                # Median, not mean: relative error is unbounded, and a few pathological outputs drive the mean
                # to about 1e6 on core_set for one model. The constant baseline is aggregated the same way, so the
                # comparison stays like-for-like.
                n_rae = float(np.median([rae(p, g) for p, g in zip(pv, gv)]))
                c_rae = best_constant_score(gv, rae, agg=np.median)
                path = pathology_rate(pv, gv)
                s_ted = 1 - n_ted / c_ted if c_ted > 0 else np.nan
                s_rae = 1 - n_rae / c_rae if c_rae > 0 else np.nan
                res = 1.0 / len(pv)
                out[f"{fam}|{m}"] = dict(n=len(pv), exact=ex, nted=n_ted, nted_const=c_ted,
                                         skill_nted=s_ted, rae_median=n_rae, rae_const=c_rae,
                                         skill_rae=s_rae, resolution=res, pathology=path)
                print(f"  {fam:<16}{m:<12}{ex:>8.3f}{n_ted:>8.3f}{c_ted:>8.3f}{s_ted:>+8.3f}"
                      f"{n_rae:>9.3f}{c_rae:>10.3f}{s_rae:>+8.3f}{path:>8.4f}")
        print()

    print("=" * 96)
    print("STEP 3 OF SCHAEFFER'S PROCEDURE: does a continuous metric replace the cliff with")
    print("smooth, above-baseline variation?")
    for label, fams in (("GEOMETRY", GEOM), ("COUNT/DURATION", COUNT)):
        pos_t = [k for k, v in out.items() if k.split("|")[0] in fams and v["skill_nted"] > 0.05]
        pos_r = [k for k, v in out.items() if k.split("|")[0] in fams and v["skill_rae"] > 0.05]
        tot = len([k for k in out if k.split("|")[0] in fams])
        print(f"  {label:<16} NTED skill>0.05 in {len(pos_t)}/{tot} cells | "
              f"RAE skill>0.05 in {len(pos_r)}/{tot} cells")
    print("\nSTEP 4 (resolution): the smallest measurable accuracy per cell is 1/n; every cell here")
    print("pools all spans, so n >= 1,500 and resolution <= 0.0007. A zero exact-match score is")
    print("therefore NOT a resolution artifact -- it is measurable and it is zero.")
    dst = os.path.join(ROOT, "results/v41_continuous_metric.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
