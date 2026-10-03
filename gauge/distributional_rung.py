#!/usr/bin/env python3
"""Distributional and ordinal measures for the magnitude-valued families, against two references.

The paper scores recovery at three levels: distributional match, ordinal agreement and exact
recovery. This script computes the first two for the seven families whose answers are magnitudes
(the four geometric families, day_distinct, core_set and longest_stay). For each model, family and
record length it compares the distribution of the model's answers across people with the
distribution of the true values across the same people, using the Jensen-Shannon divergence (base 2,
so it lies in [0, 1]) on 20 equal-mass bins of the gold values. It also computes the Spearman rank
correlation rho between the model's answers and the true values.

Two reference predictors give the divergence its scale. Neither carries any information about an
individual.
  sampler   a permutation of the gold values: a perfect distributional match by construction, with
            zero per-record information (averaged over 20 draws)
  constant  the modal gold value, repeated: no distribution at all
A model whose divergence sits near the sampler's and whose rho is near 0 reproduces the
distribution without carrying the individual. A model near the constant's has no distribution
either.

Input: results/questions.jsonl and the main runs (results/primary_gemma3_12b.json and
results/main_*.json).
Output: results/distributional_rung.json, one entry per model, family and span, with
jsd_model, jsd_constant, jsd_sampler and rho.

Usage:
  python gauge/distributional_rung.py [--items ...] [--runs ...] [--out ...]
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBINS = 20
MAGNITUDE = ["gyration_km", "max_distance", "total_distance", "longest_jump",
             "day_distinct", "core_set", "longest_stay"]


def jsd(p_vals, q_vals, edges):
    """Jensen-Shannon divergence (base 2, so it lands in [0,1]) between two samples on shared bins."""
    p, _ = np.histogram(p_vals, bins=edges)
    q, _ = np.histogram(q_vals, bins=edges)
    p = p.astype(float); q = q.astype(float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _midrank(x):
    """Average ranks for ties. Ranking with argsort(argsort(x)) gives ties arbitrary distinct ranks,
    which biases Spearman downward when models repeat answers, as they do here. This matches
    scipy.stats.spearmanr.
    """
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return np.nan
    ra, rb = _midrank(a), _midrank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="results/questions.jsonl")
    ap.add_argument("--runs", nargs="*",
                    default=["results/primary_gemma3_12b.json", "results/main_*.json"])
    ap.add_argument("--out", default="results/distributional_rung.json")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.join(ROOT, a.items))]
    gold = [it["a"].lower() for it in items]
    paths = []
    for pat in a.runs:
        paths.extend(sorted(glob.glob(os.path.join(ROOT, pat))))
    if not paths:
        sys.exit("no runs matched")
    rng = np.random.default_rng(0)
    out = {}

    for path in paths:
        tag = os.path.basename(path).replace(".json", "").replace(
            "main_", "").replace("primary_", "")
        raw = json.load(open(path))["raw"]["full"]
        if len(raw) != len(items):
            print(f"SKIP {tag}: length mismatch"); continue
        pred = [norm(raw[i], items[i]["family"], items[i].get("atype")) for i in range(len(items))]
        print(f"\n=== {tag} ===")
        print("  JSD: 0 = distribution matches exactly, 1 = disjoint. rho = per-record rank corr.")
        print(f"  {'family':<18}{'S':>5}{'JSD model':>11}{'JSD sampler':>13}"
              f"{'JSD const':>11}{'rho':>8}   verdict")
        for fam in MAGNITUDE:
            spans = sorted({it["span"] for it in items if it["family"] == fam})
            for s in spans:
                ix = [i for i, it in enumerate(items)
                      if it["family"] == fam and it["span"] == s]
                pv, gv = [], []
                for i in ix:
                    try:
                        x, y = float(pred[i]), float(gold[i])
                    except (TypeError, ValueError):
                        continue
                    pv.append(x); gv.append(y)
                if len(pv) < 30:
                    continue
                pv, gv = np.array(pv), np.array(gv)
                # Equal-mass (quantile) bins on the gold, not linear bins. Linear bins over the joint range are
                # wrong for heavy-tailed magnitudes: total_distance spans about 0 to thousands, so 20 linear bins
                # put nearly all of the gold in bin 0 and the constant predictor would score a divergence of 0.000,
                # which is impossible and shows that the binning, not the model, was being measured. Quantile bins
                # give each bin about 5% of the gold mass, so a constant lands in one bin and is penalised properly.
                # Infinite outer edges keep out-of-range predictions inside the histogram instead of clipping them.
                qs = np.quantile(gv, np.linspace(0, 1, NBINS + 1))
                inner = np.unique(qs)[1:-1]
                if len(inner) < 3:
                    continue          # too few distinct gold values to bin meaningfully
                edges = np.concatenate([[-np.inf], inner, [np.inf]])
                j_model = jsd(pv, gv, edges)
                # marginal sampler: a permutation of the SAME golds -> perfect distribution,
                # zero per-record information. Averaged over 20 draws to remove sampling noise.
                j_samp = float(np.mean([jsd(rng.permutation(gv), gv, edges) for _ in range(20)]))
                mode = collections.Counter(gv).most_common(1)[0][0]
                j_const = jsd(np.full(len(gv), mode), gv, edges)
                rho = spearman(pv, gv)
                # verdict, decided by which reference the model's JSD sits nearer
                if not np.isfinite(j_model):
                    v = "n/a"
                elif abs(j_model - j_samp) <= abs(j_model - j_const):
                    v = "DISTRIBUTIONAL" if abs(rho) < 0.2 else "ordinal+"
                else:
                    v = "NO DISTRIBUTION" if abs(rho) < 0.2 else "ordinal, poor dist"
                out[f"{tag}|{fam}@{s}"] = dict(model=tag, family=fam, span=s, jsd_model=j_model,
                                               jsd_sampler=j_samp, jsd_constant=j_const,
                                               rho=rho, verdict=v, n=len(pv))
                print(f"  {fam:<18}{s:>5}{j_model:>11.3f}{j_samp:>13.3f}"
                      f"{j_const:>11.3f}{rho:>8.3f}   {v}")

    dst = os.path.join(ROOT, a.out)
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")
    print("\nREAD THIS AS: the three-rung ladder requires cells where JSD(model) ~ JSD(sampler)")
    print("AND rho ~ 0. Cells near JSD(constant) have no distribution either, and for those the")
    print("ladder has two rungs, not three.")


def _is_num(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
