#!/usr/bin/env python3
"""Tolerance gate: is a near-zero exact match a capability result or a formatting artifact?

Lukasik et al. (2024, arXiv 2403.04182) argue that mode-seeking (greedy) decoding implicitly
optimises exact match. That makes "greedy decoding optimises the wrong metric" the default
explanation for a near-zero exact-match score on a numeric task. The geometric result rests on such
a score, so it has to survive scoring under tolerance. If models still score about zero within
+/-5% relative tolerance, the failure is not rounding or formatting.

Every tolerance metric needs its own baseline. A constant predictor scored under a +/-25% tolerance
can look excellent on a tight distribution, so raw tolerance accuracy is uninterpretable. For each
tolerance the script computes the best constant predictor's score on the same cell (the value that
maximises tolerance hits, found by scanning the gold values) and reports the normalized gain
against it. This is the tolerance analogue of the majority-class baseline. The tolerances are exact
match, +/-1 absolute, and +/-5%, +/-10% and +/-25% relative.

It also reports whether the cross-model rank spread survives tolerance scoring.

Input: results/questions.jsonl and the five main runs.
Output: results/tolerance_gate.json.

Usage:
  python gauge/tolerance_gate.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = {"phi35mini": "results/main_phi35mini.json",
        "mistral7b": "results/main_mistral7b.json",
        "llama8b": "results/main_llama8b.json",
        "gemma3_12b": "results/primary_gemma3_12b.json",
        "llama70b": "results/main_llama70b.json"}
GEOM = ["gyration_km", "max_distance", "total_distance", "longest_jump"]
COUNT = ["day_distinct", "core_set", "longest_stay"]
# (label, predicate on (pred, gold))
TOLS = [("exact", lambda p, g: p == g),
        ("abs1", lambda p, g: abs(p - g) <= 1),
        ("rel5", lambda p, g: abs(p - g) <= 0.05 * max(abs(g), 1e-9)),
        ("rel10", lambda p, g: abs(p - g) <= 0.10 * max(abs(g), 1e-9)),
        ("rel25", lambda p, g: abs(p - g) <= 0.25 * max(abs(g), 1e-9))]


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


def best_constant(gv, pred_fn):
    """Highest tolerance-score any single constant answer achieves on this cell -- the floor."""
    cands = np.unique(gv)
    if len(cands) > 400:                      # keep it cheap on wide cells
        cands = np.quantile(gv, np.linspace(0, 1, 400))
    best = 0.0
    for c in cands:
        s = float(np.mean([pred_fn(c, g) for g in gv]))
        if s > best:
            best = s
    return best


def main():
    items = [json.loads(l) for l in open(os.path.join(ROOT, "results/questions.jsonl"))]
    gold = [it["a"].lower() for it in items]
    preds = {}
    for t, p in RUNS.items():
        raw = json.load(open(os.path.join(ROOT, p)))["raw"]["full"]
        preds[t] = [norm(raw[i], items[i]["family"], items[i].get("atype"))
                    for i in range(len(items))]
    out = {}
    print("TOLERANCE GATE. gain = (score - best-constant score) / (1 - best-constant score).")
    print("A capability claim needs gain > 0.05 to be *rescued* by tolerance. Near-zero gain at")
    print("rel25 means the failure is not rounding.\n")

    for label, fams in (("GEOMETRY", GEOM), ("COUNT/DURATION", COUNT)):
        print(f"=== {label} ===")
        hdr = f"{'family':<16}{'model':<12}"
        for tl, _ in TOLS:
            hdr += f"{tl:>16}"
        print(hdr)
        print(f"{'':<28}" + "".join(f"{'score / gain':>16}" for _ in TOLS))
        for fam in fams:
            spans = sorted({it["span"] for it in items if it["family"] == fam})
            for m in RUNS:
                # pool over spans for a compact, honest per-(family,model) read
                pv, gv = [], []
                for s in spans:
                    for i in [k for k, it in enumerate(items)
                              if it["family"] == fam and it["span"] == s]:
                        try:
                            a, b = float(preds[m][i]), float(gold[i])
                        except (TypeError, ValueError):
                            continue
                        pv.append(a); gv.append(b)
                if len(pv) < 50:
                    continue
                pv, gv = np.array(pv), np.array(gv)
                row = f"{fam:<16}{m:<12}"
                rec = {}
                for tl, fn in TOLS:
                    sc = float(np.mean([fn(a, b) for a, b in zip(pv, gv)]))
                    fl = best_constant(gv, fn)
                    gn = (sc - fl) / (1 - fl) if fl < 1 else 0.0
                    rec[tl] = dict(score=sc, floor=fl, gain=gn)
                    row += f"{sc:>7.3f} /{gn:>+7.3f}"
                rec["rho"] = spearman(pv, gv)
                out[f"{fam}|{m}"] = rec
                print(row)
        print()

    # ---- the two questions this gate answers ----
    print("=" * 78)
    print("Q1. Does tolerance RESCUE geometry? (any model, any tolerance, gain > 0.05)")
    rescued = [(k, tl, v[tl]["gain"]) for k, v in out.items() if k.split("|")[0] in GEOM
               for tl, _ in TOLS if v[tl]["gain"] > 0.05]
    if rescued:
        print(f"   *** YES -- {len(rescued)} (family,model,tolerance) combinations clear 0.05:")
        for k, tl, g in sorted(rescued, key=lambda r: -r[2])[:15]:
            print(f"       {k:<28} {tl:<7} gain {g:+.3f}")
        print("   => the near-zero exact match is at least partly a PRECISION/FORMATTING effect.")
        print("      The geometry claim must be restated at the tolerance where it survives.")
    else:
        print("   NO -- no geometric family clears 0.05 gain at any tolerance up to +/-25%.")
        print("   => the failure is NOT rounding or formatting. The capability claim stands.")

    print("\nQ2. Does the CROSS-MODEL rank spread survive tolerance scoring?")
    for fam in GEOM + COUNT:
        rows = {m: out[f"{fam}|{m}"] for m in RUNS if f"{fam}|{m}" in out}
        if len(rows) < 2:
            continue
        rho = {m: v["rho"] for m, v in rows.items()}
        r25 = {m: v["rel25"]["gain"] for m, v in rows.items()}
        print(f"   {fam:<16} rho spread {min(rho.values()):+.3f}..{max(rho.values()):+.3f} "
              f"(range {max(rho.values())-min(rho.values()):.3f})   "
              f"rel25 gain spread {min(r25.values()):+.3f}..{max(r25.values()):+.3f}")

    dst = os.path.join(ROOT, "results/tolerance_gate.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
