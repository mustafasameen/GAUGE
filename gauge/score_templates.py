#!/usr/bin/env python3
"""Score the prompt-wording study: is the geometric result an artifact of how the question is asked?

The five templates are built by make_templates.py (T0 incumbent, T1 separator, T2 joiner and case,
T3 serialisation, T4 paraphrase). Each metric is taken from the paper that defines it.

From Sclar et al. (2023, arXiv 2310.11324):
  * Performance spread = max - min across templates on a fixed dataset: a range on the metric's own
    scale, not a variance or an interval. Their smallest published configuration has 10 formats and
    they call even that a lower bound, so with five templates the spread here is a lower bound
    everywhere it appears.
  * The per-template answer (parse) rate. A template that looks worse may simply produce unparseable
    output, so the rate is reported beside accuracy and never folded into it.
  * Rank reversal: whether the ordering of the models flips between templates. This matters because
    several claims are cross-model.

From Mizrahi et al. (TACL 2024, doi 10.1162/tacl_a_00681):
  * MaxP (the best template) and AvgP (the mean over templates). They recommend reporting against
    AvgP and not the original template, because the original beats the paraphrase average in 72.5%
    of cases, so single-template numbers are optimistically biased.
  * Sat = 1 - (MaxP - AvgP) and CPS = Sat x MaxP.
  * Kendall's W over models with templates as judges: W = 1 means that template choice is
    irrelevant to the ranking.
  * Divergence: how many standard deviations the original template sits from the multi-template
    mean. More than 1 is their flag.

The headline result is a failure, and a single template tends to flatter a model, so wording bias
makes a negative claim conservative. That argument would not hold for a positive result.

Input: results/questions_templates.jsonl with results/templates_<tag>.json (geometric families),
and optionally results/questions_templates_retrieval.jsonl with
results/templates-retrieval_<tag>.json (retrieval probe and position families).
Output: results/scored_templates.json.

Usage:
  python gauge/score_templates.py
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm
import expect_models

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SPECS = [("results/questions_templates.jsonl", "results/templates_*.json", "geometry"),
         ("results/questions_templates_retrieval.jsonl", "results/templates-retrieval_*.json",
          "absence+position")]
ORDER = ["T0_incumbent", "T1_separator", "T2_joiner_case", "T3_serialisation", "T4_paraphrase"]


def kendall_w(mat):
    """Kendall's W over `mat` [judges x objects]. 1 = judges agree perfectly on the ranking."""
    m, n = mat.shape
    if m < 2 or n < 2:
        return np.nan
    ranks = np.apply_along_axis(lambda r: np.argsort(np.argsort(-r)) + 1.0, 1, mat)
    R = ranks.sum(0)
    S = ((R - R.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def friedman(mat):
    """Friedman statistic over [blocks x treatments]; large chi2 = the treatments differ."""
    b, k = mat.shape
    if b < 2 or k < 2:
        return np.nan
    ranks = np.apply_along_axis(lambda r: np.argsort(np.argsort(r)) + 1.0, 1, mat)
    Rj = ranks.sum(0)
    return float(12.0 / (b * k * (k + 1)) * (Rj ** 2).sum() - 3 * b * (k + 1))


def main():
    out = {}
    for items_path, run_glob, label in SPECS:
        p = os.path.join(ROOT, items_path)
        runs = sorted(glob.glob(os.path.join(ROOT, run_glob)))
        if not os.path.exists(p) or not runs:
            print(f"({label}: no data yet -- {items_path} / {run_glob})")
            continue
        items = [json.loads(l) for l in open(p)]
        print(f"\n{'=' * 96}\n{label.upper()}: {len(items):,} items, {len(runs)} models, "
              f"{len({it['template'] for it in items})} templates")

        acc_cube = {}          # (model, family, template) -> accuracy
        for rp in runs:
            model = os.path.basename(rp).replace(".json", "").split("_", 1)[1]
            raw = json.load(open(rp))["raw"]["full"]
            if len(raw) != len(items):
                print(f"  SKIP {model}: {len(raw)} vs {len(items)}"); continue
            pred = [norm(raw[i], items[i]["family"], items[i].get("atype"))
                    for i in range(len(items))]
            for fam in sorted({it["family"] for it in items}):
                for tmpl in ORDER:
                    ix = [i for i, it in enumerate(items)
                          if it["family"] == fam and it["template"] == tmpl]
                    if not ix:
                        continue
                    fl = collections.Counter(items[i]["a"].lower()
                                             for i in ix).most_common(1)[0][1] / len(ix)
                    acc = float(np.mean([pred[i] == items[i]["a"].lower() for i in ix]))
                    parse = float(np.mean([pred[i] is not None for i in ix]))
                    acc_cube[(model, fam, tmpl)] = dict(
                        acc=acc, gain=(acc - fl) / (1 - fl) if fl < 1 else 0.0,
                        parse_rate=parse, floor=fl, n=len(ix))

        models = sorted({k[0] for k in acc_cube})
        fams = sorted({k[1] for k in acc_cube})
        print(f"\n  {'family':<18}{'model':<12}{'min':>8}{'max':>8}{'SPREAD':>9}"
              f"{'AvgP':>8}{'MaxP':>8}{'CPS':>8}{'incum':>8}{'div_SD':>8}{'parse':>8}")
        for fam in fams:
            for m in models:
                v = [acc_cube[(m, fam, t)]["acc"] for t in ORDER if (m, fam, t) in acc_cube]
                pr = [acc_cube[(m, fam, t)]["parse_rate"] for t in ORDER if (m, fam, t) in acc_cube]
                if len(v) < 2:
                    continue
                mn, mx, avg = min(v), max(v), float(np.mean(v))
                sat = 1 - (mx - avg)
                inc = acc_cube[(m, fam, "T0_incumbent")]["acc"]
                sd = float(np.std(v))
                div = (inc - avg) / sd if sd > 0 else 0.0
                out[f"{label}|{fam}|{m}"] = dict(
                    per_template={t: acc_cube[(m, fam, t)] for t in ORDER
                                  if (m, fam, t) in acc_cube},
                    min=mn, max=mx, spread=mx - mn, AvgP=avg, MaxP=mx,
                    Sat=sat, CPS=sat * mx, incumbent=inc, divergence_sd=div,
                    parse_min=min(pr), parse_max=max(pr))
                flag = "  <<< incumbent >1SD above the mean" if div > 1 else ""
                print(f"  {fam:<18}{m:<12}{mn:>8.3f}{mx:>8.3f}{mx - mn:>9.3f}"
                      f"{avg:>8.3f}{mx:>8.3f}{sat * mx:>8.3f}{inc:>8.3f}{div:>+8.2f}"
                      f"{min(pr):>8.3f}{flag}")

        # ---- does template choice reorder the MODELS? (Sclar's rank-reversal question) ----
        print(f"\n  Kendall's W over models, templates as judges (1 = template choice is "
              f"irrelevant to the ranking):")
        for fam in fams:
            mat = np.array([[acc_cube[(m, fam, t)]["acc"] for m in models]
                            for t in ORDER if all((m, fam, t) in acc_cube for m in models)])
            if mat.shape[0] < 2:
                continue
            W = kendall_w(mat)
            chi2 = friedman(mat)
            # do any two templates disagree about which model is better?
            best = [models[int(np.argmax(r))] for r in mat]
            rev = len(set(best)) > 1
            print(f"    {fam:<18} W={W:.3f}  Friedman chi2={chi2:.1f}  "
                  f"best model per template: {'DISAGREES ' + str(sorted(set(best))) if rev else best[0]}")
            out[f"{label}|{fam}|_W"] = dict(kendall_w=W, friedman_chi2=chi2,
                                            best_model_per_template=best)

        sp = [v["spread"] for k, v in out.items() if k.startswith(label) and "spread" in v]
        if sp:
            print(f"\n  SPREAD across {len(sp)} family x model cells: median {np.median(sp):.3f}, "
                  f"max {max(sp):.3f}")
            print(f"  **LOWER BOUND at K=5** (Sclar's smallest published configuration is 10 and "
                  f"they call that a lower bound too).")
            inc_hi = sum(1 for k, v in out.items()
                         if k.startswith(label) and v.get("divergence_sd", 0) > 1)
            print(f"  cells where the INCUMBENT template sits >1 SD above the multi-template mean: "
                  f"{inc_hi}/{len(sp)}")
            print(f"  (Mizrahi: the original template beats the paraphrase average 72.5% of the "
                  f"time, so report the headline against AvgP, not the incumbent.)")

    dst = os.path.join(ROOT, "results/scored_templates.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
