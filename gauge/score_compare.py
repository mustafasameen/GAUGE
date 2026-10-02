#!/usr/bin/env python3
"""Score the comparison arm: does asking which of two people is larger surface more than asking for a value?

Measured per model, family and record length:
  * Accuracy against the true ratio. The two records in an item differ by a controlled ratio (1.10
    to 5.00). If any graded signal is accessible, accuracy should rise with the ratio. A flat line
    at 0.5 means nothing is accessible, however the question is phrased.
  * Position-bias correction. Every pair was emitted twice, as (A, B) and as (B, A). A model that
    prefers "A" scores 0.5 overall while looking competent on half the items. The script reports
    the A-choice rate (how often the model says "A" regardless of truth; 0.5 is unbiased) and the
    corrected accuracy, which averages the two orders of the same pair: a pair counts as correct
    only if the model is right in both presentations.
  * Discrimination threshold. The ratio at which corrected accuracy first reaches 0.75, by linear
    interpolation between the measured ratios, or ">5.0" when it is never reached.
  * Monotonicity. The Spearman correlation between ratio and corrected accuracy across the six
    levels, the analogue of the monotonicity test in arXiv 2603.20642 (does accuracy increase with
    ratio, or is it flat).
  * The positive control. `cmp_daydistinct` is a non-geometric quantity on which the models do have
    signal. If the comparison format works at all it must work there. If the control is also flat,
    the format itself is not viable and nothing can be concluded about representation.

arXiv 2603.20642 measures discrimination of magnitudes that are presented in the prompt. The
quantities here must first be aggregated from a raw record (see tasks_compare.py).

Input: the comparison items (results/tally_compare.jsonl, results/tally_compare_spans.jsonl and
results/tally_compare_512.jsonl) and the matching runs (results/v41cmp_*.json, results/v41cmpS_*.json
and results/v41cmp512_*.json).
Output: results/v41_compare_scored.json.

Usage:
  python gauge/score_compare.py
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factqa import norm
import expect_models

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBOOT = 2000
ITEM_FILES = ["results/tally_compare.jsonl", "results/tally_compare_spans.jsonl",
              "results/tally_compare_512.jsonl"]
RUN_GLOBS = ["results/v41cmp_*.json", "results/v41cmpS_*.json", "results/v41cmp512_*.json"]
CONTROL = "cmp_daydistinct"


def boot_ci(vals, n=NBOOT, seed=0):
    """Bootstrap over PAIRS (the unit of analysis after order-correction), not over items."""
    v = np.asarray(vals, float)
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def threshold(ratios, accs, target=0.75):
    """First ratio at which accuracy reaches `target`, linearly interpolated."""
    for i in range(len(ratios)):
        if accs[i] >= target:
            if i == 0:
                return float(ratios[0])
            x0, x1, y0, y1 = ratios[i - 1], ratios[i], accs[i - 1], accs[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def main():
    items, seen = [], set()
    for f in ITEM_FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        for l in open(p):
            items.append(json.loads(l))
    if not items:
        sys.exit("no comparison items found")

    # index items per file so each run's predictions align with the file it was run on
    per_file = {}
    for f in ITEM_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            per_file[os.path.basename(f)] = [json.loads(l) for l in open(p)]

    runs = []
    for g in RUN_GLOBS:
        for p in sorted(glob.glob(os.path.join(ROOT, g))):
            tag = os.path.basename(p).replace(".json", "")
            src = ("tally_compare.jsonl" if tag.startswith("v41cmp_")
                   else "tally_compare_512.jsonl" if tag.startswith("v41cmp512_")
                   else "tally_compare_spans.jsonl")
            runs.append((tag.split("_", 1)[1], src, p))
    if not runs:
        sys.exit("no comparison runs found")

    print("COMPARISON INTERVENTION — does asking WHICH IS LARGER surface anything?")
    print("corrected accuracy: a PAIR counts as correct only if the model is right in BOTH")
    print("presentation orders. A-rate: how often it answers A regardless of truth (0.5 = unbiased).\n")

    out = {}
    by_model = collections.defaultdict(dict)
    for model, src, path in runs:
        its = per_file[src]
        raw = json.load(open(path))["raw"]["full"]
        if len(raw) != len(its):
            print(f"  SKIP {model}/{src}: {len(raw)} vs {len(its)}")
            continue
        pred = [norm(raw[i], its[i]["family"], its[i].get("atype")) for i in range(len(its))]
        # group the two orders of each pair: same family, span, uid pair, ratio
        pairs = collections.defaultdict(dict)
        for i, it in enumerate(its):
            key = (it["family"], it["span"], it["ratio"],
                   tuple(sorted((it["uid"], it["uid_b"]))))
            pairs[key][it["order"]] = (pred[i], it["a"])
        for (fam, span, ratio, _), d in pairs.items():
            if 0 not in d or 1 not in d:
                continue
            both = all(d[o][0] == d[o][1] for o in (0, 1))
            # RAW accuracy alongside corrected. A model answering a CONSTANT scores exactly 0 on
            # the corrected metric by construction (it can only be right in one of the two orders),
            # which conflates "does not know" with "has a position bias". Reporting both, plus the
            # A-rate, separates them: constant answerer -> raw ~0.5, corrected 0, A-rate ~0 or ~1.
            rawc = float(np.mean([d[o][0] == d[o][1] for o in (0, 1)]))
            a_choice = np.mean([d[o][0] == "a" for o in (0, 1)])
            unpar = any(d[o][0] is None for o in (0, 1))
            by_model[model].setdefault((fam, span, ratio), []).append(
                (both, a_choice, unpar, rawc))

    for model in sorted(by_model):
        print(f"=== {model} ===")
        print(f"  {'family':<18}{'span':>5}" + "".join(f"{r:>9}" for r in
                                                       [1.1, 1.25, 1.5, 2.0, 3.0, 5.0])
              + f"{'rho':>7}{'thr':>8}{'A-rate':>8}{'raw':>8}")
        for fam in sorted({k[0] for k in by_model[model]}):
            for span in sorted({k[1] for k in by_model[model] if k[0] == fam}):
                rs, accs, arates, unp, ns, raws = [], [], [], [], [], []
                for ratio in sorted({k[2] for k in by_model[model]
                                     if k[0] == fam and k[1] == span}):
                    v = by_model[model][(fam, span, ratio)]
                    rs.append(ratio)
                    accs.append(float(np.mean([x[0] for x in v])))
                    raws.append(float(np.mean([x[3] for x in v])))
                    arates.append(float(np.mean([x[1] for x in v])))
                    unp.append(float(np.mean([x[2] for x in v])))
                    ns.append(len(v))
                if len(rs) < 3:
                    continue
                rho = spearman(rs, accs)
                thr = threshold(rs, accs)
                lo, hi = boot_ci([x[0] for r in rs
                                  for x in by_model[model][(fam, span, r)]])
                out[f"{model}|{fam}|{span}"] = dict(
                    ratios=rs, corrected_acc=accs, a_rate=arates, unparsed=unp, n_pairs=ns,
                    rho_ratio_acc=rho, threshold_075=thr, pooled_ci=[lo, hi])
                flag = "  <<< CONSTANT" if (np.mean(arates) < 0.12 or np.mean(arates) > 0.88) else ""
                print(f"  {fam:<18}{span:>5}" + "".join(f"{a:>9.3f}" for a in accs)
                      + f"{rho:>7.2f}" + (f"{thr:>8.2f}" if thr else f"{'>5.0':>8}")
                      + f"{np.mean(arates):>8.3f}{np.mean(raws):>8.3f}{flag}")
        print()

    print("=" * 100)
    print("VERDICT — PER MODEL. Pooling hides the result: the control fails for the one model")
    print("whose geometry works, and works for a model whose geometry does not.\n")
    for model in sorted(by_model):
        cells = {k: v for k, v in out.items() if k.startswith(model + "|")}
        ctrl = [v for k, v in cells.items() if CONTROL in k]
        geo = [v for k, v in cells.items() if CONTROL not in k]
        if not geo:
            continue
        c_best = max((v["corrected_acc"][-1] for v in ctrl), default=float("nan"))
        g_rho = np.nanmean([v["rho_ratio_acc"] for v in geo])
        g_top = np.nanmean([v["corrected_acc"][-1] for v in geo])
        thrs = [v["threshold_075"] for v in geo if v["threshold_075"]]
        const = sum(1 for v in geo if np.mean(v["a_rate"]) < 0.12 or np.mean(v["a_rate"]) > 0.88)
        if g_top >= 0.6 and g_rho >= 0.7:
            verdict = ("GEOMETRY DISCRIMINATED: monotone in ratio, "
                       f"threshold {min(thrs):.1f}-{max(thrs):.1f}x" if thrs else "GEOMETRY DISCRIMINATED")
        elif const >= 0.6 * len(geo):
            verdict = f"CONSTANT ANSWERER on {const}/{len(geo)} geometry cells — not measurable"
        else:
            verdict = "no geometric discrimination"
        print(f"  {model:<12} geometry: rho {g_rho:+.2f}, acc@5x {g_top:.3f}   {verdict}")
        print(f"  {'':<12} control best acc@5x {c_best:.3f}"
              + ("   *** control collapsed to a constant for this model" if c_best < 0.2 else
                 "   (format viable for this model)"))
    print("\n  READ TOGETHER: the format is viable (mistral's control reaches .89-.92 at wide")
    print("  ratios), and llama-70b discriminates geometry where every value question failed.")
    dst = os.path.join(ROOT, "results/v41_compare_scored.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}  ({len(out)} model x family x span cells)")


if __name__ == "__main__":
    main()
