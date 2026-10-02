#!/usr/bin/env python3
"""Cluster-bootstrap checks of four claims about the recovery profile, each with the confound that would most plausibly manufacture it.

C-A  Exact recovery and rank-preserving representation come apart. Geometric descriptors have rho
     near 0 at every record length, while count and duration descriptors keep a high rho long after
     exact match collapses. Confound: rho is attenuated when the gold has little spread, so
     long-record cells with narrower gold distributions would show a declining rho that comes from
     the target and not from the model. Check: gold SD and the number of distinct gold values per
     cell, computed without any model.
C-B  Record length versus required state. At a fixed required state W, accuracy and rho fall with
     record length S, while at fixed S they do not fall with W. Confound: to answer "the last W
     records" the model must first locate the tail, so if tail location itself fails at long S, the
     W-invariance is uninformative. Check: retrieve_p90, a pure locate task near the end of the
     record, at the same lengths.
C-C  Position by length. The spread across answer positions is small at 32 rows and larger at 512.
     Position is assigned by construction and every position is a separate family, so the p10
     versus p70 contrast is reported as a marginal contrast with an interval.
C-D  Hallucination under absence. The rate of correctly answering "none" to an unanswerable query
     falls with record length. Confound: a model that never says "none" would score this way
     trivially. Check: the false-abstention rate (saying "none" when the answer is present). If it
     stays near 0, the model is not abstention-averse in general; it fails to detect absence in a
     long record.

The detailed C-A to C-D tables are computed on the first run listed in --runs (the Gemma-3-12B run
by default). The per-model spread of the headline quantities, for all runs, is reported in the
final section.

Input: results/tally_v41.jsonl and the core runs (results/v41pilot_gemma3_12b.json and
results/v41core_*.json).
Output: results/v41_claim_verification.json.

Usage:
  python gauge/verify_v41_claims.py [--items ...] [--runs ...] [--out ...]
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factqa import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBOOT = 2000

# Multi-run by default: every run matching --runs is scored separately, and the per-model spread
# is reported beside the primary run, so that a single model is never silently reported as more.
DEFAULT_RUNS = "results/v41pilot_gemma3_12b.json results/v41core_*.json"


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


def boot_stat(uids, fn, n=NBOOT, seed=0):
    """Cluster bootstrap over uid of an arbitrary statistic fn(idx)->float."""
    uids = np.asarray(uids)
    uniq = np.unique(uids)
    idx_by_u = {u: np.where(uids == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_u[u] for u in pick])
        v = fn(sel)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_runs(patterns, items):
    """Score every matching run. Returns {tag: pred_list} in a stable order."""
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not hits and os.path.exists(os.path.join(ROOT, pat)):
            hits = [os.path.join(ROOT, pat)]
        paths.extend(hits)
    seen, preds = set(), {}
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        tag = os.path.basename(p).replace(".json", "").replace("v41core_", "").replace(
            "v41pilot_", "")
        d = json.load(open(p))
        raw = d["raw"]["full"]
        if len(raw) != len(items):
            print(f"  SKIP {tag}: {len(raw)} generations vs {len(items)} items")
            continue
        preds[tag] = [norm(raw[i], it["family"], it.get("atype")) for i, it in enumerate(items)]
        st = os.stat(p)
        print(f"  {tag:<14} {os.path.basename(p):<34} {st.st_size:>10,}B  mtime {st.st_mtime:.0f}")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="results/tally_v41.jsonl")
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS.split())
    ap.add_argument("--out", default="results/v41_claim_verification.json")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.join(ROOT, a.items))]
    gold = [it["a"].lower() for it in items]
    print("=== RUNS SCORED ===")
    all_preds = load_runs(a.runs, items)
    if not all_preds:
        sys.exit("no runs matched --runs; refusing to emit verdicts")
    tags = list(all_preds)
    print(f"\nitems {len(items):,} | {len(tags)} model(s): {', '.join(tags)}"
          f" | cluster bootstrap over uid, {NBOOT} reps")
    if len(tags) == 1:
        print("*** ONE MODEL. Every number below is single-model and must be said so. ***")
    print()
    # The detailed C-A..C-D tables below are computed on the FIRST run for readability; the
    # per-model spread for every headline quantity is reported in the SPREAD section at the end,
    # which is what a multi-model claim must cite.
    pred = all_preds[tags[0]]
    out = {"models": tags, "primary": tags[0]}

    def cell(fam, span):
        return [i for i, it in enumerate(items) if it["family"] == fam and it["span"] == span]

    def parsed(ix):
        pv, gv, uu = [], [], []
        for i in ix:
            try:
                a, b = float(pred[i]), float(gold[i])
            except (TypeError, ValueError):
                continue
            pv.append(a); gv.append(b); uu.append(items[i]["uid"])
        return np.array(pv), np.array(gv), np.array(uu)

    # ---------------- C-A ----------------
    print("=== C-A  rho with CI, and the gold-spread confound (model-free) ===")
    print(f"  {'family':<22}{'S':>5}{'exact':>8}{'rho':>8}{'rho 95% CI':>18}"
          f"{'gold SD':>9}{'gold uniq':>10}")
    ca = {}
    for fam in ["gyration_km", "total_distance", "longest_jump", "max_distance",
                "day_distinct", "core_set", "longest_stay"]:
        spans = sorted({it["span"] for it in items if it["family"] == fam})
        for s in spans:
            ix = cell(fam, s)
            pv, gv, uu = parsed(ix)
            if len(pv) < 20:
                continue
            rho = spearman(pv, gv)
            lo, hi = boot_stat(uu, lambda sel: spearman(pv[sel], gv[sel]))
            ex = float(np.mean([pred[i] == gold[i] for i in ix]))
            ca[f"{fam}@{s}"] = dict(exact=ex, rho=rho, lo=lo, hi=hi,
                                    gold_sd=float(gv.std()), gold_uniq=int(len(set(gv))))
            print(f"  {fam:<22}{s:>5}{ex:>8.3f}{rho:>8.3f}   [{lo:>6.3f},{hi:>6.3f}]"
                  f"{gv.std():>9.2f}{len(set(gv)):>10}")
    out["C_A"] = ca
    geo = [v["rho"] for k, v in ca.items() if k.split("@")[0] in
           ("gyration_km", "total_distance", "longest_jump", "max_distance")]
    cnt = [v["rho"] for k, v in ca.items() if k.split("@")[0] in
           ("day_distinct", "core_set", "longest_stay")]
    print(f"\n  geometric families   rho: median {np.median(geo):.3f}  max {max(geo):.3f}")
    print(f"  count/duration       rho: median {np.median(cnt):.3f}  min {min(cnt):.3f}")
    print("  CONFOUND CHECK: if gold SD/uniqueness were driving rho, low-rho cells would have")
    print("  narrow golds. Compare the gold SD columns above between the two groups.")

    # ---------------- C-B ----------------
    print("\n=== C-B  W x S on the rank measure, plus the tail-location control ===")
    print(f"  {'W':>5}{'S':>6}{'exact':>8}{'rho':>8}{'rho 95% CI':>18}")
    cb = {}
    for W in (16, 32, 64, 128):
        fam = f"window_distinct_w{W}"
        spans = sorted({it["span"] for it in items if it["family"] == fam})
        for s in spans:
            ix = cell(fam, s)
            pv, gv, uu = parsed(ix)
            if len(pv) < 20:
                continue
            rho = spearman(pv, gv)
            lo, hi = boot_stat(uu, lambda sel: spearman(pv[sel], gv[sel]))
            cb[f"W{W}_S{s}"] = dict(W=W, S=s, rho=rho, lo=lo, hi=hi,
                                    exact=float(np.mean([pred[i] == gold[i] for i in ix])))
            print(f"  {W:>5}{s:>6}{cb[f'W{W}_S{s}']['exact']:>8.3f}{rho:>8.3f}   [{lo:>6.3f},{hi:>6.3f}]")
    print("\n  at FIXED W, rho across S:")
    for W in (16, 32, 64, 128):
        row = [(v["S"], v["rho"]) for v in cb.values() if v["W"] == W]
        if len(row) >= 2:
            row.sort()
            print(f"    W={W:<4} " + "  ".join(f"S{s}={r:.3f}" for s, r in row)
                  + f"   delta={row[-1][1]-row[0][1]:+.3f}")
    print("  at FIXED S, rho across W:")
    for S in (128, 256, 512):
        row = [(v["W"], v["rho"]) for v in cb.values() if v["S"] == S]
        if len(row) >= 2:
            row.sort()
            print(f"    S={S:<4} " + "  ".join(f"W{w}={r:.3f}" for w, r in row)
                  + f"   delta={row[-1][1]-row[0][1]:+.3f}")
    print("\n  TAIL-LOCATION CONTROL (retrieve_p90: locate a record near the record's end):")
    for s in (32, 64, 128, 256, 512):
        ix = cell("retrieve_p90", s)
        if ix:
            print(f"    S={s:<4} retrieve_p90 exact = {np.mean([pred[i]==gold[i] for i in ix]):.3f}")
    out["C_B"] = cb

    # ---------------- C-C ----------------
    print("\n=== C-C  position spread by span, with CI on the p10 - p70 contrast ===")
    cc = {}
    for s in (32, 64, 128, 256, 512):
        accs = {}
        for p in (10, 30, 50, 70, 90):
            ix = cell(f"retrieve_p{p}", s)
            accs[p] = float(np.mean([pred[i] == gold[i] for i in ix])) if ix else np.nan
        # p10 vs p70 are DIFFERENT items (separate families), so this is a marginal contrast:
        # bootstrap each arm independently and difference the replicates.
        ix10, ix70 = cell(f"retrieve_p10", s), cell(f"retrieve_p70", s)
        c10 = np.array([pred[i] == gold[i] for i in ix10], float)
        c70 = np.array([pred[i] == gold[i] for i in ix70], float)
        rng = np.random.default_rng(0)
        d = np.array([c10[rng.integers(0, len(c10), len(c10))].mean()
                      - c70[rng.integers(0, len(c70), len(c70))].mean() for _ in range(NBOOT)])
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        cc[s] = dict(accs=accs, spread=max(accs.values()) - min(accs.values()),
                     p10_minus_p70=accs[10] - accs[70], lo=lo, hi=hi)
        print(f"  S={s:<5} spread={cc[s]['spread']:.3f}   p10-p70 = {accs[10]-accs[70]:+.3f}"
              f"  95% CI [{lo:+.3f}, {hi:+.3f}]   {'SIGNIF' if lo > 0 else 'ns'}")
    out["C_C"] = cc

    # ---------------- C-D ----------------
    print("\n=== C-D  hallucination under absence, with the false-abstention control ===")
    print(f"  {'S':>5}{'says none | absent':>20}{'95% CI':>20}{'says none | present':>21}")
    cd = {}
    for s in (16, 32, 64, 128, 256, 512):
        ix = cell("retrieve_probe", s)
        una = [i for i in ix if gold[i] == "none"]
        ans = [i for i in ix if gold[i] != "none"]
        if not una or not ans:
            continue
        v = np.array([pred[i] == "none" for i in una], float)
        u = np.array([items[i]["uid"] for i in una])
        lo, hi = boot_stat(u, lambda sel: v[sel].mean())
        fa = float(np.mean([pred[i] == "none" for i in ans]))
        cd[s] = dict(none_when_absent=float(v.mean()), lo=lo, hi=hi, false_abstention=fa,
                     n_unans=len(una), n_ans=len(ans))
        print(f"  {s:>5}{v.mean():>20.3f}   [{lo:>6.3f},{hi:>6.3f}]{fa:>21.3f}")
    print("\n  If false abstention stays ~0 while 'none|absent' collapses, the model is not")
    print("  abstention-averse in general -- it fails to DETECT absence in a long record.")
    out["C_D"] = cd

    # ---------------- PER-MODEL SPREAD on every headline quantity ----------------
    # This is the section a multi-model claim cites. Anything reported as a finding must be shown
    # here across models, not read off the primary run alone.
    print("\n=== PER-MODEL SPREAD (the table a multi-model claim must cite) ===")
    spread = {}

    def per_model(name, fn):
        vals = {t: fn(all_preds[t]) for t in tags}
        fin = [v for v in vals.values() if v is not None and np.isfinite(v)]
        spread[name] = dict(per_model=vals,
                            min=float(min(fin)) if fin else None,
                            max=float(max(fin)) if fin else None,
                            range=float(max(fin) - min(fin)) if len(fin) > 1 else None)
        cells = "  ".join(f"{t}={vals[t]:.3f}" if vals[t] is not None and np.isfinite(vals[t])
                          else f"{t}=n/a" for t in tags)
        rng = spread[name]["range"]
        print(f"  {name:<34} {cells}" + (f"   range={rng:.3f}" if rng is not None else ""))

    def acc_cell(fam, span):
        ix = cell(fam, span)
        return lambda pv: float(np.mean([pv[i] == gold[i] for i in ix])) if ix else None

    def rho_cell(fam, span):
        ix = cell(fam, span)
        def f(pv):
            p, g = [], []
            for i in ix:
                try:
                    x, y = float(pv[i]), float(gold[i])
                except (TypeError, ValueError):
                    continue
                p.append(x); g.append(y)
            return spearman(p, g) if len(p) >= 20 else None
        return f

    def none_rate(span):
        ix = [i for i in cell("retrieve_probe", span) if gold[i] == "none"]
        return lambda pv: float(np.mean([pv[i] == "none" for i in ix])) if ix else None

    per_model("absence detect @16", none_rate(16))
    per_model("absence detect @512", none_rate(512))
    per_model("retrieve p10 @512", acc_cell("retrieve_p10", 512))
    per_model("retrieve p70 @512", acc_cell("retrieve_p70", 512))
    per_model("rho gyration_km @32", rho_cell("gyration_km", 32))
    per_model("rho gyration_km @512", rho_cell("gyration_km", 512))
    per_model("rho longest_stay @512", rho_cell("longest_stay", 512))
    per_model("rho day_distinct @16", rho_cell("day_distinct", 16))
    per_model("rho window_w16 @32", rho_cell("window_distinct_w16", 32))
    per_model("rho window_w16 @512", rho_cell("window_distinct_w16", 512))
    out["spread"] = spread
    if len(tags) == 1:
        print("\n  (one model: 'range' is undefined by construction, not small)")

    dst = os.path.join(ROOT, a.out)
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
