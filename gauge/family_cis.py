#!/usr/bin/env python3
"""Per-family cluster-bootstrap intervals for every model, family and record length.

Each generation is parsed with the shared parser (eval_model.norm). For every (model, family,
span) cell the script computes accuracy, the best-constant baseline (the share of the modal gold
answer), the normalized gain g = (acc - baseline) / (1 - baseline), and a 95% percentile interval
from a bootstrap over people, the unit that questions cluster on. A cell with fewer than 5 distinct
people falls back to resampling items and is labelled as such. Cells with fewer than 20 questions
are skipped. Per (model, family) the script also records whether the family ever clears its
baseline (lower interval bound above zero) and whether it decays (the interval at the longest span
lies entirely below the interval at the shortest).

Input: results/questions.jsonl and the five main runs, results/main_*.json and
results/primary_*.json (from eval_model.py). All five models must be present.
Output: results/family_cis.json, one entry per cell plus a verdict row per model and family.
This file feeds the main results table, the profile figures and several other scripts.

Usage:
  python gauge/family_cis.py        (runs on import; takes no arguments)
"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBOOT, MINCL = 2000, 5


def boot(uids, cor, seed=0):
    uids, cor = np.asarray(uids), np.asarray(cor, float)
    uniq = np.unique(uids)
    rng = np.random.default_rng(seed)
    if len(uniq) >= MINCL:
        idx = {u: np.where(uids == u)[0] for u in uniq}
        m = np.array([cor[np.concatenate([idx[u] for u in
              rng.choice(uniq, len(uniq), True)])].mean() for _ in range(NBOOT)])
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), "uid-cluster"
    m = np.array([cor[rng.integers(0, len(cor), len(cor))].mean() for _ in range(NBOOT)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), "ITEM-fallback"


items = [json.loads(l) for l in open(f"{ROOT}/results/questions.jsonl")]
# Gemma-3-12B's run is named primary_gemma3_12b.json and the other four models' runs are named
# main_<tag>.json. Both patterns are collected, and the model count is asserted below, so a
# missing model fails loudly instead of shrinking the table.
runs = sorted(glob.glob(f"{ROOT}/results/main_*.json")
              + glob.glob(f"{ROOT}/results/primary_*.json"))
if not runs:
    sys.exit("no main runs on disk")
EXPECT = {"phi35mini", "mistral7b", "llama8b", "gemma3_12b", "llama70b"}
out, nfall = {}, 0
print(f"per-family cluster CIs | {len(items):,} items | {len(runs)} models | {NBOOT} resamples\n")
for rp in runs:
    model = os.path.basename(rp).replace("main_", "").replace("primary_", "").replace(".json", "")
    raw = json.load(open(rp))["raw"]["full"]
    if len(raw) != len(items):
        print(f"  SKIP {model}: {len(raw)} vs {len(items)}"); continue
    pred = [norm(raw[i], items[i]["family"], items[i].get("atype")) for i in range(len(items))]
    for fam in sorted({it["family"] for it in items}):
        for span in sorted({it["span"] for it in items if it["family"] == fam}):
            ix = [i for i, it in enumerate(items) if it["family"] == fam and it["span"] == span]
            if len(ix) < 20:
                continue
            fl = collections.Counter(items[i]["a"] for i in ix).most_common(1)[0][1] / len(ix)
            cor = [pred[i] == items[i]["a"] for i in ix]
            lo, hi, how = boot([items[i]["uid"] for i in ix], cor)
            nfall += (how == "ITEM-fallback")
            acc = float(np.mean(cor)); g = (acc - fl) / (1 - fl) if fl < 1 else 0.0
            out[f"{model}|{fam}|{span}"] = dict(
                n=len(ix), floor=fl, acc=acc, gain=g,
                gain_ci=[(lo - fl) / (1 - fl) if fl < 1 else 0.0,
                         (hi - fl) / (1 - fl) if fl < 1 else 0.0],
                acc_ci=[lo, hi], estimator=how,
                n_clusters=int(len(np.unique([items[i]["uid"] for i in ix]))))

_got = {k.split("|")[0] for k in out}
_missing = EXPECT - _got
print(f"cells: {len(out)} | item-fallback cells: {nfall} | models: {len(_got)}/5")
if _missing:
    sys.exit(f"*** MISSING MODELS {sorted(_missing)} -- a five-model claim may not be made from "
             f"a partial table. Copy the missing runs into place or correct the glob.")
print("\nRATE CLAIMS THAT ARE NOW SUPPORTABLE — a family DECAYS only if its gain CI at the longest")
print("span sits entirely below its gain CI at the shortest (non-overlapping intervals).\n")
print(f"  {'model':<12}{'family':<22}{'short':>16}{'long':>16}  verdict")
fams = sorted({k.split('|')[1] for k in out})
for model in sorted({k.split('|')[0] for k in out}):
    for fam in fams:
        sp = sorted(int(k.split('|')[2]) for k in out if k.startswith(f"{model}|{fam}|"))
        if len(sp) < 2:
            continue
        a, b = out[f"{model}|{fam}|{sp[0]}"], out[f"{model}|{fam}|{sp[-1]}"]
        # A family can only decay from a signal it had. Geometry sits below the baseline at every span,
        # so comparing two negative gains and calling the smaller-magnitude one a "rise" would invent an
        # improvement. Check for signal first.
        spans_all = [out[f"{model}|{fam}|{x}"] for x in sp]
        ever = any(c["gain_ci"][0] > 0 for c in spans_all)
        zero_w = any(c["gain_ci"][1] - c["gain_ci"][0] < 1e-9 for c in (a, b))
        if not ever:
            v = "NEVER CLEARS FLOOR (no signal to decay)"
        else:
            dec = b["gain_ci"][1] < a["gain_ci"][0]
            v = ("DECAYS" if dec else
                 "rises" if a["gain_ci"][1] < b["gain_ci"][0] else "flat/overlapping")
        if zero_w:
            v += "  [ZERO-WIDTH CI: accuracy is constant, interval is not informative]"
        out[f"{model}|{fam}|_verdict"] = dict(verdict=v, ever_above_floor=bool(ever),
                                              zero_width=bool(zero_w))
        if v != "flat/overlapping" or fam.startswith(("gyration", "max_", "total_", "longest_")):
            print(f"  {model:<12}{fam:<22}[{a['gain_ci'][0]:+.2f},{a['gain_ci'][1]:+.2f}]"
                  f"  [{b['gain_ci'][0]:+.2f},{b['gain_ci'][1]:+.2f}]  {v}")
json.dump(out, open(f"{ROOT}/results/family_cis.json", "w"), indent=1)
print(f"\nwrote results/family_cis.json  ({len(out)} cells)")
