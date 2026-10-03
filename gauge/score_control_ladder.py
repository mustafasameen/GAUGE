#!/usr/bin/env python3
"""Score the decomposition controls and the scaffold ladder, with cluster-bootstrap intervals.

For every model the script scores two item files with the shared parser (eval_model.norm) and
reports the best-constant baseline, accuracy, the normalized gain over the baseline, and a 95%
interval from a bootstrap over people (several items share a person, so items are not independent):
  controls  results/questions_control.jsonl against results/control_<tag>.json: max_given, sum_given
            and pair_distance
  ladder    results/questions_ladder.jsonl against results/ladder_<tag>.json: the five rungs L0
            to L4, with the jump from each rung to the one above it and a flag when a rung's
            interval excludes the rung below

Input: the item files and runs named above.
Output: results/scored_control_ladder.json (keys control|<family>|<model> and
ladder|<rung>|<model>). It feeds make_ladder_figure.py.

Usage:
  python gauge/score_control_ladder.py
"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm
import expect_models

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBOOT = 2000


def cluster_boot(uids, correct, n=NBOOT, seed=0):
    """CI over USERS, not items: several items share a person and are not independent."""
    uids, correct = np.asarray(uids), np.asarray(correct, float)
    uniq = np.unique(uids)
    if len(uniq) < 3:
        return np.nan, np.nan
    idx = {u: np.where(uids == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    m = np.empty(n)
    for b in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        m[b] = correct[np.concatenate([idx[u] for u in pick])].mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(items, raw, sel):
    ix = [i for i, it in enumerate(items) if sel(it)]
    if not ix:
        return None
    fl = collections.Counter(items[i]["a"].lower() for i in ix).most_common(1)[0][1] / len(ix)
    cor = [norm(raw[i], items[i]["family"], items[i].get("atype")) == items[i]["a"].lower()
           for i in ix]
    acc = float(np.mean(cor))
    lo, hi = cluster_boot([items[i]["uid"] for i in ix], cor)
    g = (acc - fl) / (1 - fl) if fl < 1 else 0.0
    glo = (lo - fl) / (1 - fl) if fl < 1 else 0.0
    ghi = (hi - fl) / (1 - fl) if fl < 1 else 0.0
    return dict(n=len(ix), floor=fl, acc=acc, acc_ci=[lo, hi], gain=g, gain_ci=[glo, ghi])


out = {}

# ---------------- the decomposition controls ----------------
ci = os.path.join(ROOT, "results/questions_control.jsonl")
if os.path.exists(ci):
    items = [json.loads(l) for l in open(ci)]
    print("DECOMPOSITION CONTROLS — gain with cluster-bootstrap CI over users\n")
    print(f"  {'task':<16}{'model':<12}{'floor':>8}{'acc':>8}{'gain':>9}{'gain 95% CI':>20}")
    for rp in sorted(glob.glob(os.path.join(ROOT, "results/control_*.json"))):
        m = os.path.basename(rp).replace("control_", "").replace(".json", "")
        raw = json.load(open(rp))["raw"]["full"]
        if len(raw) != len(items):
            print(f"  SKIP {m}"); continue
        for fam in ["max_given", "sum_given", "pair_distance"]:
            r = cell(items, raw, lambda it, f=fam: it["family"] == f)
            if not r: continue
            out[f"control|{fam}|{m}"] = r
            print(f"  {fam:<16}{m:<12}{r['floor']:>8.3f}{r['acc']:>8.3f}{r['gain']:>+9.3f}"
                  f"   [{r['gain_ci'][0]:>+6.3f},{r['gain_ci'][1]:>+6.3f}]")
    print()

# ---------------- the scaffold ladder ----------------
li = os.path.join(ROOT, "results/questions_ladder.jsonl")
if os.path.exists(li):
    items = [json.loads(l) for l in open(li)]
    RUNGS = ["L0_raw", "L1_km", "L2_centred", "L3_dists", "L4_msd"]
    print("SCAFFOLD LADDER — gain with cluster-bootstrap CI over users\n")
    print(f"  {'rung':<14}{'model':<12}{'acc':>8}{'gain':>9}{'gain 95% CI':>20}")
    for rp in sorted(glob.glob(os.path.join(ROOT, "results/ladder_*.json"))):
        m = os.path.basename(rp).replace("ladder_", "").replace(".json", "")
        raw = json.load(open(rp))["raw"]["full"]
        if len(raw) != len(items):
            print(f"  SKIP {m}"); continue
        prev = None
        for rung in RUNGS:
            r = cell(items, raw, lambda it, g=rung: it["rung"] == g)
            if not r: continue
            out[f"ladder|{rung}|{m}"] = r
            jump = "" if prev is None else f"   jump {r['gain'] - prev:+.3f}"
            sig = ""
            if prev is not None and r["gain_ci"][0] > prev:
                sig = "  <<< CI EXCLUDES THE RUNG BELOW"
            print(f"  {rung:<14}{m:<12}{r['acc']:>8.3f}{r['gain']:>+9.3f}"
                  f"   [{r['gain_ci'][0]:>+6.3f},{r['gain_ci'][1]:>+6.3f}]{jump}{sig}")
            prev = r["gain"]
        print()

dst = os.path.join(ROOT, "results/scored_control_ladder.json")
json.dump(out, open(dst, "w"), indent=1, default=str)
print(f"wrote {dst}  ({len(out)} cells)")
