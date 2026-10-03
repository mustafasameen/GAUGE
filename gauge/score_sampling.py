#!/usr/bin/env python3
"""Score the sample-and-aggregate arm against the greedy baseline.

Lukasik et al. (2024, arXiv 2403.04182) argue that greedy decoding implicitly optimises exact
match, and propose regression-aware inference: draw k samples and reduce them, which targets
expected error instead of the mode. This script scores that fix on the geometric families and the
distinct-places control, against the identical items under greedy decoding.

The comparison is valid because the greedy numbers are taken from the existing five-model runs,
restricted to the sampled subset by item identity (the prompt), so that
  items         identical (the 4,050-item subset)
  prompt/style  identical (terse)
  max_new       identical (24)
  decoder       greedy versus k = 8 samples reduced: the only difference
A separate greedy run on the subset would add a second differing factor, a fresh GPU run, for no
gain.

Three reductions are scored, because which reduction is itself a result: median (robust to a wild
draw), mean (minimises squared error, the framing of Lukasik et al.) and mode (still targets exact
match, so it separates "sampling helped" from "aggregation helped").

Every score carries its baseline. As in tolerance_gate.py, each tolerance gets the best-constant
predictor on the same cell, and gain is reported against it. A raw accuracy that rises under a
looser metric is not evidence unless it rises faster than a constant does.

Input: results/questions.jsonl, results/questions_sampling.jsonl, the greedy main runs for
gemma3_12b and llama70b, and the k = 8 sample checkpoints results/sampling_<tag>.json.full.ckpt.
Output: results/scored_sampling.json.

Usage:
  python gauge/score_sampling.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm
import expect_models

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GREEDY = {"gemma3_12b": "results/primary_gemma3_12b.json",
          "llama70b": "results/main_llama70b.json"}
MIT = {"gemma3_12b": "results/sampling_gemma3_12b.json.full.ckpt",
       "llama70b": "results/sampling_llama70b.json.full.ckpt"}
GEOM = {"gyration_km", "max_distance", "total_distance", "longest_jump"}
TOLS = [("exact", lambda p, g: p == g),
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


def best_constant(gv, fn):
    cands = np.unique(gv)
    if len(cands) > 400:
        cands = np.quantile(gv, np.linspace(0, 1, 400))
    return max((float(np.mean([fn(c, g) for g in gv])) for c in cands), default=0.0)


def reduce_draws(draws, family, atype, how):
    """Parse k raw draws to numbers and reduce. Returns None if nothing parses."""
    vals = []
    for d in draws or []:
        p = norm(d, family, atype)
        try:
            vals.append(float(p))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    if how == "median":
        return float(np.median(vals))
    if how == "mean":
        return float(np.mean(vals))
    return float(collections.Counter(vals).most_common(1)[0][0])   # mode


def main():
    full = [json.loads(l) for l in open(os.path.join(ROOT, "results/questions.jsonl"))]
    sub = [json.loads(l) for l in open(os.path.join(ROOT, "results/questions_sampling.jsonl"))]
    # Map the subset items back into the full run by identity, so that greedy predictions can be
    # pulled without re-running the GPU. Match on the prompt, not on (family, span, uid, q): that key
    # collides, because one person can draw two items with the same question text in the same cell, and
    # matching on it would pull the wrong row's greedy prediction for some subset items. The prompt is
    # unique by construction (the rendered record differs per item). The gold-agreement assertion below
    # is a permanent guard against a wrong match.
    key = lambda it: it["prompt_full"]
    pos = {}
    for i, it in enumerate(full):
        pos.setdefault(key(it), i)
    missing = [k for k in map(key, sub) if k not in pos]
    if missing:
        sys.exit(f"{len(missing)} subsample items not found in the full items file; refusing")
    bad = [j for j, it in enumerate(sub) if full[pos[key(it)]]["a"] != it["a"]]
    if bad:
        sys.exit(f"MATCH IS WRONG: {len(bad)} items whose gold disagrees after matching; refusing")
    print(f"matched all {len(sub):,} subsample items by prompt identity; gold agrees on every one\n")

    out = {}
    for tag in MIT:
        mp = os.path.join(ROOT, MIT[tag])
        if not os.path.exists(mp):
            print(f"  (missing {MIT[tag]} -- cell not finished)")
            continue
        d = json.load(open(mp))
        samples = (d.get("samples") or {}).get("full")
        if samples is None:
            print(f"  ({tag}: no samples in checkpoint)"); continue
        k = d.get("n_samples")
        graw = json.load(open(os.path.join(ROOT, GREEDY[tag])))["raw"]["full"]
        print(f"=== {tag} | k={k} draws @ T={d.get('temperature')} vs greedy ===")
        print(f"  {'family':<16}{'metric':<8}{'greedy':>16}{'median':>16}{'mean':>16}{'mode':>16}")
        print(f"  {'':<24}" + "".join(f"{'score / gain':>16}" for _ in range(4)))

        for fam in sorted({it["family"] for it in sub}):
            idx = [j for j, it in enumerate(sub) if it["family"] == fam]
            gv, gy, red = [], [], {h: [] for h in ("median", "mean", "mode")}
            for j in idx:
                it = sub[j]
                try:
                    g = float(it["a"])
                except (TypeError, ValueError):
                    continue
                pg = norm(graw[pos[key(it)]], fam, it.get("atype"))
                try:
                    pg = float(pg)
                except (TypeError, ValueError):
                    pg = np.nan
                rs = {h: reduce_draws(samples[j], fam, it.get("atype"), h)
                      for h in red}
                if any(v is None for v in rs.values()):
                    continue
                gy.append(g); gv.append(pg)
                for h in red:
                    red[h].append(rs[h])
            if len(gy) < 30:
                continue
            gy = np.array(gy); gv = np.array(gv)
            rec = {"n": len(gy), "rho_greedy": spearman(gv[~np.isnan(gv)], gy[~np.isnan(gv)])}
            for tl, fn in TOLS:
                fl = best_constant(gy, fn)
                cells = []
                for name, vec in (("greedy", gv), ("median", np.array(red["median"])),
                                  ("mean", np.array(red["mean"])), ("mode", np.array(red["mode"]))):
                    ok = ~np.isnan(vec)
                    sc = float(np.mean([fn(a, b) for a, b in zip(vec[ok], gy[ok])])) if ok.any() else np.nan
                    gn = (sc - fl) / (1 - fl) if fl < 1 else 0.0
                    rec[f"{tl}_{name}"] = dict(score=sc, gain=gn)
                    cells.append(f"{sc:>7.3f} /{gn:>+7.3f}")
                print(f"  {fam:<16}{tl:<8}" + "".join(cells))
            for h in red:
                rec[f"rho_{h}"] = spearman(np.array(red[h]), gy)
            print(f"  {fam:<16}{'rho':<8}{rec['rho_greedy']:>16.3f}"
                  f"{rec['rho_median']:>16.3f}{rec['rho_mean']:>16.3f}{rec['rho_mode']:>16.3f}")
            out[f"{tag}|{fam}"] = rec
        print()

    if not out:
        print("nothing scored yet"); return
    print("=" * 78)
    print("VERDICT. The sampling arm SUCCEEDS on a family only if a reduction lifts gain above 0.05")
    print("where greedy was at or below it. Mode is the control: if mode helps as much as median,")
    print("the gain came from sampling variance, not from regression-aware reduction.")
    for k_, v in out.items():
        tag, fam = k_.split("|")
        if fam not in GEOM:
            continue
        g0 = v["rel25_greedy"]["gain"]
        best = max((v[f"rel25_{h}"]["gain"], h) for h in ("median", "mean", "mode"))
        verdict = ("RESCUED by " + best[1]) if (best[0] > 0.05 >= g0) else "not rescued"
        print(f"  {tag:<12}{fam:<16} greedy gain {g0:+.3f} -> best {best[0]:+.3f} ({best[1]})   {verdict}")
    dst = os.path.join(ROOT, "results/scored_sampling.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
