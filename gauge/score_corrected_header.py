#!/usr/bin/env python3
"""Score the corrected-header rerun and compare it with the reported run, cell by cell.

The coordinate prompts of the reported question set name their columns "day, timeslot, place" while
the rows show x and y. The corrected set (generate_questions.py --only-coord) names them "day,
timeslot, x, y". This script scores the five models on the corrected set and reports what the header
change did to each of the 100 geometric settings (model, family and length). It uses the same
definitions as family_cis.py: baseline = modal-answer share over the cell's golds, gain = (acc -
baseline) / (1 - baseline), and a bootstrap over people. The gold answers are identical between the
two draws and only the prompt header differs, so the baselines must match cell for cell. That is
asserted and not assumed.

Input: results/questions_corrected_header.jsonl, results/family_cis.json and the five runs
results/corrected-header_<tag>.json.
Output: results/scored_corrected_header.json.

Usage:
  python gauge/score_corrected_header.py
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBOOT, MINCL = 2000, 5
GEOM = ["gyration_km", "max_distance", "total_distance", "longest_jump"]


def boot(uids, cor, seed=0):
    """Identical to family_cis.boot -- same estimator, so old and new are on one scale."""
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


def main():
    items_p = f"{ROOT}/results/questions_corrected_header.jsonl"
    old_p = f"{ROOT}/results/family_cis.json"
    for p in (items_p, old_p):
        if not os.path.exists(p):
            sys.exit(f"MISSING: {p}")
    print(f"corrected items : {items_p}")
    print(f"reported scores : {old_p}")

    items = [json.loads(l) for l in open(items_p)]
    old = json.load(open(old_p))
    fams = sorted({it["family"] for it in items})
    print(f"{len(items):,} corrected items | families {fams}")
    if set(fams) - set(GEOM):
        sys.exit(f"corrected draw holds non-geometric families: {set(fams) - set(GEOM)}")

    runs = sorted(glob.glob(f"{ROOT}/results/corrected-header_*.json"))
    runs = [r for r in runs if not r.endswith((".ckpt", ".partial"))]
    if not runs:
        sys.exit("no corrected-header_* runs on disk")

    out, rows, skipped = {}, [], []
    for rp in runs:
        model = os.path.basename(rp).replace("corrected-header_", "").replace(".json", "")
        blob = json.load(open(rp))
        raw = (blob.get("raw") or {}).get("full")
        if raw is None:
            skipped.append((model, "no raw.full")); continue
        if len(raw) != len(items):
            skipped.append((model, f"{len(raw)} generations vs {len(items)} items")); continue
        pred = [norm(raw[i], items[i]["family"], items[i].get("atype")) for i in range(len(items))]
        for fam in fams:
            for span in sorted({it["span"] for it in items if it["family"] == fam}):
                ix = [i for i, it in enumerate(items)
                      if it["family"] == fam and it["span"] == span]
                if len(ix) < 20:
                    continue
                fl = collections.Counter(items[i]["a"] for i in ix).most_common(1)[0][1] / len(ix)
                cor = [pred[i] == items[i]["a"] for i in ix]
                lo, hi, how = boot([items[i]["uid"] for i in ix], cor)
                acc = float(np.mean(cor))
                g = (acc - fl) / (1 - fl) if fl < 1 else 0.0
                ci = [(lo - fl) / (1 - fl) if fl < 1 else 0.0,
                      (hi - fl) / (1 - fl) if fl < 1 else 0.0]
                key = f"{model}|{fam}|{span}"
                prev = old.get(key)
                rec = dict(n=len(ix), floor=fl, acc_new=acc, gain_new=g, gain_ci_new=ci,
                           estimator=how)
                if prev:
                    # Same golds in both draws, so the floors must agree. If they do not, the two
                    # runs are not on one scale and no delta computed here means anything.
                    assert abs(prev["floor"] - fl) < 1e-9, (
                        f"{key}: floor {prev['floor']} (reported) vs {fl} (corrected) -- the draws "
                        "are not comparable")
                    rec.update(gain_old=prev["gain"], gain_ci_old=prev["gain_ci"],
                               d_gain=g - prev["gain"],
                               clears_old=prev["gain_ci"][0] > 0, clears_new=ci[0] > 0)
                    rec["verdict_flip"] = rec["clears_old"] != rec["clears_new"]
                    rows.append(rec)
                out[key] = rec

    if skipped:
        for m, why in skipped:
            print(f"  SKIP {m}: {why}")
    paired = [r for r in rows if "d_gain" in r]
    models = sorted({k.split("|")[0] for k in out})
    print(f"\nmodels scored on the corrected draw : {len(models)} {models}")
    print(f"cells paired with a reported score  : {len(paired)}")
    if not paired:
        sys.exit("no cell could be paired -- nothing to report")

    d = [r["d_gain"] for r in paired]
    flips = [r for r in paired if r["verdict_flip"]]
    print(f"change in normalized gain           : {min(d):+.3f} to {max(d):+.3f} "
          f"(mean {float(np.mean(d)):+.4f})")
    print(f"cells changing their verdict        : {len(flips)} of {len(paired)}")
    for r in flips:
        print(f"    FLIP {r}")
    print(f"cells still clearing the baseline   : {sum(r['clears_new'] for r in paired)}")

    summary = dict(
        items_file=items_p, n_models=len(models), models=models,
        n_cells_paired=len(paired), d_gain_min=min(d), d_gain_max=max(d),
        d_gain_mean=float(np.mean(d)), n_verdict_flips=len(flips),
        n_clearing_new=int(sum(r["clears_new"] for r in paired)),
        n_clearing_old=int(sum(r["clears_old"] for r in paired)),
        skipped=[{"model": m, "why": w} for m, w in skipped], cells=out)
    op = f"{ROOT}/results/scored_corrected_header.json"
    json.dump(summary, open(op, "w"), indent=1)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
