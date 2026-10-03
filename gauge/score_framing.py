#!/usr/bin/env python3
"""Score the framing arm: the same geometric questions with and without mobility language, paired.

make_framing.py built one item per row that carries both `prompt_full` (mobility framing: "Here
is a person's location record") and `prompt_nomobility` (neutral framing: "Here is a set of
numbered points"), with the same coordinates, gold answer and row order, for the four geometric
families. One eval_model.py job run with `--conds full,nomobility` over that item file writes one
results JSON per model, results/framing_<tag>.json, holding `raw["full"]` and `raw["nomobility"]`:
two generation lists in item order. The golds and the coordinates never move, only the entity-framing
clause does, so the two conditions are comparable cell for cell. The expected outcome is a null:
framing should not move accuracy. Scoring a null seriously means reporting the paired delta with a
real interval, the discordant items in each direction (the quantity that carries evidence near the
baseline), and the per-condition unparsed rate (so that a parsing artifact cannot pass for a framing
effect), not just two accuracy numbers that look close.

Baseline, gain and bootstrap use the same definitions as family_cis.py and
score_corrected_header.py: baseline = modal-answer rate over a cell's golds, gain = (acc - baseline)
/ (1 - baseline). `boot()` is copied from those scripts with the same NBOOT = 2000, MINCL = 5 and
cluster-with-item-fallback shape. Only the second parameter's name is generalised, because this
script also bootstraps a signed per-item delta; the arithmetic is the same, a mean over resampled
clusters.

The baseline-equality check of score_corrected_header.py is replaced by an alignment check. Here
`full` and `nomobility` are two columns of one row of one file, so there is only one baseline to
compute. The meaningful check for this file shape is that the harness wrote both lists with the same
length, positionally aligned to the same item file. That is asserted explicitly in `resolve_items`,
in the length check and in the `items_full is items_relabelled` identity check.

Input: results/questions_framing.jsonl (or the 40-item results/questions_framing_smoke.jsonl) and
results/framing_<tag>.json.
Output: results/scored_framing.json.

Usage:
  python gauge/score_framing.py              scores the runs on disk
  python gauge/score_framing.py --selftest   synthetic known-answer checks; reads no files
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
NBOOT, MINCL = 2000, 5             # identical to score_corrected_header.py / family_cis.py
GEOM = ["gyration_km", "max_distance", "total_distance", "longest_jump"]


def boot(uids, values, seed=0):
    """Cluster bootstrap over uid, with an item-level fallback when a slice has too few distinct
    clusters (the estimator string in the return value says which was used).

    Copied from score_corrected_header.py and family_cis.py with the same NBOOT and MINCL and the
    same resampling, so every interval here sits on the same scale as the rest of the repository.
    `values` is whatever per-item numeric array the caller wants the cluster-bootstrapped mean of:
    plain correctness (0/1) for an accuracy interval, or a paired delta (-1/0/+1) for a paired
    interval. The function only resamples clusters and averages.
    """
    uids, values = np.asarray(uids), np.asarray(values, float)
    uniq = np.unique(uids)
    rng = np.random.default_rng(seed)
    if len(uniq) >= MINCL:
        idx = {u: np.where(uids == u)[0] for u in uniq}
        m = np.array([values[np.concatenate([idx[u] for u in
                      rng.choice(uniq, len(uniq), True)])].mean() for _ in range(NBOOT)])
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), "uid-cluster"
    m = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(NBOOT)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), "ITEM-fallback"


def predict(raw, items, ix):
    """norm() every generation in this slice and compare to gold. Returns (cor, parsed), lists
    aligned to `ix`. An unparsed generation (norm() -> None) is scored WRONG, never dropped --
    eval_model.py's own convention -- and `parsed` records where that happened so the unparsed
    rate can be reported per condition instead of silently folding into accuracy."""
    preds = [norm(raw[i], items[i]["family"], items[i].get("atype")) for i in ix]
    cor = [preds[k] == items[i]["a"] for k, i in enumerate(ix)]
    parsed = [p is not None for p in preds]
    return cor, parsed


def summarize(fl, cor_full, cor_relabelled, parsed_full, parsed_relabelled, uids, seed=0):
    """The one place floor/gain/paired-delta/discordant/unparsed are computed. Called once per
    (model, family, span) cell, once more per model on the pooled arrays, AND directly by
    --selftest on synthetic arrays -- the self-test exercises this real function, not a parallel
    reimplementation of its arithmetic."""
    cor_full = np.asarray(cor_full, dtype=float)
    cor_relabelled = np.asarray(cor_relabelled, dtype=float)
    n = len(cor_full)
    assert n == len(cor_relabelled) == len(parsed_full) == len(parsed_relabelled) == len(uids), \
        "summarize(): misaligned arrays passed in -- caller bug, not a data problem"
    assert n > 0, "summarize(): empty cell"
    acc_full, acc_relabelled = float(cor_full.mean()), float(cor_relabelled.mean())
    gain = lambda acc: (acc - fl) / (1 - fl) if fl < 1 else 0.0
    d = cor_full - cor_relabelled                    # paired per-item delta, in {-1, 0, +1}
    lo, hi, how = boot(uids, d, seed=seed)
    d_acc = float(d.mean())
    scale = (1 - fl) if fl < 1 else None
    d_gain = d_acc / scale if scale else 0.0
    d_gain_ci = [lo / scale, hi / scale] if scale else [0.0, 0.0]
    n10 = int(np.sum((cor_full == 1) & (cor_relabelled == 0)))  # full(mobility) right, relabelled wrong
    n01 = int(np.sum((cor_full == 0) & (cor_relabelled == 1)))  # relabelled right, full(mobility) wrong
    flag = "NON-NULL(CI excl. 0)" if (lo > 0 or hi < 0) else "null-consistent"
    return dict(
        n=n, n_clusters=int(len(np.unique(uids))), floor=fl,
        acc_full=acc_full, acc_framing=acc_relabelled,
        gain_full=gain(acc_full), gain_framing=gain(acc_relabelled),
        d_acc=d_acc, d_acc_ci=[float(lo), float(hi)],
        d_gain=float(d_gain), d_gain_ci=[float(x) for x in d_gain_ci],
        estimator=how,
        n_disc_full_right=n10, n_disc_framing_right=n01, n_discordant=n10 + n01,
        unparsed_full=1 - float(np.mean(parsed_full)),
        unparsed_framing=1 - float(np.mean(parsed_relabelled)),
        flag=flag)


def score_cell(items, ix, raw_full, raw_relabelled, seed=0):
    """One (family, span) cell, or any other index slice (the pooled full range included). Floor
    is computed ONCE from `items[i]["a"]` at `ix` -- both conditions share the same gold for the
    same item by construction, so there is one floor, handed to both sides of the gain formula
    inside summarize(). Returns (rec, per_item_arrays); the caller uses per_item_arrays to build
    the pooled-per-model numbers without re-running norm()."""
    golds = [items[i]["a"] for i in ix]
    mode_val, mode_ct = collections.Counter(golds).most_common(1)[0]
    fl = mode_ct / len(ix)
    cor_full, parsed_full = predict(raw_full, items, ix)
    cor_relabelled, parsed_relabelled = predict(raw_relabelled, items, ix)
    uids = [items[i]["uid"] for i in ix]
    maj = [items[i]["a"] == mode_val for i in ix]     # per-item "matches its OWN cell's mode"
    rec = summarize(fl, cor_full, cor_relabelled, parsed_full, parsed_relabelled, uids, seed=seed)
    arrs = dict(cor_full=cor_full, cor_relabelled=cor_relabelled, parsed_full=parsed_full,
                parsed_relabelled=parsed_relabelled, uids=uids, maj=maj)
    return rec, arrs


# --------------------------------------------------------------------------------- item-file I/O
def load_itemsets():
    """Every candidate item file the harness could have paired a results json against, keyed by
    row count. Read from disk rather than hardcoding {1500: ..., 40: ...} so a future re-draw at a
    different --per-cell does not silently go stale, and so the SAME lookup handles the smoke file
    and the full arm without a smoke/full branch anywhere in the scoring code itself."""
    sets = {}
    for fn in ("questions_framing.jsonl", "questions_framing_smoke.jsonl"):
        p = os.path.join(RESULTS, fn)
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p)]
            # Resolution is BY LENGTH, so two item files of equal length would make one silently
            # shadow the other and BOTH conditions would resolve to the same WRONG file -- which
            # the `items_full is items_relabelled` check cannot catch, because the object really is
            # shared. Refuse instead of resolving ambiguously.
            if len(rows) in sets:
                raise ValueError(
                    f"{fn} and {sets[len(rows)][0]} both have {len(rows)} rows -- item resolution "
                    f"is by row count, so the pairing is ambiguous; remove or rename one")
            sets[len(rows)] = (fn, rows)
    return sets


def resolve_items(n, itemsets):
    if n not in itemsets:
        raise ValueError(f"{n} generations match no items file on disk (have {sorted(itemsets)})")
    return itemsets[n]


# ------------------------------------------------------------------------------------- reporting
def print_row(fam, span, r):
    ci = f"[{r['d_acc_ci'][0]:+.3f},{r['d_acc_ci'][1]:+.3f}]"
    print(f"  {fam:<16}{span:>5}{r['n']:>6}{r['n_clusters']:>5}{r['floor']:>7.3f} |"
          f"{r['acc_full']:>9.3f}{r['acc_framing']:>10.3f}{r['d_acc']:>+9.3f}{ci:>18} |"
          f"{r['gain_full']:>10.3f}{r['gain_framing']:>11.3f} | "
          f"{r['n_disc_full_right']:>10}{r['n_disc_framing_right']:>11} |"
          f"{r['unparsed_full']:>9.1%}{r['unparsed_framing']:>10.1%}  {r['estimator']:<12} {r['flag']}")


HEADER = (f"  {'family':<16}{'span':>5}{'n':>6}{'ncl':>5}{'floor':>7} |"
          f"{'acc_full':>9}{'acc_relab':>10}{'d_acc':>9}{'95% CI':>18} |"
          f"{'gain_full':>10}{'gain_relab':>11} | {'full>relab':>10}{'relab>full':>11} |"
          f"{'unp_full':>9}{'unp_relab':>10}  estimator     verdict")


# ------------------------------------------------------------------------------------- self-test
def _mk_items(n, family="gyration_km", span=32, atype="numeric",
              gold_cycle=("1", "1", "1", "1", "2", "2", "2", "3", "3", "4"), uid_start=100000,
              one_uid=False):
    """n synthetic items, golds cycling through `gold_cycle` (repeats to length n; the cycle's own
    multiset fixes an exact, known floor). One distinct uid per item by default (>= MINCL clusters
    for any n >= 5); `one_uid=True` collapses every item onto a single uid, for the degenerate-
    cluster check. Only family/span/atype/a/uid are read anywhere downstream, matching exactly
    what predict()/score_cell() touch on a real item."""
    return [dict(family=family, span=span, atype=atype, a=gold_cycle[i % len(gold_cycle)],
                 uid=(7 if one_uid else uid_start + i)) for i in range(n)]


def _mk_raw(items, wrong_from):
    """Raw generation text for one condition: correct (`Answer: <gold>`) before `wrong_from`, a
    fixed wrong-but-parseable answer from `wrong_from` on. Mirrors make_framing.py's own
    stub_generate() convention (`Answer: <value>`), so this exercises the SAME norm() marker path
    a real generation would, not a hand-built shortcut around the parser."""
    out = []
    for i, it in enumerate(items):
        if i < wrong_from:
            out.append(f"Answer: {it['a']}")
        else:
            out.append(f"Answer: {'999999' if it['a'] != '999999' else '888888'}")
    return out


def selftest():
    print("=" * 100)
    print("SELFTEST -- synthetic fixtures only, no files read, no GPU/cluster/network touched")
    print("=" * 100)
    passed = True

    # ---- CHECK 1: planted margin. Non-mobility is built to be exactly 0.10 accuracy points better than
    # mobility on one 200-item cell with an exact baseline of 0.40: mobility is wrong on its last 40 of
    # 200, non-mobility on its last 20 of 200 (a subset of mobility's wrong set), so every discordant
    # item falls in the "non-mobility right / mobility wrong" direction, by exactly 20. The script must
    # recover d_acc = full - relabelled = -0.10 exactly and flag the cell NON-NULL (95% CI excludes
    # 0).
    print("\n--- CHECK 1: planted margin (non-mobility better by a known 0.10 accuracy) ---")
    n = 200
    items = _mk_items(n)
    ix = list(range(n))
    raw_full = _mk_raw(items, wrong_from=160)     # full(mobility) wrong on items[160:200] (40)
    raw_relabelled = _mk_raw(items, wrong_from=180)    # relabelled wrong on items[180:200] (20)
    rec, _ = score_cell(items, ix, raw_full, raw_relabelled, seed=1)
    print(f"  floor      = {rec['floor']:.4f}   (expect 0.4000)")
    print(f"  acc_full   = {rec['acc_full']:.4f}   acc_framing = {rec['acc_framing']:.4f}   "
          f"(expect 0.8000 / 0.9000)")
    print(f"  d_acc      = {rec['d_acc']:+.4f}   95% CI = {rec['d_acc_ci']}   (expect -0.1000, "
          f"exactly)")
    print(f"  discordant : full-right/relabelled-wrong = {rec['n_disc_full_right']} (expect 0), "
          f"relabelled-right/full-wrong = {rec['n_disc_framing_right']} (expect 20)")
    print(f"  estimator  = {rec['estimator']} on {rec['n_clusters']} distinct uids "
          f"(expect uid-cluster, 200)")
    print(f"  verdict    = {rec['flag']}   (expect NON-NULL, recovered margin flagged)")
    ok1 = (abs(rec["floor"] - 0.4) < 1e-9
           and abs(rec["acc_full"] - 0.80) < 1e-9
           and abs(rec["acc_framing"] - 0.90) < 1e-9
           and abs(rec["d_acc"] - (-0.10)) < 1e-9
           and rec["n_disc_full_right"] == 0
           and rec["n_disc_framing_right"] == 20
           and rec["estimator"] == "uid-cluster"
           and rec["d_acc_ci"][1] < 0                 # CI entirely below zero: correctly signed
           and rec["flag"].startswith("NON-NULL"))
    print(f"  CHECK 1: {'PASSED' if ok1 else '*** FAILED ***'}")
    passed = passed and ok1

    # ---- CHECK 2: pure null. Both conditions are handed the identical raw text at a non-trivial
    # accuracy (70%, baseline 0.40, so this is not a degenerate all-correct or all-wrong case). The
    # paired delta must come back exactly 0.0 (bit-exact, not merely close), with zero discordant
    # items in both directions and a CI that contains 0.
    print("\n--- CHECK 2: pure null (conditions identical) ---")
    n2 = 200
    items2 = _mk_items(n2)
    ix2 = list(range(n2))
    raw_shared = _mk_raw(items2, wrong_from=140)      # 70% correct, same text for both conditions
    rec2, _ = score_cell(items2, ix2, raw_shared, raw_shared, seed=2)
    print(f"  floor      = {rec2['floor']:.4f}   (expect 0.4000)")
    print(f"  acc_full   = {rec2['acc_full']:.4f}   acc_framing = {rec2['acc_framing']:.4f}   "
          f"(expect both 0.7000)")
    print(f"  d_acc      = {rec2['d_acc']:+.6f}   95% CI = {rec2['d_acc_ci']}   (expect exactly "
          f"0.0, CI containing 0)")
    print(f"  discordant : full-right/relabelled-wrong = {rec2['n_disc_full_right']} (expect 0), "
          f"relabelled-right/full-wrong = {rec2['n_disc_framing_right']} (expect 0)")
    print(f"  verdict    = {rec2['flag']}   (expect null-consistent)")
    ok2 = (rec2["d_acc"] == 0.0
           and rec2["d_acc_ci"][0] <= 0.0 <= rec2["d_acc_ci"][1]
           and rec2["n_disc_full_right"] == 0
           and rec2["n_disc_framing_right"] == 0
           and rec2["flag"] == "null-consistent")
    print(f"  CHECK 2: {'PASSED' if ok2 else '*** FAILED ***'}")
    passed = passed and ok2

    # ---- CHECK 3: the degenerate-cluster guard. 20 items collapsed onto one uid (far below
    # MINCL = 5): the bootstrap must fall back to item-level resampling and label itself as such,
    # rather than silently reporting uid-cluster precision it does not have.
    print("\n--- CHECK 3 (bonus): degenerate-cluster fallback fires and is labelled ---")
    items3 = _mk_items(20, one_uid=True)
    ix3 = list(range(20))
    raw3 = _mk_raw(items3, wrong_from=15)
    rec3, _ = score_cell(items3, ix3, raw3, raw3, seed=3)      # 1 uid, 20 items -> 1 cluster
    print(f"  n_clusters = {rec3['n_clusters']}   (expect 1)")
    print(f"  estimator  = {rec3['estimator']}   (expect ITEM-fallback)")
    ok3 = rec3["estimator"] == "ITEM-fallback" and rec3["n_clusters"] == 1
    print(f"  CHECK 3: {'PASSED' if ok3 else '*** FAILED ***'}")
    passed = passed and ok3

    print("\n" + "=" * 100)
    print(f"SELFTEST: {'ALL CHECKS PASSED' if passed else '*** AT LEAST ONE CHECK FAILED ***'}")
    print("=" * 100)
    return passed


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "scored_framing.json"))
    ap.add_argument("--selftest", action="store_true",
                     help="run the synthetic known-answer checks and exit; reads no files")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    runs = sorted(set(glob.glob(f"{RESULTS}/framing_*.json"))
                  | set(glob.glob(f"{RESULTS}/framingsmoke_*.json")))
    runs = [r for r in runs if not r.endswith((".ckpt", ".partial"))]
    if not runs:
        sys.exit("no results/framing_*.json or results/framingsmoke_*.json runs on disk")
    print(f"found {len(runs)} result file(s): {[os.path.basename(r) for r in runs]}")

    itemsets = load_itemsets()
    if not itemsets:
        sys.exit("no results/questions_framing*.jsonl on disk -- nothing to score against")
    print(f"item files on disk: {[(fn, n) for n, (fn, _) in itemsets.items()]}")

    out_cells, out_pooled, skipped = {}, {}, []
    for rp in runs:
        base = os.path.basename(rp)
        if base.startswith("framingsmoke_"):
            model, arm = base.replace("framingsmoke_", "").replace(".json", ""), "smoke"
        elif base.startswith("framing_"):
            model, arm = base.replace("framing_", "").replace(".json", ""), "full"
        else:
            continue

        blob = json.load(open(rp))
        raw = blob.get("raw") or {}
        raw_full, raw_relabelled = raw.get("full"), raw.get("nomobility")
        if raw_full is None or raw_relabelled is None:
            skipped.append((model, arm, "missing raw['full'] or raw['nomobility']")); continue

        # Pair by item index, and refuse rather than guess if the two conditions are not the same
        # length: a mismatch means the harness did not write them in lockstep, and no number computed from a
        # positional pairing would mean anything.
        if len(raw_full) != len(raw_relabelled):
            skipped.append((model, arm, f"full has {len(raw_full)} generations, nomobility has "
                             f"{len(raw_relabelled)} -- conditions not aligned, refusing to score"))
            continue
        try:
            fn_full, items_full = resolve_items(len(raw_full), itemsets)
            fn_relabelled, items_relabelled = resolve_items(len(raw_relabelled), itemsets)
        except ValueError as e:
            skipped.append((model, arm, str(e))); continue
        # The non-vacuous form of "assert the floors match across conditions" for a file shape
        # where both conditions are columns of one row: both generation lists must resolve to the
        # SAME items object, not merely two objects of equal length (a same-length decoy file
        # would pass a length check but not this one).
        if items_full is not items_relabelled:
            skipped.append((model, arm, f"full resolved against {fn_full}, nomobility against "
                             f"{fn_relabelled} -- not the same item file")); continue
        items = items_full
        if blob.get("n") is not None and blob["n"] != len(items):
            skipped.append((model, arm, f"stored n={blob['n']} != {len(items)} items in "
                             f"{fn_full} -- file looks corrupted")); continue

        fams = sorted({it["family"] for it in items})
        if set(fams) - set(GEOM):
            skipped.append((model, arm, f"non-geometric families present: "
                             f"{set(fams) - set(GEOM)}")); continue

        print(f"\n{'=' * 120}\n{model} [{arm}]  {base}  <- {len(items)} items from {fn_full}"
              f"\n{'=' * 120}")
        print(HEADER)
        pool = collections.defaultdict(list)
        for fam in fams:
            for span in sorted({it["span"] for it in items if it["family"] == fam}):
                ix = [i for i, it in enumerate(items) if it["family"] == fam and it["span"] == span]
                if not ix:
                    continue
                rec, arrs = score_cell(items, ix, raw_full, raw_relabelled)
                out_cells[f"{model}|{arm}|{fam}|{span}"] = rec
                print_row(fam, span, rec)
                for k in ("cor_full", "cor_relabelled", "parsed_full", "parsed_relabelled", "uids", "maj"):
                    pool[k].extend(arrs[k])
        if not pool["uids"]:
            skipped.append((model, arm, "no geometric cells scored")); continue

        fl_pool = float(np.mean(pool["maj"]))
        pooled = summarize(fl_pool, pool["cor_full"], pool["cor_relabelled"], pool["parsed_full"],
                            pool["parsed_relabelled"], pool["uids"])
        out_pooled[f"{model}|{arm}"] = pooled
        print("  " + "-" * 116)
        print_row("POOLED", "--", pooled)

    if skipped:
        print("\nSKIPPED:")
        for m, arm, why in skipped:
            print(f"  {m} [{arm}]: {why}")

    if not out_pooled:
        sys.exit("no model/arm could be scored -- nothing to write")

    summary = dict(
        nboot=NBOOT, mincl=MINCL, geom_families=GEOM,
        items_files={fn: n for n, (fn, _) in itemsets.items()},
        n_scored=len(out_pooled), scored=sorted(out_pooled),
        skipped=[{"model": m, "arm": arm, "why": w} for m, arm, w in skipped],
        cells=out_cells, pooled=out_pooled)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
