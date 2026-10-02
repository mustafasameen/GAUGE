#!/usr/bin/env python3
"""Score two decisions built from the same answers: rank people, and count people.

The paper reports that geometric descriptors never clear their baseline in exact recovery. This
script measures a dissociation: one set of answers can support a ranking decision without
supporting a counting decision.
  Decision A, counting: what share of people does the model place above the population's true 75th
      percentile? By construction the truth is 25.0% in every cell.
  Decision B, ranking: of the people the model puts in the top decile, how many really are?

Two reference points, both carrying zero information about any individual, give the scores a scale:
  permutation  a permutation of the true values: a perfect marginal in which every person's value
               belongs to someone else. It is the floor for ranking and it is exact on counting.
  oracle       the true values, the ceiling.
A gain is reported as the position between the floor and the ceiling, the same normalized form that
is used elsewhere in the paper.

Ties are the trap. Gold values are whole numbers, so many people sit exactly on the decile
boundary. A random predictor then scores above 0.10 on top-decile precision and the oracle scores
below 1.0, which can look like a lift that is only a tie artifact. Every number is therefore
referenced to the permutation floor and the oracle ceiling measured on the same cell, and ties are
broken by expectation (random shuffles, averaged) and not by sort order.

Unparsed answers are not dropped. The main results score an unparseable generation as wrong, and a
re-scoring that silently drops them would measure a more charitable quantity on a different
denominator. Cells above `--max-unparsed` are reported and excluded from the summaries, never
hidden.

The scorer tests itself first on fixtures with known answers (oracle, permutation, constant, and
every true value halved).

Input: results/tally_v41.jsonl and the five core runs.
Output: results/v41_dissociation.json.

Usage:
  python gauge/score_dissociation.py [--selftest-only] [--max-unparsed 0.05]
"""
from __future__ import annotations
import argparse, collections, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gauge"))
from eval_factqa import norm            # identical parsing to the GPU runs, by import

ITEMS = os.path.join(ROOT, "results", "tally_v41.jsonl")
OUT = os.path.join(ROOT, "results", "v41_dissociation.json")
RUNS = [("Phi-3.5-mini", "v41core_phi35mini"), ("Mistral-7B", "v41core_mistral7b"),
        ("Llama-3.1-8B", "v41core_llama8b"), ("Gemma-3-12B", "v41pilot_gemma3_12b"),
        ("Llama-3.1-70B", "v41core_llama70b")]
GEO = {"gyration_km", "longest_jump", "max_distance", "total_distance"}
BOOT = 1000
DECILE, THRESH_Q = 0.10, 0.75


def share_above(pred, cut):
    """DECISION A. Share of people the predictor places above the population's true cut."""
    return float(np.mean(np.asarray(pred, float) > cut))


def top_decile_precision(pred, true, rng, draws=30):
    """DECISION B, expected over random tie-breaks.

    Ties matter: with whole-number golds, many people share the boundary value. Breaking them by
    argsort order would credit or punish a predictor for the file's row order, so each draw shuffles
    before sorting and the score is averaged.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    n = len(pred)
    k = max(1, int(round(DECILE * n)))
    true_top = set(np.argsort(-true, kind="stable")[:k].tolist())
    # a true tie at the boundary means "top decile" is not a unique set either; expand it to every
    # person whose true value reaches the k-th largest, which is the honest target
    cut_true = np.sort(true)[::-1][k - 1]
    true_top = set(np.where(true >= cut_true)[0].tolist())
    out = []
    for _ in range(draws):
        order = rng.permutation(n)
        sel = order[np.argsort(-pred[order], kind="stable")[:k]]
        out.append(len(set(sel.tolist()) & true_top) / k)
    return float(np.mean(out))


def selftest(rng):
    """Known answers in BOTH directions before any model is scored.

    A scorer that has only ever seen real predictions cannot tell you it works. Fixtures with
    arithmetically known behaviour: the oracle must top the ranking and be exact on counting; a
    permutation must be exact on counting and near-chance on ranking; a constant must fail counting
    outright; and halving every true value must keep ranking perfect while destroying the count.
    """
    n = 300
    true = np.round(rng.gamma(2.0, 2.0, n)).astype(float)   # whole numbers, so ties are realistic
    cut = float(np.quantile(true, THRESH_Q))
    perm = rng.permutation(true)
    const = np.full(n, float(collections.Counter(true).most_common(1)[0][0]))
    halved = true / 2.0

    res = {
        "oracle":      (share_above(true, cut),   top_decile_precision(true, true, rng)),
        "permutation": (share_above(perm, cut),   top_decile_precision(perm, true, rng)),
        "constant":    (share_above(const, cut),  top_decile_precision(const, true, rng)),
        "halved":      (share_above(halved, cut), top_decile_precision(halved, true, rng)),
    }
    for k, (sh, p) in res.items():
        print("  %-12s share_above=%.3f  top-decile P=%.3f" % (k, sh, p))

    ok = True
    # the oracle and a permutation both reproduce the population count exactly: that IS the point
    ok &= abs(res["oracle"][0] - res["permutation"][0]) < 1e-9
    # ...but only the oracle knows who is who
    ok &= res["oracle"][1] > res["permutation"][1] + 0.3
    # a constant cannot place anyone above the cut, so counting collapses
    ok &= res["constant"][0] == 0.0
    # halving preserves every rank and destroys the level
    ok &= abs(res["halved"][1] - res["oracle"][1]) < 1e-9
    ok &= res["halved"][0] < res["oracle"][0] - 0.15
    print("  self-test: %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        sys.exit("SCORER IS BROKEN: a fixture with a known answer behaved wrongly. Nothing written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--max-unparsed", type=float, default=0.05)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    print("SELF-TEST on synthetic fixtures with known answers")
    selftest(rng)
    if a.selftest_only:
        return

    items = [json.loads(l) for l in open(ITEMS)]
    cells = collections.defaultdict(list)
    for i, it in enumerate(items):
        if it["family"] in GEO and it.get("mag") not in (None, "None"):
            cells[(it["family"], int(it["span"]))].append(i)

    out, skipped = {}, []
    for name, stem in RUNS:
        full = json.load(open(os.path.join(ROOT, "results", "%s.json" % stem)))["raw"]["full"]
        for (fam, span), idx in sorted(cells.items()):
            true = np.array([float(items[i]["mag"]) for i in idx])
            cut = float(np.quantile(true, THRESH_Q))
            pred, n_unp = [], 0
            for i in idx:
                p = norm(full[i], items[i]["family"], items[i].get("atype"))
                try:
                    pred.append(float(str(p).strip()))
                except Exception:
                    pred.append(np.nan); n_unp += 1
            pred = np.array(pred)
            frac_unp = n_unp / len(idx)
            # unparsed cannot be dropped silently; they are scored as "no answer", which for a
            # threshold count means the model places nobody above the cut for that person
            filled = np.where(np.isnan(pred), -np.inf, pred)
            perm = rng.permutation(true)
            key = "%s|%s|%d" % (name, fam, span)
            f_rank = top_decile_precision(perm, true, rng)
            o_rank = top_decile_precision(true, true, rng)
            m_rank = top_decile_precision(filled, true, rng)
            gain = (m_rank - f_rank) / (o_rank - f_rank) if o_rank > f_rank else float("nan")
            boot = []
            for _ in range(BOOT):
                s = rng.integers(0, len(idx), len(idx))
                b = top_decile_precision(filled[s], true[s], rng, draws=3)
                bf = top_decile_precision(rng.permutation(true[s]), true[s], rng, draws=3)
                bo = top_decile_precision(true[s], true[s], rng, draws=3)
                if bo > bf:
                    boot.append((b - bf) / (bo - bf))
            rec = dict(family=fam, span=span, n=len(idx), unparsed=frac_unp,
                       share_above_model=share_above(filled, cut),
                       share_above_perm=share_above(perm, cut),
                       share_above_truth=share_above(true, cut),
                       rank_model=m_rank, rank_perm=f_rank, rank_oracle=o_rank,
                       rank_gain=gain,
                       rank_gain_ci=[float(np.percentile(boot, 2.5)),
                                     float(np.percentile(boot, 97.5))] if boot else None)
            out[key] = rec
            if frac_unp > a.max_unparsed:
                skipped.append((key, round(frac_unp, 3)))

    print("\nCELLS EXCLUDED FROM SUMMARIES for unparsed above %.0f%%: %d"
          % (100 * a.max_unparsed, len(skipped)))
    for k, u in skipped:
        print("   %-42s %.1f%% unparsed" % (k, 100 * u))

    print("\n%-15s %7s %9s %9s   %s" % ("model", "cells", "rank gain", "clear .05", "count: truth 25.0%"))
    for name, _ in RUNS:
        rows = [v for k, v in out.items()
                if k.startswith(name + "|") and v["unparsed"] <= a.max_unparsed]
        if not rows:
            continue
        g = [r["rank_gain"] for r in rows]
        sh = [100 * r["share_above_model"] for r in rows]
        print("%-15s %7d %+9.3f %6d/%-3d   %.1f%% to %.1f%%"
              % (name, len(rows), float(np.median(g)),
                 sum(1 for x in g if x >= 0.05), len(g), min(sh), max(sh)))

    json.dump(dict(decile=DECILE, threshold_quantile=THRESH_Q, boot=BOOT, seed=0,
                   max_unparsed=a.max_unparsed, excluded=skipped, cells=out),
              open(OUT, "w"), indent=1)
    print("\nwrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
