#!/usr/bin/env python3
"""Comparison questions: ask which of two people is larger on a descriptor, not for its value.

If a model holds any usable graded signal about a descriptor, it should be able to say which of two
people has more of it even when it cannot state how much either has. Each item shows two records,
labelled A and B, and asks which person is larger. The design follows the ratio-discrimination
protocol of arXiv 2603.20642 ("Weber's Law in Transformer Magnitude Representations") where it
transfers. The difference is that the quantities here must first be aggregated from a raw record,
instead of being read from numbers written in the prompt.

Design:
  * six true ratios between the two records: 1.10, 1.25, 1.50, 2.00, 3.00, 5.00
  * every pair is emitted twice, as (A, B) and as (B, A), so that position bias is measured and the
    answers are counterbalanced
  * both records are drawn at the same record length, so length is never a cue to which is larger
  * the answer is binary, so the baseline is exactly 0.5
  * `cmp_daydistinct` (distinct places on the busiest day) is a positive control. The models have
    signal on that family, so if the comparison format works anywhere it must work there.

Quantities: cmp_gyration, cmp_maxdist, cmp_totaldist, cmp_longjump, cmp_daydistinct.

Usage: imported by generate_compare.py.
"""
from __future__ import annotations

import numpy as np

KM = 0.5
RATIOS = [1.10, 1.25, 1.50, 2.00, 3.00, 5.00]


def rg(P):
    return float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())) * KM


def maxdist(P):
    d = 0.0
    for i in range(len(P)):
        dd = np.sqrt(((P[i] - P) ** 2).sum(1)).max()
        d = max(d, float(dd))
    return d * KM


def totaldist(P):
    return float(np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1)).sum()) * KM


def longest_jump(P):
    return float(np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1)).max()) * KM


def n_distinct_day(P, days):
    """Positive control quantity: distinct places on the busiest day is not geometric."""
    best = 0
    for d in np.unique(days):
        best = max(best, len({tuple(p) for p in P[days == d]}))
    return float(best)


QUANTITY = {
    "cmp_gyration": (rg, "radius of gyration (the typical distance of their locations from "
                         "their own centre)"),
    "cmp_maxdist": (maxdist, "greatest distance between any two of their locations"),
    "cmp_totaldist": (totaldist, "total distance travelled, adding up the straight-line "
                                 "distance between consecutive records"),
    "cmp_longjump": (longest_jump, "largest single move between two consecutive records"),
    "cmp_daydistinct": (n_distinct_day, "number of distinct places visited on their busiest day"),
}


def build_item(qname, va, vb, ratio, order, bodies):
    """One comparison item. `order` 0 = (A,B) as drawn, 1 = swapped -- the position control.
    Swapping moves BOTH the rendered records and their values together, so the gold follows the
    presentation and cannot drift out of step with it."""
    _, phrase = QUANTITY[qname]
    if order == 1:
        va, vb = vb, va
        bodies = (bodies[1], bodies[0])
    gold = "a" if va > vb else "b"
    return dict(tier="F", task=qname, atype="ab", skmob="NEW", load=5,
                q=(f"Below are two people's location records. Considering the {phrase}, "
                   f"which person is larger: A or B? Answer with A or B only."),
                a=gold, ratio=ratio, order=order, val_a=va, val_b=vb,
                bodies=bodies)


def pair_for_ratio(cands, fn, ratio, rng, tol=0.08):
    """Find a pair of records whose true quantity ratio is within `tol` of the target.
    Returns (i, j, v_i, v_j) with v_i > v_j, or None."""
    vals = np.array([fn(c) if fn is not n_distinct_day else np.nan for c in cands])
    idx = np.argsort(vals)
    for _ in range(200):
        i = int(rng.integers(0, len(idx)))
        target = vals[idx[i]] / ratio
        if not np.isfinite(target) or target <= 0:
            continue
        j = int(np.argmin(np.abs(vals[idx[:i]] - target))) if i > 0 else None
        if j is None:
            continue
        hi, lo = vals[idx[i]], vals[idx[j]]
        if lo <= 0:
            continue
        if abs(hi / lo - ratio) / ratio <= tol:
            return int(idx[i]), int(idx[j]), float(hi), float(lo)
    return None
