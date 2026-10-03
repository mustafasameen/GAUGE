#!/usr/bin/env python3
"""Decomposition-control families: one link of the geometric computation at a time.

Three families isolate the links of the computation that `total_distance` requires. Each answers in
the same form as the geometric families (a whole number of kilometres), so the same parser,
decoder and scorer apply.

  pair_distance  One term of the sum. Two consecutive records are named by (day, timeslot) and the
                 model computes the straight-line distance between exactly that pair. It runs
                 across the same record lengths as the geometric families.
  sum_given      The geometry is removed: the per-segment distances are printed in the prompt and
                 the model only adds them up.
  max_given      The same kind of printed list, but the model takes the maximum. It separates
                 "cannot handle a long printed list" from "cannot add".

Reading the three together: if pair_distance and sum_given both work, total distance fails only
because the steps must be chained. If pair_distance fails, geometry fails at a single step. If
pair_distance works and sum_given fails, the failure is arithmetic over many values.

Gold answers are stratified so that they span a wide range. A narrow gold range hands a constant
predictor a high baseline and flatters the control; the reported quantity is gain over the
per-setting best constant, never raw accuracy. Berglund et al. (2023, arXiv 2309.12288, footnote
10) make the related point that a control whose answer is easier to score measures the scorer, not
the model.

Usage: imported by generate_control.py. Run `python gauge/tasks_control.py` for the self-tests.
"""
from __future__ import annotations

import numpy as np

KM = 0.5   # one grid unit = 500 m, identical to tier F and tasks_main


def C_pair_distance(s, rng, span, P):
    """One term of `total_distance`: the distance between two consecutive records named by
    (day, timeslot).

    The gold is computed from the same transformed, rounded coordinates that the model is shown. It is
    stratified so that it is not concentrated: it must be at least 1 km (so that rounding does not
    collapse the answer to 0), and a target distance is drawn uniformly across the range the record can
    produce, after which the eligible pair closest to the target is taken.
    """
    if len(P) < 3:
        return None
    seg = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1)) * KM      # distance of each consecutive pair
    ok = np.where(seg >= 1.0)[0]
    if len(ok) == 0:
        return None
    # Spread the gold. Consecutive moves are mostly short, so taking a random eligible pair piles the
    # gold up at 1-3 km and gives a high baseline. Instead, draw a target uniformly across the range
    # this record can produce and take the eligible pair closest to it.
    lo, hi = float(seg[ok].min()), float(seg[ok].max())
    if hi - lo < 2.0:
        return None                    # this record cannot support a spread gold; reject it
    target = float(rng.uniform(lo, hi))
    k = int(ok[int(np.argmin(np.abs(seg[ok] - target)))])
    d = float(seg[k])
    # The address must be unique. (day, timeslot) determines the row in real data (slot =
    # day * slots_per_day + tod, rows sorted by slot), but that is an assumption about the data, and an
    # ambiguous address would make the gold unanswerable, so it is checked for every item.
    dd = (int(s.day.iloc[k]), int(s.tod.iloc[k]))
    de = (int(s.day.iloc[k + 1]), int(s.tod.iloc[k + 1]))
    pairs = list(zip(s.day.astype(int), s.tod.astype(int)))
    if pairs.count(dd) != 1 or pairs.count(de) != 1:
        return None
    return dict(tier="F", task="pair_distance", atype="numeric", skmob="NEW", load=5,
                # Records are addressed by (day, timeslot), not by position. Day and timeslot are printed on
                # every line, but the lines carry no line numbers, so asking for "record number 15" would make the
                # model count to line 15 before any geometry could start, and would mix counting into the single
                # geometric step that this control isolates.
                q=(f"Treating each unit on the x and y axes as 500 metres, what is the "
                   f"straight-line distance between the record at day {int(s.day.iloc[k])} "
                   f"timeslot {int(s.tod.iloc[k])} and the record at day "
                   f"{int(s.day.iloc[k + 1])} timeslot {int(s.tod.iloc[k + 1])}, in kilometres? "
                   f"Answer with a whole number of kilometres."),
                a=str(int(round(d))), mag=d)


def _printed_list(rng, n, lo=1.0, hi=60.0):
    """A list of distances to print in the prompt, drawn to span a wide range so that the sum has
    wide support (the same rule against narrow golds as above).
    """
    v = np.round(rng.uniform(lo, hi, n), 1)
    return v


def C_sum_given(s, rng, span, P):
    """Berglund's in-context control: the geometry is REMOVED and only the addition remains.

    The per-segment distances are printed, so no coordinate work is required. If the model cannot
    do this, `total_distance` failing says nothing about geometry -- it is an arithmetic limit.
    The list length tracks the span so the amount of ADDITION scales exactly as it does in the
    real task.
    """
    n = max(3, min(span - 1, 40))          # cap the printed list so the prompt stays readable
    v = _printed_list(rng, n)
    # Sum in integer tenths. The printed values carry one decimal, but summing 40 float64 values that
    # only approximate one-decimal numbers accumulates error (an internal total of 703.4999 would round
    # to 703, while 703.5 recomputed from the printed text rounds to 704). The gold must be exactly the
    # sum of what the model is shown.
    total = float(np.round(v * 10).astype(np.int64).sum()) / 10.0
    body = ", ".join(f"{x:.1f}" for x in v)
    return dict(tier="F", task="sum_given", atype="numeric", skmob="NEW", load=5,
                q=(f"A person made {n} moves. The distances of those moves, in kilometres, were: "
                   f"{body}. What is the TOTAL distance they travelled, in kilometres? "
                   f"Answer with a whole number of kilometres."),
                a=str(int(round(total))), mag=total)


def C_max_given(s, rng, span, P):
    """Control for `sum_given`: same printed list, take the MAXIMUM instead of the sum.

    Separates "cannot handle a long printed list" from "cannot add". Load 1 against sum_given's
    load 5 over an identical input, which is the same paired-contrast logic as
    longest_jump-vs-total_distance but with the geometry taken out of both.
    """
    n = max(3, min(span - 1, 40))
    # Draw the answer first. Sampling n values uniformly from (1, 60) and taking the maximum gives a
    # gold of 57-60 almost every time, a baseline of .33-.39 over a handful of distinct answers. Draw the
    # intended maximum uniformly across the range, then fill the rest strictly below it, so that the
    # gold is spread by construction.
    top = float(np.round(rng.uniform(5.0, 60.0), 1))
    v = np.round(rng.uniform(0.5, top - 1.5, n - 1), 1)
    v = np.append(v, top)
    rng.shuffle(v)
    if len(v) > 1 and abs(top - np.sort(v)[-2]) < 0.5:
        return None                        # need a strict, roundable winner
    body = ", ".join(f"{x:.1f}" for x in v)
    return dict(tier="F", task="max_given", atype="numeric", skmob="NEW", load=1,
                q=(f"A person made {n} moves. The distances of those moves, in kilometres, were: "
                   f"{body}. What is the LARGEST single move distance, in kilometres? "
                   f"Answer with a whole number of kilometres."),
                a=str(int(round(top))), mag=top)


FAMILIES = {"pair_distance": (C_pair_distance, True),     # True = needs coordinates
            "sum_given": (C_sum_given, False),
            "max_given": (C_max_given, False)}


# ------------------------------------------------------------------ self-tests
if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(400):
        n = int(rng.integers(5, 60))
        P = rng.integers(0, 200, size=(n, 2)).astype(float)
        slots = np.sort(rng.choice(50 * 48, size=n, replace=False))   # unique, sorted, as in real data
        frame = pd.DataFrame({"day": slots // 48, "tod": slots % 48})
        it = C_pair_distance(frame, rng, n, P)
        if it is None:
            continue
        # independent brute-force recomputation of the gold from the question text
        import re
        dt = re.findall(r"day (\d+) timeslot (\d+)", it["q"])
        assert len(dt) == 2, it["q"]
        idx = []
        for dd, tt in dt:
            hits = frame.index[(frame.day == int(dd)) & (frame.tod == int(tt))].tolist()
            assert len(hits) == 1, f"address must be UNIQUE, got {len(hits)} rows"
            idx.append(int(hits[0]))
        assert idx[1] == idx[0] + 1, "the two addressed records must be consecutive"
        d = float(np.sqrt(((P[idx[0]] - P[idx[1]]) ** 2).sum()) * KM)
        assert int(round(d)) == int(it["a"]), (d, it["a"])
        assert d >= 1.0
        ok += 1
    print(f"pair_distance: {ok} items independently recomputed, all golds agree")

    ok = 0
    for _ in range(200):
        span = int(rng.integers(8, 512))
        it = C_sum_given(None, rng, span, None)
        vals = [float(x) for x in it["q"].split("were: ")[1].split(".", 1)[0].split(", ")
                ] if False else None
        import re
        vals = [float(x) for x in re.findall(r"(\d+\.\d)", it["q"].split("were: ")[1])]
        assert int(round(sum(vals))) == int(it["a"]), (sum(vals), it["a"])
        ok += 1
    print(f"sum_given: {ok} items, printed list re-summed from the prompt text, all golds agree")

    ok = 0
    for _ in range(300):
        span = int(rng.integers(8, 512))
        it = C_max_given(None, rng, span, None)
        if it is None:
            continue
        import re
        vals = [float(x) for x in re.findall(r"(\d+\.\d)", it["q"].split("were: ")[1])]
        assert int(round(max(vals))) == int(it["a"]), (max(vals), it["a"])
        ok += 1
    print(f"max_given: {ok} items, printed list re-maxed from the prompt text, all golds agree")
    print("all control self-tests pass")
