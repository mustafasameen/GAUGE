#!/usr/bin/env python3
"""Build the decomposition-control items file (pair_distance, sum_given, max_given).

Reuses the coordinate transform, record rendering and prompt assembly of tally_v41.py by import, so
that the controls pass through the identical pipeline as the geometric families. A control is only
informative if it differs from the failing arm in exactly one dimension, and rewriting the renderer
here would add a second one. `sum_given` and `max_given` print their own list and do not need the
record, but they are still emitted with a record body and the same prompt scaffold, so the prompt
shape is the same across all three families and the main run.

After writing the file the script prints, per cell, the majority baseline, the number of distinct
golds and the gold spread, and flags cells whose baseline is above .35.

Input: data/yjmob/yjmob_v4.parquet (from export_v4.py).
Output: results/tally_control.jsonl (2,250 items: 3 families x 5 record lengths x 150).

Usage:
  python gauge/tally_control.py --data data/yjmob/yjmob_v4.parquet --out results/tally_control.jsonl
"""
import argparse
import collections
import hashlib
import json
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_control as C
from tally_v41 import rigid, render_xy, SLOTS_PER_DAY, SPLIT_DAY, MAX_PROMPT_CHARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yjmob/yjmob_v4.parquet")
    ap.add_argument("--users", type=int, default=6000)
    ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--spans", default="32,64,128,256,512")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tally_control.jsonl")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = pd.read_parquet(a.data)
    d = d[~d.is_test].copy()
    d["day"] = d.slot // SLOTS_PER_DAY
    d["tod"] = d.slot % SLOTS_PER_DAY
    uids = d.uid.unique()[:a.users]
    by = {u: g.sort_values("slot").reset_index(drop=True)
          for u, g in d[d.uid.isin(set(uids))].groupby("uid", sort=False)}
    users = list(by)
    SPANS = [int(x) for x in a.spans.split(",")]
    print(f"{len(users):,} users | spans {SPANS} | n={a.per_cell}/cell\n", flush=True)

    items, drops, shortfall = [], collections.Counter(), []

    def _schema(body):
        """Name the columns that are actually rendered (coordinate families render "d.. t.. x.. y..")."""
        return ("day, timeslot, x, y" if " x" in body.split("\n")[0]
                else "day, timeslot, place")

    def emit(it, s, span, body):
        it["family"] = it["task"]
        it.update(uid=int(s.uid.iloc[0]), span=span, n_rows=body.count("\n") + 1,
                  n_chars=len(body),
                  prompt_full=(f"Here is a person's location record. Each line is "
                               f"{_schema(body)}.\n\n{body}\n\n"
                               f"Question: {it['q']}\nAnswer with the value only."),
                  prompt_blind=(f"Question about a person's location record: {it['q']}\n"
                                f"Answer with the value only."))
        assert it.get("atype"), f"{it['task']}: no atype"
        assert it["n_rows"] == span, f"span dial broken: {it['n_rows']} != {span}"
        assert it["n_chars"] <= MAX_PROMPT_CHARS, f"{it['task']}: over cap at span {span}"
        items.append(it)

    for span in SPANS:
        for name, (fn, needs_coord) in C.FAMILIES.items():
            made, tries = 0, 0
            while made < a.per_cell and tries < a.per_cell * 60:
                tries += 1
                u = int(rng.choice(users))
                g = by[u]
                if len(g) < span + 4:
                    drops["short"] += 1
                    continue
                st = int(rng.integers(0, len(g) - span))
                raw = g.iloc[st:st + span]
                assert int(raw.day.max()) < SPLIT_DAY
                P = rigid(raw[["x", "y"]].to_numpy(float), rng)
                it = fn(raw, rng, span, P)
                if not it:
                    drops[f"reject_{name}"] += 1
                    continue
                emit(it, raw, span, render_xy(raw.day.to_numpy(), raw.tod.to_numpy(), P))
                made += 1
            if made < a.per_cell:
                shortfall.append((name, span, made))
            print(f"  span {span:>4}  {name:<16} {made:>4}"
                  f"{'   SHORT' if made < a.per_cell else ''}", flush=True)

    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    print(f"\n{a.out}\n  {len(items):,} items | md5 {md5}")

    # ---- baselines: a control that is easy to score proves nothing ----
    print("\nPER-CELL GOLD SPREAD AND MAJORITY FLOOR "
          "(a narrow gold would flatter the control):")
    print(f"  {'family':<16}{'span':>6}{'n':>6}{'floor':>8}{'distinct':>10}{'gold p5..p95':>18}")
    for name in C.FAMILIES:
        for span in SPANS:
            ix = [it for it in items if it["family"] == name and it["span"] == span]
            if not ix:
                continue
            gs = [float(it["a"]) for it in ix]
            fl = collections.Counter(it["a"] for it in ix).most_common(1)[0][1] / len(ix)
            lo, hi = np.percentile(gs, [5, 95])
            flag = "   <<< FLOOR TOO HIGH" if fl > 0.35 else ""
            print(f"  {name:<16}{span:>6}{len(ix):>6}{fl:>8.3f}{len(set(gs)):>10}"
                  f"{f'{lo:.0f}..{hi:.0f}':>18}{flag}")
    if shortfall:
        print(f"\nSHORTFALL (reported, never refilled from an easier population): {shortfall}")
    if drops:
        print(f"drops: {dict(drops)}")


if __name__ == "__main__":
    main()
