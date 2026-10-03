#!/usr/bin/env python3
"""Build the comparison items: which of two people has the larger value of a descriptor.

Reuses the coordinate transform and renderer of generate_questions.py by import, so the two records
in each item look exactly like the records in every other arm and the only new factor is the
question form. Each item shows two people's records, labelled A and B. Pairs are chosen to hit a
target ratio between the two true values (within 8%), and every pair is written in both orders.
After writing the file the script checks that the gold is balanced between A and B, that the two
orders are equally frequent, and that no realised ratio misses its target by more than 8%.

Input: data/yjmob/yjmob_export.parquet (from export_yjmob.py).
Output: results/questions_compare.jsonl (3,000 items at the default record length of 64:
5 quantities x 6 ratios x 50 pairs x 2 orders).

Usage:
  python gauge/generate_compare.py --data data/yjmob/yjmob_export.parquet --span 64 --per-cell 50 --out results/questions_compare.jsonl
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_compare as C
from generate_questions import rigid, render_xy, SLOTS_PER_DAY, SPLIT_DAY, MAX_PROMPT_CHARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yjmob/yjmob_export.parquet")
    ap.add_argument("--users", type=int, default=6000)
    ap.add_argument("--span", type=int, default=64)
    ap.add_argument("--per-cell", type=int, default=50)   # pairs; each emitted in BOTH orders
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/questions_compare.jsonl")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = pd.read_parquet(a.data)
    d = d[~d.is_test].copy()
    d["day"] = d.slot // SLOTS_PER_DAY
    d["tod"] = d.slot % SLOTS_PER_DAY
    uids = d.uid.unique()[:a.users]
    by = {u: g.sort_values("slot").reset_index(drop=True)
          for u, g in d[d.uid.isin(set(uids))].groupby("uid", sort=False)}
    users = [u for u, g in by.items() if len(g) >= a.span + 4]
    print(f"{len(users):,} eligible users | span {a.span} | {len(C.RATIOS)} ratios "
          f"| {a.per_cell} pairs/cell x 2 orders", flush=True)

    # one pool of transformed records, reused across families so every family sees the same people
    pool = []
    for u in rng.choice(users, size=min(a.pool, len(users)), replace=False):
        g = by[int(u)]
        st = int(rng.integers(0, len(g) - a.span))
        raw = g.iloc[st:st + a.span]
        P = rigid(raw[["x", "y"]].to_numpy(float), rng)
        pool.append(dict(uid=int(u), P=P, days=raw.day.to_numpy(),
                         body=render_xy(raw.day.to_numpy(), raw.tod.to_numpy(), P)))
    print(f"pool of {len(pool)} records built", flush=True)

    items, short = [], []
    for qname, (fn, _) in C.QUANTITY.items():
        vals = np.array([fn(p["P"], p["days"]) if qname == "cmp_daydistinct"
                         else fn(p["P"]) for p in pool], dtype=float)
        for ratio in C.RATIOS:
            made, tries = 0, 0
            while made < a.per_cell and tries < a.per_cell * 200:
                tries += 1
                i = int(rng.integers(0, len(pool)))
                if not np.isfinite(vals[i]) or vals[i] <= 0:
                    continue
                target = vals[i] / ratio
                cand = np.where(np.isfinite(vals) & (vals > 0))[0]
                j = int(cand[np.argmin(np.abs(vals[cand] - target))])
                if j == i or vals[j] <= 0:
                    continue
                got = vals[i] / vals[j]
                if abs(got - ratio) / ratio > 0.08:
                    continue
                for order in (0, 1):
                    it = C.build_item(qname, vals[i], vals[j], ratio, order,
                                      (pool[i]["body"], pool[j]["body"]))
                    bodies = it.pop("bodies")
                    it.update(
                        family=qname, span=a.span,
                        uid=int(pool[i]["uid"]), uid_b=int(pool[j]["uid"]),
                        n_rows=a.span * 2, n_chars=len(bodies[0]) + len(bodies[1]),
                        prompt_full=(f"Here are two people's location records. Each line is "
                                     f"day, timeslot, x, y.\n\nPerson A:\n{bodies[0]}\n\n"
                                     f"Person B:\n{bodies[1]}\n\n"
                                     f"Question: {it['q']}\nAnswer with the value only."),
                        prompt_blind=(f"Question about two people's location records: {it['q']}\n"
                                      f"Answer with the value only."))
                    assert it["n_chars"] <= MAX_PROMPT_CHARS, f"{qname}: over cap"
                    items.append(it)
                made += 1
            if made < a.per_cell:
                short.append((qname, ratio, made))
            print(f"  {qname:<18} ratio {ratio:>4.2f}  {made:>3} pairs"
                  f"{'   SHORT' if made < a.per_cell else ''}", flush=True)

    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    print(f"\n{a.out}\n  {len(items):,} items | md5 {md5}")

    # ---- the three properties this arm depends on ----
    print("\nCHECKS:")
    ab = collections.Counter(it["a"] for it in items)
    print(f"  gold balance A/B: {dict(ab)}  (must be ~50/50 or the floor is not 0.5)")
    assert abs(ab["a"] - ab["b"]) < 0.02 * len(items), "gold is unbalanced"
    ords = collections.Counter(it["order"] for it in items)
    print(f"  order counterbalance: {dict(ords)}  (must be equal)")
    assert ords[0] == ords[1], "orders not counterbalanced"
    bad = [it for it in items if abs(max(it["val_a"], it["val_b"])
                                     / max(min(it["val_a"], it["val_b"]), 1e-9)
                                     - it["ratio"]) / it["ratio"] > 0.08]
    print(f"  items whose realised ratio misses its target by >8%: {len(bad)}")
    assert not bad
    print("  record length is identical for A and B by construction, so length is never a cue")
    if short:
        print(f"\nSHORTFALL (reported, never refilled): {short}")


if __name__ == "__main__":
    main()
