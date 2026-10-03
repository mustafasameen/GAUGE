#!/usr/bin/env python3
"""Build the scaffold-ladder items: the same people and the same golds at five levels of help.

For a given record, all five rungs carry the identical gold, so the majority baseline is identical
across rungs by construction and a jump in accuracy cannot come from an easier answer space. The
script asserts this and prints the per-rung baseline. The rungs are defined in tasks_ladder.py.

Input: data/yjmob/yjmob_export.parquet (from export_yjmob.py).
Output: results/questions_ladder.jsonl (2,000 items: 4 record lengths x 100 records x 5 rungs).

Usage:
  python gauge/generate_ladder.py --data data/yjmob/yjmob_export.parquet --out results/questions_ladder.jsonl
"""
import argparse, collections, hashlib, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_ladder as L
from generate_questions import rigid, SLOTS_PER_DAY, SPLIT_DAY, MAX_PROMPT_CHARS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/yjmob/yjmob_export.parquet")
ap.add_argument("--spans", default="32,64,128,256")
ap.add_argument("--per-cell", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/questions_ladder.jsonl")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
d = pd.read_parquet(a.data); d = d[~d.is_test].copy()
d["day"] = d.slot // SLOTS_PER_DAY; d["tod"] = d.slot % SLOTS_PER_DAY
by = {u: g.sort_values("slot").reset_index(drop=True)
      for u, g in d[d.uid.isin(set(d.uid.unique()[:6000]))].groupby("uid", sort=False)}
users = list(by)
SPANS = [int(x) for x in a.spans.split(",")]
items = []
for span in SPANS:
    made = tries = 0
    while made < a.per_cell and tries < a.per_cell * 60:
        tries += 1
        g = by[int(rng.choice(users))]
        if len(g) < span + 4: continue
        st = int(rng.integers(0, len(g) - span))
        raw = g.iloc[st:st + span]
        assert int(raw.day.max()) < SPLIT_DAY
        P = rigid(raw[["x", "y"]].to_numpy(float), rng)
        gold = L.gold(P)
        if float(gold) < 1: continue          # rounding would collapse the answer to 0
        for rung, (body, q) in L.rungs(P, raw.day.to_numpy(), raw.tod.to_numpy()).items():
            it = dict(tier="F", task=f"ladder_{rung}", family=f"ladder_{rung}", rung=rung,
                      atype="numeric", skmob="radius_of_gyration", load=5, q=q, a=gold,
                      uid=int(raw.uid.iloc[0]), span=span,
                      n_rows=body.count("\n") + 1, n_chars=len(body),
                      prompt_full=(f"Here is a person's location record.\n\n{body}\n\n"
                                   f"Question: {q}\nAnswer with the value only."),
                      prompt_blind=(f"Question about a person's location record: {q}\n"
                                    f"Answer with the value only."))
            assert it["n_chars"] <= MAX_PROMPT_CHARS
            items.append(it)
        made += 1
    print(f"  span {span:>4}  {made:>4} records x {len(L.RUNG_ORDER)} rungs", flush=True)

with open(a.out, "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
print(f"\n{a.out}\n  {len(items):,} items | md5 {hashlib.md5(open(a.out,'rb').read()).hexdigest()}")

print("\nTHE INVARIANT THE LADDER RESTS ON — identical gold distribution at every rung:")
print(f"  {'rung':<14}{'n':>6}{'floor':>8}{'distinct':>10}{'gold p50':>10}")
base = None
for rung in L.RUNG_ORDER:
    ix = [it for it in items if it["rung"] == rung]
    gs = [float(it["a"]) for it in ix]
    fl = collections.Counter(it["a"] for it in ix).most_common(1)[0][1] / len(ix)
    print(f"  {rung:<14}{len(ix):>6}{fl:>8.3f}{len(set(gs)):>10}{np.median(gs):>10.1f}")
    if base is None: base = sorted(gs)
    else: assert sorted(gs) == base, f"{rung}: gold distribution differs from L0"
print("  -> every rung carries the SAME golds, so the floor is identical and a jump in accuracy")
print("     is a jump in capability, not in answer-space difficulty")
