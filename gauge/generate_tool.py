#!/usr/bin/env python3
"""Build the program-arm items: the same records asked two ways, in one file.

Every record emits two items, one asking for the value and one asking for a Python expression that
computes it, with identical points and an identical gold. The two arms are therefore a paired
comparison on the same items within one run (a comparison across runs would not support a paired
test). The value arm here is a matched control. It is not a duplicate of the main geometric run.

The prompt presents each record as a Python list `pts` of (x, y) pairs, at record lengths of 32,
64 and 128. After writing the file the script checks that the two arms carry identical golds on
identical points, and that a correct reference expression, run through the sandbox in
tasks_tool.py, recovers the gold on every item (a ceiling of 1.000). Any shortfall therefore
belongs to the model and not to the harness.

Input: data/yjmob/yjmob_export.parquet (from export_yjmob.py).
Output: results/questions_tool.jsonl (2,400 items: 300 records x 4 families x 2 arms).

Usage:
  python gauge/generate_tool.py --data data/yjmob/yjmob_export.parquet --out results/questions_tool.jsonl
"""
import argparse, collections, hashlib, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_tool as T
from coord_records import MAX_PROMPT_CHARS, rigid
from generate_questions import SLOTS_PER_DAY, SPLIT_DAY

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/yjmob/yjmob_export.parquet")
ap.add_argument("--spans", default="32,64,128")
ap.add_argument("--per-cell", type=int, default=100)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="results/questions_tool.jsonl")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
d = pd.read_parquet(a.data); d = d[~d.is_test].copy()
d["day"] = d.slot // SLOTS_PER_DAY
by = {u: g.sort_values("slot").reset_index(drop=True)
      for u, g in d[d.uid.isin(set(pd.unique(d.uid)[:6000]))].groupby("uid", sort=False)}
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
        pts = [(int(round(x)), int(round(y))) for x, y in P]
        Pi = np.array(pts, float)
        body = "pts = [" + ", ".join(f"({x},{y})" for x, y in pts) + "]"
        ok = True
        pair = []
        for fam, (phrase, fn) in T.QUESTIONS.items():
            gold_v = fn(Pi)
            if not np.isfinite(gold_v) or gold_v < 1:
                ok = False; break
            gold = str(int(round(gold_v)))
            common = dict(tier="F", skmob=fam, load=5, a=gold, gold_value=float(gold_v),
                          uid=int(raw.uid.iloc[0]), span=span, family_base=fam,
                          pts=pts, n_rows=len(pts))
            qv = (f"Treating each unit on the x and y axes as 500 metres, what is {phrase}? "
                  f"Answer with a whole number of kilometres.")
            pv = (f"{T.PREAMBLE}\n\n{body}\n\nQuestion: {qv}\nAnswer with the value only.")
            pt = (f"{T.PREAMBLE}\n\n{body}\n\n{T.build_q(fam)}")
            if max(len(pv), len(pt)) > MAX_PROMPT_CHARS:
                ok = False; break
            pair.append(dict(common, task=f"value_{fam}", family=f"value_{fam}", arm="value",
                             atype="numeric", q=qv, n_chars=len(pv), prompt_full=pv,
                             prompt_blind=f"Question about a person's location record: {qv}\n"
                                          f"Answer with the value only."))
            pair.append(dict(common, task=f"tool_{fam}", family=f"tool_{fam}", arm="tool",
                             atype="code", q=T.build_q(fam), n_chars=len(pt), prompt_full=pt,
                             prompt_blind=T.build_q(fam)))
        if not ok: continue
        items.extend(pair)
        made += 1
    n = [x for x in items if x["span"] == span]
    print(f"  span {span:>4}  {made:>4} records x {len(T.QUESTIONS)} families x 2 arms "
          f"= {len(n)} items  max chars {max((x['n_chars'] for x in n), default=0):,}", flush=True)

with open(a.out, "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
print(f"\n{a.out}\n  {len(items):,} items | md5 {hashlib.md5(open(a.out,'rb').read()).hexdigest()}")

print("\nGATE — THE PAIRING INVARIANT: each arm must carry the IDENTICAL gold on the identical points")
for fam in T.QUESTIONS:
    v = sorted((it["uid"], it["span"], it["a"]) for it in items if it["family"] == f"value_{fam}")
    t = sorted((it["uid"], it["span"], it["a"]) for it in items if it["family"] == f"tool_{fam}")
    assert v == t, fam
    fl = collections.Counter(x[2] for x in v).most_common(1)[0][1] / len(v)
    print(f"  {fam:<16} {len(v):>4} paired records, identical golds, floor {fl:.3f}")

print("\nGATE — THE SANDBOX MUST RECOVER THE GOLD from a correct expression on these very items")
REF = {"gyration": "0.5*np.sqrt(np.mean(np.sum((np.array(pts)-np.mean(np.array(pts),axis=0))**2,axis=1)))",
       "max_distance": "0.5*max(math.hypot(a[0]-b[0],a[1]-b[1]) for a in pts for b in pts)",
       "total_distance": "0.5*sum(math.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))",
       "longest_jump": "0.5*max(math.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))"}
for fam, e in REF.items():
    ix = [it for it in items if it["family"] == f"tool_{fam}"]
    hits = 0
    for it in ix:
        v, stt = T.safe_eval(e, it["pts"])
        hits += (stt == "ok" and str(int(round(v))) == it["a"])
    print(f"  {fam:<16} a correct expression scores {hits}/{len(ix)} = {hits/len(ix):.3f}")
    assert hits == len(ix), f"{fam}: the CEILING is not 1.0 -- the arm cannot be read"
print("  -> the tool arm's ceiling is exactly 1.000, so any shortfall is the model, not the harness")
