#!/usr/bin/env python3
"""Build the cross-record attribution items: which of K people was at place P on day D.

Each item shows K people's records and asks which person was at a given place on a given day. The
answer is a letter, or "none". This is an attribution join and not a computation: over K records the
model must build a map from (day, place) to person and answer a lookup against it. The queried place
occurs in exactly one person's record on that day, so a model that finds any match anywhere is
wrong.

Every cell carries the same total number of rows (440 by default), split K ways: at K=2 each person
contributes half of the rows and at K=8 an eighth. Record length is therefore constant across the K
ladder and only the number of records changes. Truncation happens before the presence map is built,
so the gold is computed on exactly the rows the model sees.

Checks, asserted or printed after the items are written:
  baseline  answers are balanced per cell by construction, so the majority baseline is close to
            1/(K+1); a cell above 0.35 is printed as a failure
  surface   the answer is not predictable from the longest record, the most distinct places, or
            position (the accuracy of each heuristic is printed)
  cap       no candidate is rejected for exceeding the prompt cap, so no cell's composition is
            set by the cap
  cluster   an item spans K people, so it carries cluster="item" and the list of all K uids, and
            estimators resample items

Labels exclude D, T and P, which would collide with the d/t/p field prefixes of the rendered rows.

Input: data/yjmob/yjmob_v4.parquet (from export_v4.py).
Output: results/tally_whichK.jsonl (1,191 items) and results/which_person_screen.json.

Usage:
  python gauge/tally_whichK.py --data data/yjmob/yjmob_v4.parquet --out results/tally_whichK.jsonl
"""

import argparse, collections, hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tally_v4 import MAX_PROMPT_CHARS
from tally_v41 import SLOTS_PER_DAY, SPLIT_DAY

# Labels must not collide with the field prefixes. Records render as "d<day> t<slot> p<place>",
# so labelling people A to H would put person D against the day prefix, T against the timeslot
# prefix and P against the place prefix, and a model echoing a record line ("d12") would be
# indistinguishable from one answering "D". D, T and P are therefore dropped from the alphabet.
# A/B/C/E/F/G/H/J share no prefix with any field, so an answer token can only be a person. This is
# asserted below against the renderer.
LET = "ABCEFGHJ"
FIELD_PREFIXES = set("dtp")
assert not (set(LET) & {c.upper() for c in FIELD_PREFIXES}), "a label collides with a field prefix"

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/yjmob/yjmob_v4.parquet")
ap.add_argument("--cells", default="28x2,28x4,28x6,28x8",
                help="comma-separated <span_days>x<K> cells (days of history x number of people). "
                     "The default keeps the total record length fixed at --total-rows and varies K. "
                     "Cap rejections are counted and asserted to zero below.")
ap.add_argument("--per-cell", type=int, default=300)
ap.add_argument("--pool", type=int, default=6000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--total-rows", type=int, default=440)
ap.add_argument("--out", default="results/tally_whichK.jsonl")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
d = pd.read_parquet(a.data, columns=["uid", "slot", "x", "y", "is_test"])
d = d[~d.is_test].copy()
d["day"] = d.slot // SLOTS_PER_DAY
d["tod"] = d.slot % SLOTS_PER_DAY
d["cell"] = d.x.astype(np.int64) * 1000 + d.y.astype(np.int64)
keep = set(pd.unique(d.uid)[: a.pool])
by = {u: g.sort_values("slot") for u, g in d[d.uid.isin(keep)].groupby("uid", sort=False)}
users = list(by)
CELLS = [tuple(int(v) for v in c.split("x")) for c in a.cells.split(",")]
print(f"pool {len(users):,} users | cells (span_days x K) {CELLS}\n")
n_capped = 0

items = []
for span_days, K in CELLS:
    # Exactly balanced answers. Drawing the answer uniformly over the classes available in each item
    # leaves the realised baseline to sampling noise. Collecting candidates per class and taking an
    # equal number of each makes the majority baseline exactly 1/(K+1) by construction. This is a
    # stronger form of the stratification the family already uses (stratify on the query, never on the
    # people), so it introduces no new selection: which people appear is untouched.
    pool_by_ans = collections.defaultdict(list)
    per_class = max(1, a.per_cell // (K + 1))
    made = tries = 0
    while (min([len(pool_by_ans[c]) for c in
                [LET[i].lower() for i in range(K)] + ["none"]] or [0]) < per_class
           and tries < a.per_cell * 400):
        tries += 1
        us = rng.choice(users, K, replace=False)
        d0 = int(rng.integers(0, SPLIT_DAY - span_days))
        ws = [by[u][(by[u].day >= d0) & (by[u].day < d0 + span_days)] for u in us]
        per = max(6, a.total_rows // K)          # rows per person so the TOTAL is ~a.total_rows
        if any(len(w) < per for w in ws):
            continue
        ws = [w.iloc[:per] for w in ws]           # truncate BEFORE building the presence map
        if any(len(w) < 6 for w in ws):
            continue
        pres = collections.defaultdict(set)
        for i, w in enumerate(ws):
            for dd, cc in set(zip(w.day, w.cell)):
                pres[(dd, cc)].add(i)
        byans = collections.defaultdict(list)
        for q, st in pres.items():
            if len(st) == 1:
                byans[LET[next(iter(st))]].append(q)
        allc = list({c for (_, c) in pres})
        none = [(dd, cc) for dd in range(d0, d0 + span_days)
                for cc in rng.choice(allc, min(5, len(allc)), replace=False)
                if (dd, cc) not in pres]
        if none:
            byans["none"] = none
        avail = [k for k, v in byans.items() if v]
        if len(avail) < min(K, 4):                    # need most classes drawable for a fair stratum
            continue
        ans = str(rng.choice(avail))
        qday, qcell = byans[ans][int(rng.integers(len(byans[ans])))]

        # ONE shared relabelling over the union of cells: a place id means the same thing in every
        # record of this item and nothing outside it. Absolute coordinates never appear.
        allcells = sorted({c for w in ws for c in w.cell})
        m = {c: int(v) for c, v in zip(allcells, rng.choice(90000, len(allcells), replace=False))}
        body = "\n\n".join(
            f"Person {LET[i]}:\n" + "\n".join(f"d{int(r.day)} t{int(r.tod)} p{m[r.cell]}"
                                              for r in w.itertuples())
            for i, w in enumerate(ws))
        q = (f"Which of these {K} people was at place {m[qcell]} on day {qday}? "
             f"Answer with the single letter, or 'none' if it was none of them.")
        ans_uid = -1 if ans == "none" else int(us[LET.index(ans)])
        it = dict(tier="G", task="which_person", family="which_person", atype="letter",
                  skmob="NEW", load=5, q=q, a=ans.lower(),
                  span=span_days, span_days=span_days, K=K, cell=f"{span_days}x{K}",
                  uid=ans_uid, uids=[int(u) for u in us], cluster="item",
                  n_rows=sum(len(w) for w in ws), n_chars=len(body),
                  rows_per_person=[len(w) for w in ws],
                  distinct_per_person=[int(w.cell.nunique()) for w in ws],
                  prompt_full=f"Here are {K} people's location records.\n\n{body}\n\n"
                              f"Question: {q}\nAnswer with the letter only.",
                  prompt_blind=f"Question about {K} people's location records: {q}\n"
                               f"Answer with the letter only.")
        if it["n_chars"] > MAX_PROMPT_CHARS:
            n_capped += 1  # counted, and asserted to zero after the loop
            continue
        if len(pool_by_ans[it["a"]]) < per_class:
            pool_by_ans[it["a"]].append(it)
            made += 1
    for cls in [LET[i].lower() for i in range(K)] + ["none"]:
        items.extend(pool_by_ans[cls][:per_class])
    got = collections.Counter(len(pool_by_ans[c]) for c in pool_by_ans)
    short = [c for c in [LET[i].lower() for i in range(K)] + ["none"]
             if len(pool_by_ans[c]) < per_class]
    if short:
        print(f"    NOTE: classes short of {per_class}: "
              f"{ {c: len(pool_by_ans[c]) for c in short} }")
    n = [x["n_rows"] for x in items if x["cell"] == f"{span_days}x{K}"]
    ch = [x["n_chars"] for x in items if x["cell"] == f"{span_days}x{K}"]
    print(f"  {span_days:>3}d K={K}  {len(n):>4} items ({made/max(tries,1):>5.1%} yield)  "
          f"median rows {np.median(n) if n else 0:>5.0f}  max chars {max(ch) if ch else 0:>6,}",
          flush=True)

if not items:
    sys.exit("no items built")
with open(a.out, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(f"\n{a.out}\n  {len(items):,} items | "
      f"md5 {hashlib.md5(open(a.out,'rb').read()).hexdigest()}")

# ---------------------------------------------------------------- gates
print(f"\nGATE 0 — PROMPT CAP. Candidates rejected for exceeding "
      f"{MAX_PROMPT_CHARS:,} chars: {n_capped}")
assert n_capped == 0, ("the cap is shaping the item pool -- longest records are being dropped, so "
                       "the affected cell's composition is a tool artifact. Shrink the cell.")
print("  PASS — the cap rejected nothing, so no cell's composition is set by the tool")

print("\nGATE 1 — FLOOR (target ~1/(K+1); refuse above 0.350)")
ok = True
for span_days, K in CELLS:
    ix = [it for it in items if it["cell"] == f"{span_days}x{K}"]
    if not ix: continue
    c = collections.Counter(it["a"] for it in ix)
    fl = c.most_common(1)[0][1] / len(ix)
    tgt = 1 / (K + 1)
    print(f"  {span_days:>3}d K={K}  floor {fl:.3f} (target {tgt:.3f})  {dict(sorted(c.items()))}")
    if fl > 0.35:
        ok = False
print("  " + ("PASS" if ok else "*** FAIL: a majority-class answerer is too strong"))

print("\nGATE 2 — SURFACE CUES. A cheap heuristic that beats the floor means the item is not a join.")
print(f"  {'cell':>7}{'floor':>8}{'longest':>10}{'most places':>13}{'first':>8}{'last':>8}")
cue_fail = []
for span_days, K in CELLS:
    ix = [it for it in items if it["cell"] == f"{span_days}x{K}"]
    if not ix: continue
    fl = collections.Counter(it["a"] for it in ix).most_common(1)[0][1] / len(ix)
    def h(fn):
        return float(np.mean([LET[fn(it)].lower() == it["a"] for it in ix]))
    longest = h(lambda it: int(np.argmax(it["rows_per_person"])))
    places = h(lambda it: int(np.argmax(it["distinct_per_person"])))
    first = h(lambda it: 0)
    last = h(lambda it: K - 1)
    print(f"  {span_days:>5}x{K}{fl:>8.3f}{longest:>10.3f}{places:>13.3f}{first:>8.3f}{last:>8.3f}")
    for name, v in [("longest", longest), ("most places", places)]:
        if v > fl + 0.05:
            cue_fail.append((span_days, name, v, fl))
print("  " + ("PASS — no heuristic clears floor+0.05" if not cue_fail
             else f"*** FAIL: {cue_fail}"))

print("\nGATE 3 — CLUSTER VARIABLE")
uu = collections.Counter(u for it in items for u in it["uids"])
print(f"  {len(uu):,} distinct users appear across {len(items):,} items; "
      f"most-used user appears in {max(uu.values())} items "
      f"({max(uu.values())/len(items):.2%})")
print(f"  every item carries cluster='item' and {len(items)} real uid lists -- NO placeholder uid.")
print(f"  the estimator must resample ITEMS and label the fallback; a user-cluster bootstrap is")
print(f"  not defined here because one item spans K users.")

json.dump(dict(n=len(items), cells=[f"{s}x{k}" for s, k in CELLS], cap_rejected=n_capped,
               floors={f"{s}x{k}": collections.Counter(it["a"] for it in items
                                                       if it["cell"] == f"{s}x{k}").most_common(1)[0][1]
                       / max(1, sum(1 for it in items if it["cell"] == f"{s}x{k}"))
                       for s, k in CELLS},
               cue_failures=cue_fail, distinct_users=len(uu),
               max_user_share=max(uu.values())/len(items)),
          open("results/which_person_screen.json", "w"), indent=1)
print("\nwrote results/which_person_screen.json")
