#!/usr/bin/env python3
"""Generate the 34,200-question GAUGE benchmark from the YJMob100K export.

For each descriptor family (tasks_v41.py) and record length, the generator draws windows of
consecutive records from individual people, renders the prompt, and computes the gold answer from
the rows the model will see. There are 21 families, 7 record lengths (8 to 512 rows) and 108
family-by-length settings, with 300 questions per setting (600 for the retrieval probe). Place
families are relabelled per question from a pool of 512 identifiers. Coordinate families get a
per-question rigid transform (rotation, reflection, integer translation) before the gold is computed.
Coordinate prompts name their columns "day, timeslot, x, y", taken from the rendered rows.

Every item is checked as it is built:
  quota          every setting reaches its quota, or the shortfall is printed and never refilled
  answerability  every day named in a question appears in the rendered record
  atype          every item declares an answer type (the scorer needs it)
  span           the number of rendered rows equals the record length exactly
  prompt cap     MAX_PROMPT_CHARS binds nothing, so no setting is shaped by the cap

With --only-coord only the 6,000 coordinate (geometric) items are written.

Input: data/yjmob/yjmob_v4.parquet, written by export_v4.py.
Output: a JSON-lines items file (default results/tally_v41.jsonl).

Usage:
  python gauge/tally_v41.py --data data/yjmob/yjmob_v4.parquet --out results/tally_v41.jsonl
  python gauge/tally_v41.py --data data/yjmob/yjmob_v4.parquet --only-coord --out results/tally_v41_geomfix.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tally_v3 import relabel, render, SLOTS_PER_DAY, SPLIT_DAY
from tally_v4 import MAX_PROMPT_CHARS, rigid, render_xy
import tasks_v41 as T

# W starts at 16, not 8. At W=8 the gold equals W in 25-42% of items (and W is stated in the
# question, so a model can copy it), and a record-free constant scores 50-67% under +/-1 tolerance
# because distinct places over W is about 0.81. At W=16 the ratio drops to about 0.50 and both
# shortcuts fall below threshold.
WINDOWS = [16, 32, 64, 128]
POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yjmob/yjmob_v4.parquet")
    ap.add_argument("--spans", default="8,16,32,64,128,256,512")
    ap.add_argument("--per-cell", type=int, default=300)
    ap.add_argument("--users", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tally_v41.jsonl")
    # Write only the coordinate (tier F) families, the 6,000 geometric items. The place families
    # render "d.. t.. p.." and their column header already matches their rows, so they are not
    # regenerated.
    ap.add_argument("--only-coord", action="store_true")
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
    print(f"{len(d):,} train rows | {len(users):,} users | spans {SPANS} | n={a.per_cell}/cell\n",
          flush=True)

    items, drops, shortfall = [], collections.Counter(), []

    def _schema(body):
        """Name the columns that are actually rendered. Coordinate families render "d.. t.. x.. y.."
        and place families render "d.. t.. p..".
        """
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
        # --- the five asserts
        assert it.get("atype"), f"{it['task']}: no atype (norm() would score the family zero)"
        assert it["n_rows"] == span, f"span dial broken: {it['n_rows']} != {span}"
        assert it["n_chars"] <= MAX_PROMPT_CHARS, f"{it['task']}: over cap at span {span}"
        covered = {int(x) for x in s.day.unique()}
        for dd in {int(x) for x in re.findall(r"day (\d+)", it["q"])}:
            assert dd in covered, f"{it['task']}: asks about day {dd}, not in the record"
        items.append(it)

    def run_cell(span, label, fn, coord):
        """One (family, span) cell. Coord families get the rigid transform and an x/y render;
        place families get per-item relabelling and the place render."""
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
            if coord:
                P = rigid(raw[["x", "y"]].to_numpy(float), rng)
                it = fn(raw, rng, span, P)
                body = render_xy(raw.day.to_numpy(), raw.tod.to_numpy(), P)
                s = raw
            else:
                s = relabel(raw.copy(), rng)
                it = fn(s, rng, span, s["rank"].to_numpy(),
                        s.day.to_numpy(), s.tod.to_numpy())
                body = render(s)
            if not it:
                drops[f"reject_{label}"] += 1
                continue
            emit(it, s, span, body)
            made += 1
        if made < a.per_cell:
            # A setting that cannot fill its quota is reported, never silently refilled from an easier
            # population: refilling would select the items.
            shortfall.append((label, span, made))
        if made:
            print(f"  span {span:>4}  {label:<22} {made:>4}"
                  f"{'   SHORT' if made < a.per_cell else ''}", flush=True)

    # ---- coordinate families (tier F): radius of gyration, maximum displacement, total distance,
    # longest jump
    for span in [s for s in SPANS if s >= 32]:
        for fn in T.COORD_TASKS:
            run_cell(span, fn.__name__.replace("F_", ""), fn, coord=True)

    if a.only_coord:
        with open(a.out, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        import hashlib
        md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
        print(f"\n{a.out}\n  {len(items):,} coordinate items | md5 {md5}")
        hdr = {i["prompt_full"].split(chr(10))[0] for i in items}
        print(f"  distinct headers: {hdr}")
        return

    # ---- place families (tier H): coverage + controls
    # Per-family minimum span. A family is skipped below its own minimum because at shorter spans it
    # can almost never be satisfied, and the generator would spend its whole try budget rejecting
    # candidates.
    FN_MIN = {"H_recency": 8,
              # longest_stay starts at 32: in short records its answer barely varies (the baseline is .570 at
              # span 8 and .447 at 16; from 32 it is at most .32 and falls to .200 at 512).
              "H_longest_stay": 32, "H_transition": 16, "H_coreset": 16,
                            # absent and present both start at 16. H_present requires at least 10 records and H_absent does
                            # not, so at span 8 the probe would contain only absent items, a single-gold family. Both halves
                            # must exist at every span the probe occupies.
              "H_time_mode": 16, "H_day_distinct": 8, "H_absent": 16, "H_present": 16,
              "H_two_hop": 16}
    for span in SPANS:
        for fn in T.PLACE_TASKS:
            if span < FN_MIN[fn.__name__]:
                continue
            run_cell(span, fn.__name__.replace("H_", ""), fn, coord=False)

    # ---- THE CROSSED DIALS. W and p vary at FIXED span, so state volume and position are
    # identified separately from record length.
    for span in SPANS:
        for W in WINDOWS:
            if W >= span:
                continue          # the window must be a STRICT subset or it is just `distinct`
            run_cell(span, f"window_w{W}",
                     lambda s, r, sp, R, dd, tt, _W=W: T.H_window_distinct(s, r, sp, R, dd, tt, _W),
                     coord=False)
        for p in POSITIONS:
            if span < 32:
                continue          # position is not meaningfully resolvable below 32 records
            run_cell(span, f"retrieve_p{int(p * 100)}",
                     lambda s, r, sp, R, dd, tt, _p=p: T.H_retrieve_at(s, r, sp, R, dd, tt, _p),
                     coord=False)

    # ---------------------------------------------------------------- write + provenance
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()

    fam = collections.Counter(i["family"] for i in items)
    ch = sorted(i["n_chars"] for i in items)
    print(f"\nwrote {len(items):,} items -> {a.out}")
    print(f"  md5 {md5}")
    print(f"  families {len(fam)} | spans {sorted({i['span'] for i in items})}")
    print(f"  people   {len({i['uid'] for i in items}):,}")
    print(f"  n_chars  median {ch[len(ch)//2]:,} | p95 {ch[int(.95*len(ch))]:,} | max {ch[-1]:,}"
          f"  (cap {MAX_PROMPT_CHARS:,})")
    assert ch[-1] < MAX_PROMPT_CHARS, "the prompt cap binds; it would select items, not only shorten them"
    print(f"  drops    {dict(drops.most_common(6))}")
    if shortfall:
        print(f"\n  QUOTA SHORTFALL in {len(shortfall)} cells (reported, never refilled):")
        for lab, sp, n in shortfall[:20]:
            print(f"    {lab:<22} span {sp:>4}  {n}/{a.per_cell}")
    print("\nNEXT: check the cells for degeneracy before using this file "
          "(gauge/screen_degeneracy.py on a pilot run).")


if __name__ == "__main__":
    main()
