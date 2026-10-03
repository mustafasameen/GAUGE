#!/usr/bin/env python3
"""Draw the questions of the sampling arm: 150 per family-by-length setting, five families.

For each setting of the chosen families, in sorted (family, record length) order, the script draws
150 questions with python's random.Random(seed), one generator for the whole run, from the
questions of that setting in the order of the items file. The questions are written in draw order.
With the defaults this is the 4,050-question file that the sampling arm ran on (md5
908d627df177663900672cdd53c2cbbd).

gauge/make_subset.py sorts each setting by person before it draws and sorts its output, so the same
seed gives a different set there. The GPT-4o subset comes from make_subset.py, and this subset comes
from this script.

Usage:
  python tools/draw_sampling_subset.py [--items results/questions.jsonl]
      [--out results/questions_sampling.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
import random

FAMILIES = "gyration_km,max_distance,total_distance,longest_jump,day_distinct"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", default="results/questions.jsonl", help="the full items file")
    ap.add_argument("--out", default="results/questions_sampling.jsonl", help="subset to write")
    ap.add_argument("--families", default=FAMILIES, help="comma-separated family list")
    ap.add_argument("--per-cell", type=int, default=150, help="questions per setting")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    want = {f.strip() for f in a.families.split(",")}
    rows = [json.loads(line) for line in open(a.items)]
    cells = collections.defaultdict(list)
    for r in rows:
        if r["family"] in want:
            cells[(r["family"], r["span"])].append(r)
    missing = want - {k[0] for k in cells}
    assert not missing, f"families not in the items file: {sorted(missing)}"
    rng = random.Random(a.seed)
    n = 0
    with open(a.out, "w") as fh:
        for key in sorted(cells):
            pool = cells[key]
            chosen = rng.sample(pool, a.per_cell) if a.per_cell < len(pool) else pool
            for r in chosen:
                fh.write(json.dumps(r) + "\n")
            n += len(chosen)
    print(f"{n} questions in {len(cells)} settings -> {a.out}")


if __name__ == "__main__":
    main()
