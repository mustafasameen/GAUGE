#!/usr/bin/env python3
"""Write the pool of real records that gauge/program_equivalence.py draws its probes from.

program_equivalence.py checks every model-written expression against 50 real records that the
expression was not written for. The pool is the list of point sets of the program-arm items: the
points of every `tool_gyration` item of results/questions_tool.jsonl, in file order, as a JSON list
of {"pts": [[x, y], ...]}. (The four geometric families share the same 300 records, so one family is
enough.) With this pool the script gives the same counts of correct and equivalent programs as the
pool behind the paper's table.

Usage:
  python tools/make_points_pool.py [--items results/questions_tool.jsonl] [--out results/_pts_pool.json]
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", default="results/questions_tool.jsonl", help="program-arm items file")
    ap.add_argument("--out", default="results/_pts_pool.json", help="where to write the pool")
    a = ap.parse_args()
    pool = []
    with open(a.items, encoding="utf-8") as fh:
        for line in fh:
            it = json.loads(line)
            if it["family"] == "tool_gyration":
                pool.append({"pts": it["pts"]})
    if not pool:
        print(f"no tool_gyration items in {a.items}", file=sys.stderr)
        return 1
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(pool, fh)
    print(f"wrote {len(pool)} records -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
