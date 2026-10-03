#!/usr/bin/env python3
"""Turn a subset items file into the index file that the GPT-4o scripts read.

gauge/make_subset.py writes the chosen questions as a JSON-lines items file. gauge/run_frontier.py
and gauge/score_frontier.py take instead a JSON file {"indices": [...]}, the positions of the chosen
questions in the full items file (results/questions.jsonl). This script finds each subset question
in the full file and writes the sorted positions.

Two ways to find a question (--match):
  key      the first question of the full file with the same family, record length, person and
           question text. This is how the index file behind the paper's GPT-4o arm was made. For four
           people who have two questions of the same family and length it picks the earlier of the
           two, which can differ from the question that make_subset.py chose.
  content  the question whose every field is equal (the exact question that make_subset.py chose).

Usage:
  python tools/subset_to_indices.py --items results/questions.jsonl --subset results/questions_frontier.jsonl
      --out results/subset_allspans.json --note "per-cell 30, seed 0, families=6, ALL spans"
"""
from __future__ import annotations

import argparse
import json


def key_of(r):
    return (r["family"], r["span"], str(r["uid"]), r["q"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", default="results/questions.jsonl", help="the full items file")
    ap.add_argument("--subset", required=True, help="items file written by gauge/make_subset.py")
    ap.add_argument("--out", required=True, help="index file to write")
    ap.add_argument("--match", choices=["key", "content"], default="key",
                    help="how to find a subset question in the full file (default: key)")
    ap.add_argument("--note", default="", help="free text stored in the index file")
    a = ap.parse_args()
    where = {}
    for i, line in enumerate(open(a.items)):
        r = json.loads(line)
        k = key_of(r) if a.match == "key" else json.dumps(r, sort_keys=True)
        where.setdefault(k, i)
    idx = []
    for line in open(a.subset):
        r = json.loads(line)
        k = key_of(r) if a.match == "key" else json.dumps(r, sort_keys=True)
        if k not in where:
            raise SystemExit("a subset question is not in the items file")
        idx.append(where[k])
    if len(set(idx)) != len(idx):
        raise SystemExit("the subset maps two questions to one position")
    idx.sort()
    with open(a.out, "w") as fh:
        json.dump({"indices": idx, "note": a.note}, fh)
    print(f"{len(idx)} questions -> {a.out}")


if __name__ == "__main__":
    main()
