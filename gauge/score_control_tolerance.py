#!/usr/bin/env python3
"""Score the control families at exact match and within a factor of two, per model.

Exact accuracy alone cannot tell two different things apart on a continuous quantity. When
`sum_given` (hand the model a printed list of distances and ask for the total) scores 0.000 exact
for all five models, either

  (a) the models cannot add, which would contradict the program arm, where several models write
      aggregation programs that reproduce the reference on held-out records, or
  (b) the models add fine and cannot state an exact total, an arithmetic-precision limit.

Only a tolerance band separates the two. The factor-of-two rule is the one used in
error_taxonomy.py, so the controls and the geometric families are scored on one definition.

Input: results/tally_control.jsonl and results/v41ctrl2_<tag>.json for the five models.
Output: results/v41_control_tolerance.json.

Usage:
  python gauge/score_control_tolerance.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factqa import norm  # noqa: E402
import expect_models  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["phi35mini", "mistral7b", "llama8b", "gemma3_12b", "llama70b"]


def main():
    items_p = f"{ROOT}/results/tally_control.jsonl"
    if not os.path.exists(items_p):
        sys.exit(f"MISSING: {items_p}")
    items = [json.loads(l) for l in open(items_p)]
    fams = sorted({it["family"] for it in items})
    print(f"control items : {items_p}  ({len(items):,} items, families {fams})")

    found = [m for m in MODELS
             if os.path.exists(f"{ROOT}/results/v41ctrl2_{m}.json")]
    # A partial model set must fail loudly, not score quietly.
    expect_models.check(found, MODELS, label="control tolerance")

    out = {}
    for m in found:
        raw = json.load(open(f"{ROOT}/results/v41ctrl2_{m}.json"))["raw"]["full"]
        if len(raw) != len(items):
            sys.exit(f"{m}: {len(raw)} generations vs {len(items)} items")
        pred = [norm(raw[i], items[i]["family"], items[i].get("atype"))
                for i in range(len(items))]
        for fam in fams:
            ix = [i for i, it in enumerate(items) if it["family"] == fam]
            gold = [str(items[i]["a"]).lower() for i in ix]
            floor = collections.Counter(gold).most_common(1)[0][1] / len(ix)
            exact = close = unparsed = 0
            for j, i in enumerate(ix):
                p = pred[i]
                if p is None:
                    unparsed += 1
                    continue
                p = str(p).lower()
                if p == gold[j]:
                    exact += 1
                    continue
                try:
                    pv, g = float(p), float(gold[j])
                except ValueError:
                    unparsed += 1
                    continue
                # Same rule as error_taxonomy.py: within a factor of two, not exact.
                if g != 0 and not (pv / g > 2 or pv / max(g, 1e-9) < 0.5):
                    close += 1
            n = len(ix)
            out[f"{fam}|{m}"] = dict(
                n=n, floor=floor,
                exact=exact / n, within2x=(exact + close) / n,
                close_only=close / n, unparsed=unparsed / n)

    print(f"\n{'family':16s}{'model':12s}{'floor':>8s}{'exact':>9s}{'within-2x':>11s}{'unparsed':>10s}")
    for fam in fams:
        for m in found:
            r = out.get(f"{fam}|{m}")
            if r:
                print(f"{fam:16s}{m:12s}{r['floor']:8.3f}{r['exact']:9.3f}"
                      f"{r['within2x']:11.3f}{r['unparsed']:10.3f}")
        print()

    op = f"{ROOT}/results/v41_control_tolerance.json"
    json.dump(dict(items=items_p, n_items=len(items), models=found, cells=out),
              open(op, "w"), indent=1)
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
