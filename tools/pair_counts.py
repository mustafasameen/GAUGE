#!/usr/bin/env python3
"""Count how model-family pairs behave across record length (Section 4.3 and the abstract).

A model-family pair clears a record length when the lower end of the 95% interval of its normalized
gain is above zero (results/family_cis.json, written by gauge/family_cis.py). Each pair falls
in one of three groups:

  through  clears at the longest record length that the family is generated at
  never    clears at no length
  stop     clears at some shorter length and not at the longest

The abstract's share is (through + never) / all pairs. The script also prints the same share for
the two hold-outs of Section 4.3 (without the four spatial-geometric families, and without them
and the two longest windowed-count families) and the record length at which each pair in the
"stop" group last clears.

Usage:
  python tools/pair_counts.py [--cis results/family_cis.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os

GEOMETRIC = {"gyration_km", "longest_jump", "max_distance", "total_distance"}
LONG_WINDOWS = {"window_distinct_w64", "window_distinct_w128"}


def load_pairs(path):
    """{(model, family): {span: clears}} from the cell entries of the family_cis output."""
    cells = json.load(open(path))
    pairs = collections.defaultdict(dict)
    for key, v in cells.items():
        if isinstance(v, dict) and "gain_ci" in v:
            model, family, span = key.split("|")
            pairs[(model, family)][int(span)] = v["gain_ci"][0] > 0
    return pairs


def classify(spans):
    """'through', 'never' or 'stop' for one pair, given {span: clears}."""
    clears = [spans[s] for s in sorted(spans)]
    if clears[-1]:
        return "through"
    return "never" if not any(clears) else "stop"


def share(pairs, keys):
    """(number of pairs, percent that are through or never) over the given pair keys."""
    kinds = [classify(pairs[k]) for k in keys]
    return len(kinds), round(100 * sum(k != "stop" for k in kinds) / len(kinds))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cis", default="results/family_cis.json")
    a = ap.parse_args()
    pairs = load_pairs(a.cis)
    kinds = {k: classify(v) for k, v in pairs.items()}
    n = len(kinds)
    count = collections.Counter(kinds.values())
    print(f"{n} model-family pairs ({os.path.basename(a.cis)})")
    for name in ("through", "never", "stop"):
        print(f"  {name:8s}{count[name]:>4d}  ({100 * count[name] / n:.0f}%)")
    print(f"  through or never: {share(pairs, list(pairs))[1]}%")
    n2, s2 = share(pairs, [k for k in pairs if k[1] not in GEOMETRIC])
    print(f"without the geometric families:               {s2}% of {n2} pairs")
    n3, s3 = share(pairs, [k for k in pairs if k[1] not in GEOMETRIC | LONG_WINDOWS])
    print(f"without them and the two longest windows:     {s3}% of {n3} pairs")
    last = {k: max(s for s, c in pairs[k].items() if c) for k, v in kinds.items() if v == "stop"}
    if last:
        print(f"last length cleared by the {len(last)} pairs that stop: "
              f"{min(last.values())} to {max(last.values())} rows")
    print("distinct places per day (day_distinct), last length cleared:")
    for (model, family), spans in sorted(pairs.items()):
        if family == "day_distinct":
            cleared = [s for s in sorted(spans) if spans[s]]
            print(f"  {model:12s}{max(cleared) if cleared else 'none'}")


if __name__ == "__main__":
    main()
