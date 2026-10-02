#!/usr/bin/env python3
"""Check an items file by calling the real consumer, not by re-implementing its checks.

The check imports `eval_factqa` and exercises its real symbols: the same NUMERIC, BUCKET and CATEG
sets and the same `norm()` parser. It asserts on the exact keys the scorer indexes, so if the
scorer changes, this check changes with it. It verifies that:
  - every item has the keys family, atype, a, q, prompt_full and prompt_blind;
  - the parser can score every family;
  - every gold answer round-trips through `norm()`;
  - no record contains a day on or after day 60 (no test-period leakage).

Usage:  python gauge/preflight_items.py results/tally_v41.jsonl
Exit 0 means the file is safe to spend GPU time on. Any failure prints what is wrong and exits
non-zero.
"""
from __future__ import annotations
import json, os, re, sys

REQUIRED = ("family", "atype", "a", "q", "prompt_full", "prompt_blind")
SPLIT_DAY = 60


def main(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_factqa as E                       # the actual consumer

    items = [json.loads(l) for l in open(path)]
    if not items:
        raise SystemExit(f"FAIL: {path} is empty")

    missing = sorted({k for k in REQUIRED if any(k not in i for i in items)})
    if missing:
        raise SystemExit(f"FAIL: items lack keys the eval indexes: {missing}")

    # the scorer's own guard, using the scorer's own tables
    unknown = sorted({i["family"] for i in items
                      if not i.get("atype") and i["family"] not in E.NUMERIC | E.BUCKET | E.CATEG})
    if unknown:
        raise SystemExit(f"FAIL: parser cannot score {unknown} — these would report 0.000 "
                         f"at 100% unparsed and look like a model failure")

    # The scorer's own parser, on a value shaped like the golds in this file. Check every item,
    # not a sample: a sampled check can pass a file that the consumer rejects (here the first items are
    # coordinate families, which parse fine, and unscoreable items could appear later). A preflight that
    # can pass what the consumer rejects is worse than none, because it certifies the file.
    for i in items:
        got = E.norm(f"Answer: {i['a']}", i["family"], i.get("atype"))
        if got != str(i["a"]).lower():
            raise SystemExit(f"FAIL: norm() cannot round-trip a gold for {i['family']}: "
                             f"{i['a']!r} -> {got!r}")

    mx = 0
    for i in items:
        ds = [int(v) for v in re.findall(r"^d(\d+) ", i["prompt_full"], re.M)]
        if ds:
            mx = max(mx, max(ds))
    if mx >= SPLIT_DAY:
        raise SystemExit(f"FAIL: LEAK — record contains day {mx} >= {SPLIT_DAY}")

    fams = {i["family"] for i in items}
    tiers = {i.get("tier", "?") for i in items}
    spans = sorted({i.get("span") for i in items if i.get("span")})
    print(f"PREFLIGHT OK via the real consumer | {len(items):,} items | {len(fams)} families | "
          f"{len(tiers)} tiers {sorted(tiers)} | spans {spans} | max day {mx} < {SPLIT_DAY} | "
          f"golds round-trip through norm()")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/tally_v3.jsonl")
