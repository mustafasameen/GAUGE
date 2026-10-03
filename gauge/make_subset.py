#!/usr/bin/env python3
"""Build a stratified subset of the question set, balanced over family x record length.

Selection is deterministic given --seed and every cell is sampled separately, so a model run on the
subset and the full-set runs can be compared item by item. Stratification is over (family, span)
because those are the two axes every claim is read along. Within a cell the script takes a seeded
random sample (python's random.Random), after sorting the cell's items by person and by the start
of the question text, and it writes the subset sorted by family, length and person. A cell smaller
than the quota contributes all of its items, and the shortfall is reported and never absorbed
silently. A cell that vanished entirely would silently narrow the benchmark, so the script asserts
that none does.

With the arguments below it draws the 960 questions that the GPT-4o arm ran on (30 per setting, six
families). tools/subset_to_indices.py turns the output into the index file that the GPT-4o scripts
read. The sampling arm's subset is drawn without the sorting, by tools/draw_sampling_subset.py.

Input: an items file (JSON lines). Output: the subset, in the same format, and a summary on screen.

Usage:
  python gauge/make_subset.py --items results/questions.jsonl --per-cell 30 --seed 0
      --families gyration_km,max_distance,total_distance,longest_jump,day_distinct,retrieve_p50
      --out results/questions_frontier.jsonl
"""
import argparse
import collections
import hashlib
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--frac", type=float, help="fraction of each (family, span) cell to keep")
    group.add_argument("--per-cell", type=int, help="fixed count per (family, span) cell")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--families", default=None,
                    help="comma-separated family list; default is all")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.items)]
    if a.families:
        want = {f.strip() for f in a.families.split(",")}
        have = {r["family"] for r in rows}
        missing = want - have
        assert not missing, f"families not in the items file: {sorted(missing)}"
        rows = [r for r in rows if r["family"] in want]
        print(f"filtered to {len(want)} families: {len(rows)} items")
    print(f"source {a.items}: {len(rows)} items")

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["family"], r["span"])].append(r)
    print(f"strata: {len(cells)} (family x span) cells")

    rng = random.Random(a.seed)
    keep, short = [], []
    for key in sorted(cells):
        # uid is str in most families and int in a few, so coerce before sorting
        pool = sorted(cells[key], key=lambda r: (str(r.get("uid", "")), str(r.get("q", ""))[:32]))
        want = a.per_cell if a.per_cell else max(1, round(len(pool) * a.frac))
        if want >= len(pool):
            short.append((key, want, len(pool)))
            keep.extend(pool)
        else:
            keep.extend(rng.sample(pool, want))

    keep.sort(key=lambda r: (r["family"], int(r["span"]), str(r.get("uid", ""))))
    with open(a.out, "w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")

    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    fams = {r["family"] for r in keep}
    spans = sorted({int(r["span"]) for r in keep})
    print(f"\nwrote {a.out}")
    print(f"  items      {len(keep)}  ({len(keep)/len(rows):.1%} of source)")
    print(f"  families   {len(fams)}")
    print(f"  spans      {spans}")
    print(f"  md5        {md5}")
    print(f"  seed       {a.seed}")

    # Publish what the sampler could not deliver; never absorb it silently.
    if short:
        print(f"\n  {len(short)} cells smaller than the quota (took all available):")
        for key, want, have in short[:12]:
            print(f"    {key[0]}|span{key[1]}: wanted {want}, had {have}")
        if len(short) > 12:
            print(f"    ... and {len(short)-12} more")
    else:
        print("\n  every cell met its quota")

    # a cell that vanishes entirely would silently narrow the instrument
    lost = set(cells) - {(r["family"], r["span"]) for r in keep}
    print(f"  cells dropped entirely: {len(lost)}")
    assert not lost, f"FATAL: subset lost cells {sorted(lost)[:5]}"


if __name__ == "__main__":
    main()
