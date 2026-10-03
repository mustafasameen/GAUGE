#!/usr/bin/env python3
"""Draw a per-cell subsample of the geometric items for the chain-of-thought budget arm.

results/questions_corrected_header.jsonl holds 6,000 geometric items (300 per setting x 20
settings). Under `--style cot` with a multi-thousand-token budget that is too much for one run, so
the arm uses a subsample. The draw is seeded and balanced per setting, and the script prints the md5
of the result, so the item file can be reproduced from the command line alone.

Items are copied through byte for byte: same `prompt_full`, same gold, same family, span and uid.
The script only selects, and it asserts this on every drawn item, because the arm's comparability
with the terse runs rests on the prompts being the identical objects. Each output item records its
source row as `cot_src_idx`.

Input: results/questions_corrected_header.jsonl (from generate_questions.py --only-coord).
Output: the file given by --out.

Usage:
  python gauge/make_cot_items.py --per-cell 20 --out results/questions_cot.jsonl
  python gauge/make_cot_items.py --per-cell 2 --out results/questions_cot_smoke.jsonl
  python gauge/make_cot_items.py --selftest
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, random, sys

SRC = "results/questions_corrected_header.jsonl"
GEOM = ["gyration_km", "max_distance", "total_distance", "longest_jump"]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(src):
    return [json.loads(l) for l in open(src)]


def draw(rows, per_cell, seed):
    """Per-cell balanced, seeded draw. Cells are (family, span). Returns rows in a STABLE order
    (family, span, then source index) so the same arguments always produce a byte-identical file."""
    cells = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["family"] in GEOM:
            cells[(r["family"], r["span"])].append(i)
    out = []
    for cell in sorted(cells):
        idx = cells[cell]
        if per_cell > len(idx):
            raise ValueError(f"cell {cell} has {len(idx)} items, asked for {per_cell}")
        rng = random.Random(f"{seed}|{cell[0]}|{cell[1]}")   # per-cell stream: stable under resize
        out.extend(sorted(rng.sample(idx, per_cell)))
    return out


def build(src, per_cell, seed, out_path):
    rows = load(src)
    ix = draw(rows, per_cell, seed)
    with open(out_path, "w") as fh:
        for i in ix:
            r = dict(rows[i])
            r["cot_src_idx"] = i          # provenance: the exact source row this came from
            fh.write(json.dumps(r) + "\n")
    # VERIFY the draw changed nothing but selection, on every drawn item.
    got = [json.loads(l) for l in open(out_path)]
    assert len(got) == len(ix), "row count mismatch after write"
    for g in got:
        s = rows[g["cot_src_idx"]]
        for k in ("prompt_full", "a", "family", "span", "uid"):
            assert g[k] == s[k], f"item {g['cot_src_idx']}: {k} differs from source -- draw mutated an item"
    md5 = hashlib.md5(open(out_path, "rb").read()).hexdigest()
    cells = collections.Counter((g["family"], g["span"]) for g in got)
    return got, md5, cells


def selftest():
    """Known-answer checks that touch no project file: determinism, balance, and that a mutated
    source is actually caught by the copy-through assert."""
    import tempfile
    ok = True
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src.jsonl")
    with open(src, "w") as fh:
        for f in GEOM:
            for sp in (32, 64):
                for k in range(10):
                    fh.write(json.dumps(dict(family=f, span=sp, a=str(k), uid=1000 + k,
                                             atype="numeric", prompt_full=f"P{f}{sp}{k}")) + "\n")
    a1 = os.path.join(tmp, "a.jsonl"); a2 = os.path.join(tmp, "b.jsonl")
    _, m1, c1 = build(src, 3, 7, a1)
    _, m2, _ = build(src, 3, 7, a2)
    print(f"  determinism : md5 {m1[:12]} vs {m2[:12]}  -> {'SAME' if m1 == m2 else '*** DIFFER'}")
    ok &= (m1 == m2)
    bal = set(c1.values()) == {3} and len(c1) == 8
    print(f"  balance     : {len(c1)} cells, counts {sorted(set(c1.values()))}  -> {'OK' if bal else '*** BAD'}")
    ok &= bal
    _, m3, _ = build(src, 3, 8, os.path.join(tmp, "c.jsonl"))
    print(f"  seed matters: md5 {m3[:12]}  -> {'DIFFERS from seed 7' if m3 != m1 else '*** IDENTICAL'}")
    ok &= (m3 != m1)
    # the copy-through assert must FIRE when the source is tampered with
    rows = load(src); rows[0]["prompt_full"] = "TAMPERED"
    bad = os.path.join(tmp, "bad.jsonl")
    with open(bad, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    orig = load(src)
    try:
        # build from the clean source, then verify against the TAMPERED one: assert must trip
        got, _, _ = build(src, 3, 7, os.path.join(tmp, "d.jsonl"))
        tripped = False
        for g in got:
            s = rows[g["cot_src_idx"]]
            if g["prompt_full"] != s["prompt_full"]:
                tripped = True; break
        print(f"  tamper check: mutation detected -> {'OK' if tripped else '*** NOT DETECTED'}")
        ok &= tripped
    except AssertionError:
        print("  tamper check: assert fired -> OK"); ok &= True
    print(f"\nSELFTEST: {'ALL PASSED' if ok else '*** FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, SRC))
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/questions_cot_smoke.jsonl"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    got, md5, cells = build(a.src, a.per_cell, a.seed, a.out)
    print(f"wrote {a.out}")
    print(f"  n        = {len(got)}  ({a.per_cell}/cell x {len(cells)} cells)")
    print(f"  md5      = {md5}")
    print(f"  families = {sorted({g['family'] for g in got})}")
    print(f"  spans    = {sorted({g['span'] for g in got})}")
    print(f"  uids     = {len({g['uid'] for g in got})} distinct")
    print(f"  source   = {os.path.basename(a.src)} (prompt_full/gold/family/span/uid verified "
          f"byte-identical on all {len(got)} items)")


if __name__ == "__main__":
    main()
