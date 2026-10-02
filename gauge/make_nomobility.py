#!/usr/bin/env python3
"""Build the framing arm: the same geometric questions with and without mobility language.

For the four spatial-geometric families (gyration_km, max_distance, total_distance, longest_jump)
each item carries two prompts: `prompt_full` (mobility framing) and `prompt_nomobility` (neutral
framing). The arm asks whether a failure to compute these descriptors from a location record is a
failure about mobility or about numerical geometry in general. In every other geometric item the
entity is "a person's location record", so mobility framing and geometric computation are confounded.

Exactly one factor varies. The coordinates, their order, the record length, the descriptor's name,
units and formula, the gold answer, the parser and the scoring are byte-identical. Only the entity
framing changes:
  1. the wrapper sentence "Here is a person's location record." becomes "Here is a set of numbered
     points." (the schema line "Each line is day, timeslot, x, y." is untouched);
  2. one clause in the question that names the person as the mover or visitor of the points is
     swapped for a clause that names the points as a set (ENTITY_SWAP below). For the radius of
     gyration, "the visited points" becomes "the points", which carries no mathematical content.
check_pair() asserts for every item that the gold is unchanged, that the record body is
byte-identical, that the descriptor, units and formula phrases are present on both sides, and that
"person" and "location" appear only on the mobility side.

The gold is never recomputed here. Each item's "a" field was written upstream by the F_* functions
in tasks_v41.py from the transformed points, with no notion of framing. This script copies the item
(`dict(it)`) and adds `prompt_nomobility`.

Each output item records its position in the source file as `geomfix_idx`. Run eval_factqa.py with
`--conds full,nomobility` over the output to generate both conditions in one job.

Input: results/tally_v41_geomfix.jsonl (from tally_v41.py --only-coord).
Output: results/tally_nomobility.jsonl (30 items per setting by default; the paper's arm uses 75,
which gives 1,500 items).

Usage:
  python gauge/make_nomobility.py --peek 3        (dry run: prints pairs and writes nothing)
  python gauge/make_nomobility.py --per-cell 75 --out results/tally_nomobility.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "results", "tally_v41_geomfix.jsonl")
GEOM = {"gyration_km", "max_distance", "total_distance", "longest_jump"}

# ---- factor 2: the one clause inside `q` that names the person as the mover or visitor of the
# points. (old, new): `old` is copied verbatim from the `q` text in tasks_v41.py, not reconstructed,
# so a later wording change there fails loudly (AssertionError) instead of silently producing a
# mismatched pair.
ENTITY_SWAP = {
    "gyration_km":    ("this person's locations", "this set of points"),
    "max_distance":   ("any two locations this person visited", "any two points in this set"),
    "total_distance": ("this person travelled", "covered by this set of points"),
    "longest_jump":   ("this person moved between two consecutive records",
                        "between two consecutive points in this set"),
}

# ---- the technical content that must NOT move. Checked below, not just promised.
INVARIANT_PHRASES = {
    "gyration_km":    ["radius of gyration", "root-mean-square distance", "kilometres"],
    "max_distance":   ["greatest distance", "kilometres"],
    "total_distance": ["TOTAL distance",
                        "straight-line distance between each consecutive pair of records",
                        "kilometres"],
    "longest_jump":   ["LARGEST straight-line distance", "kilometres"],
}

MOB_INTRO = "Here is a person's location record."
NOMOB_INTRO = "Here is a set of numbered points."


# Residual mobility verbs that survive ENTITY_SWAP because they sit inside the descriptor's
# definition sentence and not in its entity clause. A check for person, travelled, visited and moved
# across every item catches them (the entity swap alone leaves "the visited points" in the radius of
# gyration definition). Removing them carries no mathematical content ("the root-mean-square
# distance of the points from their mean position" is the same definition), so it completes the
# single-factor manipulation instead of adding a second factor. The mobility arm keeps the original
# wording.
DEFINITION_DEMOBILISE = {
    "gyration_km": [("the visited points", "the points")],
}
MOBILITY_WORDS = r"\b(person|person's|travelled|visited|moved)\b"


def neutralize_q(family, q):
    old, new = ENTITY_SWAP[family]
    assert old in q, f"{family}: expected clause not found in q -- tasks_v41.py wording changed"
    out = q.replace(old, new, 1)
    assert out != q, f"{family}: substitution did not fire"
    for d_old, d_new in DEFINITION_DEMOBILISE.get(family, []):
        assert d_old in out, f"{family}: definition clause {d_old!r} not found -- wording changed"
        out = out.replace(d_old, d_new)
    assert not re.search(MOBILITY_WORDS, out, re.I), \
        f"{family}: mobility language survives neutralisation: {out!r}"
    return out


def render_nomobility(head, body, neutral_q):
    return (f"{NOMOB_INTRO} Each line is {head}.\n\n{body}\n\n"
            f"Question: {neutral_q}\nAnswer with the value only.")


def build_pair(it):
    """Return (item with prompt_nomobility added, body string). The mobility side -- prompt_full,
    prompt_blind, "a", and every other key -- passes through untouched via dict(it)."""
    assert it["prompt_full"].startswith(MOB_INTRO), \
        "source item's header text has changed -- check tally_v41.py::emit()"
    parts = it["prompt_full"].split("\n\n")
    head = parts[0].split("Each line is ")[1].rstrip(".")
    body = parts[1]
    neutral_q = neutralize_q(it["family"], it["q"])
    out = dict(it)
    out["prompt_nomobility"] = render_nomobility(head, body, neutral_q)
    return out, body


def check_pair(it, out, body):
    """Every assert here is the machine-checkable form of one part of the single-factor claim."""
    fam = it["family"]
    mob, nomob = out["prompt_full"], out["prompt_nomobility"]
    assert out["a"] == it["a"], f"{fam}: gold changed (must never happen)"
    nomob_body = nomob.split("\n\n")[1]
    assert nomob_body == body, f"{fam}: record body is not byte-identical across the pair"
    nums_mob = re.findall(r"-?\d+", body)
    nums_nomob = re.findall(r"-?\d+", nomob_body)
    assert nums_mob == nums_nomob, f"{fam}: coordinate digits diverged"
    for phrase in INVARIANT_PHRASES[fam]:
        assert phrase in mob and phrase in nomob, \
            f"{fam}: invariant phrase {phrase!r} missing from one side of the pair"
    assert re.search(r"\bperson\b", mob, re.I), f"{fam}: mobility prompt lost its own framing"
    assert not re.search(r"\bperson\b", nomob, re.I), f"{fam}: 'person' leaked into non-mobility prompt"
    assert not re.search(r"\blocations?\b", nomob, re.I), f"{fam}: 'location' leaked into non-mobility prompt"


def stub_generate(gold):
    """Stands in for the GPU call. NO MODEL IS LOADED OR CALLED. Returns a marker-anchored string
    so the REAL parser (eval_factqa.norm) has something to extract, proving the
    prompt -> generation -> parser -> compare-to-gold plumbing is condition-blind without spending
    any compute on it."""
    return f"(stub, no model called) Answer: {gold}"


def peek(base, n):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_factqa import norm  # the SAME parser the GPU harness uses -- imported, not reimplemented

    print(f"\n=== PEEK: {n} matched pairs, stub generation, no model loaded ===\n")
    shown_fams = set()
    shown = 0
    for geomfix_idx, it in base:
        if it["family"] in shown_fams or shown >= n:
            continue
        out, body = build_pair(it)
        check_pair(it, out, body)
        shown_fams.add(it["family"])
        shown += 1
        rows = body.split("\n")
        body_excerpt = "\n".join(rows[:2] + [f"  ... ({len(rows) - 3} rows omitted; asserted "
                                              f"byte-identical across the pair, not just displayed "
                                              f"the same) ..."] + rows[-1:])
        print(f"--- pair {shown}/{n}  family={it['family']}  span={it['span']}  uid={it['uid']}  "
              f"geomfix_idx={geomfix_idx} ---")
        print(f"[MOBILITY]\n{MOB_INTRO} Each line is "
              f"{out['prompt_full'].split(chr(10))[0].split('Each line is ')[1]}\n\n{body_excerpt}\n\n"
              f"Question: {it['q']}\nAnswer with the value only.")
        print(f"\n[NON-MOBILITY]\n{NOMOB_INTRO} Each line is "
              f"{out['prompt_full'].split(chr(10))[0].split('Each line is ')[1]}\n\n{body_excerpt}\n\n"
              f"Question: {neutralize_q(it['family'], it['q'])}\nAnswer with the value only.")
        gold_mobility = it["a"]
        gold_nonmobility = out["a"]
        assert gold_mobility == gold_nonmobility, "GOLD MISMATCH ACROSS THE PAIR"
        print(f"\ngold (mobility)     = {gold_mobility!r}")
        print(f"gold (non-mobility) = {gold_nonmobility!r}")
        print("ASSERT gold_mobility == gold_nonmobility ... OK")
        gm = stub_generate(gold_mobility)
        gn = stub_generate(gold_nonmobility)
        pm = norm(gm, it["family"], it.get("atype"))
        pn = norm(gn, it["family"], it.get("atype"))
        print(f"stub generation (mobility)     -> {gm!r} -> norm() -> {pm!r}")
        print(f"stub generation (non-mobility) -> {gn!r} -> norm() -> {pn!r}")
        assert pm == pn == str(gold_mobility).lower(), "PARSER DISAGREES ACROSS CONDITIONS"
        print("ASSERT norm(stub_mobility) == norm(stub_nonmobility) == gold ... OK\n")
    assert shown == n, f"only found {shown} distinct families to peek, wanted {n}"
    print(f"=== PEEK OK: {n}/{n} pairs printed, all asserts passed, no GPU touched ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=SRC)
    ap.add_argument("--per-cell", type=int, default=30,
                     help="items per (family, span) cell, deterministic first-N "
                          "(the convention of make_templates.py). 300 = the full corrected-header set.")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tally_nomobility.jsonl"))
    ap.add_argument("--peek", type=int, default=0,
                     help="print N matched pairs with a stubbed generation and exit; writes "
                          "nothing.")
    a = ap.parse_args()

    if not os.path.exists(a.items):
        sys.exit(f"MISSING: {a.items} -- run: python gauge/tally_v41.py --only-coord "
                  f"(the arm is built on the corrected-header coordinate items)")

    src = [json.loads(l) for l in open(a.items)]
    cells = collections.defaultdict(list)
    for idx, it in enumerate(src):
        if it["family"] in GEOM:
            cells[(it["family"], it["span"])].append((idx, it))
    if not cells:
        sys.exit(f"no geometric-family items in {a.items} -- families present: "
                  f"{sorted({it['family'] for it in src})}")

    base = []
    for k in sorted(cells):
        base.extend(cells[k][:a.per_cell])
    print(f"{len(base):,} base items over {len(cells)} (family, span) cells "
          f"| per-cell requested {a.per_cell}", flush=True)
    short = [(k, len(v)) for k, v in cells.items() if len(v) < a.per_cell]
    if short:
        print(f"  {len(short)} cells smaller than requested (took all available):")
        for k, n in sorted(short):
            print(f"    {k[0]:<16} span {k[1]:>4}  {n}/{a.per_cell}")

    if a.peek:
        peek(base, a.peek)
        return

    out_items = []
    for geomfix_idx, it in base:
        out, body = build_pair(it)
        # Position in tally_v41_geomfix.jsonl equals the position in raw["full"] of every
        # results/v41geo_*.json (the alignment rule that score_geomfix.py uses), so the mobility side of this
        # arm can be scored from generations that already exist.
        out["geomfix_idx"] = geomfix_idx
        check_pair(it, out, body)
        out_items.append(out)

    print(f"invariant OK on all {len(out_items):,} pairs: gold unchanged, record byte-identical, "
          f"descriptor/units/formula unchanged, 'person'/'location' present only on the mobility "
          f"side", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for o in out_items:
            f.write(json.dumps(o) + "\n")
    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    fam_span = collections.Counter((o["family"], o["span"]) for o in out_items)
    print(f"\nwrote {a.out}\n  {len(out_items):,} items | md5 {md5}")
    print(f"  cells {len(fam_span)} | families {sorted({o['family'] for o in out_items})}")
    print("\nNEXT (not run by this script -- no GPU/cluster/network touched here):")
    print("  python gauge/eval_factqa.py --items " + a.out + " --model <hf-id> \\")
    print("      --style terse --conds full,nomobility --max-new 24 --max-len 32768 --out <path>")


if __name__ == "__main__":
    main()
