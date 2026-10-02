#!/usr/bin/env python3
"""Build the stripped rung of the framing ladder: bare x, y rows with no day, timeslot or mobility words.

The ladder removes one thing at each step:
  full record      "a person's location record", with timestamped rows
  relabelled       "a set of numbered points", with the same timestamped rows (make_nomobility.py)
  stripped         this script: bare x, y rows, with no day and no timeslot
  two bare points  the decomposition control (pair_distance in tally_control.py)
The relabelled to stripped step removes the timestamped record structure while the entity label is
already neutral.

What changes, relative to prompt_nomobility, and nothing else:
  intro  "Here is a set of numbered points. Each line is day, timeslot, x, y." becomes
         "Here is a set of points. Each line is x, y."
         ("numbered" goes with the day and timeslot fields, which are the numbering.)
  rows   "d37 t18 x-341 y-387" becomes "x-341 y-387" (same points, same order)
  total_distance question only: "each consecutive pair of records" becomes "each consecutive pair
         of points". With the timestamps gone the rows are no longer records, and this word is the
         question's reference to the structure that was removed. The other three questions are
         untouched.
The gold is unchanged: all four descriptors depend only on the points and their order, and line
order is preserved exactly. check() verifies these invariants for every item, and `--selftest`
shows that it detects planted violations.

Input: results/tally_nomobility.jsonl (from make_nomobility.py).
Output: results/tally_stripped.jsonl.

Usage:
  python gauge/make_stripped.py
  python gauge/make_stripped.py --selftest
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "results/tally_nomobility.jsonl")
OUT = os.path.join(ROOT, "results/tally_stripped.jsonl")
INTRO_OLD = "Here is a set of numbered points. Each line is day, timeslot, x, y."
INTRO_NEW = "Here is a set of points. Each line is x, y."
ROW = re.compile(r"d(\d+) t(\d+) x(-?\d+) y(-?\d+)")
TD_OLD, TD_NEW = "each consecutive pair of records", "each consecutive pair of points"
FORBIDDEN = re.compile(r"\b(day|days|timeslot|timeslots|record|records|person|person's|"
                       r"travelled|visited|moved|location)\b|^d\d+ t\d+ ", re.I | re.M)


def strip_prompt(p: str, family: str) -> str:
    blocks = p.split("\n\n")
    assert len(blocks) == 3, f"expected intro/body/question, got {len(blocks)} blocks"
    intro, body, q = blocks
    assert intro == INTRO_OLD, f"intro changed upstream: {intro!r}"
    new_rows = []
    for line in body.split("\n"):
        m = ROW.fullmatch(line)
        assert m, f"unexpected row: {line!r}"
        new_rows.append(f"x{m.group(3)} y{m.group(4)}")
    if family == "total_distance":
        assert q.count(TD_OLD) == 1, "total_distance question wording changed upstream"
        q = q.replace(TD_OLD, TD_NEW)
    else:
        assert TD_OLD not in q
    return "\n\n".join([INTRO_NEW, "\n".join(new_rows), q])


def points(body: str, stripped: bool):
    pat = r"x(-?\d+) y(-?\d+)"
    return [(int(a), int(b)) for a, b in re.findall(pat, body)]


def check(item) -> list[str]:
    """Every invariant the arm rests on, for one item. Returns the list of violations."""
    bad = []
    s, n = item["prompt_stripped"], item["prompt_nomobility"]
    sb, nb = s.split("\n\n"), n.split("\n\n")
    if points(sb[1], True) != points(nb[1], False):
        bad.append("point sequence differs")
    if FORBIDDEN.search(s):
        bad.append(f"forbidden token: {FORBIDDEN.search(s).group(0)!r}")
    q_expected = nb[2].replace(TD_OLD, TD_NEW) if item["family"] == "total_distance" else nb[2]
    if sb[2] != q_expected:
        bad.append("question differs beyond the total_distance substitution")
    if sb[0] != INTRO_NEW:
        bad.append("intro wrong")
    return bad


def selftest() -> bool:
    ok = True
    good = {"family": "longest_jump", "prompt_nomobility":
            INTRO_OLD + "\n\nd1 t2 x3 y4\nd1 t3 x5 y-6\n\nQuestion: between two consecutive points?"}
    good["prompt_stripped"] = strip_prompt(good["prompt_nomobility"], "longest_jump")
    r = check(good); print(f"  clean item   -> violations {r}  {'OK' if not r else '*** FAIL'}"); ok &= not r
    for label, mutate in [("day leak", lambda s: s.replace("x3 y4", "d1 t2 x3 y4")),
                          ("word leak", lambda s: s + " for this person"),
                          ("point moved", lambda s: s.replace("x5 y-6", "x5 y-7"))]:
        bad = dict(good); bad["prompt_stripped"] = mutate(good["prompt_stripped"])
        r = check(bad); fired = bool(r)
        print(f"  {label:<12} -> {'fired' if fired else 'NOT DETECTED'}: {r}  {'OK' if fired else '*** FAIL'}")
        ok &= fired
    print(f"SELFTEST: {'PASSED' if ok else '*** FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    rows = [json.loads(l) for l in open(SRC)]
    for r in rows:
        r["prompt_stripped"] = strip_prompt(r["prompt_nomobility"], r["family"])
    viol = [(i, v) for i, r in enumerate(rows) for v in check(r)]
    if viol:
        sys.exit(f"{len(viol)} violations, first: {viol[:3]}")
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
    print(f"wrote {OUT}\n  n={len(rows)}  md5={md5}\n  invariants hold on all {len(rows)} items: "
          f"same points in same order, no day/timeslot/record/mobility tokens, question changed "
          f"only by the total_distance substitution, gold untouched")


if __name__ == "__main__":
    main()
