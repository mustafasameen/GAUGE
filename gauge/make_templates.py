#!/usr/bin/env python3
"""Build the prompt-wording variants: is the geometric result an artifact of how the question is asked?

For the geometric items it writes five prompt templates over the same records. The record data
(values, order and count) is identical across all five; only the scaffold changes.

  T0  incumbent        the benchmark prompt
  T1  separator        "Question:" becomes "Question --", and the answer directive likewise
  T2  joiner and case  the blank line between blocks becomes a single newline, and all descriptors
                       are lowercased
  T3  serialisation    bare `d37 t18 x-341 y-387` becomes labelled `day=37, slot=18, x=-341, y=-387`
                       (a re-serialisation that adds no information; no line numbers are introduced)
  T4  paraphrase       the instruction and the question are reworded; the quantity requested is the
                       same

The design follows two studies of prompt sensitivity. Sclar et al. (2023, arXiv 2310.11324) vary
format with the descriptor words frozen, and report performance spread (max - min) over formats;
their smallest configuration has 10 formats and they call it a lower bound. Mizrahi et al. (TACL
2024, doi 10.1162/tacl_a_00681) vary the wording and report the average and the maximum over
templates. With five templates, the spread reported here is a lower bound as well. Because the
headline result is a failure, a wording bias that flatters the original template makes the failure
conservative. That argument would not hold for a positive result.

Input: results/questions_corrected_header.jsonl, the first 50 items of each geometric family and
length.
Output: results/questions_templates.jsonl (5,000 items: 20 settings x 50 items x 5 templates).

Usage:
  python gauge/make_templates.py
"""
import collections
import hashlib
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GEOM = {"gyration_km", "max_distance", "total_distance", "longest_jump"}
PER_CELL = 50


def relabel_body(body):
    """T3: re-serialise each record line without adding or removing any information."""
    out = []
    for line in body.split("\n"):
        m = re.match(r"d(\d+)\s+t(\d+)\s+x(-?\d+)\s+y(-?\d+)", line.strip())
        out.append(f"day={m.group(1)}, slot={m.group(2)}, x={m.group(3)}, y={m.group(4)}"
                   if m else line)
    return "\n".join(out)


def paraphrase(q):
    """T4: Mizrahi's axis. Reword the framing, never the quantity asked for."""
    r = q
    r = r.replace("Treating each unit on the x and y axes as 500 metres,",
                  "Each unit on the x and y axes represents 500 metres.")
    r = r.replace("what is the radius of gyration of this person's locations, in kilometres?",
                  "How large is this person's radius of gyration, measured in kilometres?")
    r = r.replace("what is the greatest distance between any two locations this person visited, "
                  "in kilometres?",
                  "How far apart, in kilometres, are the two most distant locations this person "
                  "visited?")
    r = r.replace("what is the TOTAL distance this person travelled, adding up the straight-line "
                  "distance between each consecutive pair of records?",
                  "Adding together the straight-line distance between each consecutive pair of "
                  "records, how far did this person travel in total?")
    r = r.replace("what is the LARGEST straight-line distance this person moved between two "
                  "consecutive records, in kilometres?",
                  "Between which two consecutive records did this person move furthest, and how "
                  "many kilometres was that move?")
    r = r.replace("Answer with a whole number of kilometres.",
                  "Give your answer as a whole number of kilometres.")
    return r


TEMPLATES = {
    "T0_incumbent": lambda h, b, q: (
        f"Here is a person's location record. Each line is {h}.\n\n{b}\n\n"
        f"Question: {q}\nAnswer with the value only."),
    "T1_separator": lambda h, b, q: (
        f"Here is a person's location record. Each line is {h}.\n\n{b}\n\n"
        f"Question -- {q}\nAnswer -- give the value only."),
    "T2_joiner_case": lambda h, b, q: (
        f"here is a person's location record. each line is {h}.\n{b}\n"
        f"question: {q.lower()}\nanswer with the value only."),
    "T3_serialisation": lambda h, b, q: (
        f"Here is a person's location record. Each line is {h}.\n\n{relabel_body(b)}\n\n"
        f"Question: {q}\nAnswer with the value only."),
    "T4_paraphrase": lambda h, b, q: (
        f"Below is one person's movement history. Each row gives the {h}.\n\n{b}\n\n"
        f"{paraphrase(q)}\nReply with just the value."),
}


def main():
    src = [json.loads(l) for l in open(os.path.join(ROOT, "results/questions_corrected_header.jsonl"))]
    cells = collections.defaultdict(list)
    for it in src:
        if it["family"] in GEOM:
            cells[(it["family"], it["span"])].append(it)
    base = []
    for k in sorted(cells):
        base.extend(cells[k][:PER_CELL])          # deterministic: first N of each cell
    print(f"{len(base):,} base items | {len(TEMPLATES)} templates -> "
          f"{len(base) * len(TEMPLATES):,} prompts")

    out = []
    for it in base:
        parts = it["prompt_full"].split("\n\n")
        head = parts[0].split("Each line is ")[1].rstrip(".")
        body = parts[1]
        for name, fn in TEMPLATES.items():
            n = dict(it)
            n["template"] = name
            n["prompt_full"] = fn(head, body, it["q"])
            n["prompt_blind"] = (f"Question about a person's location record: {it['q']}\n"
                                 f"Answer with the value only.")
            out.append(n)

    # ---- the invariant that makes this a FORMAT study and not a data study ----
    for it in base:
        v = [o for o in out if o["uid"] == it["uid"] and o["q"] == it["q"]
             and o["span"] == it["span"] and o["family"] == it["family"]]
        nums = [tuple(re.findall(r"-?\d+", o["prompt_full"].split("\n\n")[1]
                                 if "\n\n" in o["prompt_full"] else o["prompt_full"]))
                for o in v if o["template"] != "T2_joiner_case"]
        assert len({len(x) for x in nums}) == 1, \
            f"a template changed how many numbers the record contains: {it['family']}"
    print("invariant OK: every template shows the same record values in the same order")

    dst = os.path.join(ROOT, "results/questions_templates.jsonl")
    with open(dst, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    md5 = hashlib.md5(open(dst, "rb").read()).hexdigest()
    print(f"\n{dst}\n  {len(out):,} items | md5 {md5}")
    for name in TEMPLATES:
        ex = next(o for o in out if o["template"] == name)
        first = ex["prompt_full"].split("\n")[0]
        rec = [l for l in ex["prompt_full"].split("\n") if l.strip()][1]
        print(f"  {name:<18} header={first[:56]!r}\n{'':22}record={rec[:56]!r}")


if __name__ == "__main__":
    main()
