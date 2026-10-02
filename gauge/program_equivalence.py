#!/usr/bin/env python3
"""Is the model's program the right computation, or did it get the right number on this record?

The program arm scores an item correct when the model's expression, executed on that record,
produces the gold. That is necessary but not sufficient for "the model knows the computation": an
expression could coincide with the gold on one record and be wrong in general (a constant, a
wrong-but-correlated formula, a lucky rounding). The stronger test is extensional equivalence: run
the model's expression and the reference implementation on records the expression was never written
for, and ask whether they agree everywhere. An expression that agrees on 50 held-out records is the
computation and not a coincidence.

The probes are held-out real records, not uniform draws. Uniform points are an easier test: a
program can agree with the reference on uniform noise and diverge on realistic mobility, which is
clustered, repetitive and heavy-tailed. Each expression is checked against 50 real records drawn
from a pool of tool-arm records, excluding the record it was written for.

Per model and family the script reports the number of expressions asked for (n), how many ran, how
many were correct on their own record ("right here"), how many of those were also equivalent on all
50 held-out records, and the difference ("lucky").

Input: results/tally_tool.jsonl, the runs results/v41tool_<tag>.json, and results/_pts_pool.json,
a JSON list of {"pts": [[x, y], ...]} records to draw probes from (tools/make_points_pool.py builds
one from the tool items).
Output: results/v41_program_equivalence.json.

Usage:
  python gauge/program_equivalence.py        (runs on import; takes no arguments)
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_tool as T
from eval_factqa import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NPROBE = 50
REF = {
 "gyration": lambda P: float(np.sqrt(((P - P.mean(0))**2).sum(1).mean())) * 0.5,
 "max_distance": lambda P: float(max(np.sqrt(((P[i]-P)**2).sum(1)).max() for i in range(len(P)))) * 0.5,
 "total_distance": lambda P: float(np.sqrt(((P[1:]-P[:-1])**2).sum(1)).sum()) * 0.5,
 "longest_jump": lambda P: float(np.sqrt(((P[1:]-P[:-1])**2).sum(1)).max()) * 0.5,
}
# Probes are held-out real records, not uniform draws (see the module docstring). Each
# expression is verified against 50 real records drawn from the pool, excluding the record it was
# written for.
rng = np.random.default_rng(0)
_pool = [np.array(it["pts"], float) for it in json.load(open(f"{ROOT}/results/_pts_pool.json"))]
PROBES_ALL = _pool

items = [json.loads(l) for l in open(f"{ROOT}/results/tally_tool.jsonl")]
out = {}
print(f"EXTENSIONAL EQUIVALENCE — each expression re-run on {NPROBE} HELD-OUT REAL records\n")
print(f"  {'model':<12}{'family':<16}{'n expr':>8}{'ran here':>10}{'right here':>12}"
      f"{'EQUIVALENT':>12}{'lucky':>8}")
for rp in sorted(__import__("glob").glob(f"{ROOT}/results/v41tool_*.json")):
    model = os.path.basename(rp).replace("v41tool_", "").replace(".json", "")
    raw = json.load(open(rp))["raw"]["full"]
    if len(raw) != len(items):
        continue
    for fam in REF:
        ix = [i for i, it in enumerate(items) if it["family"] == f"tool_{fam}"]
        ran = right = equiv = 0
        for i in ix:
            it = items[i]
            e = norm(raw[i], it["family"], "code")
            v, st = T.safe_eval(e or "", it["pts"])
            if st != "ok":
                continue
            ran += 1
            here = (v is not None and str(int(round(v))) == it["a"])
            right += here
            if not here:
                continue
            # exclude the record this expression was written for
            own = np.array(it["pts"], float)
            cand = [P for P in PROBES_ALL
                    if not (P.shape == own.shape and np.array_equal(P, own))]
            probes = [cand[j] for j in rng.choice(len(cand), min(NPROBE, len(cand)), replace=False)]
            agree = 0
            for P in probes:
                gv = REF[fam](P)
                mv, mst = T.safe_eval(e, [tuple(p) for p in P])
                if mst == "ok" and mv is not None and abs(mv - gv) <= max(1e-6, 1e-6 * abs(gv)):
                    agree += 1
            equiv += (agree == len(probes))
        if not ix:
            continue
        lucky = right - equiv
        out[f"{model}|{fam}"] = dict(n=len(ix), ran=ran, right_here=right,
                                     equivalent=equiv, right_but_not_equivalent=lucky)
        print(f"  {model:<12}{fam:<16}{len(ix):>8}{ran:>10}{right:>12}{equiv:>12}{lucky:>8}")

print("\n" + "=" * 88)
for model in sorted({k.split("|")[0] for k in out}):
    c = {k: v for k, v in out.items() if k.startswith(model + "|")}
    r = sum(v["right_here"] for v in c.values()); e = sum(v["equivalent"] for v in c.values())
    if r:
        print(f"  {model:<12} correct-on-this-record {r:>5}  |  ALSO equivalent on {NPROBE} REAL "
              f"records: {e:>5}  ({e/r:.1%})")
json.dump(out, open(f"{ROOT}/results/v41_program_equivalence.json", "w"), indent=1)
print(f"\nwrote results/v41_program_equivalence.json")
print("READ: a high equivalence rate means the tool-arm score is measuring the COMPUTATION, not a")
print("coincidence on one record -- which is what the claim that the model knows the computation requires.")
