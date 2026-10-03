#!/usr/bin/env python3
"""Score the blind control on all five models.

In the blind condition the record is withheld and only the question is shown, so the condition
bounds what a model answers from prior knowledge alone. Every main run carries a `blind` condition.
This script re-scores the generations that are already on disk (no GPU is needed) with the shared
parser (eval_model.norm). Per model it reports how many of the 21 families have blind accuracy of
exactly zero and how many sit at or below the family's best-constant baseline.

Input: results/questions.jsonl and the five main runs (results/main_<tag>.json and
results/primary_gemma3_12b.json).
Output: results/blind_all_models.json (with --check nothing is written).

Usage:
  python gauge/score_blind_all_models.py [--check]
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gauge"))
from eval_model import norm            # noqa: E402

ITEMS = os.path.join(_ROOT, "results", "questions.jsonl")
OUT = os.path.join(_ROOT, "results", "blind_all_models.json")

# Gemma's main run carries the primary prefix; the other four carry main. Named explicitly so a
# rename drops a model loudly rather than silently.
RUNS = [
    ("Phi-3.5-mini",  "microsoft/Phi-3.5-mini-instruct",      "main_phi35mini"),
    ("Mistral-7B",    "mistralai/Mistral-7B-Instruct-v0.3",   "main_mistral7b"),
    ("Llama-3.1-8B",  "meta-llama/Llama-3.1-8B-Instruct",     "main_llama8b"),
    ("Gemma-3-12B",   "google/gemma-3-12b-it",                "primary_gemma3_12b"),
    ("Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct",    "main_llama70b"),
]


def stamp(p):
    st = os.stat(p)
    return dict(path=os.path.relpath(p, _ROOT), md5=hashlib.md5(open(p, "rb").read()).hexdigest(),
                mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(ITEMS)]
    by_fam = collections.defaultdict(list)
    for i, it in enumerate(items):
        by_fam[it["family"]].append(i)
    # floor is the modal-answer share, computed once per family from the golds. It does not depend
    # on the model, so a per-model floor would be the same number five times.
    floors = {}
    for fam, idx in by_fam.items():
        g = [items[i]["a"].strip().lower() for i in idx]
        floors[fam] = collections.Counter(g).most_common(1)[0][1] / len(g)

    out, summary = {}, {}
    for name, model_id, stem in RUNS:
        p = os.path.join(_ROOT, "results", "%s.json" % stem)
        d = json.load(open(p))
        if d.get("model") and d["model"] != model_id:
            sys.exit("%s reports model %r, expected %r" % (p, d["model"], model_id))
        if "blind" not in d["raw"]:
            sys.exit("%s has no blind arm" % p)
        blind, full = d["raw"]["blind"], d["raw"]["full"]
        rows = {}
        for fam, idx in by_fam.items():
            gold = [items[i]["a"].strip().lower() for i in idx]
            b = sum(1 for i, g in zip(idx, gold)
                    if (norm(blind[i], items[i]["family"], items[i].get("atype")) or "") == g)
            f = sum(1 for i, g in zip(idx, gold)
                    if (norm(full[i], items[i]["family"], items[i].get("atype")) or "") == g)
            rows[fam] = dict(floor=floors[fam], blind=b / len(idx), full=f / len(idx), n=len(idx))
        n_zero = sum(1 for v in rows.values() if v["blind"] == 0.0)
        n_at_or_below = sum(1 for v in rows.values() if v["blind"] <= v["floor"] + 1e-9)
        worst = max(v["blind"] - v["floor"] for v in rows.values())
        out[name] = dict(model=model_id, run=stamp(p), families=rows)
        summary[name] = dict(n_families=len(rows), n_blind_zero=n_zero,
                             n_at_or_below_floor=n_at_or_below,
                             worst_excess_over_floor=worst)

    print("%-15s %9s %12s %16s %s" % ("model", "families", "blind==0", "at/below floor",
                                      "worst excess"))
    for name, s in summary.items():
        print("%-15s %9d %12d %16d %+.4f"
              % (name, s["n_families"], s["n_blind_zero"], s["n_at_or_below_floor"],
                 s["worst_excess_over_floor"]))

    universal = all(s["n_at_or_below_floor"] == s["n_families"] for s in summary.values())
    zeros = sorted(s["n_blind_zero"] for s in summary.values())
    print()
    print("ALL FIVE models at or below floor on all 21 families: %s" % universal)
    print("blind==0 family counts across models: %s" % zeros)
    if not universal:
        offenders = [(n, f, round(v["blind"] - v["floor"], 4))
                     for n, m in out.items() for f, v in m["families"].items()
                     if v["blind"] > v["floor"] + 1e-9]
        print("FAMILIES ABOVE FLOOR IN THE BLIND CONDITION (model, family, excess):")
        for o in offenders:
            print("   %s" % (o,))

    if a.check:
        print("CHECK ONLY -- nothing written")
        return
    json.dump(dict(items=stamp(ITEMS), floors=floors, summary=summary, models=out),
              open(OUT, "w"), indent=1)
    print("\nwrote %s" % os.path.relpath(OUT, _ROOT))


if __name__ == "__main__":
    main()
