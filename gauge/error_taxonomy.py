#!/usr/bin/env python3
"""Failure-mode taxonomy over the raw generations, plus a per-model abstention table.

The taxonomy shows what the models answer instead of the correct value. Its categories are read
off the raw outputs, not imposed on them. The runs are terse (24 generated tokens) and emit no
working, so a taxonomy that classifies errors at the nodes of a computation graph (Dziri et al.
2023, arXiv 2305.18654) cannot be applied here. This one follows Levy et al. (2024, doi
10.18653/v1/2024.acl-long.818), who report failure modes as their own series instead of folding
them into the error rate.

Each generation falls in exactly one category, so the six shares sum to 1 (the code checks them in
the order no_answer, correct, out_of_range, near_constant, wrong_magnitude, close):
  no_answer        nothing parseable, or an explicit refusal or "unknown"
  out_of_range     a number beyond 10x the observed gold range for that cell
  near_constant    the model's single most common answer in this cell, when that answer covers at
                   least 30% of the cell (the "answers one value without reading" mode)
  wrong_magnitude  parseable, in range, but off by more than a factor of 2
  close            off by at most a factor of 2 but not exact
  correct          exact match

The abstention table reports, per model and record length, how often the model says "none" when the
record lacks the answer, how often it says "none" when the answer is present (false abstention),
and its accuracy on the answerable half of the retrieval probe. False abstention is reported per
model because the reading "fabrication, not unwillingness to abstain" holds only where it is low
(Zhou et al., arXiv 2509.01476, find over-refusal under a different context composition).

Input: results/tally_v41.jsonl and the five core runs.
Output: results/v41_error_taxonomy.json.

Usage:
  python gauge/error_taxonomy.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factqa import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = {"phi35mini": "results/v41core_phi35mini.json",
        "mistral7b": "results/v41core_mistral7b.json",
        "llama8b": "results/v41core_llama8b.json",
        "gemma3_12b": "results/v41pilot_gemma3_12b.json",
        "llama70b": "results/v41core_llama70b.json"}
GEOM = ["gyration_km", "max_distance", "total_distance", "longest_jump"]
COUNT = ["day_distinct", "core_set", "longest_stay"]
CATS = ["correct", "close", "wrong_magnitude", "near_constant", "out_of_range", "no_answer"]


def main():
    items = [json.loads(l) for l in open(os.path.join(ROOT, "results/tally_v41.jsonl"))]
    gold = [it["a"].lower() for it in items]
    preds = {}
    for t, p in RUNS.items():
        raw = json.load(open(os.path.join(ROOT, p)))["raw"]["full"]
        preds[t] = [norm(raw[i], items[i]["family"], items[i].get("atype"))
                    for i in range(len(items))]

    out = {"taxonomy": {}, "abstention": {}}
    print("FAILURE-MODE TAXONOMY over raw generations. Categories are exclusive and sum to 1.\n")
    for label, fams in (("GEOMETRY", GEOM), ("COUNT/DURATION", COUNT)):
        print(f"=== {label} ===")
        print(f"  {'family':<16}{'model':<12}" + "".join(f"{c[:9]:>12}" for c in CATS))
        for fam in fams:
            for m in RUNS:
                ix = [i for i, it in enumerate(items) if it["family"] == fam]
                gv = []
                for i in ix:
                    try:
                        gv.append(float(gold[i]))
                    except ValueError:
                        pass
                if len(gv) < 50:
                    continue
                hi = max(gv) * 10 + 10
                # the cell's modal prediction, needed for near_constant
                pc = collections.Counter(preds[m][i] for i in ix if preds[m][i] is not None)
                modal, modal_n = (pc.most_common(1)[0] if pc else (None, 0))
                modal_share = modal_n / len(ix)
                cnt = collections.Counter()
                for i in ix:
                    p = preds[m][i]
                    if p is None:
                        cnt["no_answer"] += 1; continue
                    try:
                        pv, g = float(p), float(gold[i])
                    except ValueError:
                        cnt["no_answer"] += 1; continue
                    if p == gold[i]:
                        cnt["correct"] += 1
                    elif abs(pv) > hi:
                        cnt["out_of_range"] += 1
                    elif modal_share >= 0.30 and p == modal:
                        cnt["near_constant"] += 1
                    elif g != 0 and (pv / g > 2 or pv / max(g, 1e-9) < 0.5):
                        cnt["wrong_magnitude"] += 1
                    else:
                        cnt["close"] += 1
                n = len(ix)
                row = {c: cnt[c] / n for c in CATS}
                out["taxonomy"][f"{fam}|{m}"] = dict(n=n, modal=modal,
                                                     modal_share=modal_share, **row)
                print(f"  {fam:<16}{m:<12}" + "".join(f"{row[c]:>12.3f}" for c in CATS))
        print()

    # ---------------- per-model abstention table ----------------
    print("=" * 92)
    print("ABSTENTION, PER MODEL")
    print(f"  {'model':<12}{'span':>6}{'says none | absent':>20}{'says none | present':>21}"
          f"{'acc on answerable':>20}")
    spans = sorted({it["span"] for it in items if it["family"] == "retrieve_probe"})
    for m in RUNS:
        for s in spans:
            ix = [i for i, it in enumerate(items)
                  if it["family"] == "retrieve_probe" and it["span"] == s]
            una = [i for i in ix if gold[i] == "none"]
            ans = [i for i in ix if gold[i] != "none"]
            if not una or not ans:
                continue
            det = float(np.mean([preds[m][i] == "none" for i in una]))
            fa = float(np.mean([preds[m][i] == "none" for i in ans]))
            acc = float(np.mean([preds[m][i] == gold[i] for i in ans]))
            out["abstention"][f"{m}|{s}"] = dict(detect=det, false_abstain=fa, acc_answerable=acc)
            print(f"  {m:<12}{s:>6}{det:>20.3f}{fa:>21.3f}{acc:>20.3f}")
        print()

    fa_all = [v["false_abstain"] for v in out["abstention"].values()]
    print(f"false-abstention across all model x span cells: "
          f"min {min(fa_all):.3f}  median {np.median(fa_all):.3f}  MAX {max(fa_all):.3f}")
    print("The 'confident fabrication, not unwillingness to abstain' reading is licensed only")
    print("where false abstention is low. Report this table, never the pooled sentence.")
    dst = os.path.join(ROOT, "results/v41_error_taxonomy.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
