#!/usr/bin/env python3
"""Rebuild results/v41_absence_all_models.json, the input of the correct-abstention figure (Figure 6).

The retrieval probe (family retrieve_probe) asks for the place at a given position of the record.
Half of its questions cannot be answered from the record, and their gold answer is "none". For each
model and record length the file holds:

  abstain           share of the unanswerable questions that the model answers with an abstention
  lo, hi            95% percentile interval of that share, bootstrapped over people (2,000
                    resamples, seed 0; the resampling of family_cis.py)
  false_abstention  share of the answerable questions that the model answers with an abstention
  answerable        accuracy on the answerable questions
  n_unanswerable, n_answerable, n_clusters (the number of distinct people among the unanswerable)

A generation counts as an abstention when the shared parser (eval_factqa.norm) reads it as "none",
or when it is the bare word "no".

make_figures.py reads this file. Point estimates are the ones the paper's figure was drawn from.
The interval ends come from the bootstrap of family_cis.py and may differ from the paper's file in
the third decimal.

Input: results/tally_v41.jsonl and the five core runs (results/v41core_<tag>.json and
results/v41pilot_gemma3_12b.json).
Output: results/v41_absence_all_models.json (or the file given by --out).

Usage:
  python tools/make_absence_artifact.py [--out results/v41_absence_all_models.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gauge"))
from eval_factqa import norm  # noqa: E402

NBOOT, MINCL = 2000, 5
FAMILY = "retrieve_probe"
RUNS = {"Phi-3.5-mini": "results/v41core_phi35mini.json",
        "Mistral-7B": "results/v41core_mistral7b.json",
        "Llama-3.1-8B": "results/v41core_llama8b.json",
        "Gemma-3-12B": "results/v41pilot_gemma3_12b.json",
        "Llama-3.1-70B": "results/v41core_llama70b.json"}


def boot(uids, values, seed=0):
    """95% percentile interval of the mean of `values`, resampling people (the same estimator as
    family_cis.boot; falls back to resampling items when fewer than MINCL people are present)."""
    uids, values = np.asarray(uids), np.asarray(values, float)
    uniq = np.unique(uids)
    rng = np.random.default_rng(seed)
    if len(uniq) >= MINCL:
        idx = {u: np.where(uids == u)[0] for u in uniq}
        m = np.array([values[np.concatenate([idx[u] for u in
                      rng.choice(uniq, len(uniq), True)])].mean() for _ in range(NBOOT)])
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), "uid-cluster"
    m = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(NBOOT)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), "ITEM-fallback"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", default="results/tally_v41.jsonl")
    ap.add_argument("--out", default="results/v41_absence_all_models.json")
    a = ap.parse_args()
    items = [json.loads(line) for line in open(a.items)]
    gold = [it["a"].lower() for it in items]
    spans = sorted({it["span"] for it in items if it["family"] == FAMILY})
    out = {"spans": spans, "boot": NBOOT, "seed": 0, "estimator": "uid-cluster", "models": {}}
    for name, path in RUNS.items():
        if not os.path.exists(path):
            sys.exit(f"missing run: {path}")
        raw = json.load(open(path))["raw"]["full"]
        assert len(raw) == len(items), f"{path}: {len(raw)} generations for {len(items)} items"
        abstains = [norm(raw[i], items[i]["family"], items[i].get("atype")) == "none"
                    or raw[i].strip().lower() == "no" for i in range(len(items))]
        correct = [norm(raw[i], items[i]["family"], items[i].get("atype")) == gold[i]
                   for i in range(len(items))]
        out["models"][name] = {}
        for s in spans:
            cell = [i for i, it in enumerate(items) if it["family"] == FAMILY and it["span"] == s]
            una = [i for i in cell if gold[i] == "none"]
            ans = [i for i in cell if gold[i] != "none"]
            lo, hi, est = boot([items[i]["uid"] for i in una], [abstains[i] for i in una])
            assert est == "uid-cluster", f"{name} {s}: {est}"
            out["models"][name][str(s)] = {
                "abstain": float(np.mean([abstains[i] for i in una])), "lo": lo, "hi": hi,
                "n_clusters": len({items[i]["uid"] for i in una}),
                "false_abstention": float(np.mean([abstains[i] for i in ans])),
                "answerable": float(np.mean([correct[i] for i in ans])),
                "n_unanswerable": len(una), "n_answerable": len(ans)}
        print(f"{name:14s}" + "  ".join(f"{out['models'][name][str(s)]['abstain']:.3f}" for s in spans))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
