#!/usr/bin/env python3
"""Score the GPT-4o arm and the five open-weight models on one common subset of questions.

The GPT-4o arm answers 30 questions per setting, while the open-weight runs answer 300. Reporting
the two side by side at their native sizes would compare estimates whose intervals differ by a
factor of three for reasons unrelated to the models. Every model is therefore scored here on the
same 30 items per setting, re-scored from generations already on disk, so n is identical across
all six columns. Parsing uses the shared parser (eval_factqa.norm), so a prediction becomes an
answer by the same rule for every model.

Coverage is deliberately uneven, and the output records it: all four spatial-geometric families,
plus one count family (distinct places per day) and one retrieval family (retrieval at the median
position) as contrasts. The arm tests the geometric null. It is not a replication of the full grid.

The scorer checks itself first, in both directions: an oracle must score a normalized gain of 1.0
and the modal constant 0.0.

Input: results/tally_v41.jsonl, results/subset_allspans.json (item indices into the items file),
results/frontier_gpt4o.json (from run_frontier_probe.py) and the five core runs.
Output: results/v41_frontier.json.

Usage:
  python gauge/score_frontier.py [--selftest-only]
"""
from __future__ import annotations
import argparse, collections, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gauge"))
from eval_factqa import norm                      # identical parsing to the GPU runs, by import

ITEMS = os.path.join(ROOT, "results", "tally_v41.jsonl")
SUBSET = os.path.join(ROOT, "results", "subset_allspans.json")
FRONTIER = os.path.join(ROOT, "results", "frontier_gpt4o.json")
OUT = os.path.join(ROOT, "results", "v41_frontier.json")

OPEN_WEIGHT = [("Phi-3.5", "v41core_phi35mini"), ("Mistral-7B", "v41core_mistral7b"),
               ("Llama-8B", "v41core_llama8b"), ("Gemma-12B", "v41pilot_gemma3_12b"),
               ("Llama-70B", "v41core_llama70b")]
GEO = {"gyration_km", "max_distance", "total_distance", "longest_jump"}
PRESENCE = 0.05
BOOT = 2000


def gain(pred, gold):
    """Normalized gain over the per-setting modal-answer baseline, the paper's exact statistic."""
    b = collections.Counter(gold).most_common(1)[0][1] / len(gold)
    a = float(np.mean([str(p).strip() == str(g).strip() for p, g in zip(pred, gold)]))
    return (a - b) / (1 - b) if b < 1 else float("nan"), a, b


def boot_ci(pred, gold, rng):
    pred, gold = np.asarray(pred, object), np.asarray(gold, object)
    v = []
    for _ in range(BOOT):
        s = rng.integers(0, len(gold), len(gold))
        g, _, _ = gain(pred[s], gold[s])
        if g == g:
            v.append(g)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"),) * 2


def selftest(items, cells):
    ids = list(cells.values())[0]
    gold = [str(items[i]["a"]).strip() for i in ids]
    mode = collections.Counter(gold).most_common(1)[0][0]
    g_or, _, _ = gain(gold, gold)
    g_co, _, _ = gain([mode] * len(gold), gold)
    print("  oracle gain=%.3f (want 1.000)   modal-constant gain=%.3f (want 0.000)" % (g_or, g_co))
    ok = abs(g_or - 1) < 1e-9 and abs(g_co) < 1e-9
    print("  self-test: %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        sys.exit("SCORER IS BROKEN: a fixture with a known answer behaved wrongly. Nothing written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    items = [json.loads(l) for l in open(ITEMS)]
    idx = json.load(open(SUBSET))["indices"]
    cells = collections.defaultdict(list)
    for i in idx:
        cells[(items[i]["family"], int(items[i]["span"]))].append(i)

    print("SELF-TEST on a cell with a known answer")
    selftest(items, cells)
    if a.selftest_only:
        return

    fr = json.load(open(FRONTIER))
    ow = {n: json.load(open(os.path.join(ROOT, "results", "%s.json" % s)))["raw"]["full"]
          for n, s in OPEN_WEIGHT}
    models = [n for n, _ in OPEN_WEIGHT] + ["gpt-4o"]

    out = {}
    for (fam, span), ids in sorted(cells.items()):
        gold = [str(items[i]["a"]).strip() for i in ids]
        for m in models:
            if m == "gpt-4o":
                pred = [norm(fr["raw"][str(i)], items[i]["family"], items[i].get("atype")) for i in ids]
            else:
                pred = [norm(ow[m][i], items[i]["family"], items[i].get("atype")) for i in ids]
            g, acc, base = gain(pred, gold)
            lo, hi = boot_ci(pred, gold, rng)
            out["%s|%s|%d" % (m, fam, span)] = dict(
                family=fam, span=span, n=len(ids), gain=g, acc=acc, baseline=base, ci=[lo, hi],
                geometric=fam in GEO, clears=bool(lo > PRESENCE))

    print("\n%-12s %8s %8s   %s" % ("model", "median", "clears", "geometric cells clearing"))
    summ = {}
    for m in models:
        r = [v for k, v in out.items() if k.startswith(m + "|")]
        gv = [v for v in r if v["geometric"]]
        med = float(np.median([v["gain"] for v in r]))
        cl = sum(1 for v in r if v["clears"])
        gcl = sum(1 for v in gv if v["clears"])
        summ[m] = dict(median_gain=med, cells=len(r), clears=cl,
                       geometric_cells=len(gv), geometric_clears=gcl)
        print("%-12s %+8.3f %5d/%-3d   %d of %d" % (m, med, cl, len(r), gcl, len(gv)))

    cov = collections.Counter(f for f, _ in cells)
    json.dump(dict(presence=PRESENCE, boot=BOOT, seed=0, n_per_cell=30,
                   note="all six models scored on the SAME 30 items per cell",
                   families=sorted(cov), n_cells=len(cells),
                   frontier_usage=fr.get("usage"), summary=summ, cells=out),
              open(OUT, "w"), indent=1)
    print("\nwrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
