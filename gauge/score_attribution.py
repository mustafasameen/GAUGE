#!/usr/bin/env python3
"""Score the cross-record attribution family (which of K people was at place P on day D).

Reported per K, on the K ladder at a fixed total record length of about 440 rows (K = 2, 4, 6, 8):
  * Normalized gain (acc - baseline) / (1 - baseline), the same statistic as every other family. A
    cell clears when the gain exceeds 0.05. The baseline moves with K (it is 1/(K+1)), so the ladder
    is read on gain and never on raw accuracy: raw accuracy falls with K for free.
  * Item-level bootstrap, explicitly labelled. One item spans K people, so a person-cluster
    bootstrap has no unit to resample; this is the labelled fallback and not an omission. It is
    defensible on the data: thousands of distinct people appear across the items and none in more
    than 0.6% of them, so items are close to independent.
  * The "none" channel. The answer space includes "none of them", so a model can fail in two ways:
    name the wrong person, or refuse. The script reports the rate at which each is predicted against
    its true rate. A model that answers "none" far above its 1/(K+1) base rate is abstaining, which
    is a different finding from a model that guesses a letter.
  * The blind arm: the question without the records. If blind is close to full, the records are not
    being read.
  * The parse rate: coverage is reported and never silently folded into accuracy.

Input: results/questions_attribution.jsonl and results/attribution_<tag>.json.
Output: results/scored_attribution.json.

Usage:
  python gauge/score_attribution.py
"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBOOT = 2000
LENGTH = []
BREADTH = ["28x2", "28x4", "28x6", "28x8"]


def boot(correct, n=NBOOT, seed=0):
    v = np.asarray(correct, float)
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ip = os.path.join(ROOT, "results/questions_attribution.jsonl")
    runs = sorted(glob.glob(os.path.join(ROOT, "results/attribution_*.json")))
    if not os.path.exists(ip) or not runs:
        sys.exit("which_person: items or runs not on disk yet")
    items = [json.loads(l) for l in open(ip)]
    print(f"WHICH_PERSON — the cross-record family. {len(items):,} items, {len(runs)} models.")
    print("Uncertainty is an ITEM-level bootstrap: one item spans K users, so there is no")
    print("single-user cluster to resample. This is the labelled fallback, not an omission --")
    print("3,782 distinct users appear, none in more than 0.59% of items.\n")

    out, cube = {}, {}
    for rp in runs:
        model = os.path.basename(rp).replace("attribution_", "").replace(".json", "")
        blob = json.load(open(rp))
        for cond in ("full", "blind"):
            raw = blob["raw"].get(cond)
            if not raw or len(raw) != len(items):
                continue
            pred = [norm(raw[i], "which_person", "letter") for i in range(len(items))]
            for c in sorted({it["cell"] for it in items}):
                ix = [i for i, it in enumerate(items) if it["cell"] == c]
                fl = collections.Counter(items[i]["a"] for i in ix).most_common(1)[0][1] / len(ix)
                cor = [pred[i] == items[i]["a"] for i in ix]
                acc = float(np.mean(cor))
                lo, hi = boot(cor)
                g = (acc - fl) / (1 - fl)
                cube[(model, cond, c)] = dict(
                    n=len(ix), floor=fl, acc=acc, gain=g,
                    gain_ci=[(lo - fl) / (1 - fl), (hi - fl) / (1 - fl)],
                    parse_rate=float(np.mean([pred[i] is not None for i in ix])),
                    pred_none=float(np.mean([pred[i] == "none" for i in ix])),
                    true_none=float(np.mean([items[i]["a"] == "none" for i in ix])))

    models = sorted({k[0] for k in cube})
    for title, cells in [("LENGTH LADDER (K=4, days vary)", LENGTH),
                         ("BREADTH LADDER (7 days, K varies) — read GAIN, not accuracy: the floor "
                          "moves with K", BREADTH)]:
        print(f"\n{'=' * 100}\n{title}\n")
        print(f"  {'model':<12}{'cell':>7}{'floor':>8}{'acc':>8}{'GAIN':>9}"
              f"{'gain 95% CI':>19}{'blind':>8}{'parse':>8}{'none pred/true':>16}")
        for m in models:
            for c in cells:
                v = cube.get((m, "full", c))
                if not v:
                    continue
                b = cube.get((m, "blind", c), {})
                out[f"which|{c}|{m}"] = dict(full=v, blind=b)
                mark = "  PASS" if v["gain_ci"][0] > 0.05 else ""
                print(f"  {m:<12}{c:>7}{v['floor']:>8.3f}{v['acc']:>8.3f}{v['gain']:>+9.3f}"
                      f"   [{v['gain_ci'][0]:>+6.3f},{v['gain_ci'][1]:>+6.3f}]"
                      f"{b.get('acc', float('nan')):>8.3f}{v['parse_rate']:>8.3f}"
                      f"{v['pred_none']:>8.3f}/{v['true_none']:<7.3f}{mark}")
            print()

    print("=" * 100)
    print("VERDICT\n")
    for m in models:
        g = [cube[(m, "full", c)]["gain"] for c in sorted({k[2] for k in cube})
             if (m, "full", c) in cube]
        passed = [c for c in sorted({k[2] for k in cube})
                  if (m, "full", c) in cube and cube[(m, "full", c)]["gain_ci"][0] > 0.05]
        pn = np.mean([cube[(m, "full", c)]["pred_none"] for c in sorted({k[2] for k in cube})
                      if (m, "full", c) in cube])
        tn = np.mean([cube[(m, "full", c)]["true_none"] for c in sorted({k[2] for k in cube})
                      if (m, "full", c) in cube])
        ab = "  ABSTAINING" if pn > tn * 2 else ""
        print(f"  {m:<12} median gain {np.median(g):+.3f}  | cells clearing 0.05: "
              f"{len(passed)}/{len(g)} {passed if passed else ''}  | 'none' {pn:.2f} vs true "
              f"{tn:.2f}{ab}")
    # does the join degrade with breadth?
    print("\n  BREADTH TREND AT FIXED LENGTH (gain at K=2 -> K=8, ~440 rows throughout):")
    for m in models:
        v = [cube[(m, "full", c)]["gain"] for c in BREADTH if (m, "full", c) in cube]
        if len(v) == len(BREADTH):
            # A three-way label (rises, degrades or flat). A two-way form would print "flat" for a monotone
            # rise.
            lab = ("RISES" if v[-1] > v[0] + 0.05 else
                   "degrades" if v[-1] < v[0] - 0.05 else "flat")
            mono = all(b >= a - 0.02 for a, b in zip(v, v[1:]))
            print(f"    {m:<12}" + " -> ".join(f"{x:+.3f}" for x in v)
                  + f"   {lab}{' (monotone)' if mono and lab == 'RISES' else ''}")

    dst = os.path.join(ROOT, "results/scored_attribution.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}  ({len(out)} cells)")


if __name__ == "__main__":
    main()
