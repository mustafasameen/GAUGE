#!/usr/bin/env python3
"""Degeneracy screen: check every (family, span) cell for the two ways a cell can fail to measure anything.

Two gates, both required:

  Degeneracy gate. top_share is the fraction of a cell's items that receive the model's single
  most common answer. A model that emits a constant scores exactly the baseline while reading
  nothing. The gate fails when top_share exceeds the baseline by more than DEGEN_MARGIN (0.20) for
  a majority of models: the cell then measures a response prior and not a capability. It needs no
  gold, because it is a property of the predictions alone.

  Headroom gate. best_gain is the maximum over models of the normalized gain. A cell in which no
  model clears its own baseline has no signal to decay, so a slope fitted through it would be noise
  around zero. The gate fails when best_gain is below HEAD_MIN (0.05).

A cell that fails the headroom gate but not the degeneracy gate is a capability finding (models
engage and get it wrong) and is reported as such. A cell that fails the degeneracy gate is a format
defect: the answer space admits a constant strategy. Headroom is the primary gate, and degeneracy
is the diagnosis for a cell that lacks headroom, so degeneracy alone never condemns a cell that
demonstrably measures something. For each cell the script bootstraps the best model's gain and
reports how often the OK or not-OK call flips. A verdict that is stable in fewer than 95% of the
replicates is marked BORDERLINE.

Before any real verdict is issued the screen checks itself on planted fixtures: a constant predictor
must read DEGENERATE, a perfect predictor must read OK, and a uniform-random predictor over a wide
answer space must read NO-HEADROOM. The probe cell is derived from the loaded items file (the widest
gold answer space at n >= 100).

Input: an items file and the result JSONs of the runs to screen.
Output: a JSON file with one verdict per tier/family/span cell, and a per-family roll-up on screen.

Usage:
  python gauge/screen_degeneracy.py --items results/tally_v41.jsonl
      --runs results/v41pilot_gemma3_12b.json results/v41core_*.json
      --out results/degeneracy_screen_v41.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factqa import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CORE = ["phi35mini", "mistral7b", "llama8b", "gemma3_12b", "llama70b"]
DEGEN_MARGIN = 0.20   # top_share may exceed floor by at most this before the cell reads as constant
HEAD_MIN = 0.05       # best-model normalized gain must clear this for a rate to be fittable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="results/tally_v4.jsonl")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="result jsons; default = the five v4 core runs")
    ap.add_argument("--out", default="results/degeneracy_screen.json")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.join(ROOT, a.items))]
    runs = a.runs or [f"results/v4core_{m}.json" for m in CORE]
    preds = {}
    for r in runs:
        p = os.path.join(ROOT, r)
        if not os.path.exists(p):
            print(f"  (missing {r})")
            continue
        tag = os.path.basename(r).replace(".json", "")
        raw = json.load(open(p))["raw"]["full"]
        preds[tag] = [norm(raw[i], it["family"], it.get("atype"))
                      for i, it in enumerate(items)]
    print(f"screening {len(preds)} runs over {len(items):,} items\n")

    # ---- Check the checker: planted fixtures with known verdicts, before any real verdict is
    # trusted. A constant predictor must read DEGENERATE; a perfect predictor must read OK even though
    # its top_share tracks the gold's; a uniform-random predictor over a wide answer space must read
    # NO-HEADROOM (varied, no signal) and not DEGENERATE. The probe cell is derived from whatever items
    # file was loaded, never hardcoded: a cell pinned to one instrument would select nothing when handed
    # another. Selection rule: the widest gold answer space at n >= 100, which makes the uniform_random
    # fixture a fair NO-HEADROOM test and not an accidental hit.
    _by_cell = collections.defaultdict(list)
    for i, it in enumerate(items):
        _by_cell[(it["family"], it["span"])].append(i)
    _cands = [(len({items[i]["a"].lower() for i in ix}), k)
              for k, ix in _by_cell.items() if len(ix) >= 100]
    if not _cands:
        sys.exit("no cell with n>=100 to self-test on; refusing to emit verdicts")
    _width, _pkey = max(_cands)
    probe = _by_cell[_pkey]
    pg = [items[i]["a"].lower() for i in probe]
    pfloor = collections.Counter(pg).most_common(1)[0][1] / len(probe)
    # A constant fixture can only READ as degenerate where a constant is not already the gold.
    if pfloor + DEGEN_MARGIN >= 1.0:
        sys.exit(f"probe cell {_pkey} floor {pfloor:.3f} too high to self-test; refusing")
    rng = np.random.default_rng(0)
    _const = "sentinel"          # cannot collide with any real gold, so gain is exactly 0
    fixtures = {
        "constant": [_const] * len(probe),
        "perfect": list(pg),
        "uniform_random": [str(int(v)) for v in rng.integers(0, 10 ** 6, len(probe))],
    }
    print(f"  self-test cell {_pkey[0]}@{_pkey[1]}  n={len(probe)}  "
          f"distinct golds={_width}  floor={pfloor:.3f}")
    expect = {"constant": "DEGENERATE", "perfect": "OK", "uniform_random": "NO-HEADROOM"}
    for name, pv in fixtures.items():
        share = collections.Counter(pv).most_common(1)[0][1] / len(probe)
        acc = sum(1 for k, v in enumerate(pv) if v == pg[k]) / len(probe)
        g = (acc - pfloor) / (1 - pfloor)
        got = ("OK" if g >= HEAD_MIN else
               "DEGENERATE" if share > pfloor + DEGEN_MARGIN else "NO-HEADROOM")
        assert got == expect[name], f"SCREEN IS BROKEN: {name} -> {got}, expected {expect[name]}"
        print(f"  fixture {name:<15} top_share {share:.3f}  gain {g:+.3f}  -> {got}  OK")
    print()

    rng_screen = np.random.default_rng(0)
    cells = collections.defaultdict(list)
    for i, it in enumerate(items):
        cells[(it["tier"], it["family"], it["span"])].append(i)

    out, verdicts = {}, collections.Counter()
    print(f"{'tier':>4} {'family':<14}{'span':>5}{'floor':>7}{'topshare':>10}"
          f"{'bestgain':>10}{'p(OK)':>7}   verdict")
    for (tier, fam, span), idx in sorted(cells.items()):
        gold = [items[i]["a"].lower() for i in idx]
        floor = collections.Counter(gold).most_common(1)[0][1] / len(idx)
        shares, gains = [], []
        for tag, pv in preds.items():
            pc = collections.Counter(pv[i] for i in idx)
            shares.append(pc.most_common(1)[0][1] / len(idx))
            acc = sum(1 for i in idx if pv[i] == items[i]["a"].lower()) / len(idx)
            gains.append((acc - floor) / (1 - floor) if floor < 1 else 0.0)
        med_share, best_gain = float(np.median(shares)), float(max(gains))
        # A verdict is a DECISION built on a point estimate, so it needs its own uncertainty
        # before anything is built on it. Bootstrap the best model's gain over items and report
        # how often the OK/not-OK call flips. A cell whose verdict is stable in <95% of
        # replicates is BORDERLINE and must not silently enter or leave a rate fit.
        bi = int(np.argmax(gains))
        btag = list(preds)[bi]
        bc = np.array([preds[btag][i] == items[i]["a"].lower() for i in idx], dtype=float)
        bg = np.array([items[i]["a"].lower() for i in idx])
        _, gcode = np.unique(bg, return_inverse=True)
        W = rng_screen.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=400).astype(float)
        acc_b = (W * bc).sum(1) / W.sum(1)
        fl_b = np.array([np.bincount(gcode, weights=W[b]).max() / W[b].sum() for b in range(len(W))])
        gain_b = np.where(fl_b < 1, (acc_b - fl_b) / (1 - fl_b), 0.0)
        p_ok = float((gain_b >= HEAD_MIN).mean())
        borderline = 0.05 < p_ok < 0.95
        degen = med_share > floor + DEGEN_MARGIN
        head = best_gain >= HEAD_MIN
        # Order matters. Headroom is the primary gate and degeneracy is the diagnosis for a cell that
        # lacks headroom. A model that answers most of a cell correctly is demonstrably reading the record,
        # so a raised top_share there reflects a small answer space and mild response bias, not a constant
        # strategy. Degeneracy alone never condemns a cell that demonstrably measures something.
        v = ("OK" if head else
             "DEGENERATE" if degen else "NO-HEADROOM")
        verdicts[v] += 1
        out[f"{tier}/{fam}/{span}"] = dict(
            floor=round(floor, 4), median_top_share=round(med_share, 4),
            best_gain=round(best_gain, 4), degenerate=bool(degen),
            headroom=bool(head), verdict=v, p_ok=round(p_ok, 3),
            borderline=bool(borderline))
        flag = ("   <<< BORDERLINE" if borderline else "" if v == "OK" else "   <<<")
        print(f"{tier:>4} {fam:<14}{span:>5}{floor:>7.3f}{med_share:>10.3f}"
              f"{best_gain:>10.3f}{p_ok:>7.2f}   {v}{flag}")

    print(f"\n{dict(verdicts)}")
    # A family is unusable if a MAJORITY of its cells fail; a single failing cell is reported
    # per-cell and excluded from that family's fit rather than killing the family.
    byfam = collections.defaultdict(list)
    for k, v in out.items():
        byfam[k.split("/")[1]].append(v["verdict"])
    print("\nper-family roll-up:")
    for f, vs in sorted(byfam.items()):
        bad = sum(1 for v in vs if v != "OK")
        tag = "UNUSABLE" if bad > len(vs) / 2 else ("partial" if bad else "clean")
        print(f"  {f:<14} {bad}/{len(vs)} cells failing   {tag}")
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
