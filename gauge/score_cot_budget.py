#!/usr/bin/env python3
"""Score the chain-of-thought budget arm: the same 400 items at --max-new 4096 versus 8192.

Reports (a) accuracy and gain against each cell's own baseline at each budget, the primary
absolute question of whether geometry clears at all, and (b) the paired per-item delta, 8192 minus
4096, which is the improvement itself, with a bootstrap over people. With `--single` the script
scores one run against each cell's own baseline and invents no paired delta (used for
Llama-3.1-70B, which has the 8192 point only).

The pairing across two jobs is legitimate here, and the script checks that it is. Greedy decoding is
not bit-reproducible across batch compositions, so a cross-job pairing is normally unsafe. Here the
batching is identical (same item file, same order, same --bs) and only --max-new differs. The check:
of the items that terminated naturally below the smaller budget in both runs, all must have
byte-identical generations. The script repeats this check on every invocation and prints it, so a
run that breaks the property cannot be scored silently.

Baseline, gain and bootstrap are not reimplemented: `boot` and `summarize` are imported from
score_framing.py, so every interval in this arm sits on the same estimator as the framing arm.
summarize()'s parameter names say full and relabelled; here they carry the two budgets. The
arithmetic is a paired two-condition comparison either way.

Input: results/questions_cot.jsonl and the runs results/cot-4k_llama8b.json and
results/cot-8k_llama8b.json (with --single, one run given by --hi).
Output: results/scored_cot_budget.json (or the file given by --out).

Usage:
  python gauge/score_cot_budget.py
  python gauge/score_cot_budget.py --single --hi results/cot-8k_llama70b.json --tag llama70b
      --out results/scored_cot_70b.json
  python gauge/score_cot_budget.py --selftest
"""
from __future__ import annotations
import argparse, collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_model import norm, ANSWER_RE                      # noqa: E402
from score_framing import boot, summarize, predict        # noqa: E402  (same estimator, on purpose)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
GEOM = ["gyration_km", "max_distance", "total_distance", "longest_jump"]


def load(path, cond="full"):
    b = json.load(open(path))
    return (b["raw"][cond], (b.get("genlen") or {}).get(cond),
            (b.get("forced") or {}).get(cond), b.get("max_new"))


def determinism_check(ra, rb, ga, gb, fa, fb, lo_cap):
    """The property the cross-job pairing rests on. Items that terminated naturally below the
    SMALLER budget in both runs must be byte-identical; only the longer generations can diverge."""
    nat = [i for i in range(len(ra))
           if not fa[i] and not fb[i]
           and ga[i] is not None and gb[i] is not None and ga[i] < lo_cap and gb[i] < lo_cap]
    same = sum(1 for i in nat if ra[i] == rb[i])
    return len(nat), same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", default=os.path.join(RESULTS, "cot-4k_llama8b.json"))
    ap.add_argument("--hi", default=os.path.join(RESULTS, "cot-8k_llama8b.json"))
    ap.add_argument("--items", default=os.path.join(RESULTS, "questions_cot.jsonl"))
    ap.add_argument("--tag", default="llama8b")
    ap.add_argument("--out", default=os.path.join(RESULTS, "scored_cot_budget.json"))
    ap.add_argument("--single", action="store_true",
                     help="score ONE run against each cell's own floor (no paired budget delta). "
                          "Used for Llama-70B, which has the 8192 point only.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        # the estimator is score_framing's, already self-tested there in both directions;
        # what is NEW here is the determinism gate, so that is what gets a planted fixture.
        ra = ["x Answer: 1", "y Answer: 2", "cap no marker", "z Answer: 3"]
        rb = ["x Answer: 1", "y Answer: 2", "cap no marker LONGER", "DIFFERENT Answer: 9"]
        ga, gb = [10, 10, 100, 10], [10, 10, 200, 10]
        fa, fb = [False, False, True, False], [False, False, True, False]
        n, s = determinism_check(ra, rb, ga, gb, fa, fb, 100)
        ok = (n == 3 and s == 2)          # 3 eligible, item 3 planted to differ
        print(f"  determinism gate: {s}/{n} identical (expect 2/3, item 3 planted divergent) "
              f"-> {'OK' if ok else '*** FAILED'}")
        print(f"\nSELFTEST: {'PASSED' if ok else '*** FAILED'}")
        sys.exit(0 if ok else 1)

    items = [json.loads(l) for l in open(a.items)]
    if a.single:
        # One run against its baseline. No paired delta exists, so none is invented: the question for
        # this model is whether it clears the baseline, which is absolute.
        r, g, f, mx = load(a.hi)
        if len(r) != len(items):
            sys.exit(f"{len(r)} generations vs {len(items)} items -- refusing to score")
        print(f"single-run scoring: {a.tag} @ max_new {mx}, n={len(items)}\n")
        print(f"  {'family':<16}{'span':>5}{'n':>5}{'floor':>7}{'acc':>8}{'gain':>9}"
              f"{'95% CI (acc)':>20}{'ceiling':>9}{'forced':>8}{'unparsed':>10}")
        rows, pool = {}, collections.defaultdict(list)
        for fam in [x for x in GEOM if any(it["family"] == x for it in items)]:
            for span in sorted({it["span"] for it in items if it["family"] == fam}):
                ix = [i for i, it in enumerate(items) if it["family"] == fam and it["span"] == span]
                golds = [items[i]["a"] for i in ix]
                mode_val, mode_ct = collections.Counter(golds).most_common(1)[0]
                fl = mode_ct / len(ix)
                cor, par = predict(r, items, ix)
                uids = [items[i]["uid"] for i in ix]
                cor_a = np.asarray(cor, dtype=float)
                lo, hi, how = boot(uids, cor_a)
                gain = (cor_a.mean() - fl) / (1 - fl) if fl < 1 else 0.0
                ceil = sum(1 for i in ix if g[i] is not None and g[i] >= mx) / len(ix)
                frc = sum(1 for i in ix if f[i]) / len(ix)
                unp = 1 - float(np.mean(par))
                rows[f"{a.tag}|{fam}|{span}"] = dict(
                    n=len(ix), floor=fl, acc=float(cor_a.mean()), gain=float(gain),
                    acc_ci=[float(lo), float(hi)], ceiling=ceil, forced=frc, unparsed=unp,
                    estimator=how, clears=bool(gain >= .05))
                print(f"  {fam:<16}{span:>5}{len(ix):>5}{fl:>7.3f}{cor_a.mean():>8.3f}{gain:>+9.3f}"
                      f"{str([round(lo,3), round(hi,3)]):>20}{ceil:>8.1%}{frc:>8.1%}{unp:>10.1%}")
                for k, v in (("cor", cor), ("par", par), ("uids", uids),
                             ("maj", [items[i]["a"] == mode_val for i in ix])):
                    pool[k].extend(v)
        flp = float(np.mean(pool["maj"]))
        ca = np.asarray(pool["cor"], dtype=float)
        lo, hi, how = boot(pool["uids"], ca)
        gp = (ca.mean() - flp) / (1 - flp) if flp < 1 else 0.0
        print("  " + "-" * 96)
        print(f"  {'POOLED':<16}{'--':>5}{len(ca):>5}{flp:>7.3f}{ca.mean():>8.3f}{gp:>+9.3f}"
              f"{str([round(lo,3), round(hi,3)]):>20}")
        nclear = sum(1 for v in rows.values() if v["clears"])
        print(f"\n  cells clearing the .05 presence bar: {nclear} of {len(rows)}")
        print(f"  pooled gain vs floor: {gp:+.3f}   (negative = below the modal-answer baseline)")
        json.dump(dict(mode="single", max_new=mx, model=a.tag, n=len(items),
                       pooled=dict(n=len(ca), floor=flp, acc=float(ca.mean()), gain=float(gp),
                                    acc_ci=[float(lo), float(hi)], estimator=how),
                       cells=rows), open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
        return
    ra, ga, fa, mxa = load(a.lo)
    rb, gb, fb, mxb = load(a.hi)
    for nm, r in (("lo", ra), ("hi", rb)):
        if len(r) != len(items):
            sys.exit(f"{nm}: {len(r)} generations vs {len(items)} items -- refusing to score")
    if mxa >= mxb:
        sys.exit(f"--lo max_new {mxa} must be SMALLER than --hi {mxb}")

    n_nat, n_same = determinism_check(ra, rb, ga, gb, fa, fb, mxa)
    print(f"budget curve: {mxa} -> {mxb}   n={len(items)}   model={a.tag}")
    print(f"DETERMINISM GATE: {n_same}/{n_nat} naturally-terminating generations byte-identical "
          f"across the two jobs" + ("  [OK: pairing valid]" if n_nat and n_same == n_nat
                                    else "  *** PAIRING NOT CLEAN -- read before trusting deltas"))
    if n_nat and n_same != n_nat:
        print("  refusing to print deltas as paired; treat the two budgets as independent runs.")

    cells, pooled = {}, {}
    pool = collections.defaultdict(list)
    hdr = (f"  {'family':<16}{'span':>5}{'n':>5}{'floor':>7} |{'acc@'+str(mxa):>9}{'acc@'+str(mxb):>9}"
           f"{'delta':>8}{'95% CI':>18} |{'gain@lo':>9}{'gain@hi':>9} |{'ceil@lo':>9}{'ceil@hi':>9}")
    print("\n" + hdr)
    for fam in [f for f in GEOM if any(it["family"] == f for it in items)]:
        for span in sorted({it["span"] for it in items if it["family"] == fam}):
            ix = [i for i, it in enumerate(items) if it["family"] == fam and it["span"] == span]
            golds = [items[i]["a"] for i in ix]
            fl = collections.Counter(golds).most_common(1)[0][1] / len(ix)
            cor_a, par_a = predict(ra, items, ix)
            cor_b, par_b = predict(rb, items, ix)
            uids = [items[i]["uid"] for i in ix]
            # summarize(fl, condA, condB, ...) returns d = A - B; we want hi - lo, so pass (hi, lo)
            rec = summarize(fl, cor_b, cor_a, par_b, par_a, uids)
            rec["ceiling_lo"] = sum(1 for i in ix if ga[i] is not None and ga[i] >= mxa) / len(ix)
            rec["ceiling_hi"] = sum(1 for i in ix if gb[i] is not None and gb[i] >= mxb) / len(ix)
            rec["acc_lo"], rec["acc_hi"] = rec.pop("acc_framing"), rec.pop("acc_full")
            rec["gain_lo"], rec["gain_hi"] = rec.pop("gain_framing"), rec.pop("gain_full")
            cells[f"{a.tag}|{fam}|{span}"] = rec
            print(f"  {fam:<16}{span:>5}{rec['n']:>5}{fl:>7.3f} |{rec['acc_lo']:>9.3f}{rec['acc_hi']:>9.3f}"
                  f"{rec['d_acc']:>+8.3f}{str([round(x,3) for x in rec['d_acc_ci']]):>18} |"
                  f"{rec['gain_lo']:>9.3f}{rec['gain_hi']:>9.3f} |"
                  f"{rec['ceiling_lo']:>8.1%}{rec['ceiling_hi']:>9.1%}")
            for k, v in (("cor_lo", cor_a), ("cor_hi", cor_b), ("par_lo", par_a),
                         ("par_hi", par_b), ("uids", uids),
                         ("maj", [items[i]["a"] == collections.Counter(golds).most_common(1)[0][0] for i in ix])):
                pool[k].extend(v)
    flp = float(np.mean(pool["maj"]))
    p = summarize(flp, pool["cor_hi"], pool["cor_lo"], pool["par_hi"], pool["par_lo"], pool["uids"])
    p["acc_lo"], p["acc_hi"] = p.pop("acc_framing"), p.pop("acc_full")
    p["gain_lo"], p["gain_hi"] = p.pop("gain_framing"), p.pop("gain_full")
    pooled[a.tag] = p
    print("  " + "-" * 104)
    print(f"  {'POOLED':<16}{'--':>5}{p['n']:>5}{flp:>7.3f} |{p['acc_lo']:>9.3f}{p['acc_hi']:>9.3f}"
          f"{p['d_acc']:>+8.3f}{str([round(x,3) for x in p['d_acc_ci']]):>18} |"
          f"{p['gain_lo']:>9.3f}{p['gain_hi']:>9.3f} |")
    print(f"\n  cells clearing the .05 presence bar:  @{mxa}: "
          f"{sum(1 for v in cells.values() if v['gain_lo']>=.05)} of {len(cells)}   "
          f"@{mxb}: {sum(1 for v in cells.values() if v['gain_hi']>=.05)} of {len(cells)}")
    print(f"  cells whose paired budget CI excludes 0: "
          f"{sum(1 for v in cells.values() if v['flag'].startswith('NON-NULL'))} of {len(cells)}")
    print(f"  verdict on the pooled budget delta: {p['flag']}")
    json.dump(dict(lo_max_new=mxa, hi_max_new=mxb, model=a.tag, n=len(items),
                   determinism={"eligible": n_nat, "identical": n_same},
                   cells=cells, pooled=pooled), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
