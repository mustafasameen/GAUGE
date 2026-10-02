#!/usr/bin/env python3
"""Score the program arm: does the model know the computation it cannot perform?

This is the only authority on the `tool_*` families. The evaluation harness stores the model's raw
expression and reports 0 accuracy for them by construction, because it has no points to execute
against; that number is meaningless.

The comparison is paired and within one run: every record appears in both arms with an identical
gold, so the value-versus-program delta is a paired statistic over records. The test is a paired
bootstrap over records, the right unit because the same person's record appears in both arms.

Four numbers per cell:
  value acc    the model computes and states the number (what the benchmark measures)
  tool acc     the model writes an expression and the sandbox computes it (what it would score
               with a tool)
  well-formed  the share of expressions that pass the sandbox (coverage, never folded into accuracy)
  correct|wf   the share of well-formed expressions that are right (knowledge, given that the
               model produced code)
The last two must be read together. A model whose expressions rarely parse has not been shown to
lack the knowledge; it has been shown not to follow the output instruction, which is a different
claim and is reported as one.

Input: results/tally_tool.jsonl and results/v41tool_<tag>.json (from eval_factqa.py, which stores
the raw expressions).
Output: results/v41_tool_scored.json.

Usage:
  python gauge/score_tool.py
"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_tool as T
from eval_factqa import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBOOT = 2000


def paired_boot(delta, n=NBOOT, seed=0):
    v = np.asarray(delta, float)
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ip = os.path.join(ROOT, "results/tally_tool.jsonl")
    runs = sorted(glob.glob(os.path.join(ROOT, "results/v41tool_*.json")))
    if not os.path.exists(ip) or not runs:
        sys.exit("tool control: items or runs not on disk yet")
    items = [json.loads(l) for l in open(ip)]
    print("TOOL CONTROL — the model specifies the computation, we execute it.")
    print("Paired within-run: every record appears in both arms with the same gold.\n")

    out = {}
    for rp in runs:
        model = os.path.basename(rp).replace("v41tool_", "").replace(".json", "")
        raw = json.load(open(rp))["raw"]["full"]
        if len(raw) != len(items):
            print(f"  SKIP {model}: {len(raw)} vs {len(items)}"); continue
        print(f"=== {model} ===")
        print(f"  {'family':<16}{'span':>5}{'value':>8}{'TOOL':>8}{'delta':>8}"
              f"{'95% CI':>18}{'well-formed':>13}{'correct|wf':>12}{'top reject':>18}")
        for fam in T.QUESTIONS:
            for span in sorted({it["span"] for it in items}):
                vix = [i for i, it in enumerate(items)
                       if it["family"] == f"value_{fam}" and it["span"] == span]
                tix = [i for i, it in enumerate(items)
                       if it["family"] == f"tool_{fam}" and it["span"] == span]
                if not vix or len(vix) != len(tix):
                    continue
                # align by (uid, span) -- the pairing invariant asserted at generation
                vkey = {(items[i]["uid"], items[i]["span"]): i for i in vix}
                tkey = {(items[i]["uid"], items[i]["span"]): i for i in tix}
                keys = sorted(set(vkey) & set(tkey))
                vcor, tcor, wf, reasons = [], [], [], collections.Counter()
                for k in keys:
                    it_v, it_t = items[vkey[k]], items[tkey[k]]
                    vcor.append(norm(raw[vkey[k]], it_v["family"], "numeric") == it_v["a"])
                    expr = norm(raw[tkey[k]], it_t["family"], "code")
                    val, st = T.safe_eval(expr or "", it_t["pts"])
                    wf.append(st == "ok")
                    if st != "ok":
                        reasons[st] += 1
                    tcor.append(st == "ok" and val is not None
                                and str(int(round(val))) == it_t["a"])
                va, ta = float(np.mean(vcor)), float(np.mean(tcor))
                lo, hi = paired_boot(np.array(tcor, float) - np.array(vcor, float))
                w = float(np.mean(wf))
                cw = float(np.mean([t for t, f in zip(tcor, wf) if f])) if any(wf) else float("nan")
                top = reasons.most_common(1)[0] if reasons else ("-", 0)
                out[f"{model}|{fam}|{span}"] = dict(
                    n=len(keys), value_acc=va, tool_acc=ta, delta=ta - va, delta_ci=[lo, hi],
                    well_formed=w, correct_given_wellformed=cw,
                    reject_reasons=dict(reasons.most_common(6)), floor=collections.Counter(
                        items[i]["a"] for i in vix).most_common(1)[0][1] / len(vix))
                mark = "  <<<" if lo > 0.05 else ""
                print(f"  {fam:<16}{span:>5}{va:>8.3f}{ta:>8.3f}{ta - va:>+8.3f}"
                      f"  [{lo:>+6.3f},{hi:>+6.3f}]{w:>13.3f}{cw:>12.3f}"
                      f"{top[0][:16]:>18}{mark}")
        print()

    print("=" * 108)
    print("VERDICT — does the model know the computation it cannot perform?\n")
    for model in sorted({k.split("|")[0] for k in out}):
        c = {k: v for k, v in out.items() if k.startswith(model + "|")}
        dv = np.mean([v["delta"] for v in c.values()])
        wf = np.mean([v["well_formed"] for v in c.values()])
        ta = np.mean([v["tool_acc"] for v in c.values()])
        va = np.mean([v["value_acc"] for v in c.values()])
        sig = sum(1 for v in c.values() if v["delta_ci"][0] > 0.05)
        if wf < 0.5:
            verd = (f"NOT MEASURABLE: only {wf:.0%} of expressions are well-formed — this model "
                    f"did not follow the output instruction, which is not the same as not knowing")
        elif dv > 0.05:
            verd = ("KNOWS THE COMPUTATION, CANNOT EXECUTE IT — the failure is arithmetic and a "
                    "tool closes it")
        else:
            verd = ("CANNOT SPECIFY IT EITHER — the failure is the definition, not the arithmetic, "
                    "and no tool rescues it")
        print(f"  {model:<12} value {va:.3f} -> tool {ta:.3f} (mean delta {dv:+.3f}, "
              f"{sig}/{len(c)} cells with CI above +0.05, well-formed {wf:.0%})")
        print(f"  {'':<12} {verd}")

    dst = os.path.join(ROOT, "results/v41_tool_scored.json")
    json.dump(out, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}  ({len(out)} cells)")


if __name__ == "__main__":
    main()
