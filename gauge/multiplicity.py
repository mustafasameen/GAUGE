#!/usr/bin/env python3
"""Count every cell that was scanned and apply a multiplicity correction to the significance claims.

Schaeffer et al. (2023, arXiv 2304.15004, section 7) note that benchmarks spanning very many
task-metric-model triplets produce apparent effects by chance, and ask authors to report how many
cells were scanned and to control for multiple comparisons. This script does both:

  1. The denominator. It counts the cells in every scored result file listed in SOURCES, so that the
     paper can state the number. A file that is absent is reported as such and left out of the count.
  2. Benjamini-Hochberg control of the false discovery rate (q = 0.05) over the significance claims
     the paper makes: a cell "clears its baseline" (the gain has an interval that excludes the
     baseline), an intervention "works", a difference "is significant". Each claim becomes a
     one-sided test against its own baseline, using the stored bootstrap interval read as a normal
     approximation.

The main results table is inside the corrected family. It bolds a cell when its interval excludes the
baseline, which is a significance decision. Including those cells makes the correction stricter,
which cannot manufacture an effect where the paper reports none.

Input: the scored result files listed in SOURCES (family_cis.py, score_control_ladder.py and
score_compare.py supply the claims).
Output: results/multiplicity.json.

Usage:
  python gauge/multiplicity.py        (runs on import; takes no arguments)
"""
import glob, json, os, collections
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def bh(pvals, q=0.05):
    """Benjamini-Hochberg. Returns a boolean array of rejections and the critical p."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(n, bool), 0.0
    kmax = np.max(np.where(passed)[0])
    crit = p[order][kmax]
    return p <= crit, float(crit)


def ci_to_p(lo, hi, null=0.0):
    """One-sided p-value from a bootstrap interval, for the claim "this cell exceeds its baseline".

    The test is one-sided because every claim in this family is directional: a significantly negative
    effect must not count as a discovery. A degenerate interval (zero width, for example because
    accuracy is identically zero) returns p = 1 and is never an automatic rejection. The interval is
    read as +/-1.96 standard errors, the standard normal-approximation reading of a percentile
    interval; the paper states this.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 1.0
    if hi <= lo:                      # degenerate: no spread to test against
        return 1.0
    if lo <= null:                    # one-sided: does not clear
        return 1.0
    centre = (lo + hi) / 2
    half = (hi - lo) / 2
    z = (centre - null) / (half / 1.96)
    from math import erfc, sqrt
    return float(erfc(z / sqrt(2)) / 2)      # one-sided


# ---------------- 1. the denominator ----------------
print("CELLS SCANNED — the denominator a multiplicity claim needs\n")
counts, total = {}, 0
SOURCES = {
    # family_cis.py's output is the source of the main results table, of every "clears its baseline"
    # verdict and of the pair-level counts, so it belongs in the denominator.
    "family cells (main table)": "results/family_cis.json",
    "main screen": "results/degeneracy_screen.json",
    "tolerance gate": "results/tolerance_gate.json",
    "continuous metric": "results/continuous_metric.json",
    "distributional rung": "results/distributional_rung.json",
    "claim verification": "results/claim_verification.json",
    "error taxonomy": "results/error_taxonomy.json",
    "sampling": "results/scored_sampling.json",
    "control + ladder": "results/scored_control_ladder.json",
    "comparison arm": "results/scored_compare.json",
    "wording study": "results/scored_templates.json",
    "representation probe": "results/representation_probe.json",
}
for name, path in SOURCES.items():
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        print(f"  {name:<26}      (not present)")
        continue
    d = json.load(open(p))
    n = sum(1 for k, v in d.items() if isinstance(v, dict))
    counts[name] = n
    total += n
    print(f"  {name:<26}{n:>6} cells")
print(f"\n  TOTAL CELLS EXAMINED: {total:,}")
print("  This number goes in the paper. Schaeffer §7 asks for it explicitly.")

# ---------------- 2. BH over the claims we assert ----------------
print("\n\nBENJAMINI-HOCHBERG over the significance claims we actually make\n")
claims = []
# The main table's cell verdicts are claims. The table bolds a cell when its interval excludes
# the baseline, which is a significance decision, so those cells belong in the corrected family.
# Adding them makes the correction stricter, which cannot manufacture an effect where none is
# reported.
fc = os.path.join(ROOT, "results/family_cis.json")
if os.path.exists(fc):
    for k, v in json.load(open(fc)).items():
        if isinstance(v, dict) and "gain_ci" in v and "gain" in v:
            claims.append((f"family|{k}", v["gain"], v["gain_ci"][0], v["gain_ci"][1]))
cl = os.path.join(ROOT, "results/scored_control_ladder.json")
if os.path.exists(cl):
    for k, v in json.load(open(cl)).items():
        if "gain_ci" in v:
            claims.append((k, v["gain"], v["gain_ci"][0], v["gain_ci"][1]))
cmp_ = os.path.join(ROOT, "results/scored_compare.json")
if os.path.exists(cmp_):
    for k, v in json.load(open(cmp_)).items():
        if isinstance(v, dict) and v.get("pooled_ci") and np.isfinite(v["pooled_ci"][0]):
            claims.append((f"compare|{k}", np.mean(v["corrected_acc"]),
                           v["pooled_ci"][0], v["pooled_ci"][1]))
if not claims:
    print("  (no interval-bearing claims on disk yet)")
else:
    ps = [ci_to_p(lo, hi) for _, _, lo, hi in claims]
    rej, crit = bh(ps, q=0.05)
    print(f"  {len(claims)} interval-bearing claims | BH q=0.05 | critical p = {crit:.2e}")
    print(f"  survive correction: {int(rej.sum())} / {len(claims)}")
    killed = [(claims[i][0], ps[i]) for i in range(len(claims)) if not rej[i] and ps[i] < 0.05]
    print(f"  claims that were nominally p<0.05 but do NOT survive BH: {len(killed)}")
    for k, p in killed[:10]:
        print(f"     {k}  p={p:.3f}")
    if not killed:
        print("     none — every nominally significant claim survives the correction")

dst = os.path.join(ROOT, "results/multiplicity.json")
_by_src = collections.Counter(k.split("|")[0] for k, *_ in claims)
_surv = collections.Counter(claims[i][0].split("|")[0]
                            for i in range(len(claims)) if rej[i]) if claims else {}
json.dump(dict(cells_by_experiment=counts, total_cells=total,
               n_claims=len(claims), n_survive=int(rej.sum()) if claims else 0,
               critical_p=crit if claims else None,
               claims_by_source=dict(_by_src), survive_by_source=dict(_surv),
               test="one-sided, normal approximation to the percentile bootstrap interval",
               q=0.05), open(dst, "w"), indent=1)
print(f"\nwrote {dst}")
