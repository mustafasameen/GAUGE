#!/usr/bin/env python3
"""Score the stripped rung: relabelled (timestamped rows) versus stripped (bare x, y rows), paired.

The estimator is not reimplemented: boot, summarize and predict are imported from
score_framing.py, so every interval sits on the same cluster bootstrap as the rest of the
framing arm. summarize() labels its two conditions with the key suffixes of the framing scorer;
they are renamed here to relabelled and stripped, so that the output cannot be misread.

Two additions. (1) A replication check: the regenerated relabelled condition should be
byte-identical to the earlier run (results/framing_<tag>.json), which is what licenses comparing
this job with that job's `full` condition. (2) The paired delta per family, because the explanation
that the ordering of timestamped rows matters predicts an effect only in the order-dependent
families (longest_jump and total_distance). Decision rule: a difference is non-null if and only if
the paired 95% interval excludes 0.

Input: results/questions_stripped.jsonl and results/stripped_<tag>.json (conditions `nomobility` and
`stripped`).
Output: results/scored_stripped.json.

Usage:
  python gauge/score_stripped.py [--selftest]
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_framing import boot, summarize, predict   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
RENAME = {"acc_full": "acc_relabelled", "acc_framing": "acc_stripped",
          "gain_full": "gain_relabelled", "gain_framing": "gain_stripped",
          "unparsed_full": "unparsed_relabelled", "unparsed_framing": "unparsed_stripped",
          "n_disc_full_right": "n_disc_relabelled_right", "n_disc_framing_right": "n_disc_stripped_right"}


def rename(d):
    return {RENAME.get(k, k): v for k, v in d.items()}


def replication(new_raw, old_path):
    """Byte-identical count of regenerated relabelled generations vs the earlier run."""
    if not os.path.exists(old_path):
        return None
    old = (json.load(open(old_path)).get("raw") or {}).get("nomobility")
    if not old or len(old) != len(new_raw):
        return None
    return sum(1 for a, b in zip(new_raw, old) if a == b), len(new_raw)


def pair(items, ix, rl, st):
    golds = [items[i]["a"] for i in ix]
    mode_val, mode_ct = collections.Counter(golds).most_common(1)[0]
    cor_r, par_r = predict(rl, items, ix)
    cor_s, par_s = predict(st, items, ix)
    uids = [items[i]["uid"] for i in ix]
    rec = rename(summarize(mode_ct / len(ix), cor_r, cor_s, par_r, par_s, uids))
    return rec, dict(cor_r=cor_r, cor_s=cor_s, par_r=par_r, par_s=par_s, uids=uids,
                     maj=[items[i]["a"] == mode_val for i in ix])


def pooled(arrs):
    fl = float(np.mean(arrs["maj"]))
    return rename(summarize(fl, arrs["cor_r"], arrs["cor_s"], arrs["par_r"], arrs["par_s"], arrs["uids"]))


def selftest():
    import tempfile
    ok = True
    d = tempfile.mkdtemp(); p = os.path.join(d, "old.json")
    json.dump({"raw": {"nomobility": ["Answer: 1", "Answer: 2", "Answer: 3"]}}, open(p, "w"))
    r = replication(["Answer: 1", "Answer: 2", "Answer: 3"], p)
    print(f"  identical regeneration -> {r}  {'OK' if r == (3, 3) else '*** FAIL'}"); ok &= r == (3, 3)
    r = replication(["Answer: 1", "Answer: 9", "Answer: 3"], p)
    print(f"  planted divergence     -> {r}  {'OK' if r == (2, 3) else '*** FAIL'}"); ok &= r == (2, 3)
    x = rename({"acc_full": 1, "acc_framing": 2, "d_acc": 3})
    good = x == {"acc_relabelled": 1, "acc_stripped": 2, "d_acc": 3}
    print(f"  key renaming           -> {x}  {'OK' if good else '*** FAIL'}"); ok &= good
    print(f"SELFTEST: {'PASSED' if ok else '*** FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(RES, "questions_stripped.jsonl"))
    ap.add_argument("--out", default=os.path.join(RES, "scored_stripped.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    items = [json.loads(l) for l in open(a.items)]
    runs = sorted(r for r in glob.glob(os.path.join(RES, "stripped_*.json"))
                  if not r.endswith((".ckpt", ".partial")))
    if not runs:
        sys.exit("no results/stripped_*.json on disk")
    out = {"decision_rule": "non-null iff paired 95% CI excludes 0; no noise-floor override",
           "delta": "acc(relabelled, timestamped rows) - acc(stripped, bare x y rows)",
           "models": {}}
    for rp in runs:
        tag = os.path.basename(rp)[len("stripped_"):-len(".json")]
        raw = json.load(open(rp)).get("raw") or {}
        rl, st = raw.get("nomobility"), raw.get("stripped")
        if rl is None or st is None or len(rl) != len(items) or len(st) != len(items):
            print(f"SKIP {tag}: conditions missing or misaligned"); continue
        rep = replication(rl, os.path.join(RES, f"framing_{tag}.json"))
        cells, fam, allp = {}, collections.defaultdict(lambda: collections.defaultdict(list)), collections.defaultdict(list)
        for f in sorted({it["family"] for it in items}):
            for s in sorted({it["span"] for it in items if it["family"] == f}):
                ix = [i for i, it in enumerate(items) if it["family"] == f and it["span"] == s]
                rec, arr = pair(items, ix, rl, st)
                cells[f"{f}|{s}"] = rec
                for k, v in arr.items():
                    fam[f][k].extend(v); allp[k].extend(v)
        fams = {f: pooled(v) for f, v in fam.items()}
        P = pooled(allp)
        out["models"][tag] = dict(replication=rep, pooled=P, families=fams, cells=cells,
                                   clears_relabelled=sum(1 for c in cells.values() if c["gain_relabelled"] >= .05),
                                   clears_stripped=sum(1 for c in cells.values() if c["gain_stripped"] >= .05))
        m = out["models"][tag]
        rs = f"{rep[0]}/{rep[1]} byte-identical" if rep else "earlier run not found"
        print(f"\n{tag}: replication of relabelled condition vs earlier job: {rs}")
        print(f"  pooled  acc relabelled {P['acc_relabelled']:.3f}  stripped {P['acc_stripped']:.3f}  "
              f"delta {P['d_acc']:+.4f}  CI {[round(x, 4) for x in P['d_acc_ci']]}  -> {P['flag']}")
        print(f"  cells clearing .05: relabelled {m['clears_relabelled']}/20, stripped {m['clears_stripped']}/20   "
              f"unparsed {P['unparsed_relabelled']:.1%} / {P['unparsed_stripped']:.1%}")
        for f, v in fams.items():
            print(f"    {f:<16} delta {v['d_acc']:+.4f}  CI {[round(x, 4) for x in v['d_acc_ci']]}  {v['flag']}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
