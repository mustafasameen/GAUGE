#!/usr/bin/env python3
"""Regenerate results/v41_blind_decomposition.json with its provenance (the Gemma-3-12B blind control).

For each family the script computes the best-constant baseline ("floor"), blind accuracy and full
accuracy for the Gemma-3-12B core run, using the shared parser (eval_factqa.norm). It writes them
together with the md5 and modification time of the run and items files. The recomputation is
compared with an existing output file before anything is overwritten, so the script adds
provenance and never revises a result. The Gemma run is used because its blind arm reaches the
generation ceiling on 0.0% of questions; a truncated blind generation fails to parse and would bias
blind accuracy downward (Mistral-7B's blind arm reaches the ceiling on 23.5%).

It asserts that there are 21 families, that blind accuracy is exactly zero for 11 of them, and
that every family is at or below its baseline.

Input: results/tally_v41.jsonl and results/v41pilot_gemma3_12b.json.
Output: results/v41_blind_decomposition.json.

Usage:
  python gauge/make_blind_decomposition.py [--check]
    --check recomputes and compares with the existing file and writes nothing.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_factqa import norm            # identical parsing to the GPU run, by import

# Paths resolve against the repository, not the caller's directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(_ROOT, "results", "tally_v41.jsonl")
RUN = os.path.join(_ROOT, "results", "v41pilot_gemma3_12b.json")
MODEL = "google/gemma-3-12b-it"
OUT = os.path.join(_ROOT, "results", "v41_blind_decomposition.json")


def stamp(p):
    st = os.stat(p)
    return dict(path=p, md5=hashlib.md5(open(p, "rb").read()).hexdigest(),
                mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
                bytes=st.st_size)


def compute():
    items = [json.loads(l) for l in open(ITEMS)]
    d = json.load(open(RUN))
    if d.get("model") and d["model"] != MODEL:
        sys.exit("run file reports model %r, expected %r" % (d["model"], MODEL))
    blind, full = d["raw"]["blind"], d["raw"]["full"]
    per = collections.defaultdict(lambda: dict(b=[], f=[], gold=[]))
    for i, it in enumerate(items):
        g = it["a"].strip().lower()
        r = per[it["family"]]
        r["gold"].append(g)
        r["b"].append(1.0 if (norm(blind[i], it["family"], it.get("atype")) or "") == g else 0.0)
        r["f"].append(1.0 if (norm(full[i], it["family"], it.get("atype")) or "") == g else 0.0)
    out = {}
    for fam, r in per.items():
        n = len(r["gold"])
        # floor = modal-answer share, the same best-constant baseline every other GAUGE number uses
        floor = collections.Counter(r["gold"]).most_common(1)[0][1] / n
        out[fam] = dict(floor=floor, blind=sum(r["b"]) / n, full=sum(r["f"]) / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    fresh = compute()
    if os.path.exists(OUT):
        old = json.load(open(OUT))
        keys = {k: v for k, v in old.items() if isinstance(v, dict) and "blind" in v}
        drift = [(f, keys[f]["blind"], fresh[f]["blind"]) for f in keys
                 if f not in fresh or abs(keys[f]["blind"] - fresh[f]["blind"]) > 1e-9]
        if drift:
            print("DRIFT against the existing artifact:")
            for f, o, n in drift:
                print("   %-22s stored %.6f  recomputed %.6f" % (f, o, n))
            sys.exit("refusing to overwrite: recomputation does not reproduce the stored numbers")
        print("recomputation reproduces all %d stored families exactly" % len(keys))

    z = sum(1 for v in fresh.values() if v["blind"] == 0.0)
    below = all(v["blind"] <= v["floor"] + 1e-9 for v in fresh.values())
    print("families %d | blind exactly zero for %d | all at or below floor: %s"
          % (len(fresh), z, below))
    # The two quantities the paper states. If either moves, the sentence in the methodology section
    # is wrong.
    assert len(fresh) == 21, "paper says 21 families, got %d" % len(fresh)
    assert z == 11, "paper says exactly zero for 11 families, got %d" % z
    assert below, "paper says all families at or below baseline"

    if a.check:
        print("CHECK ONLY -- nothing written")
        return
    doc = dict(fresh)
    doc["_provenance"] = dict(
        model=MODEL, run=stamp(RUN), items=stamp(ITEMS),
        generated_by="gauge/make_blind_decomposition.py",
        note="Computed from the Gemma-3-12B core run. "
             "Its blind arm is "
             "at-cap 0.0%. Mistral-7B's blind arm is at-cap 23.5% and is NOT the source of this "
             "claim.")
    json.dump(doc, open(OUT, "w"), indent=1)
    print("wrote %s with provenance (model=%s)" % (OUT, MODEL))


if __name__ == "__main__":
    main()
