#!/usr/bin/env python3
"""Build the wording-study items for the retrieval-probe and position families.

gauge/make_templates.py writes the five prompt templates for the four geometric families. The
same code writes them for any other families. This script points it at the full question set and at
the two families of the second wording study, retrieve_probe and retrieve_p50, and writes
results/tally_templates_probe.jsonl (2,750 items: 11 settings x 50 items x 5 templates). The first
50 items of every setting are taken, as in make_templates.py. The output is byte-identical to the
file that was scored (md5 8827130d428e3f9176b9dc072fcae0d4).

Usage:
  python tools/make_probe_templates.py [--items results/tally_v41.jsonl]
                                       [--out results/tally_templates_probe.jsonl]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gauge"))
import make_templates as M          # noqa: E402

FAMILIES = {"retrieve_probe", "retrieve_p50"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", default="results/tally_v41.jsonl", help="the full question set")
    ap.add_argument("--out", default="results/tally_templates_probe.jsonl", help="output file")
    a = ap.parse_args()
    src = os.path.abspath(a.items)
    dst = os.path.abspath(a.out)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "results"))
        link = os.path.join(tmp, "results", "tally_v41_geomfix.jsonl")    # the path main() reads
        try:
            os.symlink(src, link)
        except OSError:
            shutil.copyfile(src, link)
        M.ROOT = tmp
        M.GEOM = FAMILIES
        M.main()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(tmp, "results", "tally_templates.jsonl"), dst)
    print(f"copied to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
