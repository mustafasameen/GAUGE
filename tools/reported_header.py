#!/usr/bin/env python3
"""Rewrite the column header of the coordinate prompts to the one used for the paper's main tables.

gauge/tally_v41.py names the columns of a coordinate prompt "day, timeslot, x, y", read off the
rendered rows. The main tables of the paper were computed on the same questions with the header
"day, timeslot, place" on the 6,000 coordinate prompts (the appendix reports the corrected header
as a robustness rerun, and the corrected set is what tally_v41.py --only-coord writes).

This script changes that one phrase in the coordinate prompts of an items file and copies every
other byte of the file unchanged. Applied to the output of a full tally_v41.py run, it gives the
question set that was scored for the main tables (md5 127b67859e28c8728293c459be78c047).

Usage:
  python tools/reported_header.py IN.jsonl OUT.jsonl [--expect 6000]
"""
from __future__ import annotations

import argparse
import sys

OLD = "Each line is day, timeslot, x, y."
NEW = "Each line is day, timeslot, place."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="items file written by gauge/tally_v41.py")
    ap.add_argument("dst", help="where to write the items file with the reported header")
    ap.add_argument("--expect", type=int, default=6000,
                    help="number of coordinate prompts that must be rewritten (default: 6000)")
    a = ap.parse_args()
    n = 0
    with open(a.src, "r", encoding="utf-8", newline="") as fin, \
            open(a.dst, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            if '"tier": "F"' in line and OLD in line:
                line = line.replace(OLD, NEW, 1)
                n += 1
            fout.write(line)
    print(f"rewrote the header of {n:,} coordinate prompts -> {a.dst}")
    if n != a.expect:
        print(f"expected {a.expect:,} rewrites", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
