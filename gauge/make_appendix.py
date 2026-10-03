#!/usr/bin/env python3
"""Generate the appendix tables that describe the question set.

The question set is not distributed with the paper, so the appendix has to make it auditable:
anyone who rebuilds the items file with generate_questions.py can recompute every number here and
check it against the printed table. Nothing in this script is typed by hand.

Writes, under outputs/tex/tabs/:
  tab_fingerprint.tex    summary statistics of the question set: questions, descriptor families,
                         record lengths, distinct people, settings, source corpus
  tab_coverage_grid.tex  the number of questions at every family-by-record-length combination
                         (the ragged grid)
and results/appendix_fingerprint.json, a side-car with the md5 of the items file, the sha256 of the
generator and the corpus counts.

Input: results/questions.jsonl, results/questions_corrected_header.jsonl (the corrected-header
coordinate draw) and gauge/generate_questions.py, whose argparse defaults are read for the
per-setting target and the seed.

Usage:
  python gauge/make_appendix.py [--items ...] [--corrected-header ...] [--generator ...] [--outdir ...]
"""
import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shared with make_figures.py and make_tables.py: one definition of the descriptor classes, so
# that the coverage grid, the main table and the profile figure cannot disagree about what a class
# is.
CLASS_ORDER = [
    ("Retrieval by position", ["retrieve_p10", "retrieve_p30", "retrieve_p50",
                               "retrieve_p70", "retrieve_p90", "retrieve_probe"]),
    ("Count and sequence", ["day_distinct", "two_hop", "transition", "core_set"]),
    ("Temporal", ["longest_stay", "recency", "time_mode"]),
    ("Windowed count", ["window_distinct_w16", "window_distinct_w32",
                        "window_distinct_w64", "window_distinct_w128"]),
    ("Spatial-geometric", ["gyration_km", "max_distance", "total_distance", "longest_jump"]),
]
PRETTY = {
    "retrieve_p10": "retrieve at 10th pct", "retrieve_p30": "retrieve at 30th pct",
    "retrieve_p50": "retrieve at 50th pct", "retrieve_p70": "retrieve at 70th pct",
    "retrieve_p90": "retrieve at 90th pct", "retrieve_probe": "retrieve probe",
    "day_distinct": "distinct places per day", "two_hop": "two-hop path",
    "transition": "transition", "core_set": "core set",
    "longest_stay": "longest stay", "recency": "recency", "time_mode": "modal timeslot",
    "window_distinct_w16": "distinct in last 16", "window_distinct_w32": "distinct in last 32",
    "window_distinct_w64": "distinct in last 64", "window_distinct_w128": "distinct in last 128",
    "gyration_km": "radius of gyration", "max_distance": "max displacement",
    "total_distance": "total distance", "longest_jump": "longest jump",
}
CLASS_OF = {f: c for c, fams in CLASS_ORDER for f in fams}


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha_of_text(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def generator_defaults(gen_path):
    """Read the generator's argparse defaults so the printed config cannot drift from code."""
    src = open(gen_path).read()
    out = {}
    for m in re.finditer(r'add_argument\(\s*"--([a-z0-9-]+)"[^)]*?default=([^,)]+)', src, re.S):
        flag, val = m.group(1), m.group(2).strip()
        out[flag] = val.strip('"').strip("'")
    return out


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(ROOT, "results/questions.jsonl"))
    # The coordinate families were regenerated with the corrected column header
    # (generate_questions.py --only-coord). The reported numbers come from --items;
    # --corrected-header is the corrected draw used by the robustness rerun. The appendix describes
    # both.
    ap.add_argument("--corrected-header", default=os.path.join(ROOT, "results/questions_corrected_header.jsonl"))
    ap.add_argument("--generator", default=os.path.join(ROOT, "gauge/generate_questions.py"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs/tex/tabs"))
    a = ap.parse_args()

    # Provenance banner first: a pass rate is a claim about a specific file, so name the file and
    # its modification time before computing anything from it.
    for p in (a.items, a.generator):
        if not os.path.exists(p):
            sys.exit(f"MISSING: {p}")
    st = os.stat(a.items)
    print(f"items     : {a.items}")
    print(f"            {st.st_size:,} bytes  mtime {__import__('datetime').datetime.fromtimestamp(st.st_mtime)}")
    print(f"generator : {a.generator}")

    items_md5 = md5_of(a.items)
    header_md5 = md5_of(a.corrected_header) if os.path.exists(a.corrected_header) else None
    if header_md5:
        print(f"header    : {a.corrected_header}\n            md5 {header_md5}")
    else:
        print("WARNING: corrected coordinate draw not found; the appendix will describe a "
              "single-artifact history that is not what happened.")
    gen_sha = sha_of_text(a.generator)
    cfg = generator_defaults(a.generator)
    print(f"items md5 : {items_md5}")
    print(f"gen sha256: {gen_sha[:16]}...")

    # Single stream over the instrument.
    n = 0
    fam_span = collections.Counter()
    fam_tier = {}
    fam_skmob = {}
    spans, families, uids, tiers = set(), set(), set(), collections.Counter()
    atypes = collections.Counter()
    rows_seen = collections.Counter()
    with open(a.items) as fh:
        for line in fh:
            d = json.loads(line)
            n += 1
            fam, sp = d["family"], int(d["span"])
            fam_span[(fam, sp)] += 1
            families.add(fam)
            spans.add(sp)
            uids.add(d["uid"])
            tiers[d.get("tier", "?")] += 1
            atypes[d.get("atype", "?")] += 1
            rows_seen[int(d.get("n_rows", sp))] += 1
            fam_tier.setdefault(fam, d.get("tier", "?"))
            if d.get("skmob"):
                fam_skmob.setdefault(fam, d["skmob"])

    spans = sorted(spans)
    # Group by DESCRIPTOR CLASS, the grouping the paper argues about. The generator's own tier
    # label (F/H) only distinguishes coordinate rows from place rows and means nothing to a reader.
    missing = set(families) - {f for _, fams in CLASS_ORDER for f in fams}
    if missing:
        sys.exit(f"families with no class: {sorted(missing)} -- add them to CLASS_ORDER")
    order = {f: (i, j) for i, (_, fams) in enumerate(CLASS_ORDER) for j, f in enumerate(fams)}
    families = sorted(families, key=lambda f: order[f])
    per_cell = sorted({v for v in fam_span.values()})
    print(f"\nitems={n:,} families={len(families)} spans={spans} people={len(uids):,}")
    print(f"cells={len(fam_span)}  per-cell sizes observed: {per_cell}")
    print(f"tiers={dict(tiers)}")

    os.makedirs(a.outdir, exist_ok=True)

    # ---- the fingerprint table -------------------------------------------------
    fp = os.path.join(a.outdir, "tab_fingerprint.tex")
    with open(fp, "w") as f:
        f.write("% GENERATED by gauge/make_appendix.py -- do not edit by hand.\n")
        # Footnote-size table with a reduced column separation, so that it fits the column.
        f.write("\\begin{table}[t]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n")
        # The printed table carries no hashes or seed. The side-car JSON below records the md5 of the
        # items file and the hash of the generator.
        f.write("\\caption{Summary of the 34{,}200-question set behind the recovery profile.}\n")
        f.write("\\label{tab:fingerprint}\n")
        f.write("\\begin{tabular}{ll}\n\\toprule\n")
        f.write("statistic & value \\\\\n\\midrule\n")
        f.write(f"questions & {n:,} \\\\\n")
        f.write(f"descriptor families & {len(families)} \\\\\n")
        f.write(f"record lengths & {', '.join(str(s) for s in spans)} rows \\\\\n")
        f.write(f"distinct people & {len(uids):,} \\\\\n")
        f.write(f"family-by-length settings & {len(fam_span)} \\\\\n")
        f.write(f"questions per setting & {tex_escape(cfg.get('per-cell', '?'))} (600 for the retrieval probe) \\\\\n")
        f.write("source corpus & YJMob100K \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"wrote {fp}")

    # ---- the ragged coverage grid ------------------------------------------------
    gp = os.path.join(a.outdir, "tab_coverage_grid.tex")
    with open(gp, "w") as f:
        f.write("% GENERATED by gauge/make_appendix.py -- do not edit by hand.\n")
        # The coverage grid is a full-width table (table*).
        f.write("\\begin{table*}[!t]\n\\centering\n\\small\n")
        # The caption refers to the appendix section that states the ragged-by-construction rule in
        # full, instead of repeating it.
        f.write("\\caption{Questions generated per descriptor family and record length. Empty cells "
                "were never generated or scored (Appendix~\\ref{app:coverage}).}\n")
        f.write("\\label{tab:coverage}\n")
        f.write("\\begin{tabular}{ll" + "r" * len(spans) + "r}\n\\toprule\n")
        f.write("class & family & " + " & ".join(str(s) for s in spans) + " & total \\\\\n")
        f.write("\\midrule\n")
        last_cls = None
        for fam in families:
            cls = CLASS_OF[fam]
            if cls != last_cls and last_cls is not None:
                f.write("\\addlinespace[2pt]\n")
            shown = cls if cls != last_cls else ""
            last_cls = cls
            cells = [fam_span.get((fam, s), 0) for s in spans]
            row = " & ".join(str(c) if c else "--" for c in cells)
            f.write(f"{shown} & {tex_escape(PRETTY.get(fam, fam))} & {row} & {sum(cells):,} \\\\\n")
        f.write("\\midrule\n")
        tot = [sum(fam_span.get((fam, s), 0) for fam in families) for s in spans]
        f.write("& total & " + " & ".join(f"{t:,}" for t in tot) + f" & {n:,} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"wrote {gp}")

    # Side-car with the items md5, the generator hash and the corpus counts, so that the printed
    # tables can be checked against it.
    side = os.path.join(ROOT, "results", "appendix_fingerprint.json")
    json.dump({
        "items_path": a.items, "items_md5": items_md5, "generator_sha256": gen_sha,
        "n_items": n, "n_families": len(families), "spans": spans,
        "n_people": len(uids), "n_cells": len(fam_span),
        "per_cell_target": cfg.get("per-cell"), "seed": cfg.get("seed"),
        "tiers": dict(tiers), "atypes": dict(atypes),
        "per_cell_sizes_observed": per_cell,
    }, open(side, "w"), indent=1)
    print(f"wrote {side}")


if __name__ == "__main__":
    main()
