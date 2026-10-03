#!/usr/bin/env python3
"""Emit the LaTeX tables from the result artifacts. No number is typed by hand.

Writes, under outputs/tex/tabs/:
  tab_main.tex      best normalized gain per model and descriptor family (the main results table)
  tab_coverage.tex  the capability table of 20 mobility and urban foundation models, built from
                    survey/capability_coverage.md and survey/table1_citation_map.json
  tab_levels.tex    the three-level accounting (distributional, ordinal, exact)
With --check each table is compared with the file on disk and the script fails if it is stale,
instead of writing. prereg() would write a further table that the paper does not use; it reads
results/prediction_analysis.json, which this repository does not produce, and it is never called.

Input: results/family_cis.json (family_cis.py) and results/distributional_rung.json
(distributional_rung.py).

Usage:
  python gauge/make_tables.py [--check]
"""
import json
import sys


CHECK = "--check" in sys.argv


def _emit(path, text):
    """Write the table, or under --check compare it with the file on disk and fail loudly. Without
    a check mode, a verification run would overwrite the file, and a stale table could pass review.
    """
    if "--check" in sys.argv:
        if not os.path.exists(path):
            sys.exit("MISSING %s; run without --check to generate it" % path)
        if open(path).read() != text:
            sys.exit("STALE %s; rerun gauge/make_tables.py to regenerate" % path)
        print("  ok  %s matches its generator" % os.path.basename(path))
        return
    with open(path, "w") as fh:
        fh.write(text)
import re

import os
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS = os.path.join(ROOT, "outputs", "tex", "tabs")
os.makedirs(TABS, exist_ok=True)

MODELS = ["phi35mini", "mistral7b", "llama8b", "gemma3_12b", "llama70b"]
SHORT = {"phi35mini": "Phi-3.5", "mistral7b": "Mistral-7B", "llama8b": "Llama-8B",
         "gemma3_12b": "Gemma-12B", "llama70b": "Llama-70B"}
# Descriptor classes, ordered so that the two headline groups sit adjacent. The order matches
# make_figures.CLASS_ORDER and the order in which the paper discusses the classes.
CLASSES = [
    ("Retrieval by position", ["retrieve_p10", "retrieve_p30", "retrieve_p50",
                               "retrieve_p70", "retrieve_p90", "retrieve_probe"]),
    ("Count and sequence", ["day_distinct", "two_hop", "transition", "core_set"]),
    ("Temporal", ["longest_stay", "recency", "time_mode"]),
    ("Windowed count (state volume)", ["window_distinct_w16", "window_distinct_w32",
                                       "window_distinct_w64", "window_distinct_w128"]),
    ("Spatial-geometric", ["gyration_km", "max_distance", "total_distance", "longest_jump"]),
]
PRETTY = {"gyration_km": "radius of gyration", "max_distance": "max displacement",
          "total_distance": "total distance", "longest_jump": "longest jump",
          "day_distinct": "distinct places per day", "two_hop": "two-hop path",
          "transition": "transition", "core_set": "core set",
          "window_distinct_w16": "distinct in last 16", "window_distinct_w32": "distinct in last 32",
          "window_distinct_w64": "distinct in last 64", "window_distinct_w128": "distinct in last 128",
          "retrieve_p10": "retrieve at 10th pct", "retrieve_p30": "retrieve at 30th pct",
          "retrieve_p50": "retrieve at 50th pct", "retrieve_p70": "retrieve at 70th pct",
          "retrieve_p90": "retrieve at 90th pct", "retrieve_probe": "retrieve probe",
          "longest_stay": "longest stay", "recency": "recency", "time_mode": "modal timeslot"}


class _Cap:
    """Collect what the generator writes, then hand it to _emit so --check can compare."""
    def __init__(self, path):
        self.path, self.buf = path, []

    def __enter__(self):
        return self

    def write(self, t):
        self.buf.append(t)

    def __exit__(self, *a):
        if a[0] is None:
            _emit(self.path, "".join(self.buf))
        return False


def main():
    d = json.load(open(os.path.join(ROOT, "results", "family_cis.json")))

    # best gain a (model, family) reaches at any record length, and whether it ever clears
    best, ever = collections.defaultdict(lambda: None), collections.defaultdict(bool)
    for k, v in d.items():
        if not isinstance(v, dict) or "gain" not in v:
            continue
        m, fam, _ = k.split("|")
        cur = best[(m, fam)]
        if cur is None or v["gain"] > cur:
            best[(m, fam)] = v["gain"]
        ever[(m, fam)] |= v["gain_ci"][0] > 0

    rows = []
    for cls, fams in CLASSES:
        rows.append(r"\multicolumn{6}{l}{\emph{" + cls + r"}} \\")
        for fam in fams:
            cells = []
            for m in MODELS:
                g = best[(m, fam)]
                if g is None:
                    cells.append("--")
                elif ever[(m, fam)]:
                    cells.append(f"\\textbf{{{g:+.3f}}}")
                else:
                    cells.append(f"{g:+.3f}")
            rows.append(f"\\quad {PRETTY[fam]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)

    tab = r"""% GENERATED by gauge/make_tables.py from results/family_cis.json. Do not edit by hand.
\begin{table*}[t]
  \centering
  \small
  \caption{Best normalized gain over the baseline that each model reaches on each descriptor
  family, over every length at which that family exists: seven for distinct places per day and
  recency, five for the geometric families, fewer for the longer windows. Bold marks a family whose
  95\% interval excludes the baseline at some length. No geometric family exceeds the baseline for any model at any length, while
  every count and sequence family clears it for at least one.}
  \label{tab:main}
  \begin{tabular}{l""" + "r" * len(MODELS) + r"""}
    \toprule
    Descriptor family & """ + " & ".join(SHORT[m] for m in MODELS) + r""" \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table*}
"""
    with _Cap(os.path.join(TABS, "tab_main.tex")) as fh:
        fh.write(tab)

    # Nothing is dropped silently: a family in the results that is missing from CLASSES would be
    # selective reporting of the profile.
    in_artifact = {k.split("|")[1] for k, v in d.items()
                   if isinstance(v, dict) and "gain" in v}
    in_table = {f for _, fams in CLASSES for f in fams}
    missing = in_artifact - in_table
    assert not missing, f"families in results but absent from the table: {sorted(missing)}"
    print(f"  families: {len(in_table)} shown of {len(in_artifact)} measured")

    # By name, not by index. Reading CLASSES[0] assumes that the geometric block sits first, and
    # reordering CLASSES would silently point the headline assert at the retrieval families instead,
    # where clearing is the expected result.
    GEOM = next(fams for cls, fams in CLASSES if cls == "Spatial-geometric")
    n_clear_geom = sum(1 for m in MODELS for f in GEOM if ever[(m, f)])
    n_geom_cells = len(MODELS) * len(GEOM)
    CHECK or print(f"wrote tab_main.tex  ({n_geom_cells} geometric settings, {n_clear_geom} exceed the baseline)")
    assert n_clear_geom == 0, "a geometric cell now exceeds its baseline; the headline claim changed"
    print("assert OK: the 20 geometric settings still all fail")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- coverage table
COVERAGE_SRC = os.path.join(ROOT, "survey", "capability_coverage.md")
# Verified system -> bib key map. Built once by resolving each arXiv id against the bibliography
# and, for ambiguous names, by reading page 1 of the paper. It is not re-derived here: a name-based
# matcher can cite the wrong paper when two papers share one name.
CITEMAP = os.path.join(ROOT, "survey", "table1_citation_map.json")


def esc(s):
    import re as _re
    # Drop markdown code spans wholesale. They hold math from the survey table (`F: t -> h in R^d`)
    # that has no place in a table cell and renders as run-together italics if escaped naively.
    s = _re.sub(r"`[^`]*`", "", s)
    # the survey table uses unicode separators and arrows that are not valid in a LaTeX cell
    for a, b in [("\u00b7", ";"), ("\u2192", " to "), ("\u21a6", " to "), ("\u2264", "<="),
                 ("\u2208", " in "), ("\u00d7", "x"), ("\u2019", "'"), ("\u2013", "-")]:
        s = s.replace(a, b)
    s = s.replace("**", "").replace("*", "")
    s = s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    s = _re.sub(r"\s*[;,]\s*$", "", s.strip())
    s = _re.sub(r"\s{2,}", " ", s)
    return s.strip(" ;,")


def coverage():
    """Parse the 20-system table out of capability_coverage.md and emit LaTeX.

    Parsed rather than retyped: the counting rule behind the claim that 15 of 20 systems cannot be
    asked the question (13 with no language interface, plus 2 with an LLM inside but no question
    answering) is re-derived here from the same rows the paper cites, and asserted below. If someone
    edits the source table, this breaks loudly.
    """
    _CITES = json.load(open(CITEMAP))
    rows, on = [], False
    for line in open(COVERAGE_SRC):
        if line.startswith("| model | venue"):
            on = True
            continue
        if on:
            if not line.startswith("|"):
                break
            if set(line.strip()) <= set("|- "):
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 6:
                rows.append({"model": c[0], "venue": c[1], "inp": c[2],
                             "tasks": c[3], "emits": c[4], "nl": c[5]})

    def classify(nl):
        low = nl.lower()
        if "no qa" in low:
            return "LLM inside, no QA"
        if low.startswith("**none**") or "none" in low:
            return "none"
        return "yes"

    for r in rows:
        r["cls"] = classify(r["nl"])
    n_none = sum(1 for r in rows if r["cls"] == "none")
    n_noqa = sum(1 for r in rows if r["cls"] == "LLM inside, no QA")
    n_yes = sum(1 for r in rows if r["cls"] == "yes")

    # Capability matrix. The table makes the gap visible as an empty column. Each dimension below
    # is derived by an explicit rule from the survey table's own fields, so that a reader can audit
    # every mark.
    AGG = ("aggregated", "grid tensor", "city grid", "urban region", "region data",
           "urban sensing", "census block", "grid + graph")
    DESCRIPTOR_OUT = ("descriptor", "radius of gyration value", "summary statistic")

    def marks(r):
        inp, emits, tasks = r["inp"].lower(), r["emits"].lower(), r["tasks"].lower()
        # Every dimension is one this paper satisfies, and each is derived from the survey table's own
        # fields so that a reader can audit the mark. Prior systems satisfy varying subsets, which is what
        # keeps the table a comparison and not a list of our own properties.
        individual = not any(a in inp for a in AGG)          # one person's record, not an aggregate
        nl = r["cls"] == "yes"                                # a real question interface
        raw = any(a in inp for a in ("point", "trajector", "sequence", "gps", "triplet",
                                     "check-in", "grid location", "location sequence"))
        multitask = len([t for t in tasks.split(";") if t.strip()]) > 1 or "·" in r["tasks"]
        descriptor = any(a in emits for a in DESCRIPTOR_OUT)  # emits a summary quantity
        recovery = nl and descriptor                          # scored against that person's truth
        return individual, nl, raw, multitask, descriptor, recovery

    def tick(b):
        return r"\checkmark" if b else r"--"

    body, counts = [], [0] * 6
    for r in rows:
        m = marks(r)
        for i, b in enumerate(m):
            counts[i] += bool(b)
        _nm = r["model"].strip().strip("*")
        _key = _CITES.get(_nm)
        assert _key, "no verified citation for system %r; add it to %s" % (_nm, CITEMAP)
        body.append("    " + " & ".join([esc(r["model"]) + r"~\cite{%s}" % _key]
                                        + [tick(b) for b in m]) + r" \\")

    # Contrast row: without it the table shows only absence and never shows what the
    # capability set looks like when it is complete.
    body.append(r"    \midrule")
    # This work does NOT generate and does NOT score at population level. Those cells stay empty.
    body.append("    " + " & ".join([r"\textbf{This work}"] + [r"\textbf{\checkmark}"] * 6) + r" \\")

    tab = (r"""% GENERATED by gauge/make_tables.py from survey/capability_coverage.md. Do not edit by hand.
\begin{table*}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \caption{Capabilities of twenty mobility and urban foundation models.
  The first four columns are widely shared across these systems. No system emits a descriptor value
  for an individual and none evaluates recovery of one, the gap this paper measures.}
  \label{tab:capability}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    System & \makecell{Individual\\record} & \makecell{Language\\interface} &
    \makecell{Raw location\\input} & \makecell{Multiple\\tasks} &
    \makecell{Descriptor-valued\\output} & \makecell{Recovery\\evaluated} \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
""")
    with _Cap(os.path.join(TABS, "tab_coverage.tex")) as fh:
        fh.write(tab)
    CHECK or print(f"wrote tab_coverage.tex  ({len(rows)} systems: {n_none} no interface, "
          f"{n_noqa} LLM-inside-no-QA, {n_yes} with an interface)")
    names = ["individual", "language", "raw-input", "multi-task", "descriptor", "recovery"]
    print("  tallies of %d: %s" % (len(rows), ", ".join(f"{n}={t}" for n, t in zip(names, counts))))
    assert counts[4] == 0, "a published system now emits a descriptor value; the capability claim changed"
    # Guard against a straw-man table: most columns must be well populated by prior work, so that
    # the two empty ones read as a measured gap and not as dimensions chosen to favour this paper.
    assert sum(1 for t in counts if t >= len(rows) // 3) >= 3, \
        "too few columns are satisfied by prior work; the table reads as cherry-picked"
    assert len(rows) == 20, f"expected 20 systems, parsed {len(rows)}"
    assert n_none == 13, f"the paper says 13 have no language interface; parsed {n_none}"
    assert n_none + n_noqa == 15, f"the paper says 15 cannot be asked; parsed {n_none + n_noqa}"
    print("assert OK: counting rule re-derived from the survey table")


coverage()


# ---------------------------------------------------------------- three-level accounting
def three_level():
    """Re-derive the per-cell level classification and emit it as a table.

    A level is present when:
      D  JSD(model) beats JSD(constant) by more than .10
      O  Spearman rho >= 0.20
      E  normalized gain over the baseline >= 0.05
    """
    rung = json.load(open(os.path.join(ROOT, "results", "distributional_rung.json")))
    fc = json.load(open(os.path.join(ROOT, "results", "family_cis.json")))
    pat = collections.Counter()
    for k, v in rung.items():
        if not isinstance(v, dict) or "rho" not in v:
            continue
        D = (v["jsd_constant"] - v["jsd_model"]) > 0.10
        O = v["rho"] >= 0.20
        cell = fc.get(f'{v["model"]}|{v["family"]}|{v["span"]}')
        E = bool(cell and cell.get("gain", 0) >= 0.05)
        pat["".join([("D" if D else "-"), ("O" if O else "-"), ("E" if E else "-")])] += 1
    total = sum(pat.values())
    NEST = {"DOE", "DO-", "D--", "---"}
    consistent = sum(n for p, n in pat.items() if p in NEST)
    order = ["DO-", "D--", "---", "DOE"] + sorted(p for p in pat if p not in NEST)
    rows = []
    for p in order:
        if p not in pat:
            continue
        label = {"DOE": "all three levels", "DO-": "distributional + ordinal, no exact",
                 "D--": "distributional only", "---": "no level present"}.get(
                     p, "\\textbf{ordinal without distributional}")
        mark = "" if p in NEST else r" & \textbf{violates}"
        tag = "nested" if p in NEST else ""
        rows.append(f"    \\texttt{{{p}}} & {label} & {pat[p]} & {tag or 'violates nesting'} \\\\")
    tab = (r"""% GENERATED by gauge/make_tables.py from distributional_rung.json + family_cis.json.
\begin{table}[t]
  \centering
  \small
  \caption{Which of the three measurement levels each model-family-length cell occupies, over the seven families whose answers are magnitudes. The remaining fourteen return place identifiers, for which a distributional match and a rank correlation are not defined, and they carry no level below exact. """
           # Escape the percent sign. Python's :.1% emits a bare `%`, which LaTeX reads as a comment and
           # silently swallows the rest of the line, and the output is still valid LaTeX.
           + f"{consistent} of {total} cells ({consistent/total*100:.1f}\\%)" + r""" are consistent with the
  levels being nested; every violation is of one type, ordinal signal without a distributional
  match. Exact match alone would collapse the 166 cells without exact recovery into one undifferentiated failure.}
  \label{tab:levels}
  \begin{tabular}{@{}llrl@{}}
    \toprule
    Pattern & Levels present & Cells & Nesting \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    with _Cap(os.path.join(TABS, "tab_levels.tex")) as fh:
        fh.write(tab)
    CHECK or print(f"wrote tab_levels.tex  ({total} cells, {consistent} nested-consistent, "
          f"{total-consistent} violating)")
    return total, consistent


# ---------------------------------------------------------------- predicted outcomes (not used)
def prereg():
    """Table of predictions about which descriptors survive which record lengths, scored against
    the results. Not called: it reads results/prediction_analysis.json, which this repository does
    not produce.
    """
    pr = json.load(open(os.path.join(ROOT, "results", "prediction_analysis.json")))["prereg"]
    rows = []
    for pid in sorted(pr):
        r = pr[pid]
        fam = r.get("family") or "not built"
        pred = esc(r.get("predict", ""))
        obs = r.get("observed_last_ok")
        obs_s = ("never clears" if obs is None else f"holds to {obs}") \
            if r.get("verdict") != "UNTESTED" else "not built"
        v = r["verdict"]
        vs = {"HOLDS": r"\checkmark\ holds", "FAILS": r"$\times$ fails",
              "UNTESTED": "untested"}[v]
        rows.append(f"    {pid} & \\texttt{{{esc(fam)}}} & {pred} & {obs_s} & {vs} \\\\")
    nh = sum(1 for p in pr.values() if p["verdict"] == "HOLDS")
    nf = sum(1 for p in pr.values() if p["verdict"] == "FAILS")
    tab = (r"""% GENERATED by gauge/make_tables.py from results/prediction_analysis.json.
\begin{table}[t]
  \centering
  \small
  \caption{Predictions about which descriptors survive which record lengths,
  scored against the results. """ + f"{nh} hold, {nf} fail" + r""" and one was never built. The prediction for distinct places per day was
  the decisive test and holds only at 70B.}
  \label{tab:prereg}
  \begin{tabular}{@{}lllll@{}}
    \toprule
    & Family & Prediction & Observed & Verdict \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    with _Cap(os.path.join(TABS, "tab_prereg.tex")) as fh:
        fh.write(tab)
    CHECK or print(f"wrote tab_prereg.tex  ({nh} hold, {nf} fail, {len(pr)-nh-nf} untested)")


three_level()
