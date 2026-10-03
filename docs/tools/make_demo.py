#!/usr/bin/env python3
"""Build the "Try a question" demo of the GAUGE project page.

WHAT THE DEMO IS
  One radius-of-gyration question (a spatial-geometric family) asked about a MADE-UP record. The
  record is an invented person with a home, a workplace, two cafes and a gym. It is not taken
  from YJMob100K, from any items file, or from any other dataset, and it was never derived from one.

HOW THE RECORD IS MADE (all of it is in this file, nothing is read from data)
  1. ANCHORS: six invented places on a grid of 500 m cells, and PLAN: 32 hand-written
     (day, timeslot, place) rows. Rows with the same place get the same cell, as in real records.
  2. The anchors are passed through the benchmark's own per-question transform, `rigid` from
     <root>/gauge/coord_records.py (rotation, reflection and an integer shift, seeded by
     SEED_RIGID).
  3. The question, the whole-number key and the unrounded value come from the benchmark's own
     generator function `F_rg_numeric` in <root>/gauge/tasks_main.py, called on those points.
     Nothing is re-implemented. The rows are rendered with the benchmark's own `render_xy`.

WHAT IS PROVED WHEN THE SCRIPT RUNS
  * File access is audited with an audit hook. Under <root>, only the benchmark's module sources
    (gauge/*.py, plus cached .pyc beside them) and the appendix file (--appendix) may be opened.
    Opening anything under data/, results/, hpc/, analysis/ or any .parquet/.jsonl/.csv/.json file
    stops the script. The list of files opened under <root> is printed.
  * F_rg_numeric returned a record (it returns None below 1 km).
  * The rendered text, parsed back and recomputed with plain Python (no numpy), gives the same
    unrounded value and the same whole-number key as the original function.
  * The question text produced by F_rg_numeric equals the question the appendix prints.
  * The instruction, header line, closing line and reference program shown on the page are taken
    from the paper's appendix.tex (Appendix B and C), not typed here. That file is not part of this
    repository; --appendix names it (default: <root>/paper/tex/sections/appendix.tex).
  ROOT is read-only: bytecode writing is switched off before the first import.

USAGE
  python3 -B docs/tools/make_demo.py --root "<repository root>" --appendix "<appendix.tex>" --html docs/index.html
  python3 -B docs/tools/make_demo.py --root "<repository root>" --appendix "<appendix.tex>" --html docs/index.html --check
  python3 -B docs/tools/make_demo.py --root "<repository root>" --appendix "<appendix.tex>" --json demo.json
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True  # ROOT is read-only: never let Python drop .pyc files there

SPAN = 32          # rows in the record; the geometric families start at 32 rows
# Seeds the benchmark's rigid transform (rotation, reflection, translation). Picked for display
# only, so the printed coordinates are of the same size as the paper's example and spread along
# both axes. Seeds 1 to 15 all give the same whole-number key (4); the rotation only moves the
# unrounded value between 3.93 and 4.08 through integer rounding.
SEED_RIGID = 8

# An invented person, in grid cells of 500 m (local frame, before the rigid transform).
ANCHORS = {
    "home": (0, 0),
    "work": (15, 9),
    "road": (7, 4),     # one observation on the way to work
    "cafe": (8, 6),
    "shop": (9, 5),     # a second cafe on the second day
    "gym": (-6, 5),
}
# (day, timeslot, anchor). Timeslots are half-hours, 0 to 47, as in the benchmark's records.
PLAN = [
    (23, 14, "home"), (23, 15, "home"), (23, 16, "home"), (23, 17, "home"),
    (23, 18, "road"),
    (23, 19, "work"), (23, 20, "work"), (23, 21, "work"), (23, 22, "work"), (23, 23, "work"),
    (23, 25, "cafe"), (23, 26, "cafe"),
    (23, 28, "work"), (23, 29, "work"), (23, 30, "work"), (23, 31, "work"),
    (23, 36, "gym"), (23, 37, "gym"),
    (23, 40, "home"), (23, 41, "home"), (23, 42, "home"),
    (24, 20, "home"), (24, 21, "home"), (24, 22, "home"), (24, 23, "home"),
    (24, 26, "shop"), (24, 27, "shop"),
    (24, 31, "home"), (24, 32, "home"),
    (24, 40, "home"), (24, 41, "home"), (24, 42, "home"),
]

MARK = {
    "prompt": ("<!--DEMO-PROMPT:BEGIN-->", "<!--DEMO-PROMPT:END-->"),
    "key": ("<!--DEMO-KEY:BEGIN-->", "<!--DEMO-KEY:END-->"),
}


# ----------------------------------------------------------------------------- file-access audit
OPENED: list[str] = []
_AUDIT_ROOT: list[str] = []


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            OPENED.append(os.path.abspath(os.fsdecode(args[0])))
        except Exception:
            pass


def audit_report(root: str, appendix: str):
    """Fail if anything outside the permitted set was opened under root. Returns the list."""
    root = os.path.abspath(root) + os.sep
    under = sorted({p for p in OPENED if p.startswith(root)})
    ok_prefix = (root + "gauge" + os.sep,)
    ok_files = {os.path.abspath(appendix)}
    bad = []
    for p in under:
        rel = p[len(root):]
        if p in ok_files:
            continue
        if p.startswith(ok_prefix) and (p.endswith(".py") or p.endswith(".pyc") or os.path.isdir(p)):
            continue
        bad.append(rel)
    forbidden = [r for r in (p[len(root):] for p in under)
                 if re.match(r"(data|results|hpc|analysis|prior|papers|docs|admin)/", r)
                 or r.endswith((".parquet", ".jsonl", ".csv", ".json", ".npz", ".npy", ".pkl"))]
    if bad or forbidden:
        raise SystemExit("AUDIT FAILED: files opened under ROOT outside the permitted set:\n  "
                         + "\n  ".join(sorted(set(bad + forbidden))))
    return [p[len(root):] for p in under]


# ----------------------------------------------------------------------------- appendix text
def texttt_segments(tex: str):
    """Every \\texttt{...} segment of the appendix, in order, with TeX escapes undone."""
    out = []
    for line in tex.splitlines():
        for m in re.finditer(r"\\texttt\{(.*?)\}(?=\s*(?:\\\\|\\par|$))", line):
            s = m.group(1).replace(r"\_", "_").replace(r"\ldots", "...")
            out.append(s)
    return out


def appendix_pieces(tex: str):
    seg = texttt_segments(tex)

    def find(prefix, start=0):
        for i in range(start, len(seg)):
            if seg[i].startswith(prefix):
                return i
        raise SystemExit(f"appendix.tex: no \\texttt line starting with {prefix!r}")

    i0 = find("You answer questions about location records")
    instr = " ".join(seg[i0:i0 + 4])
    # the coordinate record form is the second "Here is a person's location record" block
    h1 = find("Here is a person's location record")
    h2 = find("Here is a person's location record", h1 + 1)
    header = seg[h2]
    qi = find("Question: Treating each unit", h2)
    question = seg[qi]
    closing = seg[qi + 1]
    r0 = find("P = np.array(points)")
    ref_code = seg[r0:r0 + 2]
    return dict(instr=instr, header=header, question=question, closing=closing, ref_code=ref_code)


def typo(s: str) -> str:
    """Show TeX quotes the way the PDF prints them: `x' becomes curly quotes."""
    return s.replace("`", "\u2018").replace("'", "\u2019")


# ----------------------------------------------------------------------------- the record
def build_record(np, T, CR):
    base = np.array([ANCHORS[a] for (_, _, a) in PLAN], dtype=float)
    days = [d for (d, _, _) in PLAN]
    tods = [t for (_, t, _) in PLAN]
    assert len(PLAN) == SPAN, f"plan has {len(PLAN)} rows, expected {SPAN}"
    assert sorted(zip(days, tods)) == list(zip(days, tods)), "rows must be in time order"
    assert len(set(zip(days, tods))) == SPAN, "duplicate (day, timeslot)"
    assert all(0 <= t <= 47 for t in tods)
    rng = np.random.default_rng(SEED_RIGID)
    P = CR.rigid(base, rng)                      # benchmark's own per-question transform
    it = T.F_rg_numeric(None, rng, SPAN, P)      # benchmark's own gold, question and magnitude
    if it is None:
        raise SystemExit("F_rg_numeric returned None (radius of gyration under 1 km)")
    body = CR.render_xy(np.array(days), np.array(tods), P)   # benchmark's own renderer
    return days, tods, P, it, body


def plain_rg(body: str):
    """Independent recomputation from the rendered text, plain Python, no numpy."""
    pts = [(int(x), int(y)) for _, _, x, y in
           (re.match(r"d(\d+) t(\d+) x(-?\d+) y(-?\d+)$", ln).groups() for ln in body.splitlines())]
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    rg = math.sqrt(sum((x - mx) ** 2 + (y - my) ** 2 for x, y in pts) / n) * 0.5
    return rg, n


# ----------------------------------------------------------------------------- html fragments
def esc(s: str) -> str:
    return html.escape(s, quote=False)


def short_question(question_line: str) -> str:
    """First sentence of the printed question line (it keeps the 'Question:' label)."""
    head, sep, _ = question_line.partition("? ")
    return head + "?" if sep else question_line


def render_prompt_html(pieces, body: str) -> str:
    rows = body.splitlines()
    q = lambda t: '<span data-quote>' + esc(typo(t)) + '</span>'   # verbatim from Appendix B
    shown = esc(typo(short_question(pieces["question"])))
    header = esc(typo(pieces["header"]))
    # The full prompt, laid out as Appendix B prints it: two rows, then "...", for the record.
    full = "\n".join([
        q(pieces["instr"]),
        "",
        q(pieces["header"]),
        "<span data-madeup>" + esc(rows[0]) + "\n" + esc(rows[1]) + "</span>",
        "...",
        q(pieces["question"]),
        q(pieces["closing"]),
    ])
    return "\n".join([
        '<div class="prompt">',
        f'<pre class="p-head" data-quote>{header}</pre>',
        f'<pre class="p-rows" tabindex="0" aria-label="The {SPAN} rows of the made-up record, '
        'scrollable"><span data-madeup>' + esc(body) + '</span></pre>',
        f'<pre class="p-q" data-quote>{shown}</pre>',
        '</div>',
        '<details class="fullprompt">',
        '<summary>Show full prompt</summary>',
        f'<pre class="p-full">{full}</pre>',
        '</details>',
    ])


def render_key_html(it) -> str:
    key = it["a"]
    mag = it["mag"]
    return "\n".join([
        f'<div id="demo-key" class="keyfacts" data-key="{key}" data-value="{mag:.3f}">',
        f'<p class="ansline">Exact answer: <b class="ansnum" data-madeup>{key} km</b> '
        f'<span class="mut">(<span data-madeup>{mag:.3f}</span> before rounding)</span></p>',
        '</div>',
    ])


def splice(page: str, name: str, fragment: str) -> str:
    b, e = MARK[name]
    pat = re.compile(re.escape(b) + r".*?" + re.escape(e), re.S)
    if not pat.search(page):
        raise SystemExit(f"index.html has no {b} ... {e} markers")
    return pat.sub(lambda m: b + "\n" + fragment + "\n" + e, page, count=1)


def region(page: str, name: str) -> str:
    b, e = MARK[name]
    m = re.search(re.escape(b) + r"\n?(.*?)\n?" + re.escape(e), page, re.S)
    return m.group(1) if m else ""


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="repository root holding gauge/ (read only)")
    ap.add_argument("--appendix", help="the paper's appendix.tex (default: <root>/paper/tex/sections/appendix.tex)")
    ap.add_argument("--html", help="index.html whose DEMO markers are filled in")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the page differs from a fresh build")
    ap.add_argument("--json", help="also write the demo data to this JSON file")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    scripts = os.path.join(root, "gauge")
    for f in ("tasks_main.py", "coord_records.py"):
        if not os.path.isfile(os.path.join(scripts, f)):
            raise SystemExit(f"{f} not found under {scripts}")
    sys.addaudithook(_hook)                     # record every file opened from here on
    sys.path.insert(0, scripts)
    import numpy as np
    import tasks_main as T      # the generator: F_rg_numeric
    import coord_records as CR      # the rigid transform and renderer

    appendix = a.appendix or os.path.join(root, "paper", "tex", "sections", "appendix.tex")
    tex = open(appendix, encoding="utf-8").read()
    pieces = appendix_pieces(tex)

    days, tods, P, it, body = build_record(np, T, CR)

    # ---- checks (written to be able to fail)
    assert it["task"] == "gyration_km" and it["skmob"] == "radius_of_gyration", it["task"]
    rg_plain, n_rows = plain_rg(body)
    assert n_rows == SPAN == body.count("\n") + 1, "row count"
    assert abs(rg_plain - it["mag"]) < 1e-9, f"plain Python {rg_plain} vs original {it['mag']}"
    assert str(int(round(rg_plain))) == it["a"], f"key {it['a']} vs plain {rg_plain}"
    assert "Question: " + it["q"] == pieces["question"], \
        "question from F_rg_numeric differs from the appendix:\n  " + it["q"] + "\n  " + pieces["question"]
    assert pieces["closing"] == "Answer with the value only.", pieces["closing"]
    assert pieces["header"] == "Here is a person's location record. Each line is day, timeslot, place."
    assert pieces["ref_code"][1].startswith("r_g = 0.5 * np.sqrt("), pieces["ref_code"]

    opened = audit_report(root, appendix)   # raises if anything outside the permitted set was opened

    data = dict(
        family="gyration_km (radius of gyration, spatial-geometric)",
        made_up=True,
        rows=n_rows,
        key=it["a"],
        unrounded_km=round(float(it["mag"]), 6),
        question=it["q"],
        body=body,
        original_functions=["tasks_main.F_rg_numeric", "coord_records.rigid", "coord_records.render_xy"],
        files_opened_under_root=opened,
    )
    print(f"made-up record: {n_rows} rows over days {sorted(set(days))}")
    print(f"family        : {data['family']}")
    print(f"original code : F_rg_numeric -> key {it['a']} km, unrounded {it['mag']:.6f} km")
    print(f"plain python  : {rg_plain:.6f} km  (agrees)")
    print("question      : matches appendix.tex")
    print("first rows    :", body.splitlines()[:2])
    mods = sorted(os.path.relpath(m.__file__, root) for m in list(sys.modules.values())
                  if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(scripts + os.sep))
    print("modules       : imported from ROOT:", ", ".join(mods))
    print(f"audit         : {len(opened)} files opened under ROOT, all permitted:")
    for p in opened:
        print("                ", p)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        print("wrote", a.json)

    if a.html:
        page = open(a.html, encoding="utf-8").read()
        new_prompt = render_prompt_html(pieces, body)
        new_key = render_key_html(it)
        if a.check:
            same = region(page, "prompt") == new_prompt and region(page, "key") == new_key
            print("CHECK:", "page matches a fresh build" if same else "PAGE DIFFERS from a fresh build")
            sys.exit(0 if same else 1)
        page = splice(page, "prompt", new_prompt)
        page = splice(page, "key", new_key)
        with open(a.html, "w", encoding="utf-8") as fh:
            fh.write(page)
        print("wrote demo regions into", a.html)


if __name__ == "__main__":
    main()
