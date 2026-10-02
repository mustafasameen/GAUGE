#!/usr/bin/env python3
"""Check that every number on the GAUGE project page is printed in the paper, in the right sentence.

THE LAYERS (all must pass)
  1. Numbers. Every number in the page's content text (body, <title>, alt text, aria-labels, title
     attributes, description/Open Graph/Twitter text) must appear in `pdftotext` output of the paper
     PDF. Years 2024 to 2026, URLs and the DOI are ignored. Signs, thousands commas and percent signs
     count ("+0.932", "34,200", "89%"). The paper writes some values without the leading zero
     (".20"), so "0.20" is accepted as ".20".
  2. Quotes. Any element marked data-quote must appear in the paper text, compared on letters, digits
     and signs only (hyphenation, line breaks and punctuation spacing do not matter).
  3. Claims. A bare number match is weak: "46" occurs somewhere in any paper. So every number on the
     page must sit inside a phrase listed in CLAIMS (or inside a data-quote, or be part of a model
     or dataset name in IDENTIFIERS). A claim pairs a phrase that must be on the page, literally,
     with the sentence of the paper that supports it. The checker verifies that the paper sentence
     is in the paper, and that EVERY number in the page phrase is printed in that sentence. A number
     the page derives by arithmetic is therefore refused: it is not printed in the paper sentence.
     Number words count ("twenty" for 20, "seven" for 7). Any number left outside a claim or quote
     fails as UNBOUND.
  4. Table rows (TABLE_CLAIMS) are read from `pdftotext -layout`.

WHAT IS EXCLUDED, AND WHY
  * the BibTeX block (id="bibtex-block"): the DOI and year are citation data;
  * elements marked data-madeup: the invented record in "Try a question" and the answer computed
    from it. They are not from the paper by construction. The checker lists how many numbers it
    skips and FAILS if a data-madeup element appears anywhere outside the element with id="demo".

SOURCES
  With --tex-dir (the paper's tex folder) the checker also finds each supporting sentence in the
  LaTeX files and prints file:line, so every page number can be traced to a source line.

USAGE
  python3 -B check_numbers.py --pdf PAPER.pdf [--tex-dir PAPER/tex] [--sources] [--html ../index.html]
  (without --pdf it uses ../static/GAUGE_UMFM26.pdf once that file has been added)
  python3 -B check_numbers.py --pdf PAPER.pdf --self-test
  (--text FILE may replace --pdf when the PDF has already been run through pdftotext)
  Exit status 0 when nothing fails, 1 otherwise.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import time
import unicodedata
from html.parser import HTMLParser

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HTML = os.path.join(HERE, "..", "index.html")

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr"}
META_TEXT_KEYS = {"description", "og:title", "og:description", "og:image:alt", "twitter:title",
                  "twitter:description", "twitter:image:alt", "citation_title",
                  "citation_conference_title"}
IGNORED_YEARS = {"2024", "2025", "2026"}
MINUS = "\u2212"
WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}

# Model and dataset names that carry digits. They must be printed in the paper, exactly so.
IDENTIFIERS = ["GPT-4o", "Llama-3.1-70B", "YJMob100K"]

# The abstract's last sentence (main.tex:62-64). The fifth finding card is bound to it.
ABSTRACT_LAST = ("A permutation of the true values across people reproduces every distributional score we compute "
                 "while destroying every individual, and the models order people on geometric quantities in "
                 "53 of 100 settings while stating no individual's value")

# ------------------------------------------------------------------------------------------ claims
# (phrase literally on the page, sentence of the paper that supports it[, phrase to look for in the
# LaTeX source when it differs from the printed one, e.g. for figure, table and section labels]).
CLAIMS = [
    # the strip under the teaser, and the description text in the page head. In the PDF text this
    # sentence is split by the running header and by the Figure 1 caption, so each number is bound
    # to the half that holds it; the LaTeX phrase (third item) is the whole sentence.
    ("34,200 questions", "a benchmark of 34,200 questions across 21",
     "a benchmark of 34,200 questions across 21 descriptor families and seven record lengths"),
    ("21 descriptor families", "a benchmark of 34,200 questions across 21",
     "a benchmark of 34,200 questions across 21 descriptor families and seven record lengths"),
    ("7 record lengths", "descriptor families and seven record lengths",
     "a benchmark of 34,200 questions across 21 descriptor families and seven record lengths"),
    ("5 open-weight models", "five open-weight models from 3.8B to 70B"),
    ("16,388 distinct people", "With questions drawn from 16,388 distinct people"),
    # venue, licence, dataset
    ("2nd ACM SIGSPATIAL International Workshop", "The 2nd ACM SIGSPATIAL International Workshop on Urban Mobility Foundation Models"),
    ("November 03\u201306, 2026", "November 03\u201306, 2026"),
    ("CC BY 4.0", "Creative Commons Attribution 4.0 International License", "@main.tex:11-12 (\\setcopyright{cc}, \\setcctype{by}; acmart prints the licence text)"),
    ("Built on YJMob100K", "drawing records from YJMob100K"),
    ("Five open-weight models, plus GPT-4o on a matched subset",
     "GPT-4o, scored on a matched 960-question subset"),
    # figure captions
    ("The same model recovers distinct places per day but not radius of gyration",
     "recovers distinct places per day and does not recover radius of gyration"),
    ("Figure 1 in the paper", "Figure 1: (a) The raw material",
     "The raw material: one person's location record"),
    ("Figure 5 in the paper", "Figure 5: Normalized gain against record length for all five models",
     "Normalized gain against record length for all five models"),
    ("Figure 4 in the paper", "Figure 4: Normalized gain with 95% intervals for the decomposition controls",
     "Normalized gain with 95% intervals for the decomposition controls"),
    ("Figure 3 in the paper", "Figure 3: A frontier model on the same items",
     "A frontier model on the same items"),
    ("GPT-4o leads on the contrast families and clears no geometric cell",
     "GPT-4o leads on the contrast families and clears no geometric cell"),
    ("Largest of a list: every model clears. Distance between two points: none does",
     "Every model clears on the largest of a list; none clears on the distance between two points"),
    ("No model lifts the geometric families out of the threshold at any length",
     "no model lifts the geometric families out of the threshold at any length"),
    # finding cards
    ("20 of 20", "twenty of twenty never clear their baseline at any record length"),
    ("every record length from 32 to 512 rows", "at every length they are generated at, 32 rows to 512"),
    ("GPT-4o clears no geometric cell either",
     "GPT-4o, scored on a matched 960-question subset, clears no geometric cell either"),
    ("89%", "89% of model-family pairs either clear their baseline on the longest record we test or clear it nowhere"),
    ("Of 105 model-family pairs, 47 clear their baseline on the longest record",
     "Across the 105 model-family pairs the study covers, 47 (45%) exceed their baseline by the interval bar at the longest record"),
    ("46 never do", "46 (44%) never clear it at any length"),
    ("only 12 clear it and then stop", "only 12 (11%) clear it and then stop"),
    ("-0.414", "by -0.414, -0.464 and -0.231 for required states of 16, 32 and 64 items"),
    ("falls by 0.414 as the record grows around 16 items of state",
     "by -0.414, -0.464 and -0.231 for required states of 16, 32 and 64 items"),
    ("1,155 of 1,155", "reproduce the reference computation on 50 held-out real records, 1,155 of 1,155"),
    ("match the reference on 50 held-out records",
     "reproduce the reference computation on 50 held-out real records, 1,155 of 1,155"),
    ("Three of five models do this",
     "three of the five models write programs that compute the same descriptors correctly on 50 records they never saw"),
    # fifth card, the abstract's last result: its numbers and phrases are bound to main.tex:62-64
    ("53 of 100", ABSTRACT_LAST),
    ("on geometric quantities in 53 of 100 settings", ABSTRACT_LAST),
    ("Models order people", ABSTRACT_LAST),
    ("while stating no individual's value", ABSTRACT_LAST),
    ("A permutation of the true values", ABSTRACT_LAST),
    ("reproduces every distributional score", ABSTRACT_LAST),
    ("while destroying every individual", ABSTRACT_LAST),
    # the same count is printed in section 4.1, which the card cites
    ("53 of 100", "On the ordinal measure 53 of 100 geometric settings reach a rank correlation of at least .20"),
    # where a finding sits in the paper (section labels used on the cards)
    ("\u00a74.1", "L:4.1 Main results: the recovery profile", "Main results: the recovery profile"),
    ("\u00a74.3", "L:4.3 Effect of record length", "Effect of record length"),
    ("\u00a74.7", "L:4.7 Specifying the computation instead of",
     "Specifying the computation instead of performing it"),
    ("Section 4.1 of the paper", "L:4.1 Main results: the recovery profile", "Main results: the recovery profile"),
    ("Section 4.3 of the paper", "L:4.3 Effect of record length", "Effect of record length"),
    ("Section 4.7 of the paper", "L:4.7 Specifying the computation instead of",
     "Specifying the computation instead of performing it"),
    # the demo
    ("The 32 rows of the made-up record", "On the 32-row record printed above this returns"),
]

# Table rows, matched on `pdftotext -layout`: (page phrases, regex for the header row, regex for the row,
# phrase that locates the row in the LaTeX source). None are on the page at present; the machinery stays so a
# table value can be added later and still be checked.
TABLE_CLAIMS = []


# ------------------------------------------------------------------------------------ text helpers
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.replace(MINUS, "-").replace("\u00a0", " ")


def squash(s: str) -> str:
    """Letters, digits and signs only: line breaks and hyphenation do not matter, but a changed
    digit, decimal point, percent sign or sign does."""
    s = norm(s).lower()
    s = re.sub(r"(?<![\w])\+(?=\.?\d)", "plus", s)
    s = re.sub(r"(?<![\w])-(?=\.?\d)", "minus", s)
    s = s.replace("%", "pct")
    s = re.sub(r"(?<=\d)\.(?=\d)", "dot", s)
    return re.sub(r"[^0-9a-z]+", "", s)


def flat(parts) -> str:
    return re.sub(r"\s+", " ", norm(" ".join(parts))).strip()


NUM = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)")


def extract_numbers(text: str):
    """Numbers in text as strings such as '34,200', '+0.932', '-0.414', '89%'."""
    text = norm(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"10\.\d{4,9}/\S+", " ", text)              # DOIs
    text = re.sub(r"(?<![\w.])\.(?=\d)", "0.", text)           # the paper's ".20" style
    out = []
    for m in NUM.finditer(text):
        whole, frac, pct = m.group(1), m.group(2) or "", m.group(3)
        tok = whole + frac
        if tok in IGNORED_YEARS and not pct:
            continue
        sign = ""
        i = m.start() - 1
        if i >= 0 and text[i] in "+-" and (i == 0 or text[i - 1] in " \t\n([/"):
            sign = text[i]
        out.append(sign + tok + pct)
    return out


def canon(tok: str) -> str:
    """Compare numbers by value text: no sign, no percent sign, no thousands commas."""
    return tok.lstrip("+-").rstrip("%").replace(",", "")


def printed_numbers(sentence: str) -> set:
    nums = {canon(t) for t in extract_numbers(sentence)}
    for w in re.findall(r"[A-Za-z]+", sentence.lower()):
        if w in WORD_NUMBERS:
            nums.add(str(WORD_NUMBERS[w]))
    return nums


def find_in_pdf(token: str, pdf_text: str):
    """Context snippet if the paper holds the token as a standalone number, else None."""
    variants = [token]
    bare = token.lstrip("+-")
    sign = token[: len(token) - len(bare)]
    if bare.startswith("0."):
        variants.append(sign + bare[1:])
    for v in variants:
        pat = re.compile(r"(?<![\d.,])" + re.escape(v) + r"(?!\d)(?!\.\d)(?!,\d)")
        m = pat.search(pdf_text)
        if m:
            a, b = max(0, m.start() - 28), min(len(pdf_text), m.end() + 28)
            return re.sub(r"\s+", " ", pdf_text[a:b])
    return None


# ------------------------------------------------------------------------------------ page parsing
class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.chunks = []           # content text that must match the paper
        self.madeup = []           # text inside data-madeup elements (excluded)
        self.madeup_outside_demo = []
        self.quotes = []
        self._quote_bufs = []
        self._skip = 0
        self._madeup = 0
        self._demo = 0

    def _attr_text(self, attrs):
        d = dict(attrs)
        for k in ("alt", "title", "aria-label"):
            if d.get(k):
                self.chunks.append(d[k])

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "meta":
            key = d.get("name") or d.get("property")
            if key in META_TEXT_KEYS and d.get("content"):
                self.chunks.append(d["content"])
            return
        if not (self._madeup or self._skip):
            self._attr_text(attrs)
        if tag in VOID:
            return
        flags = {"skip": tag in ("script", "style") or d.get("id") == "bibtex-block",
                 "madeup": "data-madeup" in d, "quote": "data-quote" in d, "demo": d.get("id") == "demo"}
        if flags["madeup"] and not self._demo and not flags["demo"]:
            self.madeup_outside_demo.append(tag)
        self.stack.append((tag, flags))
        self._skip += flags["skip"]
        self._madeup += flags["madeup"]
        self._demo += flags["demo"]
        if flags["quote"]:
            self._quote_bufs.append([])

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                while len(self.stack) > i:
                    t, f = self.stack.pop()
                    self._skip -= f["skip"]
                    self._madeup -= f["madeup"]
                    self._demo -= f["demo"]
                    if f["quote"]:
                        self.quotes.append(" ".join(self._quote_bufs.pop()))
                return

    def handle_data(self, data):
        if self._skip or not data.strip():
            return
        if self._madeup:
            self.madeup.append(data)
            return
        self.chunks.append(data)
        for buf in self._quote_bufs:
            buf.append(data)


def parse_page(html: str) -> Page:
    p = Page()
    p.feed(html)
    p.close()
    return p


# ------------------------------------------------------------------------------------ the analysis
def cover_spans(text: str, phrases):
    spans = []
    for ph in phrases:
        ph = ph.strip()
        if not ph:
            continue
        start = 0
        while True:
            k = text.find(ph, start)
            if k < 0:
                break
            spans.append((k, k + len(ph)))
            start = k + 1
    return spans


def mask(text: str, spans):
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def analyze(html: str, pdf_text: str, layout_text: str | None = None):
    page = parse_page(html)
    page_text = flat(page.chunks)
    tokens = extract_numbers(page_text)
    found = {}
    for t in tokens:
        if t not in found:
            found[t] = find_in_pdf(t, pdf_text)
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    # quotes
    big = squash(pdf_text)
    big_layout = squash(layout_text) if layout_text else None
    quote_texts = [re.sub(r"\s+", " ", norm(q)).strip() for q in page.quotes]
    quotes = [(q, squash(q) in big) for q in quote_texts]

    # identifiers
    idents = [(i, i in pdf_text) for i in IDENTIFIERS]

    # claims
    claim_rows = []
    covered_phrases = []
    for c in CLAIMS:
        page_phrase, paper_phrase = c[0], c[1]
        on_page = norm(page_phrase) in page_text
        if paper_phrase.startswith("L:"):          # checked against `pdftotext -layout`
            paper_phrase = paper_phrase[2:]
            in_paper = (squash(paper_phrase) in big_layout) if big_layout else False
        else:
            in_paper = squash(paper_phrase) in big
        disagree = sorted(n for n in {canon(t) for t in extract_numbers(page_phrase)}
                          if n not in printed_numbers(paper_phrase))
        claim_rows.append(dict(page=page_phrase, paper=paper_phrase, on_page=on_page, in_paper=in_paper,
                               disagree=disagree, tex=c[2] if len(c) > 2 else None))
        if on_page:
            covered_phrases.append(norm(page_phrase))
    # table rows
    table_rows = []
    flat_layout = re.sub(r"\s+", " ", norm(layout_text)) if layout_text else None
    for needles, header_rx, row_rx, tex_phrase in TABLE_CLAIMS:
        on_page = all(norm(n) in page_text for n in needles)
        in_paper = None if flat_layout is None else bool(re.search(header_rx, flat_layout)
                                                         and re.search(row_rx, flat_layout))
        table_rows.append(dict(page=" ".join(needles), on_page=on_page, in_paper=in_paper, tex=tex_phrase))
        if on_page:
            covered_phrases.extend(norm(n) for n in needles)

    # numbers outside any claim, quote or identifier
    spans = cover_spans(page_text, covered_phrases + quote_texts + [norm(i) for i, _ in idents])
    leftover = extract_numbers(mask(page_text, spans))

    excluded = extract_numbers(flat(page.madeup))
    bad_claims = [r for r in claim_rows if not (r["on_page"] and r["in_paper"]) or r["disagree"]]
    bad_table = [r for r in table_rows if not r["on_page"] or r["in_paper"] is False]
    missing = [t for t, s in found.items() if s is None]
    bad_quotes = [q for q, ok in quotes if not ok]
    bad_idents = [i for i, ok in idents if not ok]
    unbound = sorted(set(leftover))
    ok = not (missing or bad_quotes or bad_idents or bad_claims or bad_table or unbound
              or page.madeup_outside_demo)
    return dict(page=page, found=found, counts=counts, excluded=excluded, quotes=quotes, idents=idents,
                claims=claim_rows, tables=table_rows, bad_claims=bad_claims, bad_table=bad_table,
                missing=missing, bad_quotes=bad_quotes, bad_idents=bad_idents, unbound=unbound,
                outside=page.madeup_outside_demo, ok=ok)


# ------------------------------------------------------------------------------------ LaTeX sources
def tex_plain(line: str) -> str:
    line = re.sub(r"(?<!\\)%.*$", "", line)
    line = line.replace(r"\%", "%").replace("{,}", ",").replace(r"\_", "_").replace("~", " ")
    line = line.replace("---", "\u2013").replace("--", "\u2013").replace(r"\&", "&")
    line = re.sub(r"\\(?:cite|ref|label|Description)\{[^}]*\}", "", line)
    line = re.sub(r"\\[A-Za-z]+\*?", "", line)
    return line.replace("{", "").replace("}", "").replace("$", "")


def tex_index(tex_dir: str):
    files = [os.path.join(tex_dir, "main.tex")] + sorted(glob.glob(os.path.join(tex_dir, "sections", "*.tex"))) \
        + sorted(glob.glob(os.path.join(tex_dir, "tabs", "*.tex")))
    idx = []
    for f in files:
        if not os.path.isfile(f):
            continue
        sq, ln = [], []
        for i, line in enumerate(open(f, encoding="utf-8").read().splitlines(), 1):
            s = squash(tex_plain(line))
            sq.append(s)
            ln.extend([i] * len(s))
        idx.append((os.path.relpath(f, tex_dir), "".join(sq), ln))
    return idx


def tex_locate(phrase: str, idx):
    q = squash(phrase)          # the phrase is plain text, not LaTeX: do not strip a raw '%' as a comment
    for rel, s, ln in idx:
        k = s.find(q)
        if k >= 0:
            a, b = ln[k], ln[k + len(q) - 1]
            return f"{rel}:{a}" if a == b else f"{rel}:{a}-{b}"
    return None


# ------------------------------------------------------------------------------------ reporting
def load_pdf_text(args):
    if args.text:
        return norm(open(args.text, encoding="utf-8", errors="replace").read()), args.text
    if not args.pdf:
        sys.exit("give --pdf PAPER.pdf (or --text FILE made with pdftotext)")
    r = subprocess.run(["pdftotext", args.pdf, "-"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit("pdftotext failed on " + args.pdf + ": " + r.stderr.strip())
    return norm(r.stdout), args.pdf


def load_layout_text(args):
    if not args.pdf:
        return None
    r = subprocess.run(["pdftotext", "-layout", args.pdf, "-"], capture_output=True, text=True)
    return norm(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def banner(src):
    if os.path.isfile(src):
        b = open(src, "rb").read()
        print(f"paper source : {os.path.abspath(src)}")
        print(f"               {len(b):,} bytes, modified "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(src)))}, "
              f"md5 {hashlib.md5(b).hexdigest()}")


def report(res, verbose, idx=None, sources=False):
    weak = 0
    for t, snip in res["found"].items():
        n = res["counts"][t]
        small = bool(re.fullmatch(r"[+-]?\d{1,2}", t))
        weak += small and snip is not None
        if snip is None:
            print(f"MISSING  {t:<14} x{n}")
        elif verbose:
            print(f"{'FOUND*' if small else 'FOUND '}   {t:<14} x{n:<3} paper: ...{snip}...")
        else:
            print(f"{'FOUND*' if small else 'FOUND '}   {t:<14} x{n}")
    for q, ok in res["quotes"]:
        if not ok or verbose or sources:
            loc = ""
            if idx is not None and ok:
                loc = f" [{tex_locate(q, idx) or 'tex line not found'}]"
            print(f"{'QUOTE OK     ' if ok else 'QUOTE MISSING'} {q[:78]}{'...' if len(q) > 78 else ''}{loc}")
    for i, ok in res["idents"]:
        if not ok:
            print(f"IDENTIFIER NOT IN PAPER: {i}")
    for r in res["claims"]:
        bad = not (r["on_page"] and r["in_paper"]) or r["disagree"]
        if bad or verbose or sources:
            if not r["on_page"]:
                label = "CLAIM NOT ON PAGE"
            elif not r["in_paper"]:
                label = "CLAIM NOT IN PAPER"
            elif r["disagree"]:
                label = "NUMBER NOT PRINTED IN THAT SENTENCE " + ",".join(r["disagree"])
            else:
                label = "CLAIM OK"
            loc = ""
            if idx is not None and not bad:
                tex = r["tex"]
                if tex and tex.startswith("@"):
                    loc = " [" + tex[1:] + "]"
                else:
                    where = tex_locate(tex or r["paper"], idx)
                    loc = f" [{where or 'tex line not found'}]"
            print(f"{label:<10} page: {r['page'][:60]!r}\n{'':<10} paper: {r['paper'][:92]!r}{loc}")
    for r in res["tables"]:
        bad = not r["on_page"] or r["in_paper"] is False
        if bad or verbose or sources:
            loc = ""
            if idx is not None:
                loc = f" [{tex_locate(r['tex'], idx) or 'tex line not found'}]"
            note = "" if r["in_paper"] is not None else "  (no --pdf: layout text unavailable)"
            print(f"{'TABLE MISMATCH' if bad else 'TABLE OK':<10} page: {r['page']!r}{loc}{note}")
    for t in res["unbound"]:
        print(f"UNBOUND  {t}  (a number on the page that sits in no claim, quote or identifier)")

    n_unique = len(res["found"])
    n_found = n_unique - len(res["missing"])
    print()
    print(f"numbers : {n_unique} distinct ({sum(res['counts'].values())} occurrences) checked against the "
          f"paper text; FOUND {n_found}, MISSING {len(res['missing'])}")
    print(f"binding : UNBOUND {len(res['unbound'])} (numbers outside every claim, quote and identifier); "
          f"{len(res['claims'])} claims + {len(res['tables'])} table row checked, failing "
          f"{len(res['bad_claims']) + len(res['bad_table'])}")
    print(f"quotes  : {len(res['quotes'])} verbatim quotes checked; MISSING {len(res['bad_quotes'])}")
    print(f"excluded: {len(res['excluded'])} numbers in the invented record and its answer "
          f"(data-madeup, inside #demo only)")
    print(f"note    : {weak} of the {n_found} FOUND numbers are 1 or 2 digit integers; on their own they "
          f"prove little, the binding above carries the weight")
    if res["outside"]:
        print(f"data-madeup used outside #demo on <{', '.join(res['outside'])}>: not allowed")
    print("RESULT  :", "PASS" if res["ok"] else "FAIL")


# ------------------------------------------------------------------------------------ self-test
def self_test(html: str, pdf_text: str, layout_text: str | None = None) -> bool:
    print("SELF-TEST (known answers in both directions)")
    results = []

    def expect(label, cond):
        results.append(bool(cond))
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")

    def inject(frag):
        return html.replace("</main>", frag + "</main>") if "</main>" in html else \
            html.replace("</body>", frag + "</body>")

    base = analyze(html, pdf_text, layout_text)
    expect("the real page passes", base["ok"])

    for planted, why in (("77.7%", "a number that is not in the paper"),
                         ("34,201", "one digit off 34,200"),
                         ("-0.932", "wrong sign: the paper has +0.932")):
        r = analyze(inject(f"<p>{planted}</p>"), pdf_text, layout_text)
        expect(f"planted {planted!r} ({why}) is reported MISSING", planted in r["missing"] and not r["ok"])

    ctl = find_in_pdf("34,200", pdf_text), find_in_pdf("+0.932", pdf_text), find_in_pdf("89%", pdf_text), \
        find_in_pdf("-0.099", pdf_text), find_in_pdf("0.20", pdf_text)
    expect("controls 34,200 / +0.932 / 89% / -0.099 / 0.20 (all in the paper) are FOUND",
           all(c is not None for c in ctl))

    # a wrong number that DOES occur elsewhere in the paper: only the binding layer can catch it
    wrong = html.replace("47 clear their baseline", "48 clear their baseline")
    r = analyze(wrong, pdf_text, layout_text)
    expect("'48' (it occurs elsewhere in the paper) passes the bare number check", "48" not in r["missing"])
    expect("'48 clear their baseline' is still refused (claim off the page, number unbound)",
           not r["ok"] and bool(r["bad_claims"]) and "48" in r["unbound"])
    r = analyze(html.replace("falls by 0.414", "falls by 0.464"), pdf_text, layout_text)
    expect("0.464 (printed in the paper, but for 32 items of state) is refused for the 16-item claim",
           not r["ok"])
    r = analyze(html.replace("&minus;0.414", "&minus;0.441"), pdf_text, layout_text)
    expect("a transposed digit in the big number (-0.441 for -0.414) is refused", not r["ok"])
    r = analyze(html.replace("20 of 20", "19 of 20"), pdf_text, layout_text)
    expect("'19 of 20' is refused (the paper says twenty of twenty)", not r["ok"])
    r = analyze(html.replace("53 of 100", "35 of 100"), pdf_text, layout_text)
    expect("'35 of 100' (53 transposed, on the fifth card) is refused", not r["ok"] and bool(r["bad_claims"]))
    r = analyze(html.replace("53 of 100", "53 of 101"), pdf_text, layout_text)
    expect("'53 of 101' (101 is printed elsewhere in the paper) passes the bare number check but is refused",
           "101" not in r["missing"] and not r["ok"] and "101" in r["unbound"])

    # arithmetic is refused: a number that is not printed in the claim's paper sentence
    expect("number agreement: 20 vs 'twenty of twenty' agrees",
           not [n for n in {canon(t) for t in extract_numbers("20 of 20")}
                if n not in printed_numbers("twenty of twenty never clear")])
    expect("number agreement: 93 (= 47 + 46) is not printed in the 47/46/12 sentence",
           "93" not in printed_numbers("Across the 105 model-family pairs the study covers, 47 (45%) exceed "
                                       "their baseline; 46 (44%) never clear it; only 12 (11%) clear it and stop"))

    qs = sorted((q for q in base["page"].quotes if len(q) > 40), key=len, reverse=True)
    if qs:
        longest = re.sub(r"\s+", " ", qs[0]).strip()
        edited = re.sub(r"[A-Za-z]{5,}", "zebra", longest, count=1)      # one word changed
        expect("the longest verbatim quote, with one word changed, is reported MISSING",
               squash(edited) not in squash(pdf_text))
        expect("the unedited quote is FOUND", squash(longest) in squash(pdf_text))

    sneaky = inject('<p data-madeup>987654</p>')
    expect("data-madeup outside #demo is refused", not analyze(sneaky, pdf_text, layout_text)["ok"])

    ok = all(results)
    print(f"SELF-TEST {'PASSED' if ok else 'FAILED'} ({sum(results)}/{len(results)} expectations met)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--pdf")
    ap.add_argument("--text")
    ap.add_argument("--tex-dir", help="the paper's tex folder, to print file:line for every claim")
    ap.add_argument("--sources", action="store_true", help="list every claim with its source line")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show where each number sits in the paper, and list every quote and claim")
    args = ap.parse_args()

    later = os.path.join(HERE, "..", "static", "GAUGE_UMFM26.pdf")
    if not args.pdf and not args.text and os.path.isfile(later):
        args.pdf = later                     # the paper copy that ships with the site, once it is added
    pdf_text, src = load_pdf_text(args)
    layout_text = load_layout_text(args)
    html = open(args.html, encoding="utf-8").read()
    print(f"page         : {os.path.abspath(args.html)}")
    banner(src)
    print()
    if args.self_test:
        sys.exit(0 if self_test(html, pdf_text, layout_text) else 1)
    idx = tex_index(args.tex_dir) if args.tex_dir else None
    res = analyze(html, pdf_text, layout_text)
    report(res, args.verbose, idx, args.sources)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
