#!/usr/bin/env python3
"""Count the visible words of the GAUGE project page and enforce the page's length rules.

WHAT COUNTS AS A WORD
  A whitespace-separated token that contains at least one letter or digit. Stray symbols such as
  the middle dot do not count. Text in an <input> placeholder counts, because it is on screen.

WHAT IS LEFT OUT (as the brief says: tags, nav, footer and the BibTeX block)
  * markup, <head>, scripts, styles and comments;
  * the top bar (brand line, links, theme button), the footer, and the BibTeX block with its Copy button.
  Two further groups are reported separately, because a reader does not see them on first load:
  * hidden: the answer panel (display:none until a guess) and the closed "Show full prompt" box;
  * "all text" adds the hidden text, so the strictest reading of "words on the page" is also shown.

RULES CHECKED (exit status 1 when any fails)
  visible words <= 600; finding title <= 8 words; finding body <= 25 words; figure caption <= 20
  words; no paragraph or list item over 3 sentences; Highlights has 3 bullets; every section
  heading is one sentence that ends with a period.

USAGE
  python3 -B count_words.py [--html ../index.html] [-v]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from html.parser import HTMLParser

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr"}
LIMITS = dict(visible=600, card_title=8, card_body=25, caption=20, sentences=3, highlights=3)


def words(text: str) -> int:
    return sum(1 for t in text.split() if re.search(r"[A-Za-z0-9]", t))


def sentences(text: str) -> int:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return 0
    return len(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9‘“])", text))


class Counter(HTMLParser):
    SECTION_BY_ID = {"stats": "stats", "findings": "findings", "results": "results",
                     "demo": "demo", "bibtex": "citation"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []             # [tag, attrs-dict, excluded, hidden, section, buf-or-None, kind]
        self.visible = {}           # section -> words
        self.hidden = 0
        self.excluded = {"top bar": 0, "footer": 0, "BibTeX block": 0}
        self.brand_words = 0
        self.items = []             # (kind, text) for rule checks
        self.headings = []
        self.highlight_items = 0
        self.placeholders = []

    # -- context helpers
    def _ctx(self):
        for e in reversed(self.stack):
            if e[2]:
                return "excluded"
        for e in reversed(self.stack):
            if e[3]:
                return "hidden"
        return "visible"

    def _section(self):
        for e in reversed(self.stack):
            if e[4]:
                return e[4]
        return "other"

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        cls = (d.get("class") or "").split()
        if tag == "input" and d.get("placeholder") and self._ctx() == "visible":
            self.placeholders.append(d["placeholder"])
            self.visible["demo"] = self.visible.get("demo", 0) + words(d["placeholder"])
        if tag in VOID:
            return
        excluded = (tag in ("head", "script", "style", "footer", "noscript")
                    or "topbar" in cls or "bibwrap" in cls)
        hidden = ("hidden" in d or "qreveal" in cls
                  or (tag == "details" and "open" not in d))
        section = None
        if tag == "header" and "hero" in cls:
            section = "hero"
        elif "teaser" in cls:
            section = "teaser"
        elif d.get("id") in self.SECTION_BY_ID:
            section = self.SECTION_BY_ID[d["id"]]
        kind = None
        in_card = any("card" in e[1].get("class", "").split() for e in self.stack)
        in_hl = any("hl" in e[1].get("class", "").split() for e in self.stack)
        if tag == "h3" and in_card:
            kind = "card_title"
        elif tag == "p" and "figcap" in cls:
            kind = "caption"
        elif tag == "p" and in_card and "csrc" not in cls:
            kind = "card_body"
        elif tag == "li" and in_hl:
            kind = "highlight"
        elif tag in ("p", "li"):
            kind = "para"
        elif tag == "h2":
            kind = "heading"
        d["class"] = d.get("class", "")
        self.stack.append([tag, d, excluded, hidden, section, [] if kind else None, kind])

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                while len(self.stack) > i:
                    e = self.stack.pop()
                    if e[6] and e[5] is not None:
                        text = " ".join(e[5])
                        self.items.append((e[6], text, self._ctx_for(e)))
                        if e[6] == "heading":
                            self.headings.append(text)
                return

    def _ctx_for(self, e):
        if e[2] or any(x[2] for x in self.stack):
            return "excluded"
        if e[3] or any(x[3] for x in self.stack):
            return "hidden"
        return "visible"

    def handle_data(self, data):
        if not data.strip():
            return
        ctx = self._ctx()
        n = words(data)
        # <summary> of a closed <details> is visible
        if ctx == "hidden" and self.stack and any(e[0] == "summary" for e in self.stack):
            hid = [e for e in self.stack if e[3]]
            if all(e[0] == "details" and "qreveal" not in e[1]["class"] for e in hid):
                ctx = "visible"
        if ctx == "excluded":
            if any("topbar" in e[1]["class"].split() for e in self.stack):
                self.excluded["top bar"] += n
                if any("brandline" in e[1]["class"].split() for e in self.stack):
                    self.brand_words += n
            elif any(e[0] == "footer" for e in self.stack):
                self.excluded["footer"] += n
            elif any("bibwrap" in e[1]["class"].split() for e in self.stack):
                self.excluded["BibTeX block"] += n
            # head, scripts, styles and <noscript> hold no page text: not counted anywhere
        elif ctx == "hidden":
            self.hidden += n
        else:
            sec = self._section()
            self.visible[sec] = self.visible.get(sec, 0) + n
        for e in self.stack:
            if e[5] is not None:
                e[5].append(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default=os.path.join(HERE, "..", "index.html"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    html = open(args.html, encoding="utf-8").read()
    c = Counter()
    c.feed(html)
    c.close()

    total = sum(c.visible.values())
    print(f"page: {os.path.abspath(args.html)}")
    print("visible words by section (rendered on first load):")
    for sec in ("hero", "teaser", "stats", "findings", "results", "demo", "citation", "other"):
        if sec in c.visible:
            print(f"  {sec:<9}{c.visible[sec]:>5}")
    print(f"VISIBLE WORDS (limit {LIMITS['visible']}): {total}")
    ex = ", ".join(f"{k} {v}" for k, v in c.excluded.items())
    print(f"not counted (excluded by the brief): {ex}")
    print(f"hidden until used (answer panel, closed 'Show full prompt'): {c.hidden} words")
    print(f"ALL TEXT if hidden is included (strictest reading): {total + c.hidden}")
    print(f"if the brand line (in the top bar) were counted too: {total + c.brand_words}")

    fails = []
    if total > LIMITS["visible"]:
        fails.append(f"visible words {total} > {LIMITS['visible']}")
    for kind, text, ctx in c.items:
        if ctx == "excluded":
            continue
        n = words(text)
        if kind == "card_title" and n > LIMITS["card_title"]:
            fails.append(f"finding title has {n} words (> {LIMITS['card_title']}): {text!r}")
        if kind == "card_body" and n > LIMITS["card_body"]:
            fails.append(f"finding body has {n} words (> {LIMITS['card_body']}): {text[:60]!r}")
        if kind == "caption" and n > LIMITS["caption"]:
            fails.append(f"figure caption has {n} words (> {LIMITS['caption']}): {text[:60]!r}")
        if kind in ("para", "card_body", "caption", "highlight") and sentences(text) > LIMITS["sentences"]:
            fails.append(f"{sentences(text)} sentences (> {LIMITS['sentences']}): {text[:60]!r}")
        if args.verbose and kind in ("card_title", "card_body", "caption", "highlight", "heading"):
            print(f"  {kind:<11}{n:>3}  {re.sub(chr(10), ' ', text)[:90]}")
    n_hl = sum(1 for k, _t, ctx in c.items if k == "highlight" and ctx != "excluded")
    if n_hl != LIMITS["highlights"]:
        fails.append(f"Highlights has {n_hl} bullets, expected {LIMITS['highlights']}")
    for h in c.headings:
        h1 = re.sub(r"\s+", " ", h).strip()
        if not h1.endswith(".") or sentences(h1) != 1:
            fails.append(f"section heading must be one sentence ending with a period: {h1!r}")

    print()
    if fails:
        for f in fails:
            print("FAIL", f)
        print("RESULT: FAIL")
        sys.exit(1)
    print(f"RESULT: PASS ({n_hl} highlights; {len(c.headings)} claim headings; "
          f"{sum(1 for k, _t, x in c.items if k == 'card_title' and x != 'excluded')} finding cards; "
          f"{sum(1 for k, _t, x in c.items if k == 'caption' and x != 'excluded')} captions)")


if __name__ == "__main__":
    main()
