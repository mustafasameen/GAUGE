#!/usr/bin/env python3
"""Check the GAUGE project page: links, images, text rules, required elements, credits, light-only.

Parses docs/index.html with html.parser and checks:
  * every local href/src/srcset exists (static/GAUGE_UMFM26.pdf is reported as "expected later");
  * every #anchor points at an id that exists; no placeholder or dead-looking links;
  * the only external links are the code repo, the YJMob100K DOI and the HugAgent credit, and the
    only external resource is the Google Fonts stylesheet; no arXiv link anywhere;
  * no em dash character in any text file under docs/; no hype words; every <img> has alt text;
  * docs/ is under 5 MB; no __pycache__ or .pyc under docs/;
  * head metadata (title, description, Open Graph, Twitter card; og:image is the overview PNG at its
    final URL and the file exists);
  * the content rules: venue pill text, brand line, tagline, Highlights with 3 bullets, tl;dr of at most 2
    sentences, no "Finding" callout, three hero buttons, BibTeX block, footer lines, MIT notice and
    credit to HugAgent, no leftover HugAgent content;
  * light only: no prefers-color-scheme rule, no data-theme, no theme toggle, no localStorage;
    color-scheme is light; reduced-motion handling is kept;
  * the demo: made-up label, 32 record rows in the benchmark's row format, key present.

USAGE
  python3 -B check_site.py [--docs ..] [--self-test]
Exit status 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = "https://mustafasameen.github.io/GAUGE/"
EXPECTED_LATER = {"static/GAUGE_UMFM26.pdf"}
ALLOWED_LINKS = {
    "https://github.com/mustafasameen/GAUGE",
    "https://doi.org/10.1038/s41597-024-03237-9",
    "https://github.com/jajamoa/HugAgent",
}
ALLOWED_HOSTS = {"github.com", "doi.org", "fonts.googleapis.com", "mustafasameen.github.io", "www.w3.org"}
HYPE = ["novel", "groundbreaking", "ground-breaking", "remarkable", "state-of-the-art", "cutting-edge",
        "revolutionary", "breakthrough", "unprecedented", "game-changing", "game changing", "stunning",
        "incredible", "amazing", "world-class", "first-ever", "seamless"]
LEFTOVERS = ["arxiv", "2510.15144", "chance jiajie", "jiajie@", "mit.edu", "assets/", "trace-your-thinking",
             "emnlp", "personallm", "kent larson", "robot.svg", "mit-black", "belief", "think-aloud",
             "participants", "interview", "chatbot", "mitmark", "avgnote", "easter"]
BIBTEX = """@inproceedings{sameen2026gauge,
  title     = {{GAUGE}: How Large Language Models Recover Mobility Descriptors from Individual Records},
  author    = {Sameen, Mustafa},
  booktitle = {Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Urban Mobility Foundation Models},
  series    = {UMFM '26},
  year      = {2026},
  address   = {Riverside, CA, USA},
  publisher = {ACM},
  doi       = {10.1145/3849729.3856126}
}"""
TITLE = "GAUGE: How Large Language Models Recover Mobility Descriptors from Individual Records"
TAGLINE = "Can a language model recover mobility descriptors from one person's location record?"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
        "track", "wbr"}
EM_DASH = chr(0x2014)        # written as a code point so this file itself holds no em dash
TEXT_EXT = (".html", ".css", ".js", ".py", ".json", ".md", ".txt", ".svg", ".xml")


# ------------------------------------------------------------------------------------------ mini DOM
class Node:
    def __init__(self, tag, attrs, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children = []          # Node or str

    def classes(self):
        return (self.attrs.get("class") or "").split()

    def text(self):
        out = []
        for c in self.children:
            out.append(c if isinstance(c, str) else ("" if c.tag in ("script", "style") else c.text()))
        return "".join(out)

    def walk(self):
        yield self
        for c in self.children:
            if isinstance(c, Node):
                yield from c.walk()


class DomBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {})
        self.cur = self.root
        self.errors = []
        self.doctype = False
        self.comments = []

    def handle_decl(self, decl):
        if decl.lower().startswith("doctype"):
            self.doctype = True

    def handle_comment(self, data):
        self.comments.append(data)

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is self.root:
            self.errors.append(f"stray </{tag}>")
            return
        if n is not self.cur:
            self.errors.append(f"<{self.cur.tag}> not closed before </{tag}>")
        self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)

    def close(self):
        super().close()
        if self.cur is not self.root:
            self.errors.append(f"<{self.cur.tag}> never closed")


def build(html: str) -> DomBuilder:
    b = DomBuilder()
    b.feed(html)
    b.close()
    return b


def find(root, tag=None, cls=None, id=None):
    return [n for n in root.walk() if n is not root
            and (tag is None or n.tag == tag) and (cls is None or cls in n.classes())
            and (id is None or n.attrs.get("id") == id)]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def sentences(text: str) -> int:
    text = norm_ws(text)
    return len(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9‘“])", text)) if text else 0


# ------------------------------------------------------------------------------------------ checks
def run_checks(html: str, docs: str, files_scan: bool = True):
    dom = build(html)
    root = dom.root
    results = []              # (name, ok, detail)

    def add(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # --- structure
    add("html well formed", not dom.errors and dom.doctype, "; ".join(dom.errors[:3]) or "doctype and tags balance")
    h = find(root, "html")
    add("lang attribute", bool(h) and h[0].attrs.get("lang") == "en", "lang=en")
    add("viewport meta", any(m.attrs.get("name") == "viewport" for m in find(root, "meta")), "")
    add("exactly one h1 with the paper title",
        len(find(root, "h1")) == 1 and norm_ws(find(root, "h1")[0].text()) == TITLE, "")

    # --- local references
    refs = []
    for n in root.walk():
        for attr in ("href", "src", "srcset"):
            v = n.attrs.get(attr)
            if not v:
                continue
            if attr == "srcset":
                for part in v.split(","):
                    if part.strip():
                        refs.append((n.tag, attr, part.strip().split()[0]))
            else:
                refs.append((n.tag, attr, v))
    missing, later, ok_count, ids = [], [], 0, {n.attrs["id"] for n in root.walk() if "id" in n.attrs}
    bad_anchor, bad_ext = [], []
    for tag, attr, v in refs:
        if v.startswith("data:"):
            continue
        if v.startswith("#"):
            if len(v) == 1 or v[1:] not in ids:
                bad_anchor.append(v)
            continue
        u = urlparse(v)
        if u.scheme in ("http", "https"):
            if tag == "a":
                if v not in ALLOWED_LINKS:
                    bad_ext.append(v)
            elif tag == "link" and u.netloc not in ("fonts.googleapis.com", "mustafasameen.github.io"):
                bad_ext.append(v)
            continue
        if u.scheme or v.startswith("//") or v.strip() == "" or v.startswith("javascript:"):
            bad_ext.append(v)
            continue
        rel = unquote(u.path)
        if rel in EXPECTED_LATER:
            later.append(rel)
        elif os.path.isfile(os.path.join(docs, rel)) and os.path.getsize(os.path.join(docs, rel)) > 0:
            ok_count += 1
        else:
            missing.append(rel)
    add("local files exist", not missing,
        f"{ok_count} local references resolve; missing: {missing or 'none'}; expected later: "
        f"{sorted(set(later)) or 'none'}")
    add("anchors resolve", not bad_anchor, f"bad: {bad_anchor or 'none'}")
    add("external links are the allowed ones", not bad_ext,
        f"bad: {bad_ext or 'none'}; allowed: {sorted(ALLOWED_LINKS)}")
    bad_rel = [n.attrs.get("href") for n in find(root, "a") if n.attrs.get("target") == "_blank"
               and "noopener" not in (n.attrs.get("rel") or "")]
    add("target=_blank links carry rel=noopener", not bad_rel, f"bad: {bad_rel or 'none'}")
    hosts = {urlparse(u).netloc for u in re.findall(r"https?://[^\s\"'<>)\\]+", html)}
    add("every URL in the file is on an allowed host", hosts <= ALLOWED_HOSTS,
        f"hosts: {sorted(hosts)}")
    scripts_ext = [n.attrs["src"] for n in find(root, "script") if n.attrs.get("src")]
    add("no external scripts", not scripts_ext, f"{scripts_ext or 'none'}")
    add("no placeholder text or links",
        not re.search(r"lorem ipsum|TODO|FIXME|example\.com|href=\"#\"|href=\"\"", html, re.I), "")
    add("no arXiv link or button", "arxiv" not in html.lower(), "")

    # --- images
    imgs = find(root, "img")
    bad_alt = [i.attrs.get("src") for i in imgs if len(norm_ws(i.attrs.get("alt", ""))) < 8]
    add("every <img> has alt text", imgs and not bad_alt, f"{len(imgs)} images; without alt: {bad_alt or 'none'}")
    pics = find(root, "picture")
    add("each picture offers SVG first, PNG fallback",
        len(pics) == len(imgs) and all(
            any(s.attrs.get("type") == "image/svg+xml" for s in find(p, "source")) for p in pics), f"{len(pics)} pictures")

    # --- head metadata
    metas = {(m.attrs.get("name") or m.attrs.get("property")): m.attrs.get("content", "") for m in find(root, "meta")}
    need = ["description", "og:title", "og:description", "og:image", "og:url", "twitter:card", "twitter:title",
            "twitter:description", "twitter:image"]
    add("title, description, Open Graph and Twitter tags",
        bool(find(root, "title")) and all(metas.get(k) for k in need), f"missing: {[k for k in need if not metas.get(k)] or 'none'}")
    og = metas.get("og:image", "")
    add("og:image is the overview PNG at its final URL, and the file exists",
        og == SITE + "static/img/fig_overview.png" and os.path.getsize(os.path.join(docs, "static/img/fig_overview.png")) > 0
        if os.path.isfile(os.path.join(docs, "static/img/fig_overview.png")) else False, og)
    add("twitter card is summary_large_image", metas.get("twitter:card") == "summary_large_image", "")

    # --- visible text rules
    body = find(root, "body")[0] if find(root, "body") else root
    vis = norm_ws(body.text() + " " + " ".join(i.attrs.get("alt", "") for i in imgs) + " " + find(root, "title")[0].text()
                  if find(root, "title") else body.text())
    hype = [w for w in HYPE if re.search(r"\b" + re.escape(w) + r"\b", vis, re.I)]
    add("no hype words", not hype, f"found: {hype or 'none'}")

    # --- required elements
    pills = find(root, cls="venuepill")
    add("venue pill reads exactly 'UMFM @ ACM SIGSPATIAL 2026' and sits above the title",
        len(pills) == 1 and norm_ws(pills[0].text()) == "UMFM @ ACM SIGSPATIAL 2026"
        and html.index('class="venuepill"') < html.index("<h1"), "")
    brand = find(root, cls="brandline")
    add("brand line reads 'UNIVERSITY OF FLORIDA · 2026'",
        len(brand) == 1 and norm_ws(brand[0].text()) == "UNIVERSITY OF FLORIDA · 2026", "")
    tag = find(root, cls="tagline")
    add("tagline is the paper's question, one line of text",
        len(tag) == 1 and norm_ws(tag[0].text()) == TAGLINE, "")
    hl = find(root, cls="hl")
    add("Highlights block has a label and exactly 3 bullets",
        len(hl) == 1 and "Highlights" in hl[0].text() and len(find(hl[0], "li")) == 3, "")
    box = find(root, cls="tldrbox")
    first_p = find(box[0], "p")[0].text() if box and find(box[0], "p") else ""
    add("tl;dr is at most 2 sentences", 0 < sentences(first_p.replace("tl;dr", "")) <= 2, f"{sentences(first_p.replace('tl;dr', ''))} sentences")
    labels = [norm_ws(n.text()).lower() for n in find(root, cls="lb")]
    add("no 'Finding' callout under the tl;dr", "finding" not in labels, f"labels: {labels}")
    btns = find(find(root, cls="btns")[0], "a") if find(root, cls="btns") else []
    add("hero buttons: Paper, Code, BibTeX with the right targets",
        [(norm_ws(b.text()), b.attrs.get("href")) for b in btns] ==
        [("Paper", "static/GAUGE_UMFM26.pdf"), ("Code", "https://github.com/mustafasameen/GAUGE"), ("BibTeX", "#bibtex")], "")
    nav = find(find(root, cls="topbar")[0], "nav")[0] if find(root, cls="topbar") else None
    navs = [norm_ws(a.text()) for a in find(nav, "a")] if nav else []
    add("top bar links include Paper, Code and BibTeX", all(x in navs for x in ("Paper", "Code", "BibTeX")), f"{navs}")
    bib = find(root, id="bibtex-block")
    add("BibTeX block is exactly the requested entry", bool(bib) and bib[0].text().strip() == BIBTEX, "")
    add("BibTeX section has a copy button", bool(find(root, id="copy-bib")), "")
    foot = find(root, "footer")
    ftxt = norm_ws(foot[0].text()) if foot else ""
    add("footer: copyright and licence line", "© 2026 Mustafa Sameen. The paper is licensed under CC BY 4.0." in ftxt, "")
    yab = [a for a in find(foot[0], "a")] if foot else []
    add("footer: 'Built on YJMob100K (Yabe et al., 2024)' links the DOI",
        any(norm_ws(a.text()) == "YJMob100K (Yabe et al., 2024)" and a.attrs.get("href") == "https://doi.org/10.1038/s41597-024-03237-9"
            for a in yab) and "Built on YJMob100K (Yabe et al., 2024)" in ftxt, "")
    add("footer: 'Page design adapted from HugAgent (MIT)' links the HugAgent repo",
        any(norm_ws(a.text()) == "Page design adapted from HugAgent (MIT)" and a.attrs.get("href") == "https://github.com/jajamoa/HugAgent"
            for a in yab), "")
    add("footer carries the full workshop name",
        "The 2nd ACM SIGSPATIAL International Workshop on Urban Mobility Foundation Models" in ftxt, "")
    com = "\n".join(dom.comments)
    add("MIT notice for HugAgent sits in a comment at the top",
        "MIT License" in com and "Copyright (c) 2026 The HugAgent Authors" in com
        and "Permission is hereby granted, free of charge" in com
        and html.lstrip().lower().startswith("<!doctype html>") and html.index("<!--") < html.index("<html"), "")
    stripped = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    stripped = stripped.replace("Page design adapted from HugAgent (MIT)", "").replace("https://github.com/jajamoa/HugAgent", "")
    left = [w for w in LEFTOVERS if w in stripped.lower()]
    add("no leftover HugAgent content", not left and "hugagent" not in stripped.lower(), f"found: {left or 'none'}")

    # --- headings and cards
    h2s = [norm_ws(n.text()) for n in find(root, "h2")]
    add("every section heading is one sentence ending with a period",
        h2s and all(t.endswith(".") and sentences(t) == 1 for t in h2s), f"{len(h2s)} headings")

    # --- light only
    css = "\n".join(n.text() for n in find(root, "style"))
    js = "\n".join(n.text() for n in find(root, "script"))
    low = html.lower()
    add("light only: no prefers-color-scheme, data-theme, theme toggle or localStorage",
        not any(k in low for k in ("prefers-color-scheme", "data-theme", "themebtn", "localstorage", "color-scheme:dark",
                                   "color-scheme: dark")), "")
    add("light only: color-scheme is light (CSS and meta)",
        re.search(r":root\{[^}]*color-scheme:\s*light", css) is not None and metas.get("color-scheme") == "light", "")
    add("reduced-motion handling kept", "prefers-reduced-motion" in css and "prefers-reduced-motion" in js, "")
    add("no external stylesheet other than Google Fonts",
        all(urlparse(m).netloc == "fonts.googleapis.com" for m in re.findall(r"url\((?:'|\")?(https?://[^'\")]+)", css)), "")

    # --- the demo
    demo = find(root, id="demo")
    d0 = demo[0] if demo else root
    dtxt = norm_ws(d0.text())
    rows = [r for r in (find(d0, cls="p-rows")[0].text().splitlines() if find(d0, cls="p-rows") else []) if r.strip()]
    add("demo: 'Made-up record, not from the dataset' label", "Made-up record, not from the dataset" in dtxt, "")
    add("demo: 32 record rows in the benchmark's row format",
        len(rows) == 32 and all(re.fullmatch(r"d\d+ t\d+ x-?\d+ y-?\d+", r.strip()) for r in rows), f"{len(rows)} rows")
    key = find(d0, id="demo-key")
    add("demo: answer key embedded as a number", bool(key) and re.fullmatch(r"\d+", key[0].attrs.get("data-key", "")) is not None,
        f"key {key[0].attrs.get('data-key') if key else None}")
    add("demo: 'Show full prompt' toggle, input and two buttons",
        any(norm_ws(s.text()) == "Show full prompt" for s in find(d0, "summary")) and bool(find(d0, id="guess"))
        and len(find(find(d0, id="guess-form")[0], "button")) == 2 if find(d0, id="guess-form") else False, "")
    madeup_outside = [n.tag for n in root.walk() if "data-madeup" in n.attrs
                      and not any(a.attrs.get("id") == "demo" for a in _ancestors(n))]
    add("data-madeup is used only inside the demo", not madeup_outside, f"outside: {madeup_outside or 'none'}")

    # --- files
    if files_scan:
        total, em_files, pyc = 0, [], []
        for dp, dn, fn in os.walk(docs):
            if "__pycache__" in dp.split(os.sep):
                pyc.append(dp)
            for f in fn:
                path = os.path.join(dp, f)
                total += os.path.getsize(path)
                if f.endswith(".pyc"):
                    pyc.append(path)
                if f.endswith(TEXT_EXT):
                    if EM_DASH in open(path, encoding="utf-8", errors="replace").read():
                        em_files.append(os.path.relpath(path, docs))
        add("docs/ is under 5 MB", total < 5 * 1024 * 1024, f"{total / 1024:.0f} KB")
        add("no em dash character in any text file under docs/", not em_files, f"found in: {em_files or 'none'}")
        add("no __pycache__ or .pyc under docs/", not pyc, f"{pyc or 'none'}")
    else:
        add("no em dash character in the page", EM_DASH not in html, "")
    return results


def _ancestors(n):
    while n.parent is not None:
        n = n.parent
        yield n


# ------------------------------------------------------------------------------------------ self-test
def self_test(html: str, docs: str) -> bool:
    print("SELF-TEST (each planted defect must be caught; the real page must pass)")
    base = run_checks(html, docs, files_scan=False)
    outcomes = [("the real page passes", all(ok for _, ok, _ in base))]

    def planted(label, mutate, expect_fail):
        r = run_checks(mutate(html), docs, files_scan=False)
        failed = [n for n, ok, _ in r if not ok]
        outcomes.append((f"planted: {label} -> caught by {expect_fail!r}", any(expect_fail in n for n in failed)))

    planted("an em dash", lambda h: h.replace("<h2>Recovery", "<h2>" + EM_DASH + " Recovery", 1), "em dash")
    planted("an image without alt text", lambda h: re.sub(r' alt="[^"]*"', "", h, count=1), "alt text")
    planted("a link to a file that does not exist", lambda h: h.replace("</main>", '<a href="static/nope.png">x</a></main>') if "</main>" in h
            else h.replace("</body>", '<a href="static/nope.png">x</a></body>'), "local files exist")
    planted("an external link nobody approved", lambda h: h.replace("</body>", '<a href="https://example.org/x">x</a></body>'),
            "external links")
    planted("a hype word", lambda h: h.replace("Reverses the usual direction", "A groundbreaking, novel direction", 1), "hype")
    planted("a changed venue pill", lambda h: h.replace("UMFM @ ACM SIGSPATIAL 2026</div>", "UMFM 2026</div>", 1), "venue pill")
    planted("a dark-mode rule", lambda h: h.replace("</style>", "@media (prefers-color-scheme: dark){body{background:#000}}</style>", 1),
            "light only")
    planted("a fourth highlight", lambda h: h.replace("</ul>", "<li>extra</li></ul>", 1), "Highlights")
    planted("a dead anchor", lambda h: h.replace("</body>", '<a href="#nowhere">x</a></body>'), "anchors")
    planted("an arXiv button", lambda h: h.replace("</body>", '<a href="https://arxiv.org/abs/0000.00000">arXiv</a></body>'),
            "arXiv")
    for label, ok in outcomes:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    ok = all(o for _, o in outcomes)
    print(f"SELF-TEST {'PASSED' if ok else 'FAILED'} ({sum(o for _, o in outcomes)}/{len(outcomes)})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default=os.path.join(HERE, ".."))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    docs = os.path.abspath(args.docs)
    html = open(os.path.join(docs, "index.html"), encoding="utf-8").read()
    print(f"checking {os.path.join(docs, 'index.html')}")
    if args.self_test:
        sys.exit(0 if self_test(html, docs) else 1)
    results = run_checks(html, docs)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    print("RESULT:", "PASS" if not bad else "FAIL")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
