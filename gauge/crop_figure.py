#!/usr/bin/env python3
"""Crop a one-page figure PDF to its ink bounding box plus a fixed margin, changing only the page box.

Matplotlib saves figures with white padding, and LaTeX scales a figure to the column width by its
page box, so padding wastes column height and shrinks the type unevenly across figures. The crop
takes Ghostscript's bounding box of the ink and sets MediaBox and CropBox to that box grown by the
margin. Only the page box changes. The content stream is checked to be byte-identical before the
file is written, so nothing drawn can move.

Requires the pypdf package and the ghostscript command `gs`.

Usage:
  python gauge/crop_figure.py IN.pdf [OUT.pdf] [--margin 1]    (OUT defaults to IN, in place)
  python gauge/crop_figure.py --selftest
"""
from __future__ import annotations
import argparse, hashlib, os, re, subprocess, sys, tempfile

import pypdf
from pypdf.generic import RectangleObject


def ink_bbox(path: str):
    err = subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox", path],
                         capture_output=True, text=True).stderr
    m = re.search(r"HiResBoundingBox:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", err)
    if not m:
        sys.exit(f"crop_figure: no ink bounding box for {path}")
    return tuple(map(float, m.groups()))


def crop(src: str, dst: str, margin: float = 1.0):
    x0, y0, x1, y1 = ink_bbox(src)
    r = pypdf.PdfReader(src)
    if len(r.pages) != 1:
        sys.exit(f"crop_figure: {src} has {len(r.pages)} pages, expected 1")
    pg = r.pages[0]
    before = hashlib.md5(pg.get_contents().get_data()).hexdigest()
    box = RectangleObject([x0 - margin, y0 - margin, x1 + margin, y1 + margin])
    pg.mediabox = box
    pg.cropbox = box
    w = pypdf.PdfWriter()
    w.add_page(pg)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as fh:
        w.write(fh)
    after = hashlib.md5(pypdf.PdfReader(tmp).pages[0].get_contents().get_data()).hexdigest()
    if before != after:
        os.remove(tmp)
        sys.exit(f"crop_figure: content stream changed for {src}; refusing to write")
    os.replace(tmp, dst)
    return (x1 - x0 + 2 * margin, y1 - y0 + 2 * margin)


def selftest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "t.pdf")
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1]); ax.set_title("t")
        fig.savefig(src, bbox_inches="tight", pad_inches=0.5)      # deliberately heavy padding
        plt.close(fig)
        ink = ink_bbox(src)
        page0 = [float(v) for v in pypdf.PdfReader(src).pages[0].mediabox]
        w, h = crop(src, os.path.join(d, "c.pdf"), 1.0)
        page1 = [float(v) for v in pypdf.PdfReader(os.path.join(d, "c.pdf")).pages[0].mediabox]
        grew = (page0[2] - page0[0]) > (page1[2] - page1[0]) + 50       # 0.5in pad = 72pt removed
        fits = page1[0] <= ink[0] and page1[1] <= ink[1] and page1[2] >= ink[2] and page1[3] >= ink[3]
        tight = abs((page1[2] - page1[0]) - (ink[2] - ink[0]) - 2) < 1e-6
        ok = grew and fits and tight
        print(f"  selftest: {'OK' if ok else 'FAILED'} (padding removed {grew}, ink inside box {fits}, "
              f"1pt margin exact {tight})")
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest() else 1
    if not a.src:
        ap.error("give a PDF to crop")
    w, h = crop(a.src, a.dst or a.src, a.margin)
    print(f"cropped {a.dst or a.src}: {w:.1f} x {h:.1f} pt, content stream identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
