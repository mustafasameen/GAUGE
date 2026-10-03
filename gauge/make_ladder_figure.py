#!/usr/bin/env python3
"""Draw the decomposition controls and the scaffold ladder (fig_ladder.pdf).

Both artifacts already hold `gain` and `gain_ci` for every cell, so nothing is recomputed here. A
point to the right of the dashed zero line beat the best-constant predictor on that cell. The three
controls (panel a) hand the model an intermediate quantity and ask for one link of the computation.
The five ladder rungs (panel b) hand it progressively more of the geometry. The controls place the
failure: `max_given` clears for every model, and `pair_distance` clears for none.

Input: results/scored_control_ladder.json (from score_control_ladder.py).
Output: outputs/tex/figs/fig_ladder.pdf.

Usage:
  python gauge/make_ladder_figure.py [--check]
    --check asserts that the two facts the figure's caption states still hold and that the figure
    file exists. It draws nothing.
"""
from __future__ import annotations
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import pub_style, OKABE_SEQ, FIGS, panel, SHORT, save_cropped  # noqa: E402  same style as every other figure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "scored_control_ladder.json")
OUT = os.path.join(FIGS, "fig_ladder.pdf")

MODELS = [(m, SHORT[m]) for m in ("phi35mini", "mistral7b", "llama8b", "gemma3_12b", "llama70b")]
CONTROLS = [("max_given", "largest of a list"), ("sum_given", "total of a list"),
            ("pair_distance", "distance between two points")]
RUNGS = [("L0_raw", "raw points"), ("L1_km", "in km"), ("L2_centred", "centred"),
         ("L3_dists", "distances given"), ("L4_msd", "mean sq. dist. given")]


def draw(ax, d, prefix, steps, letter, title, xlim, seen):
    """One panel. Step names are y tick labels and not text floating in the plot area, where they
    would collide with the points they label.
    """
    ticks, labels = [], []
    row = 0
    for skey, slabel in steps:
        ys = []
        for mi, (mkey, mlabel) in enumerate(MODELS):
            v = d.get("%s|%s|%s" % (prefix, skey, mkey))
            if v is None:
                continue
            y = row
            ys.append(y)
            lo, hi = v["gain_ci"]
            ax.plot([lo, hi], [y, y], color=OKABE_SEQ[mi % len(OKABE_SEQ)], lw=1.0,
                    solid_capstyle="butt", zorder=2)
            ax.plot([v["gain"]], [y], "o", ms=3.0, color=OKABE_SEQ[mi % len(OKABE_SEQ)],
                    zorder=3,
                    label=None if mkey in seen else mlabel)
            seen.add(mkey)
            row += 1
        if ys:
            ticks.append(sum(ys) / len(ys))
            labels.append(slabel)
        row += 1                                  # blank row between groups
    ax.axvline(0, color="0.35", lw=.7, ls="--", zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1.2, row - 0.6)
    ax.invert_yaxis()                             # first step at the top, reading order
    ax.set_xlim(*xlim)
    ax.set_xlabel("normalized gain over the baseline")
    panel(ax, letter, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = json.load(open(SRC))

    # The two facts the caption states. If either moves the caption is wrong, so assert before drawing.
    max_given = [d["control|max_given|%s" % m]["gain_ci"][0] > 0 for m, _ in MODELS]
    pair = [d["control|pair_distance|%s" % m]["gain_ci"][0] > 0 for m, _ in MODELS]
    assert all(max_given), "max_given no longer clears for every model"
    assert not any(pair), "pair_distance now clears for some model"
    print("assert OK: max_given clears 5/5, pair_distance clears 0/5")

    if a.check:
        if not os.path.exists(OUT):
            sys.exit("fig_ladder.pdf missing; run gauge/make_ladder_figure.py")
        print("CHECK ONLY -- asserts hold and the figure exists")
        return

    pub_style()
    # A shared x range across both panels. If each panel autoscaled, a gain of 0.6 and a gain of
    # 0.8 would occupy the same width and the two panels would read as comparable when they are not.
    allv = [v["gain_ci"] for k, v in d.items() if isinstance(v, dict) and "gain_ci" in v]
    lo = min(c[0] for c in allv)
    hi = max(c[1] for c in allv)
    pad = 0.05 * (hi - lo)
    xlim = (lo - pad, hi + pad)

    # Stacked, with height_ratios matched to the row counts. Side by side, panel (a)'s 3 groups and
    # panel (b)'s 5 groups would occupy the same axes height, so a row in (a) would be drawn nearly twice
    # as tall as a row in (b). Stacking with height_ratios gives every row the same pitch and lets the
    # two panels share one x axis.
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True,
                             gridspec_kw=dict(hspace=.22,
                                              height_ratios=[len(CONTROLS), len(RUNGS)]))
    seen = set()
    draw(axes[0], d, "control", CONTROLS, "a", "one link at a time", xlim, seen)
    draw(axes[1], d, "ladder", RUNGS, "b", "scaffolding the geometry", xlim, seen)
    axes[0].set_xlabel("")                       # shared axis: one label, on the bottom panel
    fig.tight_layout(pad=.35)
    # Legend type matches the tick labels rather than undercutting them.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(MODELS), frameon=False,
               fontsize=7.5, handletextpad=.35, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.06))
    save_cropped(fig, OUT)
    print("wrote %s  [shared x range %.2f to %.2f]" % (os.path.relpath(OUT, ROOT), *xlim))


if __name__ == "__main__":
    main()
