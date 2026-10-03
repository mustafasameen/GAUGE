#!/usr/bin/env python3
"""Draw the frontier-model figure (fig_frontier.pdf): the GPT-4o arm beside the five open-weight models.

The figure has one job: to show that adding a stronger model does not move the geometric families,
while the same model is at ceiling on retrieval. Panel (a) carries the four geometric families and
panel (b) the two contrast families (distinct places per day, and retrieval at the 50th
percentile). All six models appear in both panels, each as the median normalized gain across record
lengths with the median of its 95% interval.

Coverage is uneven on purpose, and the caption says so: the arm covers all four geometric families
and one family each from two other classes, so two panels state what was measured.

Style is imported from make_figures.py (pub_style, the colour sequence and panel()) and not
re-declared, so that a second copy cannot drift.

Input: results/frontier.json (from score_frontier.py).
Output: outputs/tex/figs/fig_frontier.pdf.

Usage:
  python gauge/make_frontier_figure.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gauge"))
from make_figures import pub_style, panel, OKABE_SEQ, SHORT, save_cropped   # single source of truth for style

SRC = os.path.join(ROOT, "results", "frontier.json")
OUT = os.path.join(ROOT, "outputs", "tex", "figs", "fig_frontier.pdf")

MODELS = ["Phi-3.5", "Mistral-7B", "Llama-8B", "Gemma-12B", "Llama-70B", "gpt-4o"]
GEO = [("gyration_km", "radius of gyration"), ("max_distance", "max displacement"),
       ("total_distance", "total distance"), ("longest_jump", "longest jump")]
CONTRAST = [("day_distinct", "distinct places/day"), ("retrieve_p50", "retrieval at 50th pct")]
PRESENCE = 0.05


def draw(ax, d, fams, letter, title):
    """One row per family, six models per row, median gain across spans with its interval."""
    ys = []
    for r, (fam, lab) in enumerate(fams):
        for m, col in zip(MODELS, OKABE_SEQ):
            cells = [v for k, v in d["cells"].items()
                     if k.startswith(m + "|") and v["family"] == fam]
            if not cells:
                continue
            g = float(np.median([c["gain"] for c in cells]))
            lo = float(np.median([c["ci"][0] for c in cells]))
            hi = float(np.median([c["ci"][1] for c in cells]))
            y = r + 0.32 - 0.128 * MODELS.index(m)
            ax.plot([lo, hi], [y, y], color=col, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot([g], [y], "o", color=col, ms=3.4, zorder=3,
                    label=SHORT.get(m, m) if r == 0 else None)
        ys.append(r)
    ax.axvline(0, color="0.35", lw=.7, ls="--", zorder=1)
    ax.axvline(PRESENCE, color="0.55", lw=.7, ls=":", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in fams])
    ax.set_ylim(-0.55, len(fams) - 0.45)
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    panel(ax, letter, title)


def main():
    pub_style()
    d = json.load(open(SRC))
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True,
                             gridspec_kw=dict(hspace=.30, height_ratios=[len(GEO), len(CONTRAST)]))
    draw(axes[0], d, GEO, "a", "spatial-geometric")
    draw(axes[1], d, CONTRAST, "b", "contrast families")
    axes[1].set_xlabel("normalized gain over the baseline")
    fig.tight_layout(pad=.35)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7,
               handletextpad=.35, columnspacing=1.4, bbox_to_anchor=(0.5, -0.10))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_cropped(fig, OUT)          # savefig.bbox is "tight" in pub_style

    # asserts: the two claims the figure exists to make
    geo = [v for k, v in d["cells"].items() if v["geometric"]]
    assert not any(v["clears"] for v in geo), "a geometric cell now clears; caption is wrong"
    g4 = [v for k, v in d["cells"].items() if k.startswith("gpt-4o|") and v["family"] == "retrieve_p50"]
    assert all(v["gain"] > 0.9 for v in g4), "gpt-4o retrieval is no longer at ceiling"
    print("asserts OK: 0 of %d geometric cells clear; gpt-4o retrieval at ceiling in %d/%d spans"
          % (len(geo), sum(1 for v in g4 if v["gain"] > 0.9), len(g4)))
    print("wrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
