#!/usr/bin/env python3
"""Generate the figures from the result artifacts. No number is typed in by hand.

Writes, under outputs/tex/figs/: fig2.pdf (normalized gain against record length, per descriptor
class, for the five models), fig4.pdf (absence detection against record length), fig_overview.pdf
(the concept figure) and fig_taxonomy.pdf (what the models answer instead of the geometric value).
fig1() (the number of families that ever clear their baseline, per model) is defined but is not part
of the default run. fig_frontier.pdf and fig_ladder.pdf are made by make_frontier_figure.py and
make_ladder_figure.py, which import the style helpers from this module.

Input: results/v41_family_cis.json, results/v41_absence_all_models.json and
results/v41_error_taxonomy.json.

Requires matplotlib, fonttools (for the macOS font-collection helper) and the ghostscript command
`gs` (used by crop_figure.py).

Usage:
  python gauge/make_figures.py
"""
import json
import os
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "outputs", "tex", "figs")
os.makedirs(FIGS, exist_ok=True)

MODELS = ["phi35mini", "mistral7b", "llama8b", "gemma3_12b", "llama70b"]
LABEL = {"phi35mini": "Phi-3.5-mini", "mistral7b": "Mistral-7B", "llama8b": "Llama-3.1-8B",
         "gemma3_12b": "Gemma-3-12B", "llama70b": "Llama-3.1-70B"}
# Display names for figures: the same short forms the paper's tables use. Prose keeps LABEL's
# full names. One mapping serves every figure, so that no two figures name a model differently.
# Full names are not used in figures because the ladder figure's one-row legend would widen, and
# LaTeX would then shrink that figure's type when scaling it to the column.
SHORT = {"phi35mini": "Phi-3.5", "mistral7b": "Mistral-7B", "llama8b": "Llama-8B",
         "gemma3_12b": "Gemma-12B", "llama70b": "Llama-70B", "gpt-4o": "GPT-4o"}
GEOM = {"gyration_km", "max_distance", "total_distance", "longest_jump"}
OKABE = {"clears": "#0072B2", "never": "#D55E00", "geom": "#E69F00"}
# Okabe-Ito qualitative sequence for line plots, colourblind-safe.
LINES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
# Six-colour extension: the retrieval class has six families, so a four-colour cycle would draw two
# different families in the same colour inside one panel.
OKABE_SEQ = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

# Descriptor classes, in the order the paper discusses them. make_tables.py uses the same
# grouping and asserts that no family is missing.
CLASS_ORDER = [
    ("retrieval by position", ["retrieve_p10", "retrieve_p30", "retrieve_p50",
                               "retrieve_p70", "retrieve_p90", "retrieve_probe"]),
    ("count and sequence", ["day_distinct", "two_hop", "transition", "core_set"]),
    ("temporal", ["longest_stay", "recency", "time_mode"]),
    ("windowed count", ["window_distinct_w16", "window_distinct_w32",
                        "window_distinct_w64", "window_distinct_w128"]),
    ("spatial-geometric", ["gyration_km", "max_distance", "total_distance", "longest_jump"]),
]
PRETTY_FIG = {
    "retrieve_p10": "10th pct", "retrieve_p30": "30th pct", "retrieve_p50": "50th pct",
    "retrieve_p70": "70th pct", "retrieve_p90": "90th pct", "retrieve_probe": "probe",
    "day_distinct": "distinct/day", "two_hop": "two-hop", "transition": "transition",
    "core_set": "core set",
    "longest_stay": "longest stay", "recency": "recency", "time_mode": "modal timeslot",
    "window_distinct_w16": "last 16", "window_distinct_w32": "last 32",
    "window_distinct_w64": "last 64", "window_distinct_w128": "last 128",
    "gyration_km": "radius of gyration", "max_distance": "max displacement",
    "total_distance": "total distance", "longest_jump": "longest jump",
}
SPANS_ALL = [8, 16, 32, 64, 128, 256, 512]


def save_cropped(fig, path):
    """Save, then crop to the ink plus a 1 pt margin (crop_figure.py). Without the crop, any
    regeneration would restore the white padding and change how large LaTeX prints each figure.
    """
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:                  # importable however this module was launched
        sys.path.insert(0, here)
    from crop_figure import crop
    fig.savefig(path)
    crop(path, path, 1.0)


def _register_ttc_faces(ttc="/System/Library/Fonts/Helvetica.ttc"):
    """Make every face of a macOS font collection visible to matplotlib; returns how many.

    Matplotlib indexes only the first face of a .ttc, so on macOS "Helvetica" resolves to Regular
    alone and every fontweight="bold" in these figures (the panel letters, and the two gains in the
    overview figure) would silently print in Regular. Each TrueType member is extracted once to a cache
    directory and registered. Returns 0 when the collection does not exist (other platforms), in which
    case matplotlib falls back to Arial or DejaVu Sans.
    """
    import pathlib
    from matplotlib import font_manager as fm
    if not os.path.exists(ttc):
        return 0
    from fontTools.ttLib import TTCollection
    cache = pathlib.Path(os.path.expanduser("~/.cache/gauge/fonts"))
    cache.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, font in enumerate(TTCollection(ttc).fonts):
        if "glyf" not in font:                 # pdf.fonttype 42 needs TrueType outlines
            continue
        dst = cache / f"{pathlib.Path(ttc).stem}-{i}.ttf"
        if not dst.exists():
            font.save(str(dst))
        fm.fontManager.addfont(str(dst))
        n += 1
    return n


def pub_style():
    """The publication style used for every figure.

    Two adaptations for a two-column ACM paper: type sizes are 8 pt (an ACM column is 3.33 in wide),
    and figure widths are pinned to the ACM column and text widths.

    `pdf.fonttype: 42` is not cosmetic. Matplotlib's default Type 3 fonts render as PostScript
    procedures, copy and search badly, and are rejected by publisher pipelines including ACM's.

    There is no background grid; reference lines carry the thresholds.
    """
    _register_ttc_faces()
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 350, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.titleweight": "normal", "axes.labelpad": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "axes.grid": False, "grid.color": "#999999", "grid.alpha": 0.18, "grid.linewidth": 0.5,
    })


def legend_below(fig, ax, bottom=0.30, y=0.012, **kw):
    """Put the legend in a band reserved inside the canvas, and save with bbox_inches=None.

    `savefig.bbox` is "tight", so the saved PDF is cropped to the drawn content and not to `figsize`.
    An `ax.legend` anchored at a negative y sits outside the axes and widens that crop, by a different
    amount for each figure. LaTeX then scales each figure to the column width, so the same 8 pt label
    would print at three different sizes. Reserving the band inside the canvas makes the saved width
    exactly `figsize`, identical across figures, so LaTeX scales all of them by 1.0 and the type
    matches.
    """
    h, l = ax.get_legend_handles_labels()
    fig.subplots_adjust(bottom=bottom)
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y), frameon=False, **kw)


def panel(ax, letter, text="", letter_size=10, dx=-22):
    """Panel header: a bold lowercase letter and a regular-weight description.

    `dx` is how far left of the axes the letter hangs. -22 suits panels with a visible y axis, where
    the letter sits outside the tick labels. Panels drawn with axis("off") have no such margin, so a
    letter at -22 would land inside the previous panel's title.
    """
    # When the letter is tucked against its own axes (dx near 0) the title has to be indented or
    # the two land on the same point and print as "aone person's record".
    if text:
        ax.set_title(text, loc="left", fontweight="normal", pad=6,
                     x=0.0 if dx <= -12 else 0.055)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, 7), textcoords="offset points",
                fontsize=letter_size, fontweight="bold", va="bottom", ha="left",
                annotation_clip=False)


def load():
    return json.load(open(os.path.join(ROOT, "results", "v41_family_cis.json")))


def fig1(d):
    """Per model: how many families ever exceed their baseline, and how many never do."""
    ever = collections.defaultdict(dict)
    for k, v in d.items():
        if not isinstance(v, dict) or "gain" not in v:
            continue
        m, fam, _ = k.split("|")
        ever[m][fam] = ever[m].get(fam, False) or (v["gain_ci"][0] > 0)

    fig, ax = plt.subplots(figsize=(3.33, 2.35))
    order = list(reversed(MODELS))   # smallest at top, so size reads downward
    ys = range(len(order))
    clears = [sum(1 for f, e in ever[m].items() if e) for m in order]
    never = [sum(1 for f, e in ever[m].items() if not e) for m in order]
    ngeom = [sum(1 for f, e in ever[m].items() if not e and f in GEOM) for m in order]

    nonlgeom = [n - g for n, g in zip(never, ngeom)]
    ax.barh(ys, clears, color=OKABE["clears"], height=.62,
            label="exceeds baseline at some length")
    ax.barh(ys, nonlgeom, left=clears, color=OKABE["never"], height=.62,
            label="never exceeds baseline")
    ax.barh(ys, ngeom, left=[c + n for c, n in zip(clears, nonlgeom)], height=.62,
            color=OKABE["never"], hatch="///", edgecolor="white", linewidth=0,
            label="of which geometric")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABEL[m] for m in order])
    ax.set_xlabel("descriptor families")
    ax.set_xlim(0, max(c + n for c, n in zip(clears, never)) + 1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=.3)
    legend_below(fig, ax, bottom=0.30, ncol=2, columnspacing=1.0, handlelength=1.5)
    fig.savefig(os.path.join(FIGS, "fig1.pdf"))
    plt.close(fig)
    return dict(zip(order, zip(clears, never, ngeom)))


def fig2(d):
    """Gain against record length, per descriptor class, for all five models.

    Each line is a model and each panel a descriptor class, averaging over the families in that class.
    This keeps the figure at its original size while letting the class split be checked across the
    whole model set that the claims range over. The per-family detail is in the main results table.
    """
    MODELS_F = [(m, SHORT[m]) for m in MODELS]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharey=True, sharex=True)
    axes = axes.ravel()
    for i, (cls, fams) in enumerate(CLASS_ORDER):
        ax = axes[i]
        drawn = 0
        for j, (mk, mlab) in enumerate(MODELS_F):
            acc = {}
            for fam in fams:
                for k, v in d.items():
                    if k.startswith(f"{mk}|{fam}|") and isinstance(v, dict) and "gain" in v:
                        acc.setdefault(int(k.split("|")[2]), []).append(v["gain"])
            pts = sorted((x, sum(g) / len(g)) for x, g in acc.items())
            if not pts:
                continue
            ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], marker="o", ms=2.6, lw=1.1,
                    color=OKABE_SEQ[j % len(OKABE_SEQ)], label=mlab)
            drawn += 1
        ax.axhline(0, color="0.35", lw=.7, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SPANS_ALL)
        ax.set_xticklabels([str(x) for x in SPANS_ALL])
        ax.minorticks_off()
        panel(ax, "abcde"[i], cls)
        if drawn == 0:
            ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes)
    axes[-1].axis("off")
    # One legend in the empty sixth cell: five models repeated in five panels would be clutter.
    _h, _l = axes[0].get_legend_handles_labels()
    axes[-1].legend(_h, _l, loc="center", frameon=False, fontsize=7.5, labelspacing=.5)
    axes[2].tick_params(labelbottom=True)
    for ax in axes[3:5]:
        ax.set_xlabel("record length (rows)")
    axes[2].set_xlabel("record length (rows)")
    for i in (3, 4):
        axes[i].axhspan(-0.05, 0.05, color="0.85", zorder=0, lw=0)
    # Label above the band, placed in data units: at axes-fraction 0.06 it would sit on the geometric
    # lines, which run inside the band (a label may not cover data).
    import matplotlib.transforms as mtrans
    axes[4].text(0.97, 0.065, "presence threshold",
                 transform=mtrans.blended_transform_factory(axes[4].transAxes, axes[4].transData),
                 ha="right", va="bottom", fontsize=5.8, color="0.35")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("normalized gain")
    fig.tight_layout(pad=.35, w_pad=.8, h_pad=1.1)
    save_cropped(fig, os.path.join(FIGS, "fig2.pdf"))
    plt.close(fig)


def fig4():
    """Correct abstention against record length, for all five models, with 95% intervals.

    The unanswerable half is the family `retrieve_probe` inside the main 34,200-question set, which
    every model ran. Correct abstention is saying "none" when the record lacks the answer. The figure
    reads results/v41_absence_all_models.json and asserts the numbers that the text states.
    """
    a = json.load(open(os.path.join(ROOT, "results", "v41_absence_all_models.json")))
    spans = a["spans"]
    order = ["Phi-3.5-mini", "Mistral-7B", "Llama-3.1-8B", "Gemma-3-12B", "Llama-3.1-70B"]
    fig, ax = plt.subplots(figsize=(3.33, 2.25))
    for k, name in enumerate(order):
        m = a["models"][name]
        y = [m[str(s)]["abstain"] for s in spans]
        lo = [m[str(s)]["lo"] for s in spans]
        hi = [m[str(s)]["hi"] for s in spans]
        c = OKABE_SEQ[k % len(OKABE_SEQ)]
        ax.fill_between(spans, lo, hi, color=c, alpha=.15, linewidth=0)
        ax.plot(spans, y, marker="o", ms=3.0, lw=1.2, color=c,
                label=SHORT[{v: k for k, v in LABEL.items()}[name]])
    ax.set_xscale("log", base=2)
    ax.set_xticks(spans)
    ax.set_xticklabels([str(x) for x in spans])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("record length (rows)")
    ax.set_ylabel("correct abstention")
    fig.tight_layout(pad=.3)
    legend_below(fig, ax, bottom=0.42, ncol=2, columnspacing=1.0, handlelength=1.4)
    save_cropped(fig, os.path.join(FIGS, "fig4.pdf"))
    plt.close(fig)
    # The numbers that the text states. If any moves, a sentence in the paper is wrong.
    g = a["models"]["Gemma-3-12B"]; l7 = a["models"]["Llama-3.1-70B"]
    assert round(g[str(spans[0])]["abstain"], 2) == 0.97 and round(g[str(spans[-1])]["abstain"], 2) == 0.13
    assert round(l7[str(spans[0])]["abstain"], 2) == 0.99 and round(l7[str(spans[-1])]["abstain"], 2) == 0.66
    assert a["models"]["Mistral-7B"][str(spans[0])]["abstain"] < 0.20
    assert a["models"]["Phi-3.5-mini"][str(spans[0])]["abstain"] < 0.10
    print("fig4: five models, abstention %.2f->%.2f (Gemma), %.2f->%.2f (Llama-70B)"
          % (g[str(spans[0])]["abstain"], g[str(spans[-1])]["abstain"],
             l7[str(spans[0])]["abstain"], l7[str(spans[-1])]["abstain"]))


# Entry point for the figures that read the scored artifacts.
if __name__ == "__main__":
    pub_style()
    _d = load()
    fig2(_d)
    fig4()


def fig_overview():
    """The overview figure, in three panels.

    (a) the raw material: one person's location record
    (b) what the field does with it: generate trajectories, score descriptor distributions
    (c) what comes back when one person's descriptor is recovered from their own record: the best gain
        of Llama-3.1-70B for distinct places per day and for radius of gyration, read from
        results/v41_family_cis.json
    """
    import matplotlib.patches as mp
    # wspace is explicit: panel() hangs the bold letter 22 pt left of its axes, so at the default
    # spacing panel c's letter would land inside panel b's title.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05),
                             gridspec_kw=dict(wspace=.42))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # (a) raw record
    a = axes[0]
    panel(a, "a", dx=-2, text="one person's record")
    rows = ["day  slot  place", "  3    18   41966", "  3    21   41966",
            "  4     7   40213", "  4    19   41966", "  5     9   38771", "  ...   ...     ..."]
    for i, r in enumerate(rows):
        a.text(.06, .80 - i * .115, r, family="monospace", fontsize=6.4,
               color="#333333" if i else "#000000",
               fontweight="normal" if i else "bold")
    a.add_patch(mp.FancyBboxPatch((.02, .06), .70, .86, boxstyle="round,pad=0.012",
                                  fc="#f4f4f1", ec="#bbbbbb", lw=.7))

    # (b) what the field does
    b = axes[1]
    panel(b, "b", dx=-2, text="what the field scores")
    xs = [i / 40 for i in range(41)]
    import math
    gen = [math.exp(-((x - .45) ** 2) / .02) for x in xs]
    real = [math.exp(-((x - .5) ** 2) / .025) for x in xs]
    b.plot([.10 + .78 * x for x in xs], [.30 + .38 * y for y in real],
           color=LINES[0], lw=1.3, label="real")
    b.plot([.10 + .78 * x for x in xs], [.30 + .38 * y for y in gen],
           color=LINES[1], lw=1.3, ls="--", label="generated")
    b.text(.5, .18, "descriptor distribution over a population", ha="center", fontsize=6.4,
           color="#333333")
    b.text(.5, .07, "scored by JSD", ha="center", fontsize=6.4, style="italic", color="#555555")
    b.legend(loc="upper right", fontsize=6.2, handlelength=1.2, borderpad=.2)

    # (c) what we ask
    c = axes[2]
    panel(c, "c", dx=-2, text="what comes back")
    # This panel carries the paper's thesis with measured gains: the same model, the same record,
    # two descriptors, opposite outcomes. Numbers are read from the scored artifact.
    _cis = json.load(open(os.path.join(ROOT, "results", "v41_family_cis.json")))
    def _best(fam):
        g = [v["gain"] for k, v in _cis.items()
             if k.startswith("llama70b|%s|" % fam) and isinstance(v, dict) and "gain" in v]
        return max(g)
    c.add_patch(mp.FancyBboxPatch((.04, .70), .92, .22, boxstyle="round,pad=0.012",
                                  fc="#eaf2f8", ec=LINES[0], lw=.8))
    c.text(.50, .81, "ask one person's record\nfor one descriptor", ha="center", va="center",
           fontsize=7.0)
    c.annotate("", xy=(.50, .60), xytext=(.50, .68),
               arrowprops=dict(arrowstyle="-|>", color="#555555", lw=.9))
    for i, (fam, lab) in enumerate([("day_distinct", "distinct places / day"),
                                    ("gyration_km", "radius of gyration")]):
        g = _best(fam)
        y = .42 - i * .26
        ok = g >= 0.05
        c.add_patch(mp.FancyBboxPatch((.04, y), .92, .19, boxstyle="round,pad=0.010",
                                      fc="#ffffff", ec="#bbbbbb", lw=.7))
        c.text(.08, y + .095, lab, ha="left", va="center", fontsize=6.0, color="#222222")
        c.text(.92, y + .095, "%+.2f" % g, ha="right", va="center", fontsize=7.0,
               fontweight="bold", color=OKABE["clears"] if ok else OKABE["never"])
    c.text(.50, .02, "best gain over the baseline, %s" % SHORT["llama70b"], ha="center", va="bottom",
           fontsize=5.9, color="#666666", style="italic")
    # The panel prints two numbers, so the numbers get an assert like every other display.
    assert round(_best("day_distinct"), 2) == 0.74, "panel c: day_distinct is no longer +0.74"
    assert round(_best("gyration_km"), 2) == -0.04, "panel c: gyration_km is no longer -0.04"

    fig.tight_layout(pad=.35)
    save_cropped(fig, os.path.join(FIGS, "fig_overview.pdf"))
    plt.close(fig)
    print("wrote fig_overview.pdf")


# Not at module level. make_ladder_figure.py imports this module for pub_style, panel and
# OKABE_SEQ, so figures drawn at import time would be regenerated before pub_style() had been
# applied, with matplotlib's default fonts.
if __name__ == "__main__":
    pub_style()
    fig_overview()


def fig_taxonomy():
    """What the models produce instead of a correct geometric answer.

    A null says only that exact recovery fails. It does not say whether the model repeats one value,
    guesses in the right neighbourhood, or answers something the wrong size. The categories come from
    error_taxonomy.py: correct; close (within a factor of 2, not exact); near-constant (the cell's modal
    answer, when that answer dominates); wrong magnitude (off by more than 2x); outside the answer
    range; no answer (nothing parseable). Averaged over the four geometric families.
    """
    tax = json.load(open(os.path.join(ROOT, "results", "v41_error_taxonomy.json")))["taxonomy"]
    # all six categories, so every bar sums to 1 and nothing is silently dropped
    CATS = [("correct", "correct"), ("close", "within a factor of 2"),
            ("near_constant", "a repeated value"), ("wrong_magnitude", "wrong magnitude"),
            ("out_of_range", "outside the answer range"), ("no_answer", "no answer")]
    COL = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7", "#999999"]
    order = list(reversed(MODELS))
    vals = []
    for m in order:
        ks = [k for k in tax if k.split("|")[1] == m and k.split("|")[0] in GEOM]
        vals.append([sum(tax[k][c] for k in ks) / len(ks) for c, _ in CATS])

    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    ys = range(len(order))
    left = [0.0] * len(order)
    for i, ((c, lab), col) in enumerate(zip(CATS, COL)):
        w = [v[i] for v in vals]
        ax.barh(ys, w, left=left, height=.62, color=col, label=lab)
        left = [a + b for a, b in zip(left, w)]
    ax.set_yticks(list(ys)); ax.set_yticklabels([SHORT[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("share of answers on geometric families")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=.3)
    legend_below(fig, ax, bottom=0.46, ncol=2, columnspacing=1.0, handlelength=1.4)
    save_cropped(fig, os.path.join(FIGS, "fig_taxonomy.pdf"))
    plt.close(fig)
    for m, v in zip(order, vals):
        assert abs(sum(v) - 1.0) < 1e-6, f"{m}: categories sum to {sum(v):.3f}, not 1"
        print(f"  {LABEL[m]:<15} correct {v[0]:.3f}  close {v[1]:.3f}  repeated {v[2]:.3f}  "
              f"wrong-mag {v[3]:.3f}  sum {sum(v):.3f}")


# Not at module level. make_ladder_figure.py imports this module for pub_style, panel and
# OKABE_SEQ, so figures drawn at import time would be regenerated before pub_style() had been
# applied, with matplotlib's default fonts.
if __name__ == "__main__":
    pub_style()
    fig_taxonomy()
