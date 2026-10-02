#!/usr/bin/env python3
"""Geometry and multi-record question families of the earlier v4 generator, plus shared helpers.

The v4.1 generator (tally_v41.py) imports MAX_PROMPT_CHARS, rigid and render_xy from this module.
The rest of the file builds the earlier v4 item set: the v3 tasks at 300 questions per cell, a
binned radius-of-gyration family (tier F) and a which-person family (tier G). That item set is not
part of the 34,200-question benchmark.

Coordinates are released only under a per-item random rigid transform: rotation, reflection and
integer translation. Distances stay exact and absolute position is destroyed, which closes the route
through memorised coordinates in the same way that per-item place relabelling does for place ids.
Gold answers are computed from the transformed, rounded coordinates that the prompt shows. The data
provider does not disclose the city (Yabe et al. 2024), so no place is mapped, geocoded or named.

Input: data/yjmob/yjmob_v4.parquet (written by export_v4.py).
Output: results/tally_v4.jsonl and results/tally_v4_screen.json.

Usage:
  python gauge/tally_v4.py --data data/yjmob/yjmob_v4.parquet --out results/tally_v4.jsonl
"""
from __future__ import annotations
import argparse, collections, itertools, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tally_v3 import (TASKS as T3, MIN_OBS as MIN3, relabel, render, NIGHT, SLOTS_PER_DAY, SPLIT_DAY)

RG_EDGES = [3.1, 5.7, 8.9, 13.6]  # pooled quintiles of the radius of gyration, in km
RG_LABEL = ["A", "B", "C", "D", "E"]
LET = "ABCDEFGH"
# Longest prompt allowed, in characters. Measured with the tokenizers of the evaluated models over
# the longest items, the worst case is about 1.19 characters per token, because these records are
# digit-dense and several tokenizers split digits individually. The smallest context window among the
# evaluated models is 32,768 tokens, and 37,000 characters fit inside it with about 1,200 tokens to
# spare. The longest prompt the generator produces is about 33,000 characters, so the cap never
# binds. That matters: a generator that rejects over-cap candidates and refills to quota does not
# shorten the tail of the distribution, it selects against it.
MAX_PROMPT_CHARS = 37000


def rigid(xy, rng):
    """Random rotation + reflection + integer translation. Distances exact, position destroyed."""
    th = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if rng.random() < .5:
        R = R @ np.array([[-1., 0.], [0., 1.]])
    return np.rint((xy - xy.mean(0)) @ R.T + rng.integers(-400, 400, 2)).astype(int)


def render_xy(days, tods, P):
    return "\n".join(f"d{d} t{t} x{x} y{y}" for d, t, (x, y) in zip(days, tods, P))


def F_rg(s, rng, span):
    """Binned radius of gyration. Bin edges are FIXED, stated in the question, identical across
    spans, so bins are never a per-span confound."""
    P = rigid(s[["x", "y"]].to_numpy(float), rng)
    rg = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()) * 0.5)
    k = int(np.digitize(rg, RG_EDGES))
    return dict(tier="F", task="gyration", atype="categorical",
                q=("Treating each unit on the x and y axes as 500 metres, how spread out is this "
                   "person's movement? Report the radius of gyration band: "
                   "A under 3.1 km, B 3.1-5.7, C 5.7-8.9, D 8.9-13.6, E over 13.6. "
                   "Answer with the single letter."),
                a=RG_LABEL[k], mag=None, _P=P)


def F_farthest(s, rng, span):
    """Day on which the person went farthest from where they sleep. Order/position sensitive."""
    P = rigid(s[["x", "y"]].to_numpy(float), rng)
    nightmask = s.tod.isin(NIGHT).to_numpy()
    if nightmask.sum() < 5:
        return None
    hc = collections.Counter(map(tuple, P[nightmask])).most_common(1)[0][0]
    dist = np.sqrt(((P - np.array(hc)) ** 2).sum(1)) * 0.5
    dd = pd.Series(dist).groupby(s.day.to_numpy()).max()
    if len(dd) < 3 or dd.nlargest(2).iloc[0] == dd.nlargest(2).iloc[1]:
        return None
    return dict(tier="F", task="farthest_day", atype="numeric",
                q=("Treating each unit on the x and y axes as 500 metres, and taking the place they "
                   "are at most often during timeslots 0-11 as home: on which day did this person "
                   "travel farthest from home? Answer with the day number."),
                a=str(int(dd.idxmax())), mag=None, _P=P)


TASKS_F = [F_rg, F_farthest]
MIN_F = {"gyration": 32, "farthest_day": 32}


def make_G(by, users, rng, K, span_days):
    """Which of K people was at place P on day D. Answer in {A..K, none}.
    Stratified over the ANSWER by choosing the QUERY, never by choosing the people."""
    us = rng.choice(users, K, replace=False)
    d0 = int(rng.integers(0, SPLIT_DAY - span_days))
    ws = [by[u][(by[u].day >= d0) & (by[u].day < d0 + span_days)] for u in us]
    if any(len(w) < 12 for w in ws):
        return None
    pres = collections.defaultdict(set)
    for i, w in enumerate(ws):
        for dd, cc in set(zip(w.day, w.cell)):
            pres[(dd, cc)].add(i)
    byans = collections.defaultdict(list)
    for q, st in pres.items():
        if len(st) == 1:
            byans[LET[next(iter(st))]].append(q)
    allc = list({c for (_, c) in pres})
    none = [(dd, cc) for dd in range(d0, d0 + span_days)
            for cc in rng.choice(allc, min(5, len(allc)), replace=False) if (dd, cc) not in pres]
    if none:
        byans["none"] = none
    avail = [k for k, v in byans.items() if v]
    if len(avail) < K:
        return None
    ans = str(rng.choice(avail)); q = byans[ans][int(rng.integers(len(byans[ans])))]
    # ONE SHARED relabelling over the union of cells: an id means the same thing in every record of
    # this item and nothing outside it.
    allcells = sorted({c for w in ws for c in w.cell})
    m = {c: int(v) for c, v in zip(allcells, rng.choice(4096, len(allcells), replace=False))}
    body = "\n\n".join(f"Person {LET[i]}:\n" +
                       "\n".join(f"d{int(r.day)} t{int(r.slot % SLOTS_PER_DAY)} p{m[r.cell]}"
                                 for r in w.itertuples())
                       for i, w in enumerate(ws))
    return dict(tier="G", task="which_person", atype="categorical",
                q=(f"Which of these {K} people was at place {m[q[1]]} on day {q[0]}? "
                   f"Answer with the single letter, or 'none'."),
                a=ans, mag=None, K=K, span_days=span_days, _body=body,
                n_rows=sum(len(w) for w in ws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yjmob/yjmob_v4.parquet")
    ap.add_argument("--spans", default="8,16,32,64,128,256,512")
    ap.add_argument("--per-cell", type=int, default=300)
    ap.add_argument("--users", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tally_v4.jsonl")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = pd.read_parquet(a.data)
    d = d[~d.is_test].copy()
    d["day"] = d.slot // SLOTS_PER_DAY
    d["tod"] = d.slot % SLOTS_PER_DAY
    d["cell"] = d.x.astype(np.int32) * 1000 + d.y
    uids = d.uid.unique()[:a.users]
    by = {u: g.sort_values("slot").reset_index(drop=True)
          for u, g in d[d.uid.isin(set(uids))].groupby("uid", sort=False)}
    users = list(by)
    SPANS = [int(x) for x in a.spans.split(",")]
    print(f"{len(d):,} train rows | {len(users):,} users | spans {SPANS} | n={a.per_cell}/cell\n",
          flush=True)

    items, drops = [], collections.Counter()

    def emit(it, s, span, body=None):
        it["family"] = it["task"]
        rec = body if body is not None else render(s)
        it.update(uid=int(s.uid.iloc[0]) if body is None else -1, span=span,
                  n_rows=it.get("n_rows", rec.count("\n") + 1), n_chars=len(rec),
                  prompt_full=(f"Here is a person's location record. Each line is "
                               f"day, timeslot, place.\n\n{rec}\n\n"
                               f"Question: {it['q']}\nAnswer with the value only."),
                  prompt_blind=(f"Question about a person's location record: {it['q']}\n"
                                f"Answer with the value only."))
        items.append(it)

    # ---- tiers A-E, ported verbatim from v3 -------------------------------------------------
    for span, fn in itertools.product(SPANS, T3):
        made, tries = 0, 0
        while made < a.per_cell and tries < a.per_cell * 300:
            tries += 1
            u = int(rng.choice(users)); g = by[u]
            if len(g) < span + 4:
                drops["short"] += 1; continue
            st = int(rng.integers(0, len(g) - span))
            s = relabel(g.iloc[st:st + span], rng)
            it = fn(s, rng, span)
            if not it:
                drops[f"reject_{fn.__name__}"] += 1; continue
            if span < MIN3[it["task"]]:
                break
            assert int(s.day.max()) < SPLIT_DAY
            emit(it, s, span)
            for dd in {int(x) for x in re.findall(r"day (\d+)", it["q"])}:
                assert dd in set(int(x) for x in s.day.unique()), "unanswerable"
            assert items[-1]["n_rows"] == span, f"span dial broken {items[-1]['n_rows']}!={span}"
            made += 1
        if made:
            print(f"  span {span:>4}  {fn.__name__:<14} {made:>4}", flush=True)

    # ---- tier F ------------------------------------------------------------------------------
    for span, fn in itertools.product([s for s in SPANS if s >= 32], TASKS_F):
        made, tries = 0, 0
        while made < a.per_cell and tries < a.per_cell * 300:
            tries += 1
            u = int(rng.choice(users)); g = by[u]
            if len(g) < span + 4:
                continue
            st = int(rng.integers(0, len(g) - span))
            s = g.iloc[st:st + span]
            it = fn(s, rng, span)
            if not it:
                drops[f"reject_{fn.__name__}"] += 1; continue
            P = it.pop("_P")
            body = render_xy(s.day.to_numpy(), s.tod.to_numpy(), P)
            it["family"] = it["task"]
            it.update(uid=int(u), span=span, n_rows=span, n_chars=len(body),
                      prompt_full=(f"Here is a person's location record. Each line is "
                                   f"day, timeslot, x, y.\n\n{body}\n\n"
                                   f"Question: {it['q']}\nAnswer with the value only."),
                      prompt_blind=(f"Question about a person's location record: {it['q']}\n"
                                    f"Answer with the value only."))
            assert it["n_rows"] == span
            items.append(it); made += 1
        if made:
            print(f"  span {span:>4}  {fn.__name__:<14} {made:>4}", flush=True)

    # ---- tier G: span_days is the length dial at K=4; K is a SECOND dial at 7 days ------------
    G_CELLS = [(4, sd) for sd in (3, 7, 14, 21)] + [(k, 7) for k in (2, 3, 5)]
    for K, sd in G_CELLS:
        made, tries = 0, 0
        while made < a.per_cell and tries < a.per_cell * 300:
            tries += 1
            it = make_G(by, users, rng, K, sd)
            if not it:
                drops["reject_G"] += 1; continue
            body = it.pop("_body")
            it["family"] = it["task"]
            if len(body) > MAX_PROMPT_CHARS - 400:
                drops["G_too_long"] += 1; continue
            it.update(uid=-1, span=sd, n_chars=len(body),
                      prompt_full=(f"Here are location records for {K} people. Each line is "
                                   f"day, timeslot, place.\n\n{body}\n\n"
                                   f"Question: {it['q']}\nAnswer with the value only."),
                      prompt_blind=(f"Question about location records for {K} people: {it['q']}\n"
                                    f"Answer with the value only."))
            items.append(it); made += 1
        if made:
            print(f"  K={K} days={sd:<3}  which_person   {made:>4}", flush=True)

    short = {k: v for k, v in collections.Counter(
        (i["family"], i["span"], i.get("K")) for i in items).items() if v != a.per_cell}
    assert not short, f"cells that did not reach {a.per_cell}: {short}"
    print(f"  quota assertion OK: every cell filled to {a.per_cell}")

    # Nothing may exceed the smallest evaluated model's usable context. The generator and the
    # evaluator must agree on this by construction.
    over = [i for i in items if len(i["prompt_full"]) > MAX_PROMPT_CHARS]
    assert not over, f"{len(over)} items exceed MAX_PROMPT_CHARS ({MAX_PROMPT_CHARS})"
    print(f"  context assertion OK: longest prompt {max(len(i['prompt_full']) for i in items):,} chars"
          f" <= {MAX_PROMPT_CHARS:,}")

    # ---- screens -----------------------------------------------------------------------------
    print(f"\n{len(items):,} items\n")
    print(f"  {'tier':<5}{'task':<15}{'n':>6}{'floor':>8}   verdict")
    screen = {}
    for t in sorted({i["task"] for i in items},
                    key=lambda x: [i for i in items if i["task"] == x][0]["tier"] + x):
        sub = [i for i in items if i["task"] == t]
        c = collections.Counter(i["a"] for i in sub)
        fl = c.most_common(1)[0][1] / len(sub)
        v = "EXPLOITABLE" if fl >= .5 else "weak" if fl >= .35 else "OK"
        screen[t] = dict(n=len(sub), floor=fl, tier=sub[0]["tier"],
                         spans=sorted({i["span"] for i in sub}), verdict=v)
        print(f"  {sub[0]['tier']:<5}{t:<15}{len(sub):>6}{fl:>8.3f}   {v}")
    print(f"\n  distinct users touched: {len({i['uid'] for i in items if i['uid']>=0}):,}")
    print(f"  drops: {dict(drops.most_common(6))}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        for i in items:
            fh.write(json.dumps(i) + "\n")
    json.dump(screen, open(a.out.replace(".jsonl", "_screen.json"), "w"), indent=1)
    print(f"\nwrote {a.out} + _screen.json")


if __name__ == "__main__":
    main()
