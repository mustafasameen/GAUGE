#!/usr/bin/env python3
"""Place-record question families (tiers A to E) and the helpers the benchmark generator reuses.

generate_questions.py imports relabel, render, SLOTS_PER_DAY and SPLIT_DAY from this module, and
coord_records.py imports the task list. The task families defined here form a separate question
set: retrieval, aggregation, ranking, home and work inference, and day-level structure. They are
not part of the 34,200-question benchmark.

Input: a parquet file of per-person place-rank records, for example data/yjmob/yjmob_ranks.parquet.
Output: an items file (JSON lines) and a screen file with one answer-concentration row per family.

Usage:
  python gauge/place_records.py --data data/yjmob/yjmob_ranks.parquet --out results/place_questions.jsonl
"""
from __future__ import annotations
import argparse, collections, itertools, json, os, re
import numpy as np, pandas as pd

SLOTS_PER_DAY = 48
SPLIT_DAY = 60
# Timeslots are 30-minute bins of the day. In the data, the share of records at a person's top
# place peaks near slot 3 and is lowest near slot 22, which is where local night and midday fall.
# Slot 0 is therefore taken as local midnight, and the windows below are local-time windows.
NIGHT = list(range(0, 12))                 # 00:00-06:00 local -> "overnight"
WORKHRS = list(range(18, 36))              # 09:00-18:00 local -> "daytime"


# ----------------------------------------------------------------- A. retrieve (control)
def A_retrieve(s, rng, span_obs):
    r = s.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
    return dict(tier="A", task="retrieve",
                q=f"What place was this person at on day {int(r.day)}, timeslot {int(r.tod)}?",
                a=str(int(r["rank"])), atype="numeric", mag=None)


# ----------------------------------------------------------------- B. aggregate
def B_distinct(s, rng, span_obs):
    n = int(s["rank"].nunique())
    return dict(tier="B", task="distinct",
                q="How many distinct places does this record contain?",
                a=str(n), atype="numeric", mag=n)


def B_occurrence(s, rng, span_obs):
    vc = s["rank"].value_counts()
    byc = collections.defaultdict(list)
    for pl, n in vc.items():
        byc[int(n)].append(int(pl))
    k = int(rng.choice(sorted(byc)))                 # stratify over the COUNT, not the place:
    p = int(rng.choice(byc[k]))                      # uniform-over-places makes "1" the answer 74%
    return dict(tier="B", task="occurrence",
                q=f"How many times does place {p} appear in this record?",
                a=str(k), atype="numeric", mag=k)


def B_difference(s, rng, span_obs):
    half = len(s) // 2
    if half < 4:
        return None
    a, b = set(s.iloc[:half]["rank"]), set(s.iloc[half:]["rank"])
    d1, d2 = int(s.iloc[half - 1].day), int(s.iloc[half].day)
    if d1 == d2:
        return None                      # split must fall on a day boundary to be stateable
    return dict(tier="B", task="difference",
                q=f"Consider the part of the record up to and including day {d1}, and the part from "
                  f"day {d2} onward. How many places appear in the LATER part but NOT the earlier?",
                a=str(len(b - a)), atype="numeric", mag=len(b - a))


def B_intersection(s, rng, span_obs):
    half = len(s) // 2
    if half < 4:
        return None
    a, b = set(s.iloc[:half]["rank"]), set(s.iloc[half:]["rank"])
    d1, d2 = int(s.iloc[half - 1].day), int(s.iloc[half].day)
    if d1 == d2:
        return None
    return dict(tier="B", task="intersection",
                q=f"Consider the part of the record up to and including day {d1}, and the part from "
                  f"day {d2} onward. How many places appear in BOTH parts?",
                a=str(len(a & b)), atype="numeric", mag=len(a & b))


# ----------------------------------------------------------------- C. rank / compare
def C_second_place(s, rng, span_obs):
    vc = s["rank"].value_counts()
    if len(vc) < 3 or vc.iloc[1] == vc.iloc[2]:      # require a strict 2nd to avoid ties
        return None
    return dict(tier="C", task="second_place",
                q="Which place is the SECOND most frequent in this record? Answer with the place "
                  "number.",
                a=str(int(vc.index[1])), atype="numeric", mag=None)


def C_busier_half(s, rng, span_obs):
    half = len(s) // 2
    if half < 4:
        return None
    na, nb = s.iloc[:half]["rank"].nunique(), s.iloc[half:]["rank"].nunique()
    d1, d2 = int(s.iloc[half - 1].day), int(s.iloc[half].day)
    if na == nb or d1 == d2:
        return None
    return dict(tier="C", task="busier_half",
                q=f"Did this person visit more distinct places from day {d2} onward than up to and "
                  f"including day {d1}? Answer yes or no.",
                a="yes" if nb > na else "no", atype="binary", mag=None)


# ----------------------------------------------------------------- D. infer semantics
def D_home(s, rng, span_obs):
    """CONDITIONAL aggregation: filter to overnight slots, THEN take the mode."""
    nt = s[s.tod.isin(NIGHT)]
    if len(nt) < 5 or nt["rank"].nunique() < 2:
        return None
    vc = nt["rank"].value_counts()
    if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
        return None
    # Reject items where the overnight filter does not change the answer. Otherwise a model that
    # ignores the filter and reports the overall most frequent place would score well.
    if int(s["rank"].mode().iloc[0]) == int(vc.index[0]):
        return None
    return dict(tier="D", task="home",
                q="Where does this person most likely live? Answer with the place number they are "
                  "at most often during overnight hours (timeslots 0-11).",
                a=str(int(vc.index[0])), atype="numeric", mag=None)


def D_work(s, rng, span_obs):
    """Same shape as home but a different filter AND excluding home -- two conditions at once."""
    nt = s[s.tod.isin(NIGHT)]
    if len(nt) < 3:
        return None
    home = int(nt["rank"].mode().iloc[0])
    dt = s[s.tod.isin(WORKHRS) & (s["rank"] != home)]
    if len(dt) < 5 or dt["rank"].nunique() < 2:
        return None
    vc = dt["rank"].value_counts()
    if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
        return None
    if int(s["rank"].mode().iloc[0]) == int(vc.index[0]):
        return None
    return dict(tier="D", task="work",
                q="Excluding the place where they sleep, where does this person most likely work? "
                  "Answer with the place number they are at most often during daytime hours "
                  "(timeslots 18-35).",
                a=str(int(vc.index[0])), atype="numeric", mag=None)


# ----------------------------------------------------------------- E. structure / regularity
def E_regular_day(s, rng, span_obs):
    """Which day-of-record is MOST routine: highest share at that day's own modal place."""
    days = sorted(s.day.unique())
    if len(days) < 3:
        return None
    sc = {}
    for d in days:
        sd = s[s.day == d]
        if len(sd) < 3:
            continue
        sc[int(d)] = float(sd["rank"].value_counts().iloc[0] / len(sd))
    if len(sc) < 3:
        return None
    order = sorted(sc.items(), key=lambda kv: -kv[1])
    if order[0][1] == order[1][1]:
        return None
    return dict(tier="E", task="regular_day",
                q="On which day in this record did the person spend the LARGEST share of their "
                  "observations at a single place? Answer with the day number.",
                a=str(order[0][0]), atype="numeric", mag=None)


def E_odd_day(s, rng, span_obs):
    """Which day is most UNLIKE the person's overall pattern (lowest share at their global top)."""
    days = sorted(s.day.unique())
    if len(days) < 3:
        return None
    top = int(s["rank"].mode().iloc[0])
    sc = {}
    for d in days:
        sd = s[s.day == d]
        if len(sd) < 3:
            continue
        sc[int(d)] = float((sd["rank"] == top).mean())
    if len(sc) < 3:
        return None
    order = sorted(sc.items(), key=lambda kv: kv[1])
    if order[0][1] == order[1][1]:
        return None
    return dict(tier="E", task="odd_day",
                q=f"Place {top} is where this person is most often overall. On which day in this "
                  f"record did they spend the SMALLEST share of their time there? Answer with the "
                  f"day number.",
                a=str(order[0][0]), atype="numeric", mag=None)


# ----------------------------------------------------------------- E. structure (cont.)
def E_busiest_day(s, rng, span_obs):
    """Day with the most distinct places: a strict argmax over a per-day count.

    Structurally the same as regular_day and odd_day (an argmax over a per-day quantity), so it
    belongs to the structure tier. Ties for the maximum are rejected.
    """
    days = sorted(s.day.unique())
    if len(days) < 4:
        return None
    cnt = {int(d): int(s[s.day == d]["rank"].nunique()) for d in days if len(s[s.day == d]) >= 3}
    if len(cnt) < 4:
        return None
    order = sorted(cnt.items(), key=lambda kv: -kv[1])
    if order[0][1] == order[1][1]:                 # strict argmax only
        return None
    return dict(tier="E", task="busiest_day",
                q="On which day in this record did the person visit the most distinct places? "
                  "Answer with the day number.",
                a=str(order[0][0]), atype="numeric", mag=order[0][1])


TASKS = [A_retrieve, B_distinct, B_occurrence, B_difference, B_intersection,
         C_second_place, C_busier_half, D_home, D_work, E_regular_day, E_odd_day, E_busiest_day]
# minimum observations for a task to be well-posed at all
MIN_OBS = {"retrieve": 8, "distinct": 8, "occurrence": 8, "difference": 12, "intersection": 12,
           "second_place": 12, "busier_half": 12, "home": 24, "work": 24,
           "regular_day": 24, "odd_day": 24, "busiest_day": 32}


def relabel(s, rng, pool=512):
    """Replace frequency-ordered ranks with arbitrary ids, drawn per item.

    Ranks are assigned by global visit frequency, so the label itself leaks the answer to frequency
    questions (the most visited place is rank 0). Random ids remove that. Distinct-count,
    occurrence-count, retrieval, difference and intersection answers do not depend on the labels.
    Gold answers are computed after relabelling, never before.
    """
    places = s["rank"].unique()
    new = rng.choice(pool, size=len(places), replace=False)
    m = {int(o): int(n) for o, n in zip(places, new)}
    out = s.copy()
    out["rank"] = out["rank"].map(m)
    return out


def render(s):
    return "\n".join(f"d{int(r.day)} t{int(r.tod)} p{int(r['rank'])}" for _, r in s.iterrows())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yjmob/yjmob_ranks.parquet")
    ap.add_argument("--spans", default="8,16,32,64,128,256")
    ap.add_argument("--per-cell", type=int, default=40)
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--split-day", type=int, default=SPLIT_DAY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/place_questions.jsonl")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = pd.read_parquet(a.data)
    d["day"] = d.slot // SLOTS_PER_DAY
    d["tod"] = d.slot % SLOTS_PER_DAY
    d = d[d.day < a.split_day]                                   # LEAKAGE GATE
    uids = d.uid.unique()[:a.users]
    by_uid = {u: g.sort_values("slot") for u, g in d[d.uid.isin(set(uids))].groupby("uid", sort=False)}
    SPANS = [int(x) for x in a.spans.split(",")]
    print(f"{len(d):,} train-period rows | {len(uids):,} users | spans {SPANS}\n")

    items = []
    for span, fn in itertools.product(SPANS, TASKS):
        made, tries = 0, 0
        probe = fn.__name__
        while made < a.per_cell and tries < a.per_cell * 400:
            tries += 1
            u = int(rng.choice(uids))
            g = by_uid.get(u)
            if g is None or len(g) < span + 4:
                continue
            st = int(rng.integers(0, len(g) - span))
            s = g.iloc[st:st + span]                             # SPAN = observation count, exactly
            s = relabel(s, rng)                                  # kill the frequency-label leak
            it = fn(s, rng, span)
            if not it:
                continue
            if span < MIN_OBS[it["task"]]:
                break                                            # task not well-posed at this span
            rec = render(s)
            # `family` is the key every downstream script reads; emit it alongside tier and task.
            it["family"] = it["task"]
            it.update(uid=u, span=span, n_rows=rec.count("\n") + 1, n_chars=len(rec),
                      prompt_full=f"Here is a person's location record. Each line is "
                                  f"day, timeslot, place.\n\n{rec}\n\n"
                                  f"Question: {it['q']}\nAnswer with the value only.",
                      prompt_blind=f"Question about a person's location record: {it['q']}\n"
                                   f"Answer with the value only.")
            # ---- screens, asserted per item
            assert int(s.day.max()) < a.split_day, f"LEAK day {int(s.day.max())}"
            assert it["n_rows"] == span, f"span dial broken: {it['n_rows']} rows for span {span}"
            for dd in {int(x) for x in re.findall(r"day (\d+)", it["q"])}:
                assert dd in set(int(x) for x in s.day.unique()), f"unanswerable day {dd}"
            items.append(it); made += 1
        if made:
            print(f"  span {span:>4}  {it['task']:<14} {made:>4}")

    print(f"\n{len(items):,} items\n")
    print(f"  {'tier':<5}{'task':<15}{'n':>6}{'spans':>28}{'floor':>8}   verdict")
    screen = {}
    for t in sorted({i["task"] for i in items}, key=lambda x: [i for i in items if i["task"] == x][0]["tier"] + x):
        sub = [i for i in items if i["task"] == t]
        c = collections.Counter(i["a"] for i in sub)
        fl = c.most_common(1)[0][1] / len(sub)
        sp = sorted({i["span"] for i in sub})
        v = "EXPLOITABLE" if fl >= .5 else "weak" if fl >= .35 else "OK"
        screen[t] = dict(n=len(sub), floor=fl, tier=sub[0]["tier"], spans=sp, verdict=v)
        print(f"  {sub[0]['tier']:<5}{t:<15}{len(sub):>6}{str(sp):>28}{fl:>8.3f}   {v}")

    print(f"\n  {'span':>6}{'items':>8}{'rows med':>10}{'chars med':>11}")
    for sp in SPANS:
        sub = [i for i in items if i["span"] == sp]
        if sub:
            print(f"  {sp:>6}{len(sub):>8}{np.median([i['n_rows'] for i in sub]):>10.0f}"
                  f"{np.median([i['n_chars'] for i in sub]):>11.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        for i in items:
            fh.write(json.dumps(i) + "\n")
    json.dump(screen, open(a.out.replace(".jsonl", "_screen.json"), "w"), indent=1)
    print(f"\nwrote {a.out} + _screen.json")


if __name__ == "__main__":
    main()
