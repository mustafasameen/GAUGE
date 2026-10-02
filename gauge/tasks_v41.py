#!/usr/bin/env python3
"""Question families for the v4.1 benchmark: four geometric families and the place families.

The coordinate families (radius of gyration, maximum displacement, total distance, longest jump) and
the place families (recency, longest stay, transition, core set, time mode, distinct places on a day,
windowed distinct count, retrieval at a position, the retrieval probe and a two-hop composition)
follow scikit-mobility's individual measures (Pappalardo et al.), the field's reference list of what
is computed about one person. Each family records the measure it covers (`skmob`, or NEW for a
construction made here) and a `load` label: 0 locate, 1 extremum or discriminate, 2 tally one
target, 3 dedupe the whole record, 4 combine two sets, 5 exact arithmetic over all coordinates.

Continuous quantities are asked as integers in natural units with wide support, never as bands. A
small categorical answer space lets a model reach the baseline by repeating one letter while reading
nothing. A +/-1 tolerance is a scoring choice applied afterwards, not a parser change.

Gold answers are computed from the transformed, rounded coordinates that the model is shown, so the
prompt and the answer agree by construction. The self-test at the bottom recomputes every gold by an
independent brute-force route on 300 random records.

Usage: imported by tally_v41.py. Run `python gauge/tasks_v41.py` for the self-test.
"""
from __future__ import annotations

import collections

import numpy as np

KM = 0.5   # one grid unit = 500 m, matching v4's tier F


# ---------------------------------------------------------------- coordinate families (load 1,5)
def F_rg_numeric(s, rng, span, P):
    """Radius of gyration in whole kilometres. skmob: radius_of_gyration. load 5.
    The earlier binned form (five bands) let a model reach the baseline with one constant letter, so
    this family asks for an integer in natural units instead.
    """
    rg = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()) * KM)
    if rg < 1.0:
        return None                     # sub-km spread rounds to 0/1 and collapses the answer space
    return dict(tier="F", task="gyration_km", atype="numeric", skmob="radius_of_gyration", load=5,
                q=("Treating each unit on the x and y axes as 500 metres, what is the radius of "
                   "gyration of this person's locations, in kilometres? The radius of gyration is "
                   "the root-mean-square distance of the visited points from their mean position. "
                   "Answer with a whole number of kilometres."),
                a=str(int(round(rg))), mag=rg)


def F_maxdist(s, rng, span, P):
    """skmob: maximum_distance -- greatest distance between any two visited points. load 5."""
    U = np.unique(P, axis=0)
    if len(U) < 3:
        return None
    d = np.sqrt(((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)).max() * KM
    if d < 1.0:
        return None
    return dict(tier="F", task="max_distance", atype="numeric", skmob="maximum_distance", load=5,
                q=("Treating each unit on the x and y axes as 500 metres, what is the greatest "
                   "distance between any two locations this person visited, in kilometres? "
                   "Answer with a whole number of kilometres."),
                a=str(int(round(d))), mag=float(d))


def F_totaldist(s, rng, span, P):
    """skmob: distance_straight_line -- total distance travelled over the record. load 5."""
    d = float(np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1)).sum() * KM)
    if d < 1.0:
        return None
    return dict(tier="F", task="total_distance", atype="numeric",
                skmob="distance_straight_line", load=5,
                q=("Treating each unit on the x and y axes as 500 metres, what is the TOTAL "
                   "distance this person travelled, adding up the straight-line distance between "
                   "each consecutive pair of records? Answer with a whole number of kilometres."),
                a=str(int(round(d))), mag=d)


def F_longest_jump(s, rng, span, P):
    """skmob: jump_lengths, reduced to its extremum. load 1: find the largest single step.
    Paired with F_totaldist, which sums the same per-step distances.
    """
    if len(P) < 3:
        return None
    j = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1)) * KM
    top = np.sort(j)[::-1]
    if top[0] < 1.0 or (len(top) > 1 and abs(top[0] - top[1]) < 0.5):
        return None                     # need a strict, roundable winner
    return dict(tier="F", task="longest_jump", atype="numeric", skmob="jump_lengths", load=1,
                q=("Treating each unit on the x and y axes as 500 metres, what is the LARGEST "
                   "straight-line distance this person moved between two consecutive records, in "
                   "kilometres? Answer with a whole number of kilometres."),
                a=str(int(round(top[0]))), mag=float(top[0]))


# ---------------------------------------------------------------- place families (load 0,1,3)
def H_recency(s, rng, span, R, days, tods):
    """skmob: recency_rank, reduced to rank 1. load 0-1 -- locate the last record before a cutoff."""
    if len(R) < 6:
        return None
    k = int(rng.integers(3, len(R)))            # cutoff strictly inside the record
    d0, t0 = int(days[k]), int(tods[k])
    # Gold comes from the question's meaning ("last record strictly before this instant"), never
    # from the array index k-1. The positional version is right only if the record is perfectly ordered
    # by (day, timeslot), which is an assumption about the data and not a property of the question.
    prior = [i for i in range(len(R)) if (int(days[i]), int(tods[i])) < (d0, t0)]
    if not prior:
        return None
    last = max(prior, key=lambda i: (int(days[i]), int(tods[i])))
    # The instant must be unambiguous: a tie at the maximum has no single "last place".
    ties = [i for i in prior if (int(days[i]), int(tods[i]))
            == (int(days[last]), int(tods[last]))]
    if len({int(R[i]) for i in ties}) != 1:
        return None
    return dict(tier="H", task="recency", atype="numeric", skmob="recency_rank", load=0,
                q=(f"What is the last place this person was at STRICTLY BEFORE day {d0} "
                   f"timeslot {t0}? Answer with the place number."),
                a=str(int(R[last])), mag=None)


def H_longest_stay(s, rng, span, R, days, tods):
    """skmob: waiting_times, reduced to its extremum. load 1."""
    runs, cur = [], 1
    for i in range(1, len(R)):
        cur = cur + 1 if R[i] == R[i - 1] else 1
        runs.append(cur)
    if not runs:
        return None
    best = max(runs)
    if best < 2 or sorted(runs)[-1] == sorted(runs)[-2:][0] and runs.count(best) > 3:
        return None                     # ambiguous / degenerate ties
    return dict(tier="H", task="longest_stay", atype="numeric", skmob="waiting_times", load=1,
                q=("What is the largest number of CONSECUTIVE records in which this person stayed "
                   "at the same place? Answer with a whole number."),
                a=str(int(best)), mag=int(best))


def H_transition(s, rng, span, R, days, tods):
    """skmob: individual_mobility_network, reduced to the modal out-edge. load 3 -- tally the
    successors of one place over the whole record."""
    succ = collections.defaultdict(collections.Counter)
    for i in range(len(R) - 1):
        if R[i] != R[i + 1]:
            succ[R[i]][R[i + 1]] += 1
    # Same anti-shortcut as `time_mode`: people return home, so the modal successor of a randomly
    # chosen place equalled the global mode for 209/500 users (42%). The SOURCE PLACE is chosen so
    # the answer is not simply the person's most-frequent place.
    glob = collections.Counter(R.tolist()).most_common(1)[0][0]
    cands = [p for p, c in succ.items() if sum(c.values()) >= 3 and
             len(c) >= 2 and c.most_common(2)[0][1] > c.most_common(2)[1][1]
             and c.most_common(1)[0][0] != glob]
    if not cands:
        return None
    p = int(rng.choice(cands))
    return dict(tier="H", task="transition", atype="numeric",
                skmob="individual_mobility_network", load=3,
                q=(f"When this person leaves place {p} and moves somewhere different, which place "
                   f"do they go to most often? Answer with the place number."),
                a=str(int(succ[p].most_common(1)[0][0])), mag=None)


def H_coreset(s, rng, span, R, days, tods):
    """Wide-answer proxy for skmob: uncorrelated_entropy. load 3-4.
    Entropy is continuous and would need bands. This asks for an integer with wide support that
    captures the same concentration construct: the size of the smallest set of places that covers
    half of the visits.
    """
    vc = collections.Counter(R.tolist()).most_common()
    tot, run = len(R), 0
    for i, (_, c) in enumerate(vc, 1):
        run += c
        if run * 2 >= tot:
            if i < 2 or i > len(vc) - 1:
                return None             # degenerate at both extremes
            return dict(tier="H", task="core_set", atype="numeric",
                        skmob="uncorrelated_entropy(proxy)", load=4,
                        q=("What is the SMALLEST number of distinct places that together account "
                           "for at least half of all the records in this history? Answer with a "
                           "whole number."),
                        a=str(i), mag=i)
    return None


def H_time_mode(s, rng, span, R, days, tods):
    """skmob: NEW (generalises home_location to an arbitrary window). load 1."""
    # Anti-shortcut. With a window chosen at random, the windowed mode equals the person's overall
    # modal place for roughly 59% of people, so a model that answers "wherever they are most often
    # overall" would score well without restricting to the window. The window is therefore chosen so
    # that the answer differs from the overall mode. This stratifies over the answer by choosing the
    # query, never by choosing the people.
    if len(R) < 12:
        return None
    glob = collections.Counter(R.tolist()).most_common(1)[0][0]
    los = list(range(0, 36))
    rng.shuffle(los)
    for lo in los:
        hi = lo + 11
        m = (tods >= lo) & (tods <= hi)
        if m.sum() < 6:
            continue
        vc = collections.Counter(R[m].tolist()).most_common(2)
        if len(vc) < 2 or vc[0][1] == vc[1][1] or vc[0][0] == glob:
            continue
        return dict(tier="H", task="time_mode", atype="numeric", skmob="NEW", load=1,
                    q=(f"Considering only records in timeslots {lo} to {hi} inclusive, which place "
                       f"is this person at most often? Answer with the place number."),
                    a=str(int(vc[0][0])), mag=None)
    return None


def H_day_distinct(s, rng, span, R, days, tods):
    """Distinct places on one day. The operation is the same as counting distinct places in the
    whole record, over one day's worth of state.
    """
    cnt = collections.Counter(days.tolist())
    cands = [d for d, c in cnt.items() if c >= 4]
    if not cands:
        return None
    d = int(rng.choice(cands))
    return dict(tier="H", task="day_distinct", atype="numeric",
                skmob="number_of_locations(windowed)", load=3,
                q=(f"How many DISTINCT places did this person visit on day {d}? Answer with a "
                   f"whole number."),
                a=str(int(len(set(R[days == d].tolist())))), mag=None)


def H_window_distinct(s, rng, span, R, days, tods, W):
    """State-volume dial: distinct places among the last W records while the full record stays at
    span S.

    W is the amount of state the answer needs and S is the input length, and the two are crossed:

        accuracy(W, S)   W in {16, 32, 64, 128}   x   S in {32, 64, 128, 256, 512}

    Varying one of them alone cannot separate "the input is long" from "the answer needs a lot of
    state", because the two usually move together. `distinct` over the whole record is the diagonal
    W = S. The window must be a strict subset of the record, so a cell exists only where W < S.
    """
    if W >= len(R) or W < 3:
        return None
    tail = R[-W:]
    return dict(tier="H", task=f"window_distinct_w{W}", atype="numeric",
                skmob="number_of_locations(windowed)", load=3, window=W,
                q=(f"Considering ONLY the last {W} records in this history, how many DISTINCT "
                   f"places did this person visit? Answer with a whole number."),
                a=str(int(len(set(tail.tolist())))), mag=int(len(set(tail.tolist()))))


def H_retrieve_at(s, rng, span, R, days, tods, pos):
    """Retrieval at a controlled relative position.

    The queried record sits at relative position `pos` in the record, so the effect of position is
    separated from the effect of record length:

        accuracy(p, S)   p in {0.1, 0.3, 0.5, 0.7, 0.9}   x   S in {32, ..., 512}

    The queried (day, timeslot) must occur exactly once in the record.
    """
    if len(R) < 10:
        return None
    k = int(round(pos * (len(R) - 1)))
    d, t = int(days[k]), int(tods[k])
    if sum(1 for i in range(len(R)) if int(days[i]) == d and int(tods[i]) == t) != 1:
        return None                     # the queried instant must be unique
    return dict(tier="H", task=f"retrieve_p{int(pos*100)}", atype="numeric", skmob="NEW",
                load=0, position=pos,
                q=(f"What place was this person at on day {d}, timeslot {t}? Answer with the "
                   f"place number."),
                a=str(int(R[k])), mag=None)


def H_absent(s, rng, span, R, days, tods):
    """Unanswerable control. skmob: NEW (a control, not a mobility measure). load 0.

    The queried instant is plausible but absent: the day is one the record covers, and the timeslot is
    one with no observation on that day. An implausible query (day 900) could be rejected without
    reading the record, which would measure query screening and not grounding. Gold is "none".

    Read together with the answerable half (H_present), never alone. A model that answers "none" to
    everything scores 100% here and 0% on the answerable half; a model that never abstains scores the
    reverse. The reported quantity is the rate at which unanswerable items are given a place instead of
    an abstention.
    """
    present = set(zip(days.tolist(), tods.tolist()))
    covered = sorted({int(d) for d in days})
    if len(covered) < 2:
        return None
    # Draw the absent instant's timeslot from the person's own timeslot distribution. Sampling
    # slots uniformly would put far more queries in night slots than real records occupy (people are
    # observed less at night), and a model could then answer "none" to a night query without reading
    # the record.
    tod_pool = tods.tolist()
    for _ in range(60):
        t = int(tod_pool[int(rng.integers(0, len(tod_pool)))])
        d = int(covered[int(rng.integers(0, len(covered)))])
        if (d, t) not in present:
            return dict(tier="H", task="retrieve_probe", atype="numeric_or_none", skmob="NEW",
                        load=0, answerable=False,
                        q=(f"What place was this person at on day {d}, timeslot {t}? If the "
                           f"record does not say, answer none."),
                        a="none", mag=None)
    return None


def H_present(s, rng, span, R, days, tods):
    """The answerable half of the retrieval probe: same family name, same question form, gold is a place.

    The two halves must share one family. Alone, the absent items have a single gold ("none"), so a
    constant answer would score perfectly. Mixed roughly 50/50, "none" is the majority class at a
    baseline near .5, a constant "none" earns a normalized gain of 0, and only a model that tells
    present instants from absent ones scores above the baseline.
    """
    if len(R) < 10:
        return None
    k = int(rng.integers(0, len(R)))
    d, t = int(days[k]), int(tods[k])
    if sum(1 for i in range(len(R)) if int(days[i]) == d and int(tods[i]) == t) != 1:
        return None
    return dict(tier="H", task="retrieve_probe", atype="numeric_or_none", skmob="NEW", load=0,
                answerable=True,
                q=(f"What place was this person at on day {d}, timeslot {t}? If the record does "
                   f"not say, answer none."),
                a=str(int(R[k])), mag=None)


def H_two_hop(s, rng, span, R, days, tods):
    """Explicit two-hop composition. skmob: NEW. load 1+0 composed.

    Hop 1 is the day with the most distinct places (load 1). Hop 2 is a retrieval on that day (load 0).
    Both components are individually clean at every span, so if this family fails earlier than either
    one, the cost is the composition itself and not a weak component. This poses, over a mobility
    record, the question raised by Dziri et al. (2023, arXiv 2305.18654) about multi-step composition.
    """
    per = collections.defaultdict(set)
    for i in range(len(R)):
        per[int(days[i])].add(int(R[i]))
    if len(per) < 3:
        return None
    ranked = sorted(per.items(), key=lambda kv: -len(kv[1]))
    if len(ranked[0][1]) == len(ranked[1][1]):
        return None                     # hop 1 must have a unique answer
    d = ranked[0][0]
    onday = [i for i in range(len(R)) if int(days[i]) == d]
    slots = collections.Counter(int(tods[i]) for i in onday)
    uniq = [t for t, c in slots.items() if c == 1]
    if not uniq:
        return None                     # hop 2 must be unambiguous on that day
    t = int(rng.choice(uniq))
    ans = [int(R[i]) for i in onday if int(tods[i]) == t][0]
    return dict(tier="H", task="two_hop", atype="numeric", skmob="NEW", load=1,
                q=("First find the day on which this person visited the most DISTINCT places. "
                   f"Then answer: what place were they at on that day at timeslot {t}? "
                   "Answer with the place number."),
                a=str(ans), mag=None)


COORD_TASKS = [F_rg_numeric, F_maxdist, F_totaldist, F_longest_jump]
PLACE_TASKS = [H_recency, H_longest_stay, H_transition, H_coreset, H_time_mode, H_day_distinct,
               H_absent, H_present, H_two_hop]
MIN_OBS_41 = {"gyration_km": 32, "max_distance": 32, "total_distance": 32, "longest_jump": 32,
              "recency": 8, "longest_stay": 8, "transition": 16, "core_set": 16,
              "time_mode": 16, "day_distinct": 8, "retrieve_probe": 8, "two_hop": 16}


# ------------------------------------------------------------------------------- SELF-TEST
def _selftest():
    """Recompute every gold by an independent brute-force route and assert equality.
    A generator whose gold is produced only by the code under test is unverified.
    """
    rng = np.random.default_rng(0)
    for trial in range(300):
        n = int(rng.integers(20, 120))
        R = rng.integers(0, 12, n)
        days = np.sort(rng.integers(0, 20, n))
        tods = rng.integers(0, 48, n)
        P = rng.integers(-60, 60, (n, 2)).astype(float)

        # ---- coordinate families
        r = F_rg_numeric(None, rng, n, P)
        if r:
            mu = [sum(p[0] for p in P) / n, sum(p[1] for p in P) / n]
            bf = (sum((p[0] - mu[0]) ** 2 + (p[1] - mu[1]) ** 2 for p in P) / n) ** .5 * KM
            assert r["a"] == str(int(round(bf))), f"rg {r['a']} vs {bf}"
        r = F_maxdist(None, rng, n, P)
        if r:
            U = {tuple(p) for p in P}
            bf = max(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5 for a in U for b in U) * KM
            assert r["a"] == str(int(round(bf))), f"maxdist {r['a']} vs {bf}"
        r = F_totaldist(None, rng, n, P)
        if r:
            bf = sum(((P[i][0] - P[i - 1][0]) ** 2 + (P[i][1] - P[i - 1][1]) ** 2) ** .5
                     for i in range(1, n)) * KM
            assert r["a"] == str(int(round(bf))), f"totaldist {r['a']} vs {bf}"
        r = F_longest_jump(None, rng, n, P)
        if r:
            bf = max(((P[i][0] - P[i - 1][0]) ** 2 + (P[i][1] - P[i - 1][1]) ** 2) ** .5
                     for i in range(1, n)) * KM
            assert r["a"] == str(int(round(bf))), f"jump {r['a']} vs {bf}"

        # ---- place families
        r = H_recency(None, rng, n, R, days, tods)
        if r:
            d0 = int(r["q"].split("day ")[1].split(" ")[0])
            t0 = int(r["q"].split("timeslot ")[1].split("?")[0])
            prior = [i for i in range(n) if (int(days[i]), int(tods[i])) < (d0, t0)]
            bf = max(prior, key=lambda i: (int(days[i]), int(tods[i])))
            assert r["a"] == str(int(R[bf])), "recency"
        r = H_longest_stay(None, rng, n, R, days, tods)
        if r:
            best = mx = 1
            for i in range(1, n):
                mx = mx + 1 if R[i] == R[i - 1] else 1
                best = max(best, mx)
            assert r["a"] == str(best), f"stay {r['a']} vs {best}"
        r = H_transition(None, rng, n, R, days, tods)
        if r:
            p = int(r["q"].split("place ")[1].split(" ")[0])
            c = collections.Counter(R[i + 1] for i in range(n - 1) if R[i] == p and R[i + 1] != p)
            assert r["a"] == str(int(c.most_common(1)[0][0])), "transition"
        r = H_coreset(None, rng, n, R, days, tods)
        if r:
            vc = sorted(collections.Counter(R.tolist()).values(), reverse=True)
            run = k = 0
            for c in vc:
                run += c; k += 1
                if run * 2 >= n:
                    break
            assert r["a"] == str(k), f"coreset {r['a']} vs {k}"
        r = H_time_mode(None, rng, n, R, days, tods)
        if r:
            lo = int(r["q"].split("timeslots ")[1].split(" ")[0])
            hi = int(r["q"].split("to ")[1].split(" ")[0])
            c = collections.Counter(R[i] for i in range(n) if lo <= tods[i] <= hi)
            assert r["a"] == str(int(c.most_common(1)[0][0])), "time_mode"
        r = H_day_distinct(None, rng, n, R, days, tods)
        if r:
            d = int(r["q"].split("day ")[1].split("?")[0])
            assert r["a"] == str(len({R[i] for i in range(n) if days[i] == d})), "day_distinct"
        for W in (8, 16, 32, 64):
            r = H_window_distinct(None, rng, n, R, days, tods, W)
            if r:
                bf = len({R[i] for i in range(n - W, n)})
                assert r["a"] == str(bf), f"window_distinct W={W}"
                assert r["window"] == W and f"last {W} records" in r["q"], "window label mismatch"
        for pos in (0.1, 0.5, 0.9):
            r = H_retrieve_at(None, rng, n, R, days, tods, pos)
            if r:
                d = int(r["q"].split("day ")[1].split(",")[0])
                t = int(r["q"].split("timeslot ")[1].split("?")[0])
                hits = [int(R[i]) for i in range(n) if int(days[i]) == d and int(tods[i]) == t]
                assert len(hits) == 1 and r["a"] == str(hits[0]), f"retrieve_at p={pos}"
        r = H_absent(None, rng, n, R, days, tods)
        if r:
            d = int(r["q"].split("day ")[1].split(",")[0])
            t = int(r["q"].split("timeslot ")[1].split("?")[0])
            # the queried instant must be ABSENT, its day COVERED, and its timeslot must be one
            # the person is actually observed at (the anti-leak stratification)
            assert not any(int(days[i]) == d and int(tods[i]) == t for i in range(n)), "absent-present"
            assert d in {int(x) for x in days}, "absent-day-not-covered"
            assert t in set(tods.tolist()), "absent-tod-not-stratified"
            assert r["a"] == "none" and r["answerable"] is False, "absent-gold"
        r = H_present(None, rng, n, R, days, tods)
        if r:
            d = int(r["q"].split("day ")[1].split(",")[0])
            t = int(r["q"].split("timeslot ")[1].split("?")[0])
            hits = [int(R[i]) for i in range(n) if int(days[i]) == d and int(tods[i]) == t]
            assert len(hits) == 1 and r["a"] == str(hits[0]) and r["answerable"] is True, "present"
            assert r["task"] == "retrieve_probe", "present must share the family name"
        r = H_two_hop(None, rng, n, R, days, tods)
        if r:
            t = int(r["q"].split("timeslot ")[1].split("?")[0])
            per = collections.defaultdict(set)
            for i in range(n):
                per[int(days[i])].add(int(R[i]))
            d = max(per, key=lambda k: len(per[k]))
            hits = [int(R[i]) for i in range(n) if int(days[i]) == d and int(tods[i]) == t]
            assert len(hits) == 1 and r["a"] == str(hits[0]), "two_hop"
    print("SELF-TEST PASSED: 300 random records, every gold matches an independent recomputation")


if __name__ == "__main__":
    _selftest()
