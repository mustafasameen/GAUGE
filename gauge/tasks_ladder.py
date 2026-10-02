#!/usr/bin/env python3
"""Scaffold ladder: hand the model progressively more of the radius-of-gyration computation.

The radius of gyration is r_g = sqrt(mean_i ||p_i - centroid||^2), in km, from a grid in which one
unit is 500 m. A model has to perform these steps in order, and each rung of the ladder removes one
of them:

  L0  raw              parse coordinates, find the centroid, square the deviations, average, take
                       the square root, convert units (the task as the benchmark asks it)
  L1  km units         coordinates are already in km (removes the unit conversion)
  L2  centred          coordinates are already centred on the person's own centroid (removes the
                       centroid step; r_g is now sqrt(mean(x^2 + y^2)))
  L3  distances given  the distance of each point from the centre is printed (removes all geometry;
                       only a mean of squares and a square root remain)
  L4  mean given       the mean squared distance is printed (only the square root remains)

Every rung holds the same quantity, the same people, the same answer format and the same scorer;
only the amount of pre-computation changes. The rung where accuracy jumps is where the failure
sits. The gold distribution is identical across rungs by construction, so the majority baseline is
identical too and a jump cannot come from an easier answer space; tally_ladder.py asserts this.

Usage: imported by tally_ladder.py. Run `python gauge/tasks_ladder.py` for the self-tests.
"""
from __future__ import annotations

import numpy as np

KM = 0.5


def _rg_km(P):
    return float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())) * KM


def _fmt(v):
    return f"{v:.1f}"


def rungs(P, days, tods):
    """Return {rung_name: (body, question)} for one record. Gold is identical across rungs."""
    C = P.mean(0)
    dev = P - C
    dist_km = np.sqrt((dev ** 2).sum(1)) * KM
    msd = float((dist_km ** 2).mean())
    ASK = ("what is the radius of gyration of this person's locations, in kilometres? "
           "The radius of gyration is the square root of the mean squared distance of their "
           "locations from their own centre. Answer with a whole number of kilometres.")

    out = {}
    # L0: exactly the production rendering and question
    out["L0_raw"] = (
        "\n".join(f"d{d} t{t} x{int(round(x))} y{int(round(y))}"
                  for d, t, (x, y) in zip(days, tods, P)),
        "Treating each unit on the x and y axes as 500 metres, " + ASK)
    # L1: coordinates already in kilometres
    out["L1_km"] = (
        "\n".join(f"d{d} t{t} x{_fmt(x * KM)} y{_fmt(y * KM)}"
                  for d, t, (x, y) in zip(days, tods, P)),
        "The x and y values below are already in kilometres, so " + ASK)
    # L2: centred on the person's own centroid
    out["L2_centred"] = (
        "\n".join(f"d{d} t{t} x{_fmt(x * KM)} y{_fmt(y * KM)}"
                  for d, t, (x, y) in zip(days, tods, dev)),
        "The x and y values below are in kilometres and are already measured FROM this person's "
        "own centre, so their centre is at x0.0 y0.0. Given that, " + ASK)
    # L3: every point's distance from the centre is printed; no geometry left
    out["L3_dists"] = (
        "\n".join(f"d{d} t{t} dist{_fmt(r)}"
                  for d, t, r in zip(days, tods, dist_km)),
        "Each line below gives how far this person was from their own centre, in kilometres. "
        "Given those distances, " + ASK)
    # L4: the mean squared distance is printed; only the square root remains
    out["L4_msd"] = (
        f"This person's mean squared distance from their own centre is {msd:.2f} "
        f"square kilometres.",
        "Given that mean squared distance, " + ASK)
    return out


def gold(P):
    return str(int(round(_rg_km(P))))


RUNG_ORDER = ["L0_raw", "L1_km", "L2_centred", "L3_dists", "L4_msd"]


# ------------------------------------------------------------------ self-tests
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    import re
    ok = 0
    for _ in range(300):
        n = int(rng.integers(20, 200))
        P = rng.integers(-500, 500, size=(n, 2)).astype(float)
        days = rng.integers(0, 50, n)
        tods = rng.integers(0, 48, n)
        g = float(gold(P))
        r = rungs(P, days, tods)
        assert set(r) == set(RUNG_ORDER)
        # L4's printed mean-squared-distance must reproduce the gold via sqrt
        msd = float(re.search(r"centre is ([\d.]+) ", r["L4_msd"][0]).group(1))
        assert abs(round(np.sqrt(msd)) - g) <= 1, (np.sqrt(msd), g)
        # L3's printed distances must reproduce the gold via mean-of-squares then sqrt
        ds = [float(x) for x in re.findall(r"dist([\d.]+)", r["L3_dists"][0])]
        assert len(ds) == n
        assert abs(round(np.sqrt(np.mean(np.square(ds)))) - g) <= 1, (ds[:3], g)
        # L2's centred coords must reproduce the gold directly
        xs = [float(x) for x in re.findall(r"x(-?[\d.]+)", r["L2_centred"][0])]
        ys = [float(y) for y in re.findall(r"y(-?[\d.]+)", r["L2_centred"][0])]
        assert abs(round(np.sqrt(np.mean(np.square(xs) + np.square(ys)))) - g) <= 1
        ok += 1
    print(f"{ok} records: every rung independently reproduces the SAME gold from its own "
          f"printed content")
    print("ladder self-tests pass")
