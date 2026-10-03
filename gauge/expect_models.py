#!/usr/bin/env python3
"""Guard: a scorer states which models it expects and fails on a partial set.

A scorer that globs result files would otherwise score whatever happens to be on disk and print a
confident, well-formatted result for a partial set. `check` prints how many of the expected models
were found. It exits with the list of missing ones unless partial scoring is allowed explicitly
(the --allow-partial flag or ALLOW_PARTIAL=1), in which case the shortfall is printed beside the
result.

Used by score_compare.py, score_control_ladder.py, score_control_tolerance.py, score_sampling.py
and score_templates.py.
"""
import os, sys

FIVE = ["phi35mini", "mistral7b", "llama8b", "gemma3_12b", "llama70b"]


def check(found, expected=None, label="", strict=True):
    """`found` = iterable of model tags discovered. Returns the sorted list, or exits."""
    expected = expected or FIVE
    found = sorted(set(found))
    missing = [m for m in expected if m not in found]
    extra = [m for m in found if m not in expected]
    print(f"  MODEL SET [{label}]: {len(found)}/{len(expected)} present")
    if extra:
        print(f"    unexpected: {extra}")
    if missing:
        msg = (f"    *** MISSING {len(missing)}: {missing}\n"
               f"    A partial set produces a confident but unrepresentative result.\n"
               f"    Copy the missing runs into place, or pass --allow-partial to score anyway and have the\n"
               f"    shortfall printed beside every number.")
        if strict and os.environ.get("ALLOW_PARTIAL") != "1" and "--allow-partial" not in sys.argv:
            sys.exit(msg)
        print(msg)
    return found, missing
