#!/usr/bin/env bash
# Test that gauge/generate_questions.py rebuilds the question set that was scored for the paper.
#
# usage: tools/test_generator_reproduces.sh PARQUET WORKDIR
#   PARQUET  the file written by gauge/export_yjmob.py (for example data/yjmob/yjmob_export.parquet)
#   WORKDIR  a new or empty directory for the outputs; nothing else is written
#
# Three checks, each printing PASS or FAIL:
#   1. generate_questions.py --only-coord writes the 6,000 coordinate items of the corrected-header set.
#   2. A full run writes 34,200 items.
#   3. After tools/reported_header.py restores the header "day, timeslot, place" on the coordinate
#      prompts, the full run matches the question set that was scored, byte for byte.
# The exit status is 0 only if all three pass.
#
# A full run of generate_questions.py ends with an UnboundLocalError about `hashlib`, raised after
# the items file has been written completely (a local import inside main() hides the module-level
# one on the full path). The generator is left exactly as it was used, so this script accepts that
# one error and nothing else.
#
# Published values. REPORTED is the md5 of the 34,200-question set behind the paper's main tables.
# CORRECTED_COORD is the md5 of the 6,000 coordinate items with the corrected column header.
set -u

REPORTED="${EXPECT_REPORTED_MD5:-127b67859e28c8728293c459be78c047}"
CORRECTED_COORD="${EXPECT_COORD_MD5:-51e0445d2f39ea7e562dffd6b59aa81d}"
PYTHON="${PYTHON:-python3}"

if [ "$#" -ne 2 ]; then
    echo "usage: $0 PARQUET WORKDIR" >&2
    exit 2
fi
PQ="$1"
W="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
[ -f "$PQ" ] || { echo "no such file: $PQ" >&2; exit 2; }
mkdir -p "$W" || exit 2

md5f() {
    "$PYTHON" -c 'import hashlib,sys; print(hashlib.md5(open(sys.argv[1],"rb").read()).hexdigest())' "$1"
}
nlines() { "$PYTHON" -c 'import sys; print(sum(1 for _ in open(sys.argv[1],"rb")))' "$1"; }

fail=0
cd "$REPO" || exit 2

# 1. coordinate items only
"$PYTHON" gauge/generate_questions.py --data "$PQ" --only-coord --out "$W/coord.jsonl" > "$W/coord.log" 2>&1
rc=$?
got="$(md5f "$W/coord.jsonl" 2>/dev/null)"
if [ "$rc" -eq 0 ] && [ "$got" = "$CORRECTED_COORD" ]; then
    echo "PASS 1  --only-coord: 6,000 items, md5 $got"
else
    echo "FAIL 1  --only-coord: exit $rc, md5 ${got:-none}, expected $CORRECTED_COORD"; fail=1
fi

# 2. full run
"$PYTHON" gauge/generate_questions.py --data "$PQ" --out "$W/full.jsonl" > "$W/full.log" 2>&1
rc=$?
n="$(nlines "$W/full.jsonl" 2>/dev/null)"
if [ "$rc" -ne 0 ] && ! grep -q "UnboundLocalError.*hashlib" "$W/full.log"; then
    echo "FAIL 2  full run: exit $rc without the known hashlib error; see $W/full.log"; fail=1
elif [ "${n:-0}" -ne 34200 ]; then
    echo "FAIL 2  full run: ${n:-0} lines, expected 34200"; fail=1
else
    note=""
    [ "$rc" -ne 0 ] && note=" (exit $rc: the known hashlib error after the file was written)"
    echo "PASS 2  full run: 34,200 items, md5 as generated $(md5f "$W/full.jsonl")$note"
fi

# 3. restore the header the main tables were scored with, then compare
if "$PYTHON" tools/reported_header.py "$W/full.jsonl" "$W/full_reported.jsonl" --expect 6000 > "$W/header.log" 2>&1; then
    got="$(md5f "$W/full_reported.jsonl")"
    if [ "$got" = "$REPORTED" ]; then
        echo "PASS 3  with the reported header: md5 $got equals the scored question set"
    else
        echo "FAIL 3  with the reported header: md5 $got, expected $REPORTED"; fail=1
    fi
else
    echo "FAIL 3  tools/reported_header.py failed; see $W/header.log"; fail=1
fi

if [ "$fail" -eq 0 ]; then echo "ALL PASS"; else echo "FAILED"; fi
exit "$fail"
