#!/bin/bash
# Generic launcher: one model per array task, one items file per submission.
#
#   ITEMS=results/tally_v41.jsonl PREFIX=v41core CONDS=full,blind sbatch slurm/run_model.sbatch
#
# Task 0 to 4 run Phi-3.5-mini, Mistral-7B, Llama-3.1-8B, Gemma-3-12B and Llama-3.1-70B. Use
# --array=3 for a single model. The settings of every run in the paper are in slurm/submit_all.sh.
#
# Environment variables (all read when the job starts):
#   ITEMS        items file (required)
#   PREFIX       output prefix, for example v41core; the output is results/<PREFIX>_<tag>.json
#   CONDS        conditions to run (default full,blind)
#   STYLE        prompt style: terse, cot, procedure or fewshot (default terse)
#   BS           batch size cap (default 64)
#   TOK_BUDGET   padded-token budget per batch (default 40000)
#   MAX_NEW      generation cap in tokens (default 24)
#   MAX_LEN      longest prompt in tokens (default 32768)
#   EXTRA_ARGS   further arguments for eval_factqa.py, for example "--n-samples 8 --temperature 0.7"
#   EXPECT_MD5   if set, the items file must have this md5
#   PREFLIGHT    1 (default) runs gauge/preflight_items.py first; use 0 for the program arm, whose
#                gold is a number while the model writes an expression
#   GAUGE_ENV    optional file to source first (module loads, conda or venv activation)
# The resource lines below are placeholders. Add your site's account and partition (and change the
# time limit and memory if needed) as extra sbatch options on the command line.
#SBATCH --job-name=gauge
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results
if [ -n "${GAUGE_ENV:-}" ]; then source "$GAUGE_ENV"; fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

MODELS=(microsoft/Phi-3.5-mini-instruct mistralai/Mistral-7B-Instruct-v0.3
        meta-llama/Llama-3.1-8B-Instruct google/gemma-3-12b-it meta-llama/Llama-3.1-70B-Instruct)
TAGS=(phi35mini mistral7b llama8b gemma3_12b llama70b)

i="${SLURM_ARRAY_TASK_ID:-${TASK:?set TASK when not running under slurm}}"
ITEMS="${ITEMS:?set ITEMS}"
PREFIX="${PREFIX:?set PREFIX}"
TAG="${TAGS[$i]}"
# The scorers look for the Gemma-3-12B main run under the name v41pilot_gemma3_12b.json.
if [ "$PREFIX" = "v41core" ] && [ "$TAG" = "gemma3_12b" ]; then NAME="v41pilot_${TAG}"; else NAME="${PREFIX}_${TAG}"; fi
OUT="results/${NAME}.json"

if [ -n "${EXPECT_MD5:-}" ]; then
    GOT="$(python -c 'import hashlib,sys; print(hashlib.md5(open(sys.argv[1],"rb").read()).hexdigest())' "$ITEMS")"
    [ "$GOT" = "$EXPECT_MD5" ] || { echo "items md5 $GOT, expected $EXPECT_MD5" >&2; exit 3; }
fi
echo "model=${MODELS[$i]} items=$ITEMS out=$OUT"
if [ "${PREFLIGHT:-1}" = "1" ]; then python gauge/preflight_items.py "$ITEMS"; fi

# shellcheck disable=SC2086
python gauge/eval_factqa.py --items "$ITEMS" --model "${MODELS[$i]}" \
    --style "${STYLE:-terse}" --bs "${BS:-64}" --tok-budget "${TOK_BUDGET:-40000}" \
    --max-new "${MAX_NEW:-24}" --max-len "${MAX_LEN:-32768}" --conds "${CONDS:-full,blind}" \
    --log-every "${LOG_EVERY:-10}" --ckpt-every "${CKPT_EVERY:-20}" ${EXTRA_ARGS:-} \
    --out "$OUT"
echo "done: $OUT"
