#!/bin/bash
# Print (or submit) the sbatch command for every model run in the paper.
#
#   slurm/submit_all.sh                                  print the commands
#   SUBMIT=1 SBATCH_OPTS="<your sbatch options>" slurm/submit_all.sh    submit them
#
# SBATCH_OPTS carries whatever your site needs, such as an account and a partition.
#
# Each line sets the environment variables that slurm/run_model.sbatch reads. The question files
# must already exist in results/ (see the README for the commands that build them). Settings are
# the ones used for the runs in the paper: greedy decoding unless EXTRA_ARGS says otherwise.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SBATCH_OPTS="${SBATCH_OPTS:-}"

go() {   # go ARRAY "VAR=value ..."
    local array="$1"; shift
    local cmd="env $* sbatch --array=${array} ${SBATCH_OPTS} slurm/run_model.sbatch"
    if [ "${SUBMIT:-0}" = "1" ]; then eval "$cmd"; else echo "$cmd"; fi
}

# main profile: all 34,200 questions, answer and blind condition, five models (main_*, and
# primary_gemma3_12b for Gemma-3-12B)
go 0-4 ITEMS=results/questions.jsonl PREFIX=main CONDS=full,blind BS=64 TOK_BUDGET=40000 LOG_EVERY=40
# corrected-header coordinate questions (Appendix B, and the source of the framing arms)
go 0-4 ITEMS=results/questions_corrected_header.jsonl PREFIX=corrected-header CONDS=full,blind
# decomposition controls and scaffold ladder (Section 4.2, Figure 4)
go 0-4 ITEMS=results/questions_control.jsonl PREFIX=control CONDS=full,blind
go 0-4 ITEMS=results/questions_ladder.jsonl PREFIX=ladder CONDS=full
# comparison arm (Section 4.5): record length 64, then 32/128/256, then 512
go 0-4 ITEMS=results/questions_compare.jsonl PREFIX=compare CONDS=full,blind BS=32
go 0-4 ITEMS=results/questions_compare_spans.jsonl PREFIX=compare-spans CONDS=full BS=32
go 0-4 ITEMS=results/questions_compare_512.jsonl PREFIX=compare-512 CONDS=full,blind BS=8 TOK_BUDGET=26000 MAX_LEN=24576
# wording study (Section 4.5)
go 0-4 ITEMS=results/questions_templates.jsonl PREFIX=templates CONDS=full
go 0-4 ITEMS=results/questions_templates_retrieval.jsonl PREFIX=templates-retrieval CONDS=full BS=32
# cross-record attribution at fixed total length (Section 4.4)
go 0-4 ITEMS=results/questions_attribution.jsonl PREFIX=attribution CONDS=full,blind BS=8 TOK_BUDGET=26000 MAX_NEW=16 MAX_LEN=24576
# program arm (Section 4.7; the tool condition of Section 4.6): the model writes an expression
go 0-4 ITEMS=results/questions_tool.jsonl PREFIX=tool CONDS=full BS=32 MAX_NEW=96 MAX_LEN=8192 PREFLIGHT=0
# framing arms (Section 4.6): mobility versus relabelled, then relabelled versus stripped
go 0-4 ITEMS=results/questions_framing.jsonl PREFIX=framing CONDS=full,nomobility
go 0-4 ITEMS=results/questions_stripped.jsonl PREFIX=stripped CONDS=nomobility,stripped
# sampling arm (Section 4.5): eight draws, Gemma-3-12B and Llama-3.1-70B only
go 3-4 ITEMS=results/questions_sampling.jsonl PREFIX=sampling CONDS=full BS=4 \
   "EXTRA_ARGS='--n-samples 8 --temperature 0.7 --top-p 0.95'"
# reasoning-budget arm (Section 4.6): chain of thought with forced answers. Llama-3.1-8B at 4,096
# and 8,192 tokens, Llama-3.1-70B at 8,192 tokens. Keep BS the same for the two Llama-3.1-8B runs.
go 2 ITEMS=results/questions_cot.jsonl PREFIX=cot-4k CONDS=full STYLE=cot BS=8 MAX_NEW=4096 LOG_EVERY=5 CKPT_EVERY=10 \
   "EXTRA_ARGS='--require-marker --marker-above 32 --force-answer --force-tokens 8'"
go 2 ITEMS=results/questions_cot.jsonl PREFIX=cot-8k CONDS=full STYLE=cot BS=8 MAX_NEW=8192 LOG_EVERY=5 CKPT_EVERY=10 \
   "EXTRA_ARGS='--require-marker --marker-above 32 --force-answer --force-tokens 8'"
go 4 ITEMS=results/questions_cot.jsonl PREFIX=cot-8k CONDS=full STYLE=cot BS=8 MAX_NEW=8192 LOG_EVERY=5 CKPT_EVERY=10 \
   "EXTRA_ARGS='--require-marker --marker-above 32 --force-answer --force-tokens 8'"
