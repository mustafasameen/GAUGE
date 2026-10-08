# GAUGE

**GAUGE: How Large Language Models Recover Mobility Descriptors from Individual Records**

Mustafa Sameen. Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Urban Mobility
Foundation Models (UMFM '26), Riverside, CA, USA, 2026.
DOI: [10.1145/3849729.3856126](https://doi.org/10.1145/3849729.3856126).
Project page: <https://mustafasameen.github.io/GAUGE/>

Mobility science compresses a person's movement into descriptors such as radius of gyration, the
count of distinct places, and the longest single jump. Mobility models take those descriptors as
generation constraints, as routing inputs, and as scoring functions over populations, and never ask
a model to recover one from a record. GAUGE is a benchmark that reverses the direction and measures
what comes back. Across five open-weight models and 34,200 questions over individual location
records, recovery splits by descriptor class rather than by difficulty. Asked to state a value
directly under greedy decoding, spatial-geometric descriptors never clear their baseline, at any
record length or tolerance we score. Count, sequence and retrieval descriptors do. GPT-4o, scored on
a matched 960-question subset, clears no geometric cell either. The profile is bimodal in record
length: 89% of model-family pairs either clear their baseline on the longest record we test or clear
it nowhere. A mean accuracy therefore describes no pair. What binds recovery is the amount of record
rather than the state an answer requires, established by two designs that vary different
quantities, and the profile survives changes to wording, scaffolding and answer format. The failure
is in carrying the computation out rather than in knowing what it is: asked for a Python expression
instead of a value, three of the five models write programs that compute the same descriptors
correctly on 50 records they never saw. A permutation of the true values across people reproduces
every distributional score we compute while destroying every individual, and the models order people
on geometric quantities in 75 of 100 settings while stating no individual's value. We give the field
a second axis for evaluating mobility foundation models: recovery from one person's record, with
exact answers and exact baselines, beside the distributional scores now used.

This repository holds the code that turns YJMob100K into the benchmark questions, runs the models on
them, scores the answers and builds the tables and figures of the paper. It holds no data, no model
outputs and no per-person records. The generator rebuilds the question set from YJMob100K, and the
steps below show how.

## Layout

| Path | Contents |
| --- | --- |
| `gauge/` | question generators, model runner, scorers, table and figure builders |
| `slurm/` | generic example launchers for the model runs |
| `tools/` | small helpers that connect the steps, and the reproduction test |
| `survey/` | the inputs of Table 1 (the capability survey) |

File names follow one scheme, and the scorers look for these names. Question files are
`results/questions*.jsonl`. A model run is `results/<arm>_<tag>.json`, where `<tag>` is the model
tag of the table in the Models section and `<arm>` is the name of the arm (step 3 lists them):
`main_<tag>.json` holds the runs on the 34,200-question set, except that the Gemma-3-12B run is
`primary_gemma3_12b.json`. Scored results are `results/<name>.json`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data results outputs logs
```

Run every script from the repository root. Scripts read and write `data/`, `results/` and
`outputs/` relative to it. Question generation, scoring, tables and figures run on a CPU. Running
the five open-weight models needs GPUs: Llama-3.1-70B in bf16 needs about 150 GB of GPU memory. The
figure scripts call Ghostscript (`gs`). The paper's model runs used Python 3.12, torch 2.11.0 and
transformers 5.14.1.

## Data

The questions are built from YJMob100K, dataset 1 (Yabe et al., 2024). The data are public on
Zenodo (DOI [10.5281/zenodo.10836269](https://doi.org/10.5281/zenodo.10836269)) under CC BY 4.0.
Download `yjmob100k-dataset1.csv.gz` into `data/yjmob100k/`. The file we used is 419,026,821 bytes
with md5 `3781f6f03a118b5f639bdb4f94dcfdb8`.

Please cite the dataset paper when you use it: T. Yabe, K. Tsubouchi, T. Shimizu and Y. Sekimoto,
"YJMob100K: City-scale and longitudinal dataset of anonymized human mobility trajectories",
Scientific Data 11:397 (2024), doi:10.1038/s41597-024-03237-9.

YJMob100K is anonymised: places are 500 m cells, time is in 30-minute bins, dates are masked and the
city is not disclosed. The code uses coordinates only as relative distances, under a rigid
transform drawn per question, and never maps a place.

## Pipeline

Each step lists what it reproduces in the paper.

### 1. Export YJMob100K

```bash
python gauge/export_yjmob.py --out data/yjmob/yjmob_export.parquet
```

Takes the first 20,000 people in the file and writes one parquet file (23,441,802 rows, md5
`ca3ba3a691dfdfe3bd8d1043cd85df58`) and a manifest beside it. Section 3.2.

### 2. Generate the questions

```bash
# The 34,200-question set (Sections 3.2 and 4.1 to 4.3, Tables 2, 3, 6 and 7)
python gauge/generate_questions.py --data data/yjmob/yjmob_export.parquet --out results/questions_generated.jsonl
python tools/reported_header.py results/questions_generated.jsonl results/questions.jsonl

# The 6,000 coordinate questions with the corrected column header (Appendix B)
python gauge/generate_questions.py --data data/yjmob/yjmob_export.parquet --only-coord --out results/questions_corrected_header.jsonl

# Decomposition controls and scaffold ladder (Sections 4.2 and 4.5, Figure 4)
python gauge/generate_control.py --data data/yjmob/yjmob_export.parquet --out results/questions_control.jsonl
python gauge/generate_ladder.py --data data/yjmob/yjmob_export.parquet --out results/questions_ladder.jsonl

# Cross-record attribution (Section 4.4)
python gauge/generate_attribution.py --data data/yjmob/yjmob_export.parquet --out results/questions_attribution.jsonl

# Which of two people is larger, at record length 64, then 32, 128 and 256, then 512 (Section 4.5)
python gauge/generate_compare.py --data data/yjmob/yjmob_export.parquet --span 64 --per-cell 50 --out results/questions_compare.jsonl
python gauge/generate_compare.py --data data/yjmob/yjmob_export.parquet --span 32 --per-cell 25 --out results/questions_compare_s32.jsonl
python gauge/generate_compare.py --data data/yjmob/yjmob_export.parquet --span 128 --per-cell 25 --out results/questions_compare_s128.jsonl
python gauge/generate_compare.py --data data/yjmob/yjmob_export.parquet --span 256 --per-cell 25 --out results/questions_compare_s256.jsonl
cat results/questions_compare_s32.jsonl results/questions_compare_s128.jsonl results/questions_compare_s256.jsonl > results/questions_compare_spans.jsonl
python gauge/generate_compare.py --data data/yjmob/yjmob_export.parquet --span 512 --seed 512 --out results/questions_compare_512.jsonl

# Five prompt wordings (Section 4.5)
python gauge/make_templates.py
python tools/make_retrieval_templates.py

# Program arm: the same records asked for a value and for a Python expression (Sections 4.6 and 4.7)
python gauge/generate_tool.py --data data/yjmob/yjmob_export.parquet --out results/questions_tool.jsonl

# Framing arm and its stripped rung (Section 4.6, Appendix E)
python gauge/make_framing.py --per-cell 75 --out results/questions_framing.jsonl
python gauge/make_stripped.py

# Reasoning-budget arm (Section 4.6, Appendix E)
python gauge/make_cot_items.py --per-cell 20 --out results/questions_cot.jsonl

# Subsets for the GPT-4o arm (Section 4.1, Figure 3) and the sampling arm (Section 4.5)
python gauge/make_subset.py --items results/questions.jsonl --per-cell 30 --seed 0 --families gyration_km,max_distance,total_distance,longest_jump,day_distinct,retrieve_p50 --out results/questions_frontier.jsonl
python tools/subset_to_indices.py --items results/questions.jsonl --subset results/questions_frontier.jsonl --out results/subset_allspans.json --note "per-cell 30, seed 0, families=6, ALL spans"
python tools/draw_sampling_subset.py --items results/questions.jsonl --out results/questions_sampling.jsonl
```

Notes on step 2:

- A full run of `generate_questions.py` takes about a minute and a half. It ends with a traceback
  (`UnboundLocalError ... hashlib`) after the file is complete. The file has 34,200 lines.
- `generate_questions.py` names the columns of a coordinate prompt `day, timeslot, x, y`. The
  paper's main tables were scored with `day, timeslot, place` on those 6,000 prompts, and Appendix B
  reports the `x, y` version as a rerun. `tools/reported_header.py` changes that one phrase and
  nothing else, which gives the question set that was scored.
- The questions are drawn with a fixed seed. These files rebuild byte for byte:

| File | Items | md5 |
| --- | --- | --- |
| `questions.jsonl` | 34,200 | `127b67859e28c8728293c459be78c047` |
| `questions_corrected_header.jsonl` | 6,000 | `51e0445d2f39ea7e562dffd6b59aa81d` |
| `questions_control.jsonl` | 2,250 | `3df5b504696160b0864b72d7f239a7af` |
| `questions_ladder.jsonl` | 2,000 | `5f8e45208c0bba4b238e06f39eb41d57` |
| `questions_attribution.jsonl` | 1,191 | `068bd388a889bb1f28dea7dd254b47c5` |
| `questions_compare.jsonl` | 3,000 | `7c0b2e38734cb3d78bc395cf87fd2116` |
| `questions_compare_spans.jsonl` | 4,500 | `e5aa026b2287512fa9e4dc253ccee6fd` |
| `questions_compare_512.jsonl` | 3,000 | `88cee185da7dc6ad8506034532358103` |
| `questions_templates.jsonl` | 5,000 | `fcec812105ffe2af096afd3f4615d916` |
| `questions_templates_retrieval.jsonl` | 2,750 | `8827130d428e3f9176b9dc072fcae0d4` |
| `questions_tool.jsonl` | 2,400 | `92a3e7ba833500d8ccdf67271002620e` |
| `questions_framing.jsonl` | 1,500 | `24700ebadac456a30e4e33b1e34e602f` |
| `questions_stripped.jsonl` | 1,500 | `ac29acbac7884a07d667512365a00152` |
| `questions_cot.jsonl` | 400 | `94502382f363ada635d6288b99bc3456` |
| `questions_sampling.jsonl` | 4,050 | `908d627df177663900672cdd53c2cbbd` |
| `subset_allspans.json` | 960 | `e133cccd24fd1f8c547538992a120d6a` |

`tools/test_generator_reproduces.sh PARQUET WORKDIR` checks the first two rows.

### 3. Run the models

Every question is asked with the record (`full`) and without it (`blind`). Check an items file
first, then run one model:

```bash
python gauge/preflight_items.py results/questions.jsonl
python gauge/eval_model.py --items results/questions.jsonl --model google/gemma-3-12b-it --style terse --bs 64 --tok-budget 40000 --max-new 24 --max-len 32768 --conds full,blind --out results/primary_gemma3_12b.json
```

`slurm/run_model.sbatch` runs one model per array task, and `slurm/submit_all.sh` prints (or, with
`SUBMIT=1`, submits) the command for every run in the paper. They hold no account or partition. Pass
those as `SBATCH_OPTS`. Without a scheduler, run the `python` command inside the launcher by hand. The
runs and the names the scorers look for:

| Arm | Items file | Output | Paper |
| --- | --- | --- | --- |
| Main set | `questions.jsonl` | `main_<tag>.json` (`primary_gemma3_12b.json` for Gemma-3-12B) | 4.1 to 4.3 |
| Corrected header | `questions_corrected_header.jsonl` | `corrected-header_<tag>.json` | Appendix B |
| Controls | `questions_control.jsonl` | `control_<tag>.json` | 4.2 |
| Ladder | `questions_ladder.jsonl` | `ladder_<tag>.json` | 4.5 |
| Attribution | `questions_attribution.jsonl` | `attribution_<tag>.json` | 4.4 |
| Comparison | `questions_compare*.jsonl` | `compare_`, `compare-spans_`, `compare-512_<tag>.json` | 4.5 |
| Wording | `questions_templates*.jsonl` | `templates_`, `templates-retrieval_<tag>.json` | 4.5 |
| Sampling | `questions_sampling.jsonl` | `sampling_<tag>.json.full.ckpt` | 4.5 |
| Framing | `questions_framing.jsonl`, `questions_stripped.jsonl` | `framing_`, `stripped_<tag>.json` | 4.6 |
| Reasoning | `questions_cot.jsonl` | `cot-4k_`, `cot-8k_<tag>.json` | 4.6 |
| Program | `questions_tool.jsonl` | `tool_<tag>.json` | 4.6, 4.7 |

`preflight_items.py` fails on `questions_tool.jsonl`, because the key of a program question is a number
while the model writes an expression. Skip it for that file (`PREFLIGHT=0` in the launcher).

The GPT-4o arm answers the 960-question subset through the OpenAI API:

```bash
export OPENAI_API_KEY=...
python gauge/run_frontier.py --peek 2 --subset results/subset_allspans.json
python gauge/run_frontier.py --model gpt-4o --subset results/subset_allspans.json --out results/frontier_gpt4o.json
```

### 4. Score

Run these in order. Each reads the question files and runs named in its docstring and writes one
JSON file to `results/`.

```bash
python gauge/family_cis.py                       # Table 2, Figure 5, Section 4.3: gain, baseline, intervals
python gauge/distributional_rung.py              # Table 3, Section 4.1: three-level measurement
python gauge/error_taxonomy.py                   # Figure 2 and the abstention rates of Section 4.3: what the models answer instead
python gauge/tolerance_gate.py                   # Section 4.1: tolerance scoring
python gauge/continuous_metric.py                # Section 4.1: continuous error metrics
python gauge/screen_degeneracy.py --items results/questions.jsonl --runs results/primary_gemma3_12b.json results/main_*.json --out results/degeneracy_screen.json   # Section 3.4
python gauge/score_blind_all_models.py           # Section 3.4: blind control, five models
python gauge/make_blind_decomposition.py         # Section 3.4: blind control, Gemma-3-12B
python gauge/score_corrected_header.py           # Appendix B: corrected header against reported header
python gauge/score_control_ladder.py             # Section 4.2 and 4.5, Figure 4
python gauge/score_control_tolerance.py          # Section 4.2: controls within a factor of two
python gauge/score_compare.py                    # Section 4.5: comparison arm
python gauge/score_templates.py                  # Section 4.5: wording study
python gauge/score_sampling.py                   # Section 4.5: sampling arm
python gauge/score_attribution.py                # Section 4.4: cross-record attribution
python gauge/score_framing.py                    # Section 4.6: framing
python gauge/score_stripped.py                   # Section 4.6: stripped rung
python gauge/score_cot_budget.py                 # Section 4.6: reasoning budget, Llama-3.1-8B
python gauge/score_cot_budget.py --single --hi results/cot-8k_llama70b.json --tag llama70b --out results/scored_cot_70b.json   # Section 4.6: Llama-3.1-70B
python gauge/score_tool.py                       # Section 4.6: value against program
python tools/make_points_pool.py                 # records used as held-out probes
python gauge/program_equivalence.py              # Section 4.7, Table 4: extensional equivalence
python gauge/score_dissociation.py               # Discussion, Table 5: rank people, count people
python gauge/score_frontier.py                   # Section 4.1, Figure 3: GPT-4o against the five models
python gauge/verify_claims.py                    # Section 4.3: required state, answer position, absence
python gauge/multiplicity.py                     # Section 3.5: cells scanned and Benjamini-Hochberg correction
python tools/make_absence_artifact.py            # input of Figure 6
python tools/pair_counts.py                      # Section 4.3 and abstract: 105 pairs, 47 / 46 / 12, 89%
```

### 5. Tables and figures

```bash
python gauge/make_tables.py                      # Table 1, Table 2, Table 3
python gauge/make_appendix.py                    # Table 6 (Appendix A), Table 7 (Appendix D)
python gauge/make_program_table.py               # Table 4
python gauge/make_dissociation_table.py          # Table 5
python gauge/make_ceiling_table.py --check       # Section 3.5, Appendix B: how often generation hit the 24-token cap
python gauge/make_figures.py                     # Figures 1, 2, 5 and 6
python gauge/make_frontier_figure.py             # Figure 3
python gauge/make_ladder_figure.py               # Figure 4
```

The LaTeX tables go to `outputs/tex/tabs/` and the figures to `outputs/tex/figs/`.
`gauge/tasks_*.py` hold the reference computations behind every answer key (Appendix C). Run
`python gauge/tasks_main.py` (and `tasks_control.py`, `tasks_ladder.py`, `tasks_tool.py`) to execute
their self-tests.

## Models

| Tag | Hugging Face id |
| --- | --- |
| `phi35mini` | `microsoft/Phi-3.5-mini-instruct` |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `llama8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `gemma3_12b` | `google/gemma-3-12b-it` |
| `llama70b` | `meta-llama/Llama-3.1-70B-Instruct` |

All five run through Hugging Face Transformers with greedy decoding, bf16 weights and a 24-token
cap on the answer. Some of these weights need you to accept the model licence on Hugging Face.
GPT-4o (`gpt-4o`) is called through the OpenAI API with temperature 0.

## Checks

`tools/test_generator_reproduces.sh PARQUET WORKDIR` rebuilds the question set and compares md5
values with the question set that was scored.

## Licence

The code is released under the MIT licence (see `LICENSE`). YJMob100K has its own licence (CC BY
4.0), and the models have theirs.

## Citation

```bibtex
@inproceedings{sameen2026gauge,
  author    = {Sameen, Mustafa},
  title     = {{GAUGE}: How Large Language Models Recover Mobility Descriptors from Individual Records},
  booktitle = {Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Urban Mobility Foundation Models},
  series    = {UMFM '26},
  year      = {2026},
  publisher = {ACM},
  address   = {Riverside, CA, USA},
  doi       = {10.1145/3849729.3856126}
}
```
