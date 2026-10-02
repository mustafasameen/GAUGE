#!/usr/bin/env python3
"""Run a model over an items file and score its generations against the gold answers.

Every question is asked twice: once with the record in the prompt (condition `full`) and once with
the record withheld (condition `blind`). The blind condition bounds what the model can answer from
the question alone. Each generation is parsed by `norm`, the single scoring rule that every other
script in this repository imports. A generation that cannot be parsed counts as wrong and is never
dropped, and the unparsed rate is printed per condition.

Printed per family: the number of questions, accuracy with and without the record, their
difference, the majority-class baseline (read from a screen file beside the items file, if there
is one), and the unparsed rates.

Input: an items file (JSON lines) from the generators, and a Hugging Face model id.
Output: a results JSON with the raw generations (raw[condition]), per-item generated lengths and a
per-family summary. A checkpoint per condition is written beside it as {out}.{cond}.ckpt.

Usage:
  python gauge/eval_factqa.py --items results/tally_v41.jsonl --model google/gemma-3-12b-it
      --style terse --bs 64 --tok-budget 40000 --max-new 24 --max-len 32768
      --conds full,blind --out results/v41pilot_gemma3_12b.json
"""
from __future__ import annotations
import argparse, collections, json, os, re, time
import numpy as np

# The prompt style is a controlled factor, not a default. Under an unconstrained style one model
# (Llama-3.1-8B) wrote chain-of-thought spontaneously while the others answered in a few
# characters, and so got far more test-time compute. Every model is run under the same named style.
STYLES = {
    # terse: no externalisation. Used with a 24-token cap so that a verbose model cannot buy
    # test-time compute.
    "terse": ("You answer questions about location records. Reply with ONLY the value, on a single "
              "line of the form `Answer: <value>`. Do not explain. If you cannot determine the "
              "value, reply `Answer: unknown`."),
    # cot: free-form chain of thought that ends in a final `Answer:` line.
    "cot":   ("You answer questions about location records. Work through it step by step, showing "
              "your counting explicitly, then end with a final line of exactly the form "
              "`Answer: <value>`. If you cannot determine the value, end with `Answer: unknown`."),
    # procedure: least-to-most style decomposition (Zhou et al., arXiv:2205.10625). The
    # decomposition is stated explicitly instead of being left to the model: enumerate, then reduce.
    "procedure": ("You answer questions about location records. Follow this procedure exactly:\n"
                  "  Step 1. Go through the record line by line and write out the list of items the "
                  "question is about (for example, every place that appears, in order).\n"
                  "  Step 2. Remove duplicates from that list if the question asks about DISTINCT "
                  "things.\n"
                  "  Step 3. Count or select from the list you wrote, never from memory of the "
                  "record.\n"
                  "Then end with a final line of exactly the form `Answer: <value>`. If you cannot "
                  "determine the value, end with `Answer: unknown`."),
    # fewshot: the same procedure as `procedure`, demonstrated with two worked examples instead of
    # being instructed. Run at the same token budget as the other trace styles, so only the prompt
    # differs.
    "fewshot": ("You answer questions about location records. Here are two worked examples.\n\n"
                "Record:\nd1 t3 p7\nd1 t9 p7\nd1 t14 p2\nd2 t4 p7\n"
                "Question: How many distinct places does this record contain?\n"
                "Working: places in order = 7, 7, 2, 7. Unique = {7, 2}. That is 2.\n"
                "Answer: 2\n\n"
                "Record:\nd5 t2 p3\nd5 t8 p3\nd5 t19 p9\nd6 t1 p3\nd6 t7 p4\n"
                "Question: How many times does place 3 appear in this record?\n"
                "Working: scan for p3 -> d5 t2, d5 t8, d6 t1. That is 3.\n"
                "Answer: 3\n\n"
                "Now do the same: write your working, then end with a final line of exactly the "
                "form `Answer: <value>`. If you cannot determine the value, end with "
                "`Answer: unknown`."),
}
SYS_INSTRUCTION = STYLES["terse"]

NUMERIC = {"distinct_week", "longest_streak", "new_places", "top_place_week",
           "retrieval", "count_day"}  # families whose answers are numbers
BUCKET = {"share_top_bucket"}
CATEG = {"busiest_daypart", "compare_weeks"}


ANSWER_RE = re.compile(r"answer\s*[:=]\s*(.+)", re.I | re.S)


def norm(text, family, atype=None):
    """Extract a comparable answer from a generation. Returns None if nothing is parseable; the
    caller scores None as wrong and does not drop the item.

    The answer is the text after an explicit `Answer:` marker when there is one. Otherwise it is the
    last number in the generation, never the first, because reasoning precedes the conclusion and a
    model often restates the question first ("To find the number of distinct places visited in week
    4..."), so the first integer can belong to the question.
    """
    t = (text or "").strip()
    if not t:
        return None
    m = ANSWER_RE.search(t)
    if m:
        t = m.group(1)
    t = t.strip().lower()
    if not t:
        return None
    if atype == "binary" or family in ("membership", "compare_weeks"):
        if re.search(r"\byes\b", t):
            return "yes"
        if re.search(r"\bno\b", t):
            return "no"
        return None
    # Categorical answers: a letter, or "none". The answer follows the marker, so take the first
    # letter token after it, not the last. An answer type that the parser does not handle would score a
    # whole family at 0.000 with every generation unparsed, and that reads as a model failure.
    if atype == "categorical":
        if re.search(r"\bnone\b", t):
            return "none"
        mm = re.search(r"\b([a-h])\b", t)
        return mm.group(1) if mm else None
    # Code (the program arm). The model emits a Python expression, not an answer, so there is
    # nothing to normalise and comparing it with the gold string is meaningless. The raw text is
    # returned so that it survives into `raw`. score_tool.py, which holds the points, is the only
    # authority on this family's accuracy. The in-eval accuracy for the tool_* families is 0 by
    # construction and must never be cited; score_tool.py asserts it.
    if atype == "code":
        return text.strip() if text and text.strip() else None
    # Letter or none (cross-record attribution). Not `categorical`: its bare [a-h] match picks up
    # the article "a" in a reply like "a person was at place 3011" and silently scores option A. Prefer
    # the bare answer, then an explicit "person X", then a leading token, and only then a standalone
    # letter, which must not be a lone "a" before a noun.
    if atype == "letter":
        # The alphabet is A/B/C/E/F/G/H/J. D, T and P are excluded at generation because they collide
        # with the d/t/p field prefixes in the rendered record (see tally_whichK.py). The character classes
        # below exclude d, t and p for the same reason.
        if re.search(r"\bnone\b", t) or re.search(r"\bno(?:body|t any| one)\b", t):
            return "none"
        if re.fullmatch(r"[abcefghj]", t):
            return t
        mm = re.search(r"\bperson\s+([a-h])\b", t)
        if mm:
            return mm.group(1)
        mm = re.search(r"^\W*([a-h])\b(?!\s+[a-z])", t)     # "A." / "(C)" but not "a person"
        if mm:
            return mm.group(1)
        mm = re.search(r"\banswer\s*(?:is)?[:\s]\s*([a-h])\b", t)
        if mm:
            return mm.group(1)
        mm = re.search(r"\b([b-h])\b", t)                    # 'a' excluded: it is the article
        return mm.group(1) if mm else None
    # A/B forced choice (comparison arm). Not `categorical`: its bare [a-h] match would pick up the
    # article "a" in a reply like "B has a larger radius" and score the wrong option. Order of
    # preference: the whole answer is a bare a or b, then an explicit "person a/b", then the first
    # standalone a or b.
    if atype == "ab":
        if t in ("a", "b"):
            return t
        mm = re.search(r"\bperson\s+([ab])\b", t)
        if mm:
            return mm.group(1)
        mm = re.search(r"^\W*([ab])\b", t)          # leading token, e.g. "A." or "(B)"
        if mm:
            return mm.group(1)
        mm = re.search(r"\banswer\s*(?:is)?\s*([ab])\b", t)
        return mm.group(1) if mm else None
    # Numeric or none (the retrieval probe). The family mixes an answerable half (a place number)
    # with an unanswerable half (gold "none"), so neither `numeric` nor `categorical` fits: categorical
    # extracts a single letter a-h and returns None for "417". The answer space has two types, so the
    # family gets its own atype.
    if atype == "numeric_or_none":
        n = re.findall(r"-?\d+", t)
        # Absence vocabulary. The prompt says "answer none", and models comply in spirit far more often
        # than in letter: at span 512, 33 of 300 unanswerable items came back "Answer: unknown", which a
        # bare test for the word "none" would score as unparseable and so understate absence detection.
        # This is conservative by construction: an absence phrase counts only when the answer carries no
        # digit, so "the record does not say, but likely 417" still scores as the answer 417 and not as an
        # abstention.
        if not n and re.search(r"\b(none|unknown|n/?a|unavailable|unclear|indeterminate)\b|"
                               r"not\s+(recorded|specified|given|available|present|listed|shown)|"
                               r"no\s+(record|data|entry|information)|does\s+not\s+(say|specify)|"
                               r"cannot\s+be\s+determined", t):
            return "none"
        if re.search(r"\bnone\b", t):
            return "none"
        return n[-1] if n else None
    if atype == "numeric" or family in NUMERIC:
        # Decimals are parsed as numbers, not split. Taking the last run of digits from a decimal answer
        # returns its fractional part ("11.9" would become 9), which would corrupt exactly the distance
        # families, since distances are naturally answered as decimals. Digit-grouping commas are stripped
        # first: "13,111" would otherwise be read as 111.
        t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
        n = re.findall(r"-?\d+(?:\.\d+)?", t)
        if not n:
            return None
        v = n[-1]  # the LAST number, not the first (reasoning precedes the conclusion)
        try:                                       # "11.0" and "11" are the same answer to a
            f = float(v)                           # "whole number" question; render canonically so
            if f.is_integer():                     # a model is not penalised for writing ".0".
                return str(int(f))
        except ValueError:
            pass
        return v  # a genuine decimal stays a decimal: it is not rounded into correctness;
                                                   # the tolerance and continuous metrics are what handle near-misses
    if family in BUCKET:
        m = re.search(r"(\d{1,3})\s*[-–]\s*(\d{1,3})\s*%?", t)
        return f"{int(m.group(1))}-{int(m.group(2))}%" if m else None
    if family == "compare_weeks":
        if re.search(r"\byes\b", t):
            return "yes"
        if re.search(r"\bno\b", t):
            return "no"
        return None
    for w in ("night", "morning", "afternoon", "evening"):
        if re.search(rf"\b{w}\b", t):
            return w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="results/factqa_v1.jsonl")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--init-from", default=None,
                    help="LoRA adapter directory to load on top of --model (requires peft).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--style", choices=["terse", "cot", "procedure", "fewshot"],
                    default="terse")
    ap.add_argument("--max-new", type=int, default=192)
    # Conditions to run, comma-separated: full, blind, and nomobility or stripped for the framing
    # arms. `--conds full` halves the cost when the blind control is not needed.
    ap.add_argument("--conds", default="full,blind")
    # Optional chat-template flag for models that have a thinking mode (for example Qwen3), which is
    # toggled by one boolean in the template. `default` passes no flag.
    ap.add_argument("--thinking", choices=["default", "on", "off"], default="default")
    # Maximum prompt length in tokens. A longer prompt aborts the run instead of being truncated.
    # The benchmark runs use 32768.
    ap.add_argument("--max-len", type=int, default=4096)
    # Optional filter: keep only items whose prompt has at most this many characters (0 keeps
    # everything). It lets a model with a short context window run on the subset it can hold; the
    # dropped items are printed per family.
    ap.add_argument("--max-chars", type=int, default=0, help="0 = no filter")
    # Length-aware batching. Prompts run from about 100 to 33,000 characters, so a fixed batch size
    # either runs out of memory on long items or wastes the GPU on short ones. Batches are filled to a
    # padded-token budget instead, longest first, so that an out-of-memory error shows up in the first
    # minute of a run.
    ap.add_argument("--tok-budget", type=int, default=0, help="0 = fixed batches of --bs")
    # Print interval, in batches. Affects printing only.
    ap.add_argument("--log-every", type=int, default=40)
    # Write a partial checkpoint every N batches (0 writes only at the end of each condition). A
    # job that is killed at its wall-time limit would otherwise lose everything it had generated.
    ap.add_argument("--ckpt-every", type=int, default=0, help="0 = only at end of condition")
    ap.add_argument("--resume", action="store_true",
                    help="reload {out}.{cond}.partial/.ckpt and regenerate only the missing items. "
                         "A job that times out would otherwise lose everything it generated.")
    # Scoring policy for reasoning-style arms. Terse generations are bare answers ('197', 'yes'), so
    # extracting from the whole generation is the designed path. Chain-of-thought and procedure
    # generations are reasoning traces: if a trace never reaches the anchored 'Answer:' marker
    # (truncation, or no termination), extracting a number from mid-trace would score the model's
    # enumeration counter and not an answer. With this flag a markerless trace is scored as unparsed:
    # wrong, and counted separately from wrong answers.
    ap.add_argument("--require-marker", action="store_true")
    # A generation without a marker is unparsed only if it is a trace. Bare answers fit in a couple
    # of dozen tokens and traces run to hundreds, so the marker is required only above this many
    # generated tokens. That is the terse/trace boundary on which the three-way accounting (correct,
    # wrong, unparsed) rests.
    ap.add_argument("--marker-above", type=int, default=32)
    # Forced answer (budget forcing). A generation that hits the token ceiling without ever emitting
    # the `Answer:` marker would be lost and scored unparsed. At a fixed budget a longer record yields
    # a longer trace, so the answer rate would then fall with record length by arithmetic and not
    # because of the model. With this flag such a generation gets its own trace back plus "Answer:" and
    # a few more tokens to state a conclusion. The answer rate is then close to 100% by construction and
    # accuracy is the only thing that varies. The forced fraction is recorded per item and reported.
    ap.add_argument("--force-answer", action="store_true")
    ap.add_argument("--force-tokens", type=int, default=8)
    # The forcing pass re-runs generation on the prompt plus the whole trace, so its sequences are
    # several times longer than in the main pass, and a --bs that was safe while generating may not be
    # safe here. 0 means a quarter of --bs, at least 1.
    ap.add_argument("--force-bs", type=int, default=0)
    # Regression-aware inference arm (Lukasik et al., arXiv 2403.04182). n_samples=1 means greedy
    # decoding.
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default="results/factqa_gate1.json")
    a = ap.parse_args()
    CONDS = tuple(c.strip() for c in a.conds.split(",") if c.strip())
    CHAT_KW = {} if a.thinking == "default" else {"enable_thinking": a.thinking == "on"}

    global SYS_INSTRUCTION
    SYS_INSTRUCTION = STYLES[a.style]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(l) for l in open(a.items)]
    if a.limit:
        items = items[:a.limit]
    if a.max_chars:
        keep = [i for i in items if len(i["prompt_full"]) <= a.max_chars]
        dropped = collections.Counter(i["family"] for i in items if len(i["prompt_full"]) > a.max_chars)
        print(f"COVERAGE FILTER --max-chars {a.max_chars:,}: keeping {len(keep):,}/{len(items):,} "
              f"({len(keep)/len(items):.1%}); dropped by family {dict(dropped.most_common())}", flush=True)
        items = keep
    print(f"{len(items):,} items | {len({i['family'] for i in items})} families", flush=True)
    # Parser guard. A family the parser cannot handle would score 0.000 with every generation
    # unparsed and read as a model failure, so refuse to run in that case: every gold is round-tripped
    # through the real scorer. The atype "code" is exempt, narrowly and checked. Its gold is a number
    # and the model emits an expression, so a round trip is not defined. Instead every code item must
    # carry the points needed to execute an expression, and score_tool.py is the authority on the
    # family.
    unscoreable = collections.Counter()
    for i in items:
        if i.get("atype") == "code":
            if not i.get("pts") or "gold_value" not in i:
                unscoreable[(i["family"], "code:MISSING pts/gold_value")] += 1
            continue
        if norm(f"Answer: {i['a']}", i["family"], i.get("atype")) != str(i["a"]).lower():
            unscoreable[(i["family"], i.get("atype"))] += 1
    if unscoreable:
        raise SystemExit(f"PARSER CANNOT SCORE: {dict(unscoreable)}. Golds do not round-trip through "
                         f"norm(). Refusing to run and report 0.000 for these.")
    print(f"parser guard OK: all {len(items):,} golds round-trip through the real scorer", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                     # required for correct batched generation
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
    if a.init_from:
        from peft import PeftModel
        d0 = a.init_from if os.path.isdir(os.path.join(a.init_from, "adapter_config.json")) \
            else os.path.join(a.init_from, "seed0")
        model = PeftModel.from_pretrained(model, d0)
        print(f"LoRA adapter loaded: {d0}", flush=True)
    model.eval()
    try:
        probe = tok.apply_chat_template([{"role": "user", "content": "probe"}],
                                        tokenize=False, add_generation_prompt=True, **CHAT_KW)
        if CHAT_KW:
            alt = tok.apply_chat_template([{"role": "user", "content": "probe"}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=not CHAT_KW["enable_thinking"])
            # Refuse to run if the chat template ignores the flag: both renderings would be identical and
            # the contrast between thinking and non-thinking would silently be no contrast.
            if probe == alt:
                raise SystemExit(f"enable_thinking HAS NO EFFECT on {a.model}'s chat template — "
                                 f"both renderings identical. The thinking contrast would be void.")
            print(f"thinking toggle VERIFIED active: enable_thinking={CHAT_KW['enable_thinking']} "
                  f"renders {len(probe)} chars vs {len(alt)} for the opposite setting", flush=True)
    except Exception as e:
        raise SystemExit(f"CHAT TEMPLATE INCOMPATIBLE for {a.model}: {e}")
    # Tokenise once and reuse the lengths for both the truncation guard and the length-aware
    # batching.
    TOKLEN = {c: [len(tok(SYS_INSTRUCTION + "\n\n" + x[f"prompt_{c}"])["input_ids"]) for x in items]
              for c in CONDS}
    longest = max(max(v) for v in TOKLEN.values())
    print(f"longest prompt {longest:,} tokens | max_len {a.max_len:,}", flush=True)
    if longest >= a.max_len:
        raise SystemExit(f"TRUNCATION would occur ({longest} >= {a.max_len}). Raise --max-len or "
                         f"regenerate items under the cap. Refusing to score clipped records.")
    # State the loaded precision and the resident size, so that a silently dequantised model is
    # visible in the run's own output.
    _dt = str(next(model.parameters()).dtype)
    _nb = sum(x.numel() * x.element_size() for x in model.parameters()) / 1e9
    print(f"model ready on {model.device} | dtype {_dt} | {_nb:.1f} GB resident | "
          f"chat template OK | no truncation", flush=True)

    out, genlens, forceds = {}, {}, {}
    for cond in CONDS:
        key = f"prompt_{cond}"
        preds, t0 = [None] * len(items), time.time()
        # Generated-token counts, kept per item, so that hitting the token ceiling is visible while the
        # job is running and the ceiling rate can be checked per span offline.
        genlen = [None] * len(items)
        samples = [None] * len(items)
        # ---- RESUME ------------------------------------------------------------------------
        # Reload whatever a previous run of this same command already generated. Only items whose
        # prediction is still None are regenerated, so a timeout costs the unfinished tail and not the whole
        # run. A checkpoint is used only if its item count, max_new and dtype match this run. Check the
        # items file's md5 before resuming so that two different item sets cannot be spliced.
        if a.resume:
            for suffix in (".ckpt", ".partial"):
                pth = a.out + f".{cond}{suffix}"
                if not os.path.exists(pth):
                    continue
                prev = json.load(open(pth))
                praw = (prev.get("raw") or {}).get(cond)
                pgl = (prev.get("genlen") or {}).get(cond)
                if not praw or len(praw) != len(items):
                    print(f"  {cond}: {os.path.basename(pth)} has "
                          f"{0 if not praw else len(praw)} slots for {len(items)} items -- IGNORED",
                          flush=True)
                    continue
                _now = str(next(model.parameters()).dtype)
                if prev.get("dtype") is not None and prev["dtype"] != _now:
                    print(f"  {cond}: {os.path.basename(pth)} was generated at dtype "
                          f"{prev['dtype']}, this run loads {_now} -- IGNORED. Mixing precisions "
                          f"in one result would be a silent confound.", flush=True)
                    continue
                if prev.get("max_new") != a.max_new:
                    print(f"  {cond}: {os.path.basename(pth)} was written at max_new "
                          f"{prev.get('max_new')}, this run is {a.max_new} -- IGNORED", flush=True)
                    continue
                for k in range(len(items)):
                    if preds[k] is None and praw[k] is not None:
                        preds[k] = praw[k]
                        if pgl:
                            genlen[k] = pgl[k]
                print(f"  {cond}: resumed {sum(p is not None for p in preds):,}/{len(items):,} "
                      f"generations from {os.path.basename(pth)}", flush=True)
                break
        # The token budget is per sequence, and n_samples multiplies sequences: with
        # num_return_sequences=k, Hugging Face expands the batch to k copies before the prefill, so a batch
        # built to 40,000 padded tokens would prefill k times that. The budget is divided by k here.
        _eff_budget = a.tok_budget // max(a.n_samples, 1)
        if a.n_samples > 1:
            print(f"  tok-budget {a.tok_budget:,} / {a.n_samples} samples "
                  f"-> {_eff_budget:,} effective", flush=True)
        todo = [k for k in range(len(items)) if preds[k] is None]
        if a.resume and len(todo) < len(items):
            print(f"  {cond}: {len(todo):,} items still to generate", flush=True)
        if not todo:
            print(f"  {cond}: nothing left to generate, skipping to scoring", flush=True)
        if a.tok_budget:
            lens = TOKLEN[cond]
            order = sorted(todo, key=lambda k: -lens[k])
            groups, cur, curmax = [], [], 0
            for k in order:
                m = max(curmax, lens[k])
                if cur and (m * (len(cur) + 1) > _eff_budget or len(cur) >= a.bs):
                    groups.append(cur); cur, curmax = [], 0
                    m = lens[k]
                cur.append(k); curmax = m
            if cur:
                groups.append(cur)
            print(f"  {cond}: {len(groups)} length-aware batches "
                  f"(longest item {max(lens):,} tok, budget {_eff_budget:,})", flush=True)
        else:
            groups = [todo[i:i + a.bs] for i in range(0, len(todo), a.bs)]
        for gi, idx in enumerate(groups):
            b = [items[k] for k in idx]
            # Every model gets the instruction folded into the user turn: identical text and identical
            # role everywhere. Some chat templates reject a system role, and using system for some families and
            # user for others would put a prompt-construction difference inside the cross-model comparison.
            msgs = [tok.apply_chat_template(
                [{"role": "user", "content": SYS_INSTRUCTION + "\n\n" + x[key]}],
                tokenize=False, add_generation_prompt=True, **CHAT_KW) for x in b]
            enc = tok(msgs, return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_len).to(model.device)
            # Sample-and-aggregate arm. With the default n_samples=1 this branch is not taken and decoding
            # is greedy. Lukasik et al. (arXiv 2403.04182) argue that mode-seeking decoding implicitly
            # optimises exact match, and propose regression-aware inference: draw k samples and reduce them.
            # All k draws are stored and reduced offline, so median, mean and mode can be compared without
            # re-running the GPU.
            if a.n_samples > 1:
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True,
                                       temperature=a.temperature, top_p=a.top_p,
                                       num_return_sequences=a.n_samples,
                                       pad_token_id=tok.pad_token_id)
                for j, k in enumerate(idx):
                    draws = []
                    for s in range(a.n_samples):
                        new = g[j * a.n_samples + s][enc["input_ids"].shape[1]:]
                        draws.append(tok.decode(new, skip_special_tokens=True))
                    samples[k] = draws
                    preds[k] = draws[0]          # draw 0 keeps the file shape identical
                    genlen[k] = int((g[j * a.n_samples][enc["input_ids"].shape[1]:]
                                     != tok.pad_token_id).sum())
            else:
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                for j, k in enumerate(idx):
                    new = g[j][enc["input_ids"].shape[1]:]
                    preds[k] = tok.decode(new, skip_special_tokens=True)
                    genlen[k] = int((new != tok.pad_token_id).sum())
            if gi % a.log_every == 0:
                done = sum(p is not None for p in preds)
                seen = sorted(x for x in genlen if x is not None)
                p95 = seen[int(.95 * (len(seen) - 1))] if seen else 0
                # at_cap counts items whose generation ran to the ceiling, that is, answers that were cut off
                # rather than finished.
                cap = sum(x >= a.max_new for x in seen)
                print(f"  {cond}: {done}/{len(items)}  ({time.time()-t0:.0f}s)  "
                      f"gen p95 {p95}/{a.max_new}, at_cap {cap} ({cap/max(len(seen),1):.1%})",
                      flush=True)
            if a.ckpt_every and gi and gi % a.ckpt_every == 0:
                os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
                # Close and fsync explicitly so that a partial file on disk is always a readable partial.
                with open(a.out + f".{cond}.partial", "w") as _pf:
                    json.dump(dict(model=a.model, stage=f"partial_{cond}", n=len(items),
                                   n_done=sum(p is not None for p in preds), max_new=a.max_new,
                                   dtype=str(next(model.parameters()).dtype),
                                   genlen={cond: genlen}, raw={cond: preds}), _pf)
                    _pf.flush(); os.fsync(_pf.fileno())
        # Durability checkpoint, written before forcing. The forcing pass can itself fail (it re-runs
        # the model on much longer sequences), and generation is the expensive artifact, so it is made
        # durable here, before anything that can fail touches the GPU again.
        _pre = dict(model=a.model, stage=f"pregen_{cond}", n=len(items),
                    n_done=sum(p is not None for p in preds), max_new=a.max_new,
                    # dtype must be carried: --resume skips a checkpoint whose precision differs, and a checkpoint
                    # without it would bypass that guard.
                    dtype=str(next(model.parameters()).dtype),
                    genlen={cond: genlen}, forced={cond: [False] * len(items)},
                    raw={cond: preds})
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out + f".{cond}.ckpt", "w") as _fh:
            json.dump(_pre, _fh)
            _fh.flush(); os.fsync(_fh.fileno())
        print(f"  {cond}: generation checkpointed ({_pre['n_done']}/{len(items)}) BEFORE forcing",
              flush=True)

        # ---- FORCED-ANSWER SECOND PASS -------------------------------------------------------
        forced = [False] * len(items)
        if a.force_answer:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fbs = a.force_bs if a.force_bs > 0 else max(1, a.bs // 4)
            need = [k for k in range(len(items))
                    if genlen[k] is not None and genlen[k] >= a.max_new
                    and not ANSWER_RE.search(preds[k] or "")]
            print(f"  {cond}: FORCING an answer for {len(need):,} of {len(items):,} "
                  f"({len(need)/len(items):.1%}) generations that hit the ceiling without one",
                  flush=True)
            for bi in range(0, len(need), fbs):
                blk = need[bi:bi + fbs]
                msgs = [tok.apply_chat_template(
                    [{"role": "user", "content": SYS_INSTRUCTION + "\n\n" + items[k][key]}],
                    tokenize=False, add_generation_prompt=True, **CHAT_KW)
                    + (preds[k] or "") + "\nAnswer:" for k in blk]
                enc2 = tok(msgs, return_tensors="pt", padding=True, truncation=True,
                           max_length=a.max_len + a.max_new).to(model.device)
                with torch.no_grad():
                    g2 = model.generate(**enc2, max_new_tokens=a.force_tokens, do_sample=False,
                                        pad_token_id=tok.pad_token_id)
                for j, k in enumerate(blk):
                    tail = tok.decode(g2[j][enc2["input_ids"].shape[1]:], skip_special_tokens=True)
                    preds[k] = (preds[k] or "") + "\nAnswer:" + tail
                    forced[k] = True
                if bi % (fbs * 10) == 0:
                    print(f"    forced {min(bi + fbs, len(need)):,}/{len(need):,}", flush=True)
            still = sum(1 for k in need if not ANSWER_RE.search(preds[k] or ""))
            print(f"  {cond}: forcing done | still unparseable after forcing: {still:,} "
                  f"({still/max(1,len(need)):.1%} of forced)", flush=True)

        # Checkpoint before any check that can kill the job. The assertion below is a safety net that
        # should never fire, but if it does the generations are already on disk and can be re-scored
        # offline.
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        # `samples` holds all k draws per item when --n-samples > 1 (None otherwise, so the file shape is
        # unchanged for greedy runs). It goes in the per-condition checkpoint, which is written before any
        # assertion can kill the job, because the k draws are the expensive artifact.
        _ck = dict(model=a.model, stage=f"generations_{cond}", n=len(items),
                       max_new=a.max_new, dtype=str(next(model.parameters()).dtype),
                       genlen={cond: genlen},
                       n_samples=a.n_samples, temperature=a.temperature,
                       samples={cond: samples} if a.n_samples > 1 else None,
                       forced={cond: forced}, raw={cond: preds})
        with open(a.out + f".{cond}.ckpt", "w") as _cf:
            json.dump(_ck, _cf); _cf.flush(); os.fsync(_cf.fileno())
        print(f"  {cond}: checkpointed {len(preds):,} generations before scoring", flush=True)
        # Assert that every slot is filled exactly once, in item order: a batching change that silently
        # mis-maps outputs to items would look like a model result.
        assert all(p is not None for p in preds), \
            f"{sum(p is None for p in preds)} items never generated"
        out[cond] = preds
        genlens[cond] = genlen
        # `forced` is a per-condition local; without accumulating it here the final dump
        # would silently carry only the LAST condition's flags.
        forceds[cond] = forced
        if torch.cuda.is_available():
            # Sampled in-process: a separate python call after the eval exits reads a fresh CUDA context and
            # would print 0.0 GiB every time.
            print(f"  {cond} PEAK GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB of "
                  f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB", flush=True)
        print(f"  {cond} done in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ score
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, it in enumerate(items):
        f = it["family"]
        for cond in CONDS:
            p = norm(out[cond][n], f, it.get("atype"))
            raw = out[cond][n]
            # Use ANSWER_RE, the same pattern norm() keys on. A looser test such as `"answer" in raw.lower()`
            # admits traces that merely echo the prompt ("...Answer with the place number...") and traces that
            # state a tentative value mid-reasoning ("So answer should be 469. But we must double-check") and
            # then run away. norm() would then fall back to extracting a number from both.
            if (a.require_marker and not ANSWER_RE.search(raw or "")
                    and (genlens.get(cond, [None] * len(items))[n] or 0) > a.marker_above):
                p = None  # markerless TRACE: unparsed, never fallback-extracted
            # "unknown" is our own escape hatch, so a model that uses it is abstaining and not failing to
            # parse. Scoring that identically to a confidently wrong number discards the signal, so it is
            # counted and reported on its own axis.
            abstain = int(p is None and re.search(r"\bunknown\b", raw or "", re.I) is not None)
            res[f][f"{cond}_correct"].append(int(p is not None and p == it["a"].lower()))
            res[f][f"{cond}_parsed"].append(int(p is not None))
            res[f][f"{cond}_abstain"].append(abstain)

    screen = {}
    sp = a.items.replace(".jsonl", "_screen.json")
    if os.path.exists(sp):
        screen = json.load(open(sp))

    print("\n" + "=" * 108)
    print("GATE 1 (headroom) + BLIND CONTROL")
    print("=" * 108)
    print(f"  {'family':<20}{'n':>6}{'full':>9}{'blind':>9}{'DELTA':>9}"
          f"{'maj base':>10}{'unparsed f/b':>15}   screen verdict")
    summary = {}
    for f in sorted(res):
        r = res[f]
        n = len(r["full_correct"])
        # A condition that was not run is nan, never 0.0: a missing measurement and a measured zero are
        # different facts and must not print the same.
        mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
        fa, ba = mean(r["full_correct"]), mean(r["blind_correct"])
        uf = 1 - mean(r["full_parsed"]); ub = 1 - mean(r["blind_parsed"])
        mb = screen.get(f, {}).get("maj_share", float("nan"))
        sv = screen.get(f, {}).get("verdict", "?")
        summary[f] = dict(n=n, full=fa, blind=ba, delta=fa - ba, maj_base=mb,
                          unparsed_full=uf, unparsed_blind=ub, screen=sv)
        print(f"  {f:<20}{n:>6}{fa:>9.3f}{ba:>9.3f}{fa-ba:>+9.3f}{mb:>10.3f}"
              f"{uf:>7.1%}/{ub:<7.1%}   {sv}")

    keep = [f for f in summary if summary[f]["screen"] == "OK"]
    if keep and "blind" in CONDS:
        F = float(np.mean([summary[f]["full"] for f in keep]))
        B = float(np.mean([summary[f]["blind"] for f in keep]))
        print(f"\n  SURVIVING FAMILIES ONLY ({len(keep)}): full {F:.3f} | blind {B:.3f} "
              f"| delta {F-B:+.3f}")
        print(f"  GATE 1 (headroom to train):   {'PASS' if F < 0.85 else 'FAIL - near ceiling'}"
              f"   (full={F:.3f})")
        print(f"  BLIND CONTROL (reward needs the record): "
              f"{'PASS' if F - B > 0.10 else 'FAIL - reward is a prior'}   (delta={F-B:+.3f})")

    if keep and "blind" not in CONDS:
        F = float(np.mean([summary[f]["full"] for f in keep]))
        print(f"\n  SURVIVING FAMILIES ONLY ({len(keep)}): full {F:.3f} | blind NOT RUN "
              f"(--conds {a.conds}) — no delta reported, by design")
    print("\n  SCREEN-VS-MODEL CHECK (does blind_acc track the CPU majority baseline?)"
          if "blind" in CONDS else "\n  (blind not run — screen-vs-model check skipped)")
    for f in (sorted(summary) if "blind" in CONDS else []):
        s = summary[f]
        if s["maj_base"] == s["maj_base"]:
            d = s["blind"] - s["maj_base"]
            print(f"    {f:<20} blind {s['blind']:.3f} vs maj {s['maj_base']:.3f}  ({d:+.3f})"
                  f"{'   <-- screen mispredicted' if abs(d) > 0.25 else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # genlen, forced, style and max_new are written to the final file as well as to the
    # per-condition checkpoints, so that the artifact can report the forced fraction and the generated
    # length. Existing readers index only ["raw"] and ["summary"].
    json.dump(dict(summary=summary, model=a.model, n=len(items),
                   style=a.style, max_new=a.max_new,
                   require_marker=a.require_marker, marker_above=a.marker_above,
                   force_answer=a.force_answer,
                   genlen=genlens, forced=forceds,
                   raw={c: out[c] for c in out}), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
