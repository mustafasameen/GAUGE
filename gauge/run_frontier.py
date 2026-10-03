#!/usr/bin/env python3
"""Run a hosted model (GPT-4o) over a subset of the questions, with the same prompts and parser.

The prompt is `prompt_full` from the items file, verbatim. It is not rewritten for the API, so the
hosted model answers the same question as the open-weight models. Generation is capped at 24 tokens
at temperature 0, as in the open-weight runs. Items are read from results/questions.jsonl at the
positions listed in an index file (--subset), so the item set is identical to the one the
open-weight runs answered. Scoring happens in score_frontier.py, which imports `norm` from
eval_model, so a prediction becomes an answer by the same rule for every model.

`--peek N` runs N items and prints exactly what came back, without scoring: look at raw output
before trusting any parsed value. The script resumes from --out, so a generation that was paid for
is never requested twice, and it checkpoints every 25 items.

Requires the openai package and an API key in OPENAI_API_KEY (or in a file named by --key-file).

Input: results/questions.jsonl and an index file. The index file is JSON with the key "indices",
a list of positions in the items file. The paper's arm used 30 questions per setting over 32
settings (960 in total).
Output: the JSON file given by --out, with the raw generations keyed by item position.

Usage:
  export OPENAI_API_KEY=...
  python gauge/run_frontier.py --peek 2 --subset results/subset_allspans.json
  python gauge/run_frontier.py --model gpt-4o --subset results/subset_allspans.json
      --out results/frontier_gpt4o.json
"""
from __future__ import annotations
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gauge"))
from eval_model import norm                      # identical parsing to the GPU runs, by import

SUBSET = os.path.join(ROOT, "results", "subset_stratified.json")
ITEMS = os.path.join(ROOT, "results", "questions.jsonl")
# the instrument's own instruction; the GPU runs cap generation at 24 tokens
MAX_TOKENS = 24


def load_items(subset_path, items_path):
    items = [json.loads(l) for l in open(items_path)]
    if os.path.exists(subset_path):
        idx = json.load(open(subset_path))
        idx = idx.get("indices", idx) if isinstance(idx, dict) else idx
        return [(i, items[i]) for i in idx]
    print("NOTE: %s not found; pass an index file with --subset. Falling back to a stratified\n"
          "      draw here would give a DIFFERENT item set than the open-weight runs scored."
          % os.path.relpath(subset_path, ROOT))
    return [(i, it) for i, it in enumerate(items)]


def get_key(a):
    if a.key_file:
        return open(os.path.expanduser(a.key_file)).read().strip()
    k = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not k:
        sys.exit("No API key. Set OPENAI_API_KEY in THIS shell, or pass --key-file <path>.\n"
                 "A key exported in an interactive terminal does not reach this process.")
    return k


def _save(a, out, tin, tout):
    json.dump({"model": a.model, "max_tokens": MAX_TOKENS, "n": len(out),
               "subset": os.path.relpath(a.subset, ROOT),
               "usage": {"input_tokens": tin, "output_tokens": tout,
                         "est_usd": round(tin / 1e6 * 2.50 + tout / 1e6 * 10.00, 2)},
               "raw": out}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--peek", type=int, default=0, help="run N items and print RAW output, no scoring")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--key-file")
    ap.add_argument("--subset", default=SUBSET, help="path to the stratified item-index file")
    ap.add_argument("--out")
    a = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI(api_key=get_key(a))

    pairs = load_items(a.subset, ITEMS)
    if a.peek:
        pairs = pairs[:a.peek]
    elif a.limit:
        pairs = pairs[:a.limit]
    print("model=%s  items=%d  max_tokens=%d" % (a.model, len(pairs), MAX_TOKENS))

    # RESUME: never pay twice for a generation we already have
    out, n_ok = {}, 0
    if a.out and os.path.exists(a.out):
        out = json.load(open(a.out)).get("raw", {})
        print("resuming: %d generations already on disk" % len(out))
    tin = tout = 0
    for k, (i, it) in enumerate(pairs):
        if str(i) in out and not a.peek:
            continue
        try:
            r = client.chat.completions.create(
                model=a.model, max_tokens=MAX_TOKENS, temperature=0,
                messages=[{"role": "user", "content": it["prompt_full"]}])
            raw = r.choices[0].message.content
            tin += r.usage.prompt_tokens; tout += r.usage.completion_tokens
        except Exception as e:
            print("  [%d] API ERROR: %s" % (i, e)); continue
        out[str(i)] = raw
        if a.peek:
            # RAW FIRST. Never judge legibility from a parsed value.
            print("\n--- item %d  family=%s span=%s  gold=%s" % (i, it["family"], it["span"], it.get("mag")))
            print("    RAW  : %r" % raw)
            print("    parsed: %r" % norm(raw, it["family"], it.get("atype")))
        else:
            n_ok += 1
            if k % 50 == 0:
                print("  %d/%d  in=%dk out=%dk  ~$%.2f"
                      % (k, len(pairs), tin//1000, tout//1000, tin/1e6*2.50 + tout/1e6*10.00), flush=True)
        if a.out and not a.peek and len(out) % 25 == 0:
            _save(a, out, tin, tout)          # checkpoint: a kill must never lose paid generations
        time.sleep(0.05)

    if a.out and not a.peek:
        json.dump({"model": a.model, "max_tokens": MAX_TOKENS, "n": len(out),
                   "subset": os.path.relpath(a.subset, ROOT),
                   "usage": {"input_tokens": tin, "output_tokens": tout,
                             "est_usd": round(tin/1e6*2.50 + tout/1e6*10.00, 2)},
                   "raw": out},
                  open(a.out, "w"), indent=1)
        print("wrote %s (%d generations) | in=%d out=%d ~$%.2f"
              % (a.out, len(out), tin, tout, tin/1e6*2.50 + tout/1e6*10.00))


if __name__ == "__main__":
    main()
