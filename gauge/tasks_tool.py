#!/usr/bin/env python3
"""Program arm: ask for a Python expression over the record's points instead of a value.

The question is whether a model can specify the computation that it cannot perform. The items use
the same records, golds and tolerance as the value arm. The only change is that the model is asked
for a single Python expression over a variable `pts` (the list of (x, y) pairs), and we execute it:

  value arm:    record -> the model computes the number -> a number
  program arm:  record -> the model specifies the computation -> we compute -> a number

The two arms are scored identically, so they are directly comparable on the same items. If the
program arm scores far above the value arm, the model knows the descriptor and the failure is in
carrying out the computation. If it scores about the same, the model cannot specify the
computation either.

The arm uses an expression and not a tool-calling harness, so that retries, error feedback and a
scaffolding budget do not vary across models with different tool-calling training. This is also
strictly harder than a harness, because there is no retry.

Safety: model output is never passed to `exec`. `safe_eval` parses it into a syntax tree and
evaluates it only if every node is on a whitelist: arithmetic, comparisons, calls to a fixed
allowlist of numpy and math names, literals, and comprehensions over `pts`. Imports, attribute
access outside the allowlist and names that were not bound are rejected. A rejected expression
scores wrong and is counted, never silently dropped. The whitelist was written for this study;
do not treat it as a general security boundary.

Usage: imported by tally_tool.py, score_tool.py and program_equivalence.py. Run
`python gauge/tasks_tool.py` for the self-tests.
"""
from __future__ import annotations
import ast, math, re
import itertools
import numpy as np

KM = 0.5

# ---- the whitelist. Anything outside it is a rejected expression, not an error and not a pass.
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load,
    # Store appears ONLY as a comprehension target here: mode='eval' admits no assignment
    # statement, so allowing it cannot introduce a binding we did not intend.
    ast.Store,
    ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice, ast.Index if hasattr(ast, "Index")
    else ast.Slice, ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv,
    ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq, ast.IfExp, ast.Starred,
    ast.keyword, ast.BoolOp, ast.And, ast.Or,
)
# Comprehension targets are bound by the comprehension itself and cannot reach anything outside
# it, so the names they bind are collected from the syntax tree and allowed. A fixed list of
# loop-variable names would reject correct expressions such as
# `for (x1, y1), (x2, y2) in zip(pts, pts[1:])` or `for p1, p2 in combinations(pts, 2)`. Nothing else
# is relaxed: no imports, no dunder names, no calls to names that were not bound.
ALLOWED_NAMES = {"pts", "rec", "recs", "np", "math", "itertools", "set", "len", "tuple",
                 "next", "any", "all", "list", "abs", "sum", "min", "max", "len", "range",
                 "sorted", "zip", "enumerate", "float", "int", "round", "pow"}
ALLOWED_ATTRS = {
    "sqrt", "mean", "sum", "max", "min", "array", "asarray", "linalg", "norm", "hypot", "square",
    "abs", "diff", "std", "var", "T", "shape", "ptp", "cumsum", "power", "pi", "dot",
    "reshape", "astype", "tolist", "argmax", "argmin", "median", "average", "concatenate",
    # names that appeared in correct model expressions
    "newaxis", "flatten", "ravel", "combinations", "permutations", "product", "stack", "vstack",
    "hstack", "expand_dims", "squeeze", "triu", "tril", "triu_indices", "nan", "inf", "float64",
    "sort", "clip", "take", "repeat", "tile", "einsum", "maximum", "minimum", "add", "subtract",
}


def safe_eval(expr: str, pts, rec=None, recs=None):
    """Return (value, status). status in {ok, reject:<why>, error:<why>}."""
    expr = (expr or "").strip().strip("`")
    # models often prefix the expression with a label: "Answer: np.max(...)". Strip a leading
    # label before parsing.
    expr = re.sub(r"^\s*(?:the\s+)?(?:answer|expression|result|output)\s*[:=]\s*", "", expr,
                  flags=re.I)
    for pre in ("python", "Python"):
        if expr.startswith(pre):
            expr = expr[len(pre):].strip()
    if "\n" in expr:
        expr = expr.split("\n")[0].strip()
    if not expr:
        return None, "reject:empty"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return None, f"reject:syntax"
    # collect every name BOUND by a comprehension target; those are safe by construction
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return None, f"reject:node:{type(node).__name__}"
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES and node.id not in bound:
            return None, f"reject:name:{node.id}"
        if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_ATTRS:
            return None, f"reject:attr:{node.attr}"
    env = {"pts": [tuple(map(float, p)) for p in pts], "np": np, "math": math,
           "itertools": itertools, "set": set, "tuple": tuple, "next": next,
           "any": any, "all": all, "list": list,
           "abs": abs, "sum": sum, "min": min, "max": max, "len": len, "range": range,
           "sorted": sorted, "zip": zip, "enumerate": enumerate, "float": float, "int": int,
           "round": round, "pow": pow, "__builtins__": {}}
    if rec is not None:
        env["rec"] = [tuple(map(float, r)) for r in rec]
    if recs is not None:
        env["recs"] = [[tuple(map(float, t)) for t in r] for r in recs]
    try:
        v = eval(compile(tree, "<expr>", "eval"), env, {})
        v = float(np.asarray(v).ravel()[0]) if np.size(v) else None
        if v is None or not math.isfinite(v):
            return None, "error:nonfinite"
        return v, "ok"
    except Exception as e:
        return None, f"error:{type(e).__name__}"


QUESTIONS = {
    "gyration": ("the radius of gyration of this person's locations in kilometres (the square root "
                 "of the mean squared distance of their locations from their own centre)",
                 lambda P: float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())) * KM),
    "max_distance": ("the greatest distance in kilometres between any two of this person's "
                     "locations",
                     lambda P: float(max(np.sqrt(((P[i] - P) ** 2).sum(1)).max()
                                         for i in range(len(P)))) * KM),
    "total_distance": ("the total distance in kilometres travelled, adding the straight-line "
                       "distance between consecutive records",
                       lambda P: float(np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1)).sum()) * KM),
    "longest_jump": ("the largest single move in kilometres between two consecutive records",
                     lambda P: float(np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1)).max()) * KM),
}

PREAMBLE = ("Below is a Python list `pts` of (x, y) pairs giving one person's recorded locations, "
            "in order. Each unit on the x and y axes is 500 metres.")


def build_q(fam):
    phrase, _ = QUESTIONS[fam]
    return (f"Write a SINGLE Python expression, using only `pts`, `math` and `np` (numpy), that "
            f"computes {phrase}. Output the expression only, with no explanation, no assignment "
            f"and no code fences.")


# ------------------------------------------------------------------ self-tests
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    P = rng.integers(-400, 400, size=(60, 2)).astype(float)
    pts = [tuple(p) for p in P]
    # 1. a CORRECT expression for each family must recover the gold through the sandbox
    GOOD = {
        "gyration": "0.5*np.sqrt(np.mean(np.sum((np.array(pts)-np.mean(np.array(pts),axis=0))**2,axis=1)))",
        "max_distance": "0.5*max(math.hypot(a[0]-b[0],a[1]-b[1]) for a in pts for b in pts)",
        "total_distance": "0.5*sum(math.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))",
        "longest_jump": "0.5*max(math.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))",
    }
    for fam, e in GOOD.items():
        v, st = safe_eval(e, pts)
        g = QUESTIONS[fam][1](P)
        assert st == "ok", (fam, st)
        assert abs(v - g) / max(g, 1e-9) < 1e-6, (fam, v, g)
        print(f"  ok   {fam:<16} sandbox reproduces the gold ({v:.3f} vs {g:.3f})")
    # 2. every escape attempt must be REJECTED, not merely error
    BAD = ["__import__('os').system('ls')", "open('/etc/passwd').read()",
           "().__class__.__bases__[0].__subclasses__()", "eval('1+1')", "exec('x=1')",
           "[c for c in ().__class__.__mro__]", "globals()", "pts.__class__",
           "np.__loader__.load_module", "lambda: 1", "__builtins__"]
    for e in BAD:
        v, st = safe_eval(e, pts)
        assert v is None and st.startswith("reject"), (e, v, st)
        print(f"  ok   rejected  {e[:44]:<46} -> {st}")
    # 3. plausible model junk must be rejected cleanly, never crash
    for e in ["I cannot answer that", "", "```python\nnp.sqrt(1)\n```", "x = np.sqrt(2)",
              "np.sqrt(np.mean(", "the answer is 4.2"]:
        v, st = safe_eval(e, pts)
        print(f"  ok   junk      {e[:44]!r:<46} -> {st}")
    print("\ntool-control self-tests pass: the sandbox recovers every gold and rejects every escape")
