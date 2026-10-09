"""
Exact graders for the model-routing benchmark, as an evaluator for Bedrock Advanced Prompt Optimization.
Handler: lambda_function.lambda_handler

Bedrock validates this file before the job runs: it allows only a short list of imports (it rejected
os, subprocess, sys, tempfile and __future__) and rejects the builtins compile() and exec(). So this
file uses json, re and logging only, and code answers get a static shape check instead of their tests.
"""

import json
import logging
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _norm(s):
    return re.sub(r"[^a-z0-9.@+\- ]", "", s.lower()).strip().rstrip(".")


def score_value(pred, gold):
    """1.0 when the answer is the value (a few extra words allowed), else 0.0."""
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return 1.0
    return 1.0 if g and g in p and len(p) <= len(g) + 25 else 0.0


def score_yesno(pred, gold):
    """1.0 when the last yes/no word in the answer matches the reference, else 0.0."""
    words = re.findall(r"\b(yes|no)\b", pred.lower())
    return 1.0 if words and words[-1] == gold.lower().strip() else 0.0


def score_number(pred, gold):
    """1.0 when the final 'Answer: <number>' (or else the last number) equals the reference."""
    m = re.search(r"answer\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+)", pred, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", pred)
    if not nums:
        return 0.0
    try:
        return 1.0 if abs(float(nums[-1].replace(",", "")) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def score_option(pred, gold):
    """1.0 when the final 'Answer: (X)' option letter matches the reference option, else 0.0."""
    m = re.findall(r"answer\s*:?\s*\(?([A-R])\)", pred, re.I) or re.findall(r"\(([A-R])\)", pred)
    return 1.0 if m and "(" + m[-1].upper() + ")" == gold.strip() else 0.0


def score_json(pred, gold):
    """Fraction of the reference JSON fields the answer reproduces exactly; 0.0 if it is not one JSON object."""
    s = re.sub(r"^```(?:json)?|```$", "", pred.strip(), flags=re.M).strip()
    i, j = s.find("{"), s.rfind("}")
    try:
        obj = json.loads(s[i : j + 1]) if i >= 0 and j > i else None
    except ValueError:
        return 0.0
    if not isinstance(obj, dict):
        return 0.0
    ref = json.loads(gold)
    hits = 0
    for k, v in ref.items():
        a = obj.get(k)
        if v is None:
            hits += a in (None, "", "null")
        else:
            hits += isinstance(a, str) and a.strip().lower() == v.strip().lower()
    return hits / len(ref)


def score_code(pred, gold):
    """A static check of a Python answer (Bedrock does not allow running code in this evaluator).

    The reference is a working solution, a line '# tests', then asserts such as `assert f(a, b) == c`.
    From the first assert we take the function name and its number of arguments. The answer's largest
    ```python block earns:
    - 0.4 for defining a function with that exact name,
    - 0.3 more when that function takes the same number of arguments as the example call,
    - 0.3 more when its body has a return statement and no placeholder (pass, ..., TODO).
    1.0 means the answer has the right shape; the benchmark runs the real tests separately.
    """
    tests = [t for t in gold.split("\n# tests\n", 1)[1].splitlines() if t.strip().startswith("assert")]
    m = (
        re.match(r"\s*assert\s+(?:not\s+)?(?:\(?\s*)?(?:set\(|math\.isclose\(|abs\()?\s*([A-Za-z_]\w*)\((.*)", tests[0])
        if tests
        else None
    )
    if not m:
        return 0.0
    name, rest = m.group(1), m.group(2)
    depth, args, cur = 1, 0, ""
    for ch in rest:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                break
        if ch == "," and depth == 1:
            args += 1
        cur += ch
    nargs = args + (1 if cur.strip() else 0)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", pred, re.S)
    src = max(blocks, key=len) if blocks else pred
    d = re.search(r"^def\s+" + re.escape(name) + r"\s*\(([^)]*)\)", src, re.M)
    if not d:
        return 0.0
    score = 0.4
    params = [p for p in d.group(1).split(",") if p.strip() and not p.strip().startswith("*")]
    if len(params) == nargs:
        score += 0.3
    body = src[d.end() :]
    if re.search(r"\breturn\b", body) and not re.search(r"^\s*(pass|\.\.\.)\s*$|TODO", body, re.M):
        score += 0.3
    return round(score, 2)


def kind_of(gold):
    g = gold.strip()
    if "\n# tests\n" in g:
        return "code"
    if g.startswith("{"):
        return "json"
    if g.startswith("(") and g.endswith(")") and len(g) <= 4:
        return "option"
    if g.lower() in ("yes", "no"):
        return "yesno"
    try:
        float(g)
        return "number"
    except ValueError:
        return "value"


SCORERS = {
    "code": score_code,
    "json": score_json,
    "option": score_option,
    "yesno": score_yesno,
    "number": score_number,
    "value": score_value,
}


def compute_score(preds, golds):
    """Score each model answer against its reference with an exact, task-specific check (higher is better).

    The reference decides the check:
    - Python code, a line '# tests', then asserts: fraction of the asserts the answer's code passes.
    - A JSON object: fraction of its fields the answer's JSON object reproduces exactly.
    - An option letter like (C): 1 if the answer ends with 'Answer: (C)'.
    - yes or no: 1 if the answer's last yes/no matches.
    - A number: 1 if the answer's final 'Answer: <number>' equals it.
    - Any other value: 1 if the answer is just that value.
    Answers in exactly the requested format score best; extra prose can break the format checks.
    """
    scores, details = [], []
    for pred, gold in zip(preds, golds, strict=False):
        kind = kind_of(gold)
        try:
            s = float(SCORERS[kind](pred or "", gold))
        except Exception as e:
            s = 0.0
            kind = kind + " error: " + str(e)[:80]
        scores.append(s)
        details.append(kind if s >= 1 else kind + ": did not fully match the reference")
    return {"score": sum(scores) / len(scores) if scores else 0.0, "scores": scores, "details": details}


def lambda_handler(event, context):
    try:
        preds, golds = event.get("preds", []), event.get("golds", [])
        if not preds:
            return {"score": 0.0, "scores": []}
        return compute_score(preds, golds)
    except Exception as e:
        logger.error("evaluator error: %s", e)
        return {"score": 0.0, "scores": [0.0] * len(event.get("preds", [])), "error": str(e)}
