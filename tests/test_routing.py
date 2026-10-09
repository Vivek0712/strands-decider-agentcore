"""Unit tests for the model-routing use case: graders, routing policy, the APO evaluator, cost maths."""

import json
import pathlib
import sys

import pytest

UC = pathlib.Path(__file__).resolve().parent.parent / "usecases" / "model-routing"
sys.path.insert(0, str(UC / "bench"))
sys.path.insert(0, str(UC / "apo"))

graders = pytest.importorskip("graders")
import lambda_function as apo_eval  # noqa: E402

GOLD_JSON = json.dumps({"name": "Priya Khan", "email": None, "phone": "+1 972 501 9948", "city": "Lisbon",
                        "company": "Northwind"})
CODE_GOLD = json.dumps({"tests": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"], "imports": []})


@pytest.mark.parametrize("grader,pred,gold,score", [
    ("extract", "Lena Haddad", "Lena Haddad", 1.0),
    ("extract", "The order was placed by Lena Haddad.", "Lena Haddad", 1.0),
    ("extract", "Hannah Costa", "Lena Haddad", 0.0),
    ("yesno", "No.\n\nThe player is a footballer.", "no", 1.0),
    ("yesno", "yes", "no", 0.0),
    ("math", "3 x 4 = 12\nAnswer: 12", "12", 1.0),
    ("math", "Answer: 1,250", "1250", 1.0),
    ("math", "Answer: 13", "12", 0.0),
    ("choice", "step by step...\nAnswer: (C)", "(C)", 1.0),
    ("choice", "Answer: (B)", "(C)", 0.0),
    ("json", '```json\n{"name": "Priya Khan", "email": null, "phone": "+1 972 501 9948", "city": "Lisbon", '
             '"company": "Northwind"}\n```', GOLD_JSON, 1.0),
    ("json", '{"name": "Priya Khan", "email": null, "phone": null, "city": "Lisbon", "company": "Northwind"}',
     GOLD_JSON, 0.8),
    ("json", "not json", GOLD_JSON, 0.0),
    ("code", "```python\ndef add(a, b):\n    return a + b\n```", CODE_GOLD, 1.0),
    ("code", "```python\ndef add(a, b):\n    return a + b if a > 0 else 99\n```", CODE_GOLD, 0.5),
    ("code", "```python\ndef add(a, b):\n    while True: pass\n```", CODE_GOLD, 0.0),
])
def test_graders(grader, pred, gold, score):
    timeout = {"timeout_s": 2} if grader == "code" else {}
    got = graders.GRADERS[grader](pred, gold, **timeout) if timeout else graders.grade(grader, pred, gold)
    assert got == pytest.approx(score)


def test_grade_never_raises():
    assert graders.grade("json", "{broken", "{also broken") == 0.0


def test_decider_policy_is_monotonic_in_threshold():
    from evaluate import decider_policy

    probs = {"micro": 0.3, "lite": 0.5, "pro": 0.2}
    assert decider_policy(probs, 0.2) == "micro"
    assert decider_policy(probs, 0.5) == "lite"  # 0.3 + 0.5 >= 0.5
    assert decider_policy(probs, 0.9) == "pro"
    order = ["micro", "lite", "pro"]
    picks = [order.index(decider_policy(probs, t / 20)) for t in range(21)]
    assert picks == sorted(picks), "a higher threshold must never pick a cheaper tier"


def test_apo_evaluator_matches_local_graders_except_code():
    """The Lambda evaluator Bedrock calls must agree with the benchmark graders (code is a static check there)."""
    cases = [("Lena Haddad", "Lena Haddad"), ("no", "no"), ("Answer: 12", "12"), ("Answer: (C)", "(B)"),
             ('{"name": "Priya Khan", "email": null, "phone": "+1 972 501 9948", "city": "Lisbon", '
              '"company": "Northwind"}', GOLD_JSON)]
    kinds = ["extract", "yesno", "math", "choice", "json"]
    out = apo_eval.compute_score([p for p, _ in cases], [g for _, g in cases])
    local = [graders.grade(k, p, g) for k, (p, g) in zip(kinds, cases, strict=True)]
    assert out["scores"] == pytest.approx(local)
    assert out["score"] == pytest.approx(sum(local) / len(local))


def test_apo_evaluator_uses_only_allowed_constructs():
    """Bedrock rejects evaluators that import os/subprocess/sys/tempfile/__future__ or call exec/compile."""
    import ast

    tree = ast.parse((UC / "apo" / "lambda_function.py").read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert imported <= {"json", "re", "logging"}, imported
    assert not called & {"exec", "compile", "eval", "__import__", "open"}, called & {"exec", "compile", "eval"}


def test_apo_static_code_check():
    gold = "def add(a, b):\n    return a + b\n# tests\nassert add(2, 3) == 5"
    assert apo_eval.score_code("```python\ndef add(x, y):\n    return x + y\n```", gold) == pytest.approx(1.0)
    assert apo_eval.score_code("```python\ndef add(x):\n    return x\n```", gold) == pytest.approx(0.7)
    assert apo_eval.score_code("```python\ndef other(x, y):\n    return x\n```", gold) == 0.0


def test_score_strategy_counts_router_cost():
    from evaluate import score_strategy

    ans = {"t1": {m: {"score": 1.0, "cost": c, "latency_s": 1.0} for m, c in (("micro", 0.001), ("lite", 0.002),
                                                                           ("pro", 0.01))}}
    plain = score_strategy("x", {"t1": "micro"}, ans, ["t1"])
    routed = score_strategy("x", {"t1": "micro"}, ans, ["t1"], router={"t1": {"cost": 0.005, "latency_s": 2.0}})
    assert plain["cost_per_1k"] == pytest.approx(1.0)
    assert routed["cost_per_1k"] == pytest.approx(6.0)
    assert routed["router_cost_per_1k"] == pytest.approx(5.0)
    assert routed["latency_p50"] == pytest.approx(3.0)
    assert routed["oracle_match"] == 1.0
