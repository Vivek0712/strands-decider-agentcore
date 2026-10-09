"""Automatic graders, one per answer type. Each returns a score in [0, 1] (1 = correct).

This file is self-contained (standard library only) so the same code runs locally and inside the
AWS Lambda evaluator that Bedrock Advanced Prompt Optimization calls.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.@+\- ]", "", s.lower()).strip().rstrip(".")


def extract(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return 1.0
    return 1.0 if g and g in p and len(p) <= len(g) + 25 else 0.0  # the value, maybe with a short lead-in


def yesno(pred: str, gold: str) -> float:
    words = re.findall(r"\b(yes|no)\b", pred.lower())
    return 1.0 if words and words[-1] == gold.lower().strip() else 0.0


def _last_number(s: str) -> str | None:
    m = re.search(r"answer\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+)", s, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", s)
    return nums[-1].replace(",", "") if nums else None


def math(pred: str, gold: str) -> float:
    n = _last_number(pred)
    try:
        return 1.0 if n is not None and abs(float(n) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def choice(pred: str, gold: str) -> float:
    m = re.findall(r"answer\s*:?\s*\(?([A-R])\)", pred, re.I) or re.findall(r"\(([A-R])\)", pred)
    return 1.0 if m and f"({m[-1].upper()})" == gold.strip() else 0.0


def _json_obj(s: str) -> dict | None:
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    i, j = s.find("{"), s.rfind("}")
    try:
        obj = json.loads(s[i : j + 1]) if i >= 0 and j > i else None
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def json_fields(pred: str, gold: str) -> float:
    """Fraction of the 5 fields that match exactly (after trimming and lower-casing); 0 if not JSON."""
    obj, ref = _json_obj(pred), json.loads(gold)
    if obj is None:
        return 0.0

    def same(a: object, b: object) -> bool:
        if a in (None, "", "null") and b is None:
            return True
        return isinstance(a, str) and isinstance(b, str) and a.strip().lower() == b.strip().lower()

    return sum(same(obj.get(k), v) for k, v in ref.items()) / len(ref)


def _code_block(s: str) -> str:
    m = re.findall(r"```(?:python)?\n(.*?)```", s, re.S)
    return max(m, key=len) if m else s


def code(pred: str, gold: str, timeout_s: float = 10) -> float:
    """Fraction of the MBPP asserts that pass, run in a separate Python process with a timeout."""
    spec = json.loads(gold)
    tests = spec["tests"]
    prog = "\n".join(spec.get("imports", [])) + "\n" + _code_block(pred) + "\n"
    passed = 0
    for t in tests:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog + t + "\n")
        try:
            r = subprocess.run(
                [sys.executable, "-I", f.name],
                capture_output=True,
                timeout=timeout_s,
                cwd=tempfile.gettempdir(),
                env={"PATH": os.environ.get("PATH", "")},
            )
            passed += r.returncode == 0
        except subprocess.TimeoutExpired:
            pass
        finally:
            os.unlink(f.name)
    return passed / len(tests)


GRADERS = {"extract": extract, "yesno": yesno, "math": math, "choice": choice, "json": json_fields, "code": code}


def grade(grader: str, pred: str, gold: str) -> float:
    try:
        return float(GRADERS[grader](pred or "", gold))
    except Exception:  # noqa: BLE001 - a broken answer scores 0, never crashes a run
        return 0.0
