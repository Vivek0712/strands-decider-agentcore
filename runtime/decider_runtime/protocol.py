"""The invocation payloads this runtime accepts, and the answers it returns.

    {"state": ..., "questions": {...}, "images": [...]}   one System One request (strands-decider's API)
    {"batch": [request, request, ...]}                     several requests, answered in order
    {"action": "health"}                                   model, load timings and memory; no inference

Questions use strands-decider's three types: `noul` (yes/no, returns P(yes)), `choice` (pick one of
named options, returns a probability per option) and `score` (an ordered rubric, returns the
expected level). Every answer carries calibrated probabilities, so a caller can act only when the
model is confident and defer otherwise.
"""

from __future__ import annotations

import time
from typing import Any

from .config import Settings


class BadRequest(ValueError):
    pass


def error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, **extra}}


def check_request(req: Any, settings: Settings, where: str = "request") -> dict[str, Any]:
    if not isinstance(req, dict):
        raise BadRequest(f"{where} must be a JSON object")
    if "state" not in req or "questions" not in req:
        raise BadRequest(f"{where} needs `state` and `questions` (see the README for the format)")
    qs = req["questions"]
    if not isinstance(qs, dict) or not qs:
        raise BadRequest(f"{where}.questions must be a non-empty object of name -> question")
    if len(qs) > settings.max_questions:
        raise BadRequest(f"{where} has {len(qs)} questions; this runtime answers at most "
                         f"{settings.max_questions} per request (DECIDER_MAX_QUESTIONS)")
    if req.get("images"):
        raise BadRequest("image input is not enabled on this runtime (text-only CPU deployment)")
    return {k: v for k, v in req.items() if k in ("state", "questions", "model")}


def handle(payload: Any, engine: Any, settings: Settings) -> dict[str, Any]:
    """Route one invocation payload; never raises (errors come back as {"error": ...})."""
    if not isinstance(payload, dict):
        return error("bad_request", "the payload must be a JSON object")
    if payload.get("action") == "health":
        from .engine import memory_now, rss_gb

        live = {"memory_now_gb": memory_now(), "peak_rss_gb": rss_gb()}
        return {"health": {**engine.info, **live}, "ready": engine.ready.is_set() and not engine.error}
    try:
        if "batch" in payload:
            reqs = payload["batch"]
            if not isinstance(reqs, list) or not reqs:
                raise BadRequest("batch must be a non-empty list of requests")
            if len(reqs) > settings.max_batch:
                raise BadRequest(f"batch has {len(reqs)} requests; at most {settings.max_batch} (DECIDER_MAX_BATCH)")
            checked = [check_request(r, settings, f"batch[{i}]") for i, r in enumerate(reqs)]
            t = time.perf_counter()
            results = [engine.evaluate(r) for r in checked]
            return {"results": results, "latency_ms": round((time.perf_counter() - t) * 1000)}
        req = check_request(payload, settings)
        t = time.perf_counter()
        out = engine.evaluate(req)
        out["latency_ms"] = round((time.perf_counter() - t) * 1000)
        return out
    except BadRequest as e:
        return error("bad_request", str(e))
    except TimeoutError as e:
        return error("loading", str(e), state=engine.info.get("state"))
    except Exception as e:  # noqa: BLE001 - a validation error from the schema, or a model failure
        name = type(e).__name__
        return error("invalid_request" if name == "ValidationError" else "inference_error", f"{name}: {e}")
