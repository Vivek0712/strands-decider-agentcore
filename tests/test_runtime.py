"""The runtime's settings and payload handling, with a fake engine (no model, no torch)."""

import threading

import pytest

from decider_runtime.config import Settings
from decider_runtime.protocol import handle


class FakeEngine:
    def __init__(self, ready=True, error=None):
        self.ready = threading.Event()
        if ready:
            self.ready.set()
        self.error = error
        self.info = {"model": "fake", "state": "ready" if ready else "loading"}
        self.calls = []

    def evaluate(self, req):
        if not self.ready.is_set():
            raise TimeoutError("the model is still loading")
        self.calls.append(req)
        return {"model": "fake", "answers": {k: {"type": v["type"], "noul": 0.9} for k, v in req["questions"].items()},
                "usage": {"input_tokens": 10, "output_tokens": len(req["questions"])}}


S = Settings(max_questions=3, max_batch=2)
Q = {"q": {"type": "noul", "instructions": "Is it?"}}


def test_settings_from_env_and_names():
    s = Settings.from_env({"DECIDER_MODEL": "org/m", "DECIDER_REVISION": "a" * 40, "DECIDER_QUANT": "bf16",
                           "DECIDER_MAX_TOKENS": "2048"})
    assert (s.model, s.quant, s.max_tokens) == ("org/m", "bf16", 2048)
    assert s.display_name == "m@aaaaaaa"
    assert Settings.from_env({}).quant == "int8"
    with pytest.raises(ValueError, match="DECIDER_QUANT"):
        Settings.from_env({"DECIDER_QUANT": "int4"})


def test_single_request_is_answered_with_latency():
    e = FakeEngine()
    out = handle({"state": "x", "questions": Q}, e, S)
    assert out["answers"]["q"]["noul"] == 0.9 and "latency_ms" in out
    assert e.calls == [{"state": "x", "questions": Q}]


def test_extra_fields_are_dropped_before_the_model():
    e = FakeEngine()
    handle({"state": "x", "questions": Q, "secret_debug": 1}, e, S)
    assert "secret_debug" not in e.calls[0]


def test_batch_in_order_and_limits():
    e = FakeEngine()
    out = handle({"batch": [{"state": "a", "questions": Q}, {"state": "b", "questions": Q}]}, e, S)
    assert [c["state"] for c in e.calls] == ["a", "b"] and len(out["results"]) == 2
    out = handle({"batch": [{"state": "a", "questions": Q}] * 3}, e, S)
    assert out["error"]["code"] == "bad_request" and "DECIDER_MAX_BATCH" in out["error"]["message"]


@pytest.mark.parametrize("payload, fragment", [
    ("not an object", "JSON object"),
    ({"state": "x"}, "needs `state` and `questions`"),
    ({"state": "x", "questions": {}}, "non-empty"),
    ({"state": "x", "questions": {f"q{i}": Q["q"] for i in range(4)}}, "at most 3"),
    ({"state": "x", "questions": Q, "images": ["data:image/png;base64,AA"]}, "image input is not enabled"),
    ({"batch": []}, "non-empty list"),
])
def test_bad_requests_return_errors_not_exceptions(payload, fragment):
    out = handle(payload, FakeEngine(), S)
    assert out["error"]["code"] == "bad_request" and fragment in out["error"]["message"]


def test_loading_and_health():
    e = FakeEngine(ready=False)
    out = handle({"state": "x", "questions": Q}, e, S)
    assert out["error"]["code"] == "loading"
    h = handle({"action": "health"}, e, S)
    assert h["ready"] is False and h["health"]["state"] == "loading" and "memory_now_gb" in h["health"]


def test_model_errors_are_reported():
    class Broken(FakeEngine):
        def evaluate(self, req):
            raise RuntimeError("boom")

    out = handle({"state": "x", "questions": Q}, Broken(), S)
    assert out["error"]["code"] == "inference_error" and "boom" in out["error"]["message"]
