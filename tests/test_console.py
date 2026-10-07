import importlib.util
import pathlib

import pytest

pytest.importorskip("fastapi")
spec = importlib.util.spec_from_file_location(
    "console", pathlib.Path(__file__).parent.parent / "examples" / "04_triage_console" / "app.py")
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)


def answers(p_escalate, queue, p_queue, urgency):
    probs = {k: (1 - p_queue) / (len(console.QUEUES) - 1) for k in console.QUEUES}
    probs[queue] = p_queue
    return {"escalate": {"noul": p_escalate}, "queue": {"choice": queue, "probabilities": probs},
            "urgency": {"probabilities": {str(i): (0.8 if i == urgency else 0.1) for i in range(3)}}}


def test_confident_escalation_routes_to_team():
    r = console.route(answers(0.9, "legal", 0.92, 2))
    assert r == {"handler": "human agent", "team": "legal", "sla": "today", "auto": True}


def test_confident_automation():
    r = console.route(answers(0.1, "general", 0.7, 0))
    assert r["handler"] == "automated reply" and r["auto"] and r["sla"] == "1 week"


def test_uncertain_goes_to_review():
    assert not console.route(answers(0.5, "billing", 0.9, 1))["auto"]
    r = console.route(answers(0.9, "billing", 0.4, 1))
    assert r["team"] == "triage desk" and not r["auto"]


def test_app_serves_tickets_and_triage():
    from fastapi.testclient import TestClient

    class FakeClient:
        def decide(self, state, questions):
            assert set(questions) == {"escalate", "queue", "urgency"} and "Customer message" in state
            return {"model": "fake", "answers": answers(0.1, "general", 0.9, 0)}

        def health(self):
            return {"ready": True}

    c = TestClient(console.create_app(FakeClient(), "test"))
    tickets = c.get("/api/tickets").json()["tickets"]
    r = c.post(f"/api/triage/{tickets[0]['id']}").json()
    assert r["route"]["auto"] and r["model"] == "fake"
    assert c.post("/api/triage/nope").status_code == 404
    assert "triage console" in c.get("/").text
