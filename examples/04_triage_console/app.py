"""Ticket triage console: a Strands Decider on AgentCore Runtime routes support tickets.

    python examples/04_triage_console/app.py --arn <decider runtime ARN>      # deployed decider
    python examples/04_triage_console/app.py --url http://localhost:8080      # local container
    open http://127.0.0.1:8501

Each ticket gets one request with three typed questions (escalate? which queue? how urgent?).
The console acts on an answer only when the decider is confident, and sends the rest to a person:
that is the point of calibrated probabilities.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from client.decider_client import DeciderClient, choice, noul, score  # noqa: E402

QUEUES = {"returns": "refunds, returns and damaged items", "billing": "payments, invoices and charges",
          "shipping": "late or lost deliveries and address changes", "account": "login, password and app problems",
          "legal": "legal threats, injuries and compliance", "general": "thanks, feedback and anything else"}
QUESTIONS = {
    "escalate": noul("Does this ticket need a human agent?",
                     true="legal threats, injuries, safety, angry repeat complaints, money disputes, or anything a "
                          "canned reply cannot settle",
                     false="routine how-to questions, status checks, thanks and feedback that a standard reply "
                           "can handle"),
    "queue": choice("Which team should handle this ticket?", QUEUES),
    "urgency": score("How urgent is this ticket?", ["low: can wait a week", "medium: answer within two days",
                                                    "high: answer today"]),
}
# The policy: act on an answer only when the decider is confident. Pick the thresholds on your own
# tickets: they trade how much is automated against how often a person has to look.
ESCALATE_YES, ESCALATE_NO, QUEUE_MIN = 0.75, 0.25, 0.6


def route(answers: dict) -> dict:
    p = answers["escalate"]["noul"]
    q = answers["queue"]
    u = answers["urgency"]["probabilities"]
    lvl = int(max(u, key=u.get))  # the most likely level
    handler = "human agent" if p >= ESCALATE_YES else "automated reply" if p <= ESCALATE_NO else "human review"
    team = q["choice"] if q["probabilities"][q["choice"]] >= QUEUE_MIN else "triage desk"
    sla = ("1 week", "2 days", "today")[lvl]
    confident = handler != "human review" and team != "triage desk"
    return {"handler": handler, "team": team, "sla": sla, "auto": confident}


def create_app(client: DeciderClient, source: str) -> FastAPI:
    app = FastAPI(title="Decider triage console")
    tickets = json.loads((ROOT / "tickets.json").read_text())

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(ROOT / "index.html")

    @app.get("/api/tickets")
    def list_tickets() -> dict:
        return {"tickets": tickets, "source": source, "policy": {"escalate_yes": ESCALATE_YES,
                "escalate_no": ESCALATE_NO, "queue_min": QUEUE_MIN}}

    @app.get("/api/health")
    def health() -> dict:
        return client.health()

    @app.post("/api/triage/{ticket_id}")
    def triage(ticket_id: str) -> dict:
        t = next((x for x in tickets if x["id"] == ticket_id), None)
        if t is None:
            raise HTTPException(404, "no such ticket")
        t0 = time.time()
        r = client.decide(f"Channel: {t['channel']}\nCustomer message: {t['text']}", QUESTIONS)
        return {"id": ticket_id, "answers": r["answers"], "model": r.get("model"),
                "latency_s": round(time.time() - t0, 2), "route": route(r["answers"])}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--arn")
    g.add_argument("--url")
    ap.add_argument("--profile")
    ap.add_argument("--port", type=int, default=8501)
    a = ap.parse_args()
    client = DeciderClient(arn=a.arn, profile=a.profile, session_prefix="triage-console") if a.arn \
        else DeciderClient(url=a.url)
    source = "AgentCore Runtime: " + a.arn.split("/")[-1] if a.arn else "local container: " + a.url
    uvicorn.run(create_app(client, source), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
