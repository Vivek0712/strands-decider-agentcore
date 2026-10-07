"""A small client for a Strands Decider running on Amazon Bedrock AgentCore Runtime.

    from client.decider_client import DeciderClient, noul, choice, score

    decider = DeciderClient(arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/deciders_triage-AbCd")
    r = decider.decide("Order #4411 arrived broken, I want a refund.", {
        "escalate": noul("Should this ticket go to a human agent?"),
        "queue": choice("Which queue?", {"returns": "refunds and returns", "billing": "payments"}),
        "urgency": score("How urgent?", ["low", "medium", "high"]),
    })
    r["answers"]["escalate"]["noul"]      # P(yes), calibrated

How AgentCore sessions map to capacity: every runtime session id gets its own microVM
(2 vCPU / 8 GB) that stays warm until it is idle for the runtime's idle timeout. The model loads
once per session, so this client reuses a small pool of session ids: `pool=1` (the default) keeps
one warm replica; `pool=N` spreads calls over N warm replicas for N times the throughput.

Also works against a local container (`url="http://localhost:8080"`), for development and tests.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.request
import uuid
from typing import Any


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    """A yes/no question; the answer is P(yes)."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def choice(instructions: str, options: dict[str, str]) -> dict[str, Any]:
    """Pick one of named options ({name: description}); the answer has a probability per option."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    """Rate on an ordered rubric, low to high; the answer is the expected level and its distribution."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


class DeciderError(RuntimeError):
    """The runtime answered with an error (bad request, still loading, inference failure)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class DeciderClient:
    def __init__(self, arn: str | None = None, *, url: str | None = None, region: str | None = None,
                 profile: str | None = None, pool: int = 1, session_prefix: str = "decider",
                 qualifier: str | None = None, timeout_s: float = 900, retries: int = 3) -> None:
        if bool(arn) == bool(url):
            raise ValueError("give exactly one of arn= (AgentCore) or url= (a local container)")
        self.arn, self.url, self.qualifier = arn, url.rstrip("/") if url else None, qualifier
        self.timeout_s, self.retries = timeout_s, retries
        # AgentCore session ids must be at least 33 characters
        self.sessions = [f"{session_prefix}-{i}-{uuid.uuid4()}" for i in range(max(1, pool))]
        self._next = itertools.cycle(self.sessions)
        self._lock = threading.Lock()
        self._client = None
        if arn:
            import boto3
            from botocore.config import Config

            region = region or arn.split(":")[3]
            session = boto3.Session(profile_name=profile, region_name=region)
            # a cold session downloads and loads the model before it answers: allow minutes, not seconds
            self._client = session.client("bedrock-agentcore", config=Config(
                read_timeout=int(timeout_s), connect_timeout=30, retries={"max_attempts": 2}))

    # ---- calls ----------------------------------------------------------------------------------

    def invoke(self, payload: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        """Send one raw payload (see the README for formats) and return the JSON answer."""
        with self._lock:
            sid = session_id or next(self._next)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                out = self._invoke_once(payload, sid)
            except Exception as e:  # noqa: BLE001 - network or throttling: retry with backoff
                last = e
            else:
                err = out.get("error") if isinstance(out, dict) else None
                if not err:
                    return out
                if err.get("code") != "loading":
                    raise DeciderError(err.get("code", "error"), err.get("message", ""))
                last = DeciderError("loading", err.get("message", "the model is loading"))
            time.sleep(min(2 ** attempt * 5, 60))
        raise last if last else RuntimeError("no answer")

    def _invoke_once(self, payload: dict[str, Any], sid: str) -> dict[str, Any]:
        body = json.dumps(payload).encode()
        if self.url:
            req = urllib.request.Request(f"{self.url}/invocations", data=body,
                                         headers={"Content-Type": "application/json",
                                                  "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": sid})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())
        kwargs: dict[str, Any] = {"agentRuntimeArn": self.arn, "runtimeSessionId": sid, "payload": body,
                                  "contentType": "application/json", "accept": "application/json"}
        if self.qualifier:
            kwargs["qualifier"] = self.qualifier
        resp = self._client.invoke_agent_runtime(**kwargs)
        return json.loads(resp["response"].read())

    def decide(self, state: Any, questions: dict[str, dict[str, Any]], **kw: Any) -> dict[str, Any]:
        """Ask questions about `state` (text or JSON-able data); returns answers with probabilities."""
        return self.invoke({"state": state, "questions": questions}, **kw)

    def decide_batch(self, requests: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
        """Several {"state", "questions"} requests in one call (one round trip, answered in order)."""
        return self.invoke({"batch": requests}, **kw)["results"]

    def health(self, session_id: str | None = None) -> dict[str, Any]:
        """Model, load timings and memory of one session's microVM (starts it if it is cold)."""
        return self.invoke({"action": "health"}, session_id=session_id)

    def warm(self) -> list[dict[str, Any]]:
        """Start every session in the pool now, so later calls do not pay the cold start; waits until ready."""
        threads, out = [], [None] * len(self.sessions)

        def one(i: int, sid: str) -> None:
            t0 = time.time()
            for _ in range(60):
                h = self.invoke({"action": "health"}, session_id=sid)
                if h.get("ready") or h.get("health", {}).get("state") == "failed":
                    h["ready_after_s"] = round(time.time() - t0, 1)  # measured here, from the first call
                    out[i] = h
                    return
                time.sleep(10)

        for i, sid in enumerate(self.sessions):
            t = threading.Thread(target=one, args=(i, sid))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return out  # type: ignore[return-value]
