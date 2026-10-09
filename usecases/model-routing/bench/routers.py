"""The online routers: each looks at one request and picks a tier (micro, lite or pro).

- DeciderRouter: a Strands Decider on AgentCore Runtime answers one `choice` question and returns a
  calibrated probability per tier, so the routing threshold can be tuned (see policy.py).
- LLMClassifierRouter: a small Bedrock model is asked the same question and answers with one word.
- Bedrock Intelligent Prompt Routing is different: it routes and answers in one call, so it lives in
  run_routers.py (bedrock_router).

Every router gets the same description of the tiers, so the comparison is about the router, not the
prompt.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root: client/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from client.decider_client import DeciderClient, choice, noul  # noqa: E402
from models import AGENTCORE_GB_HOUR, AGENTCORE_VCPU_HOUR, MODELS, converse  # noqa: E402

TIER_DESCRIPTIONS = {
    "micro": "a small, fast model: enough for looking up a value in the text, yes/no or short classification, "
    "and simple rewriting",
    "lite": "a mid-size model: needed for structured output such as JSON with several fields, summaries, "
    "and light reasoning",
    "pro": "a large model: needed for multi-step maths, writing code that must pass tests, and hard logic puzzles",
}
ROUTING_QUESTION = (
    "Which is the cheapest model that will answer this request correctly? Prefer the smaller "
    "model whenever it is enough."
)
MAX_STATE_CHARS = 6000

# The second question: a property of the request itself, not knowledge about particular models.
# A decider answers it with a calibrated P(yes); the escalation threshold is then tuned on data.
HARD_QUESTION = "Does answering this request correctly need careful multi-step reasoning?"
HARD_CRITERIA = {
    "true": "a logic puzzle, a calculation with several steps, or code that must pass tests",
    "false": "looking something up in the given text, a yes/no judgement, or reformatting given facts",
}


class DeciderRouter:
    name = "decider"

    def __init__(self, arn: str, pool: int = 4, profile: str | None = None, question: str = "tier") -> None:
        self.question = question
        self.client = DeciderClient(arn=arn, pool=pool, profile=profile, session_prefix="router")
        self.memory_gb = 4.2  # resident memory of a warm session (measured)

    def warm(self) -> list:
        return self.client.warm()

    def route(self, prompt: str) -> dict:
        t = time.time()
        if self.question == "hard":
            r = self.client.decide(
                {"request": prompt[:MAX_STATE_CHARS]},
                {"hard": noul(HARD_QUESTION, true=HARD_CRITERIA["true"], false=HARD_CRITERIA["false"])},
            )
        else:
            r = self.client.decide(
                {"request": prompt[:MAX_STATE_CHARS]}, {"tier": choice(ROUTING_QUESTION, TIER_DESCRIPTIONS)}
            )
        dt = time.time() - t
        # AgentCore bills CPU while it works (both vCPUs during inference) and memory per second
        compute_s = r.get("latency_ms", dt * 1000) / 1000
        cost = compute_s / 3600 * (2 * AGENTCORE_VCPU_HOUR + self.memory_gb * AGENTCORE_GB_HOUR)
        if self.question == "hard":
            p = r["answers"]["hard"]["noul"]
            return {
                "tier": "pro" if p >= 0.5 else "micro",
                "p_hard": p,
                "latency_s": round(dt, 3),
                "compute_s": round(compute_s, 3),
                "cost": cost,
            }
        a = r["answers"]["tier"]
        return {
            "tier": a["choice"],
            "probs": a["probabilities"],
            "latency_s": round(dt, 3),
            "compute_s": round(compute_s, 3),
            "cost": cost,
        }


CLASSIFIER_PROMPT = """You route each user request to the cheapest model that will answer it correctly.

Models:
- micro: {micro}
- lite: {lite}
- pro: {pro}

{question} Reply with exactly one word: micro, lite or pro.

<request>
{request}
</request>"""


HARD_PROMPT = """{question} Yes means {true}. No means {false}. Reply with exactly one word: yes or no.

<request>
{request}
</request>"""


class LLMClassifierRouter:
    def __init__(self, model_key: str, question: str = "tier") -> None:
        self.model = MODELS[model_key]
        self.question = question
        self.name = f"classifier-{model_key}" + ("-hard" if question == "hard" else "")

    def route(self, prompt: str) -> dict:
        if self.question == "hard":
            r = converse(
                self.model.model_id,
                HARD_PROMPT.format(question=HARD_QUESTION, request=prompt[:MAX_STATE_CHARS], **HARD_CRITERIA),
                max_tokens=5,
            )
            w = re.findall(r"\b(yes|no)\b", r["text"].lower())
            return {
                "tier": "micro" if w and w[0] == "no" else "pro",
                "raw": r["text"][:40],
                "latency_s": r["latency_s"],
                "in": r["in"],
                "out": r["out"],
                "cost": self.model.cost(r["in"], r["out"]),
                "parsed": bool(w),
            }
        r = converse(
            self.model.model_id,
            CLASSIFIER_PROMPT.format(question=ROUTING_QUESTION, request=prompt[:MAX_STATE_CHARS], **TIER_DESCRIPTIONS),
            max_tokens=5,
        )
        words = re.findall(r"\b(micro|lite|pro)\b", r["text"].lower())
        tier = words[0] if words else "pro"  # unparseable: play safe
        return {
            "tier": tier,
            "raw": r["text"][:40],
            "latency_s": r["latency_s"],
            "in": r["in"],
            "out": r["out"],
            "cost": self.model.cost(r["in"], r["out"]),
            "parsed": bool(words),
        }


def decider_arn() -> str:
    arn = os.environ.get("DECIDER_ARN")
    if arn:
        return arn
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    import deciderctl

    return deciderctl.deployed()["triage"]["runtimeArn"]
