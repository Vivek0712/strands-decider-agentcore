"""Target models, their on-demand prices, and a Bedrock Converse wrapper that records tokens and time.

Prices: AWS Price List API, us-east-1, on-demand, fetched 2026-10-08 (USD per 1,000 tokens).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE")


@dataclass(frozen=True)
class Model:
    key: str
    model_id: str
    in_per_1k: float
    out_per_1k: float
    label: str

    def cost(self, tin: int, tout: int) -> float:
        return tin / 1000 * self.in_per_1k + tout / 1000 * self.out_per_1k


MODELS = {
    m.key: m
    for m in [
        Model("micro", "us.amazon.nova-micro-v1:0", 0.000035, 0.00014, "Nova Micro"),
        Model("lite", "us.amazon.nova-lite-v1:0", 0.00006, 0.00024, "Nova Lite"),
        Model("pro", "us.amazon.nova-pro-v1:0", 0.0008, 0.0032, "Nova Pro"),
        Model("scout", "us.meta.llama4-scout-17b-instruct-v1:0", 0.00017, 0.00066, "Llama 4 Scout 17B"),
        Model("llama8b", "us.meta.llama3-1-8b-instruct-v1:0", 0.00022, 0.00022, "Llama 3.1 8B"),
        # Anthropic models on Bedrock: regional (us.) cross-Region inference prices, AWS Marketplace listing
        Model("haiku45", "us.anthropic.claude-haiku-4-5-20251001-v1:0", 0.0011, 0.0055, "Claude Haiku 4.5"),
        # Claude Sonnet 4.6 stands in for Sonnet 5.5, which is not yet available to this account
        Model("sonnet46", "us.anthropic.claude-sonnet-4-6", 0.0033, 0.0165, "Claude Sonnet 4.6"),
    ]
}
TIERS = ["micro", "lite", "pro"]  # the cheap-to-expensive ladder the online routers choose from

# AgentCore Runtime consumption pricing (v2): CPU only while active, memory while the session lives.
AGENTCORE_VCPU_HOUR = 0.1276
AGENTCORE_GB_HOUR = 0.0169

_local = threading.local()
_sem = threading.BoundedSemaphore(int(os.environ.get("BEDROCK_CONCURRENCY", "5")))


def runtime():
    if not hasattr(_local, "rt"):
        s = boto3.Session(profile_name=PROFILE, region_name=REGION)
        _local.rt = s.client(
            "bedrock-runtime", config=Config(read_timeout=120, retries={"max_attempts": 8, "mode": "adaptive"})
        )
    return _local.rt


def converse(model_id: str, prompt: str, max_tokens: int = 1024, system: str | None = None) -> dict:
    """One Converse call at temperature 0. Returns text, tokens, latency and the model that answered."""
    kw = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
    }
    if system:
        kw["system"] = [{"text": system}]
    with _sem:
        t = time.time()
        r = runtime().converse(**kw)
        dt = time.time() - t
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    invoked = (r.get("trace", {}).get("promptRouter", {}) or {}).get("invokedModelId")
    return {
        "text": text,
        "in": r["usage"]["inputTokens"],
        "out": r["usage"]["outputTokens"],
        "latency_s": round(dt, 3),
        "server_ms": r.get("metrics", {}).get("latencyMs"),
        "invoked": invoked,
    }
