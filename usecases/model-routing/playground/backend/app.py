"""Playground API (AWS Lambda behind API Gateway and CloudFront). No API key; rate limited.

Routes (all JSON):
  GET  /api/status                 limits left for this caller, global usage totals, decider state
  POST /api/route/decider          {"prompt"} -> calibrated tier probabilities from the decider on AgentCore
  POST /api/route/classifier       {"prompt"} -> Nova Micro's one-word tier choice
  POST /api/route/bedrock          {"prompt"} -> Bedrock Intelligent Prompt Routing: chosen model + its answer
  POST /api/answer                 {"prompt", "tier", "router"} -> the answer from that tier, cost, and what
                                    always using Nova Pro would have cost for the same tokens
  (EventBridge) {"warm": true}     keeps one decider session warm

Limits: per IP per hour (PER_IP_HOUR), all callers per day (GLOBAL_DAY), prompt length, output tokens.
Every route call counts once; a playground "run" is about 4 calls.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import time

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-east-1")
DECIDER_ARN = os.environ["DECIDER_ARN"]
ROUTER_ARN = os.environ["ROUTER_ARN"]
TABLE = os.environ["TABLE"]
ORIGIN_SECRET = os.environ.get("ORIGIN_SECRET", "")  # CloudFront adds it; direct calls to the API are refused
PER_IP_HOUR = int(os.environ.get("PER_IP_HOUR", "40"))
GLOBAL_DAY = int(os.environ.get("GLOBAL_DAY", "1500"))
MAX_PROMPT = int(os.environ.get("MAX_PROMPT", "2500"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "700"))
# one shared session id (>= 33 chars) so every visitor reaches the same warm microVM
SESSION = os.environ.get("DECIDER_SESSION", "playground-model-routing-decider-session-01")

MODELS = {  # USD per 1,000 tokens, AWS Price List API, us-east-1 on-demand
    "micro": ("us.amazon.nova-micro-v1:0", 0.000035, 0.00014, "Nova Micro"),
    "lite": ("us.amazon.nova-lite-v1:0", 0.00006, 0.00024, "Nova Lite"),
    "pro": ("us.amazon.nova-pro-v1:0", 0.0008, 0.0032, "Nova Pro"),
}
VCPU_HOUR, GB_HOUR, DECIDER_GB = 0.1276, 0.0169, 4.2
TIERS = {
    "micro": "a small, fast model: enough for looking up a value in the text, yes/no or short classification, "
    "and simple rewriting",
    "lite": "a mid-size model: needed for structured output such as JSON with several fields, summaries, "
    "and light reasoning",
    "pro": "a large model: needed for multi-step maths, writing code that must pass tests, and hard logic puzzles",
}
QUESTION = (
    "Which is the cheapest model that will answer this request correctly? Prefer the smaller "
    "model whenever it is enough."
)
CLASSIFIER = (
    "You route each user request to the cheapest model that will answer it correctly.\n\nModels:\n"
    "- micro: {micro}\n- lite: {lite}\n- pro: {pro}\n\n{q} Reply with exactly one word: micro, lite or pro."
    "\n\n<request>\n{request}\n</request>"
)
DEFAULT_TAU = float(os.environ.get("DECIDER_TAU", "0.5"))

ddb = boto3.client("dynamodb")
br = boto3.client("bedrock-runtime", config=Config(read_timeout=25, retries={"max_attempts": 2}))
ac = boto3.client("bedrock-agentcore", config=Config(read_timeout=27, connect_timeout=3, retries={"max_attempts": 0}))


def resp(code: int, body: dict) -> dict:
    return {
        "statusCode": code,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


# ---- limits and usage ----------------------------------------------------------------------------


def _incr(key: str, ttl_s: int, by: int = 1) -> int:
    r = ddb.update_item(
        TableName=TABLE,
        Key={"pk": {"S": key}},
        UpdateExpression="ADD n :one SET expires = if_not_exists(expires, :exp)",
        ExpressionAttributeValues={":one": {"N": str(by)}, ":exp": {"N": str(int(time.time()) + ttl_s)}},
        ReturnValues="UPDATED_NEW",
    )
    return int(r["Attributes"]["n"]["N"])


def _get(key: str) -> int:
    it = ddb.get_item(TableName=TABLE, Key={"pk": {"S": key}}).get("Item")
    return int(it["n"]["N"]) if it else 0


def keys(ip: str) -> tuple[str, str]:
    return f"ip#{ip}#{time.strftime('%Y%m%d%H', time.gmtime())}", f"day#{time.strftime('%Y%m%d', time.gmtime())}"


def check_limits(ip: str) -> dict | None:
    kip, kday = keys(ip)
    if _incr(kday, 3 * 86400) > GLOBAL_DAY:
        return resp(
            429, {"error": "daily_limit", "message": "The playground has reached today's limit. Try again tomorrow."}
        )
    if _incr(kip, 7200) > PER_IP_HOUR:
        return resp(
            429,
            {"error": "rate_limited", "message": f"Limit of {PER_IP_HOUR} calls per hour reached. Try again later."},
        )
    return None


def add_usage(cost: float, pro_cost: float, router_cost: float = 0.0) -> None:
    """Running totals for the page: answers served, what they cost, what routing them cost (decider
    compute), and what Nova Pro would have cost for the same tokens. Stored in micro-dollars."""
    ddb.update_item(
        TableName=TABLE,
        Key={"pk": {"S": "totals"}},
        UpdateExpression="ADD answers :one, cost_micro_usd :c, pro_micro_usd :p, router_micro_usd :r",
        ExpressionAttributeValues={
            ":one": {"N": "1"},
            ":c": {"N": str(round(cost * 1e6))},
            ":p": {"N": str(round(pro_cost * 1e6))},
            ":r": {"N": str(round(router_cost * 1e6))},
        },
    )


# ---- routers -------------------------------------------------------------------------------------


def decider(payload: dict) -> dict:
    r = ac.invoke_agent_runtime(
        agentRuntimeArn=DECIDER_ARN,
        runtimeSessionId=SESSION,
        payload=json.dumps(payload).encode(),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(r["response"].read())


def route_decider(prompt: str, tau: float) -> dict:
    h = decider({"action": "health"})
    if not h.get("ready"):
        return {
            "status": "warming",
            "stage": h.get("health", {}).get("state"),
            "message": "The decider's microVM is starting (about a minute). Retrying shortly.",
        }
    t = time.time()
    r = decider(
        {
            "state": {"request": prompt},
            "questions": {"tier": {"type": "choice", "instructions": QUESTION, "criteria": TIERS}},
        }
    )
    if "error" in r:
        return {"status": "error", "message": r["error"].get("message", "")}
    probs = r["answers"]["tier"]["probabilities"]
    cum, tier = 0.0, "pro"
    for k in ("micro", "lite"):
        cum += probs.get(k, 0)
        if cum >= tau:
            tier = k
            break
    compute_s = r.get("latency_ms", 0) / 1000
    return {
        "status": "ok",
        "tier": tier,
        "argmax": r["answers"]["tier"]["choice"],
        "probs": probs,
        "tau": tau,
        "latency_s": round(time.time() - t, 2),
        "cost": compute_s / 3600 * (2 * VCPU_HOUR + DECIDER_GB * GB_HOUR),
        "model": r.get("model"),
    }


def converse(model_id: str, prompt: str, max_tokens: int) -> dict:
    t = time.time()
    r = br.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    return {
        "text": "".join(c.get("text", "") for c in r["output"]["message"]["content"]),
        "in": r["usage"]["inputTokens"],
        "out": r["usage"]["outputTokens"],
        "latency_s": round(time.time() - t, 2),
        "invoked": (r.get("trace", {}).get("promptRouter", {}) or {}).get("invokedModelId", ""),
    }


def cost_of(tier: str, tin: int, tout: int) -> float:
    _, i, o, _ = MODELS[tier]
    return tin / 1000 * i + tout / 1000 * o


def route_classifier(prompt: str) -> dict:
    r = converse(MODELS["micro"][0], CLASSIFIER.format(q=QUESTION, request=prompt, **TIERS), 5)
    w = re.findall(r"\b(micro|lite|pro)\b", r["text"].lower())
    return {
        "status": "ok",
        "tier": w[0] if w else "pro",
        "raw": r["text"][:30],
        "latency_s": r["latency_s"],
        "cost": cost_of("micro", r["in"], r["out"]),
    }


def route_bedrock(prompt: str) -> dict:
    r = converse(ROUTER_ARN, prompt, MAX_TOKENS)
    inv = r["invoked"].split("/")[-1]
    tier = "lite" if "nova-lite" in inv else "pro" if "nova-pro" in inv else inv
    c = cost_of(tier, r["in"], r["out"]) if tier in MODELS else 0.0
    add_usage(c, cost_of("pro", r["in"], r["out"]))
    return {
        "status": "ok",
        "tier": tier,
        "answer": r["text"],
        "latency_s": r["latency_s"],
        "cost": c,
        "tokens": {"in": r["in"], "out": r["out"]},
        "pro_cost": cost_of("pro", r["in"], r["out"]),
    }


def answer(prompt: str, tier: str, router_cost: float = 0.0) -> dict:
    if tier not in MODELS:
        return {"status": "error", "message": "tier must be micro, lite or pro"}
    r = converse(MODELS[tier][0], prompt, MAX_TOKENS)
    c, pc = cost_of(tier, r["in"], r["out"]), cost_of("pro", r["in"], r["out"])
    add_usage(c, pc, router_cost)
    return {
        "status": "ok",
        "tier": tier,
        "model": MODELS[tier][3],
        "answer": r["text"],
        "latency_s": r["latency_s"],
        "tokens": {"in": r["in"], "out": r["out"]},
        "cost": c,
        "pro_cost": pc,
    }


def status(ip: str) -> dict:
    kip, kday = keys(ip)
    tot = ddb.get_item(TableName=TABLE, Key={"pk": {"S": "totals"}}).get("Item") or {}
    n = lambda k: int(tot.get(k, {}).get("N", "0"))  # noqa: E731
    try:
        h = decider({"action": "health"})
        dstate = "ready" if h.get("ready") else h.get("health", {}).get("state", "starting")
    except Exception:  # noqa: BLE001
        dstate = "unavailable"
    return {
        "limits": {
            "per_ip_hour": PER_IP_HOUR,
            "used_this_hour": _get(kip),
            "global_day": GLOBAL_DAY,
            "used_today": _get(kday),
            "max_prompt_chars": MAX_PROMPT,
            "max_output_tokens": MAX_TOKENS,
        },
        "totals": {
            "answers": n("answers"),
            "cost_usd": n("cost_micro_usd") / 1e6,
            "pro_usd": n("pro_micro_usd") / 1e6,
            "router_usd": n("router_micro_usd") / 1e6,
        },
        "decider": dstate,
        "tau": DEFAULT_TAU,
        "prices": {k: {"name": v[3], "in_per_1k": v[1], "out_per_1k": v[2]} for k, v in MODELS.items()},
    }


def handler(event: dict, context: object) -> dict:
    if event.get("warm"):
        return decider({"action": "health"})
    path, method = event.get("rawPath", ""), event.get("requestContext", {}).get("http", {}).get("method", "GET")
    ip = event.get("requestContext", {}).get("http", {}).get("sourceIp", "unknown")
    hdrs = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if ORIGIN_SECRET and hdrs.get("x-origin-verify") != ORIGIN_SECRET:
        return resp(403, {"error": "forbidden", "message": "Use the playground page."})
    ip = hdrs.get("cloudfront-viewer-address", "").rsplit(":", 1)[0] or ip  # the viewer, not the CloudFront edge
    try:
        if path == "/api/status" and method == "GET":
            return resp(200, status(ip))
        if method != "POST":
            return resp(404, {"error": "not_found"})
        body = json.loads(event.get("body") or "{}")
        prompt = str(body.get("prompt", "")).strip()
        if not prompt or len(prompt) > MAX_PROMPT:
            return resp(400, {"error": "bad_request", "message": f"Send a prompt of 1 to {MAX_PROMPT} characters."})
        limited = check_limits(ip)
        if limited:
            return limited
        if path == "/api/route/decider":
            tau = min(0.95, max(0.05, float(body.get("tau", DEFAULT_TAU))))
            return resp(200, route_decider(prompt, tau))
        if path == "/api/route/classifier":
            return resp(200, route_classifier(prompt))
        if path == "/api/route/bedrock":
            return resp(200, route_bedrock(prompt))
        if path == "/api/answer":
            rc = min(0.01, max(0.0, float(body.get("router_cost", 0) or 0)))  # the decider's compute, for the totals
            return resp(200, answer(prompt, str(body.get("tier", "")), rc))
        return resp(404, {"error": "not_found"})
    except cf.TimeoutError:
        return resp(504, {"error": "timeout"})
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "ReadTimeout" in type(e).__name__ or "timed out" in msg.lower():
            return resp(
                200,
                {
                    "status": "busy",
                    "message": "The decider is busy with another visitor's request. Try again in a few seconds.",
                },
            )
        return resp(500, {"error": "internal", "message": msg[:300]})
