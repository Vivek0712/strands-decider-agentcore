"""Run every router over every task and record its choice, cost and latency.

    python bench/run_routers.py --routers decider,classifier-micro,classifier-lite,classifier-llama8b,bedrock

Writes results/routes-<router>.jsonl (resumable). The answers themselves come from
results/answers.jsonl, except for Bedrock Intelligent Prompt Routing, which routes and answers in one
call and is graded here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from graders import grade  # noqa: E402
from models import MODELS, converse  # noqa: E402
from routers import DeciderRouter, LLMClassifierRouter, decider_arn  # noqa: E402
from run_targets import MAX_TOKENS, load_tasks  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOVA_ROUTER = "arn:aws:bedrock:us-east-1:{account}:default-prompt-router/amazon.nova:1"
INVOKED = {
    "us.amazon.nova-lite-v1:0": "lite",
    "us.amazon.nova-pro-v1:0": "pro",
    "amazon.nova-lite-v1:0": "lite",
    "amazon.nova-pro-v1:0": "pro",
}


def bedrock_router_fn(account: str):
    arn = NOVA_ROUTER.format(account=account)

    def route(task: dict) -> dict:
        r = converse(arn, task["prompt"], MAX_TOKENS[task["family"]])
        inv = (r["invoked"] or "").split("/")[-1]
        tier = INVOKED.get(inv, inv)
        return {
            "tier": tier,
            "invoked": inv,
            "latency_s": r["latency_s"],
            "in": r["in"],
            "out": r["out"],
            "answer_cost": MODELS[tier].cost(r["in"], r["out"]) if tier in MODELS else None,
            "cost": 0.0,
            "score": grade(task["grader"], r["text"], task["answer"]),
        }

    return route


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", default="decider,classifier-micro,classifier-lite,classifier-llama8b,bedrock")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--pool", type=int, default=4, help="warm decider sessions")
    a = ap.parse_args()
    tasks = load_tasks()
    for name in a.routers.split(","):
        out = ROOT / "results" / f"routes-{name}.jsonl"
        done = {json.loads(x)["id"] for x in out.read_text().splitlines()} if out.exists() else set()
        todo = [t for t in tasks if t["id"] not in done]
        if not todo:
            print(name, "complete")
            continue
        if name in ("decider", "decider-hard"):
            router = DeciderRouter(
                decider_arn(), pool=a.pool, profile=a.profile, question="hard" if name.endswith("-hard") else "tier"
            )
            print("warming", a.pool, "decider sessions:", [w.get("ready_after_s") for w in router.warm()])
            fn, workers = (lambda t, r=router: r.route(t["prompt"])), a.pool
        elif name == "bedrock":
            import boto3

            acct = boto3.Session(profile_name=a.profile).client("sts").get_caller_identity()["Account"]
            fn, workers = bedrock_router_fn(acct), 5
        else:
            parts = name.split("-")
            router_c = LLMClassifierRouter(parts[1], question="hard" if parts[-1] == "hard" else "tier")
            fn, workers = (lambda t, r=router_c: r.route(t["prompt"])), 5
        print(f"{name}: {len(todo)} tasks")
        with out.open("a") as f, ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fn, t): t for t in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                t = futs[fu]
                try:
                    f.write(
                        json.dumps({"id": t["id"], "family": t["family"], "split": t["split"], **fu.result()}) + "\n"
                    )
                    f.flush()
                except Exception as e:  # noqa: BLE001
                    print(name, t["id"], "error:", str(e)[:200])
                if i % 40 == 0:
                    print(f"  {name} {i}/{len(todo)}")


if __name__ == "__main__":
    main()
