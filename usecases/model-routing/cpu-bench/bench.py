"""CPU benchmark of a Strands Decider served by the runtime/ container code, on one EC2 instance.

Runs against http://127.0.0.1:8080/invocations (runtime/main.py started by userdata.sh) and writes JSON.
Workloads: one routing question per request (sequential, 4 and 8 concurrent, batched by 8), and a
three-question ticket decision.
"""

import argparse
import json
import platform
import statistics as st
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8080/invocations"
HARD = {
    "type": "noul",
    "instructions": "Does answering this request correctly need careful multi-step reasoning?",
    "criteria": {
        "true": "a logic puzzle, a calculation with several steps, or code that must pass tests",
        "false": "looking something up in the given text, a yes/no judgement, or reformatting given facts",
    },
}
TICKET = {
    "state": "Order #4411 arrived with a cracked screen. I want my money back today, this is the second time.",
    "questions": {
        "escalate": {"type": "noul", "instructions": "Should this ticket go to a human agent?"},
        "queue": {
            "type": "choice",
            "instructions": "Which queue handles this ticket?",
            "criteria": {
                "returns": "refunds and damaged items",
                "billing": "payments",
                "shipping": "deliveries",
                "account": "login and profile",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["low: can wait a week", "medium: within two days", "high: today"],
        },
    },
}


def call(payload, timeout=600):
    t = time.time()
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return time.time() - t, out


def summary(lat, wall=None, n=None):
    s = sorted(lat)
    d = {
        "n": len(s),
        "p50_s": round(st.median(s), 3),
        "p95_s": round(s[int(0.95 * (len(s) - 1))], 3),
        "mean_s": round(st.mean(s), 3),
    }
    if wall:
        d["wall_s"] = round(wall, 2)
        d["decisions_per_s"] = round((n or len(s)) / wall, 3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [json.loads(line) for line in open("/opt/sda/usecases/model-routing/data/tasks.jsonl")]
    test = [t for t in tasks if t["split"] == "test"]
    fams = {}
    for t in test:
        fams.setdefault(t["family"], []).append(t)
    prompts = [fams[f][i]["prompt"] for i in range(4) for f in sorted(fams)]  # 24 prompts, 4 per family
    reqs = [{"state": {"request": p}, "questions": {"hard": HARD}} for p in prompts]
    res = {"quant": a.quant, "machine": platform.machine(), "cpu": "", "health": call({"action": "health"})[1]}
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name") or line.startswith("CPU part"):
            res["cpu"] = line.split(":", 1)[1].strip()
            break
    flags = open("/proc/cpuinfo").read()
    res["isa"] = {
        k: (k in flags) for k in ("amx_bf16", "amx_int8", "avx512_bf16", "avx512_vnni", "bf16", "i8mm", "sve2")
    }
    for r in reqs[:2]:
        call(r)  # warm
    lat = [call(r)[0] for r in reqs]
    tok = [call(r)[1].get("usage", {}).get("input_tokens") for r in reqs[:6]]
    res["one_question_sequential"] = {**summary(lat, sum(lat), len(lat)), "input_tokens_sample": tok}
    for c in (4, 8):
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            lat_c = [x[0] for x in ex.map(call, reqs)]
        res[f"one_question_concurrency_{c}"] = summary(lat_c, time.time() - t0, len(reqs))
    blat = [call({"batch": reqs[i : i + 8]})[0] for i in range(0, 24, 8)]
    res["one_question_batch_8"] = {
        "calls": len(blat),
        "per_call": summary(blat),
        "decisions_per_s": round(24 / sum(blat), 3),
    }
    tl = [call(TICKET)[0] for _ in range(6)]
    res["ticket_three_questions_sequential"] = summary(tl, sum(tl), len(tl))
    res["health_after"] = call({"action": "health"})[1]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("health", "health_after")}, indent=1))


if __name__ == "__main__":
    main()
