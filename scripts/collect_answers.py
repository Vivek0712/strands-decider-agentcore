"""Answer every decision in a JSONL file with one decider endpoint and save the answers.

    python scripts/collect_answers.py --url http://localhost:8080 --out answers-int8.json
    python scripts/collect_answers.py --arn arn:aws:bedrock-agentcore:...:runtime/... --out answers.json

Used to check that int8 serving keeps the reference (bf16) decisions: run it against both and
compare with scripts/compare_answers.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from client.decider_client import DeciderClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="a local runtime container, e.g. http://localhost:8080")
    g.add_argument("--arn", help="an AgentCore runtime ARN")
    ap.add_argument("--data", default="examples/data/decisions.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", default=None)
    a = ap.parse_args()
    client = DeciderClient(url=a.url) if a.url else DeciderClient(arn=a.arn, region=a.region)
    out = {}
    for line in pathlib.Path(a.data).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        t = time.time()
        r = client.decide(d["state"], d["questions"])
        out[d["id"]] = {"answers": r["answers"], "latency_s": round(time.time() - t, 2)}
        print(f"{d['id']:<22} {out[d['id']]['latency_s']:>6.1f}s", flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
