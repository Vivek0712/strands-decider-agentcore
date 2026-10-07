"""Throughput and latency of a deployed decider, and how session pools scale it.

    python examples/03_batch_and_scale.py --arn <runtime ARN> --pool 1
    python examples/03_batch_and_scale.py --arn <runtime ARN> --pool 4

Each AgentCore session id is its own microVM (2 vCPU / 8 GB) with its own copy of the model, so a
pool of N session ids is N warm replicas. `--pool 4` warms four sessions first (cold starts happen
in parallel), then sends the decisions in examples/data/decisions.jsonl across them.
"""

import argparse
import json
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from client.decider_client import DeciderClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arn", required=True)
ap.add_argument("--pool", type=int, default=1)
ap.add_argument("--repeat", type=int, default=2, help="send the file this many times")
ap.add_argument("--profile")
a = ap.parse_args()

DATA = pathlib.Path(__file__).parent / "data" / "decisions.jsonl"
rows = [json.loads(x) for x in DATA.read_text().splitlines() if x]
work = rows * a.repeat
client = DeciderClient(arn=a.arn, pool=a.pool, profile=a.profile, session_prefix=f"scale{a.pool}")

t = time.time()
warm = client.warm()
print(f"warmed {a.pool} session(s) in {time.time() - t:.0f}s:",
      [w["ready_after_s"] for w in warm], "s to ready each")

lat = []


def one(row):
    t0 = time.time()
    client.decide(row["state"], row["questions"])
    lat.append(time.time() - t0)


t = time.time()
with ThreadPoolExecutor(max_workers=a.pool) as ex:
    list(ex.map(one, work))
wall = time.time() - t
q = sum(len(r["questions"]) for r in work)
lat.sort()
print(f"{len(work)} decisions ({q} questions) in {wall:.0f}s with {a.pool} warm session(s): "
      f"{len(work) / wall * 60:.1f} decisions/min, median {statistics.median(lat):.1f}s, "
      f"p95 {lat[int(0.95 * (len(lat) - 1))]:.1f}s per decision")
