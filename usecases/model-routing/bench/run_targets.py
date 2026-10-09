"""Answer every task with every target model and grade it: the ground truth for routing.

    python bench/run_targets.py                 # all models in MODELS
    python bench/run_targets.py --models micro,lite,pro --prompts optimized   # APO-optimized prompts

Writes results/answers[-<prompts>].jsonl, one line per (task, model). Resumable: rows already in
the file are skipped. Temperature 0, so a rerun gives (nearly) the same answers.
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_TOKENS = {"code": 1024, "math": 1024, "logic": 1500, "json": 300, "extract": 100, "sports": 50}


def load_tasks() -> list[dict]:
    return [json.loads(x) for x in (ROOT / "data" / "tasks.jsonl").read_text().splitlines() if x]


def prompt_for(task: dict, model: str, optimized: dict | None) -> str:
    if optimized is None:
        return task["prompt"]
    tpl = optimized.get(task["family"], {}).get(model)
    return tpl.replace("{{input}}", task["input"]) if tpl else task["prompt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--prompts", default="original", help="original, or optimized (results/apo_prompts.json)")
    ap.add_argument("--split", default=None, help="only this split (train/test)")
    a = ap.parse_args()
    optimized = json.loads((ROOT / "results" / "apo_prompts.json").read_text()) if a.prompts == "optimized" else None
    out = ROOT / "results" / ("answers.jsonl" if a.prompts == "original" else f"answers-{a.prompts}.jsonl")
    out.parent.mkdir(exist_ok=True)
    done = {(r["id"], r["model"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    tasks = [t for t in load_tasks() if a.split in (None, t["split"])]
    jobs = [(t, m) for t in tasks for m in a.models.split(",") if (t["id"], m) not in done]
    if optimized is not None:  # only the (family, model) pairs that APO produced a prompt for
        jobs = [(t, m) for t, m in jobs if optimized.get(t["family"], {}).get(m)]
    print(f"{len(jobs)} calls to make ({len(done)} already done)")

    def one(t: dict, m: str) -> dict:
        r = converse(MODELS[m].model_id, prompt_for(t, m, optimized), MAX_TOKENS[t["family"]])
        return {
            "id": t["id"],
            "family": t["family"],
            "split": t["split"],
            "model": m,
            "score": grade(t["grader"], r["text"], t["answer"]),
            "in": r["in"],
            "out": r["out"],
            "cost": MODELS[m].cost(r["in"], r["out"]),
            "latency_s": r["latency_s"],
            "text": r["text"],
        }

    with out.open("a") as f, ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(one, t, m) for t, m in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fu.result()) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001 - log and continue; a rerun fills the gap
                print("error:", str(e)[:200])
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}")


if __name__ == "__main__":
    main()
