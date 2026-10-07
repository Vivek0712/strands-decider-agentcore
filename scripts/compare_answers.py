"""Compare two answer files from collect_answers.py: same decisions, and how far probabilities moved.

    python scripts/compare_answers.py answers-bf16.json answers-int8.json
"""

from __future__ import annotations

import json
import sys


def top(a: dict) -> str:
    if a["type"] == "noul":
        p = a["noul"]
        return "yes" if p >= 0.8 else "no" if p <= 0.2 else "unsure"
    if a["type"] == "choice":
        return a["choice"]
    return str(round(a["score"]))


def probs(a: dict) -> list[float]:
    return [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def main(ref_path: str, new_path: str) -> int:
    ref, new = json.load(open(ref_path)), json.load(open(new_path))
    same = total = 0
    worst = 0.0
    print(f"{'decision':<22} {'question':<12} {'reference':>10} {'new':>10} {'max |dp|':>9}")
    for did, r in ref.items():
        for q, ra in r["answers"].items():
            na = new[did]["answers"][q]
            dp = max(abs(x - y) for x, y in zip(probs(ra), probs(na), strict=True))
            worst = max(worst, dp)
            total += 1
            same += top(ra) == top(na)
            print(f"{did:<22} {q:<12} {top(ra):>10} {top(na):>10} {dp:>9.3f}")
    lat = [v["latency_s"] for v in new.values()]
    print(f"\nsame decision on {same}/{total} questions; largest probability change {worst:.3f}; "
          f"median latency {sorted(lat)[len(lat) // 2]:.1f}s")
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
