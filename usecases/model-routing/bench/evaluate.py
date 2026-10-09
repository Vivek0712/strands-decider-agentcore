"""Score every routing strategy on the held-out test split: quality, cost, latency, routing accuracy.

    python bench/evaluate.py          # results/summary.json, results/curve-decider.json, printed tables

Quality is the mean grader score of the answer the strategy would have returned (1 = correct).
Cost is per 1,000 requests and includes the router itself. Latency is router + answer, per request.
Anything tuned (the decider threshold, per-family tables) is fitted on the train split only.
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from models import MODELS, TIERS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAMILIES = ["extract", "sports", "json", "math", "code", "logic"]
CORRECT = 0.999  # "correct" for the oracle: a full score


def jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x] if p.exists() else []


def load_answers(name: str = "answers.jsonl") -> dict:
    out: dict = collections.defaultdict(dict)
    for r in jsonl(RES / name):
        out[r["id"]][r["model"]] = r
    return out


def tasks() -> list[dict]:
    return jsonl(ROOT / "data" / "tasks.jsonl")


def oracle_tier(ans: dict, tid: str) -> str:
    for t in TIERS:
        if ans[tid][t]["score"] >= CORRECT:
            return t
    return "pro"  # nobody is fully right: the oracle still pays for the best model


def decider_policy(probs: dict, tau: float) -> str:
    """Cheapest tier whose cumulative probability of being enough reaches tau."""
    cum = 0.0
    for t in TIERS[:-1]:
        cum += probs.get(t, 0.0)
        if cum >= tau:
            return t
    return TIERS[-1]


def score_strategy(
    name: str, choices: dict, ans: dict, ids: list, router: dict | None = None, own_answers: dict | None = None
) -> dict:
    """choices: id -> model key. router: id -> {cost, latency_s}. own_answers: id -> {score, cost, latency_s}."""
    q, cost, rcost, lat, rlat, hit, dist = [], 0.0, 0.0, [], [], 0, collections.Counter()
    answer_costs = {}
    for i in ids:
        m = choices[i]
        a = own_answers[i] if own_answers else ans[i][m]
        r = (router or {}).get(i, {})
        q.append(a["score"])
        cost += a["cost"] + r.get("cost", 0.0)
        rcost += r.get("cost", 0.0)
        answer_costs[i] = (m, a["cost"])
        lat.append(a["latency_s"] + r.get("latency_s", 0.0))
        rlat.append(r.get("latency_s", 0.0))
        hit += m == oracle_tier(ans, i)
        dist[m] += 1
    n = len(ids)
    return {
        "strategy": name,
        "quality": sum(q) / n,
        "cost_per_1k": cost / n * 1000,
        "router_cost_per_1k": rcost / n * 1000,
        "latency_p50": st.median(lat),
        "latency_p95": sorted(lat)[int(0.95 * (n - 1))],
        "router_latency_p50": st.median(rlat),
        "oracle_match": hit / n,
        "mix": {k: dist[k] / n for k in sorted(dist)},
        "_answer_costs": answer_costs,
    }


def main() -> None:
    T = {t["id"]: t for t in tasks()}
    ans = load_answers()
    complete = [i for i in T if all(m in ans[i] for m in MODELS)]
    test = [i for i in complete if T[i]["split"] == "test"]
    train = [i for i in complete if T[i]["split"] == "train"]
    print(f"{len(complete)}/{len(T)} tasks answered by every model; test {len(test)}, train {len(train)}")
    rows = []

    for m in MODELS:
        rows.append(score_strategy(f"always {m}", {i: m for i in test}, ans, test))
    rows.append(score_strategy("oracle (micro/lite/pro)", {i: oracle_tier(ans, i) for i in test}, ans, test))

    # per-family table fitted on train: cheapest tier within 0.03 of the best tier's train score
    def family_table(train_scores: dict, models: list[str], tol: float = 0.03) -> dict:
        table = {}
        for f in FAMILIES:
            best = max(train_scores[(f, m)] for m in models)
            ok = [m for m in models if train_scores[(f, m)] >= best - tol]
            table[f] = min(ok, key=lambda m: MODELS[m].in_per_1k + MODELS[m].out_per_1k)
        return table

    tr = {(f, m): st.mean(ans[i][m]["score"] for i in train if T[i]["family"] == f) for f in FAMILIES for m in MODELS}
    tbl = family_table(tr, TIERS)
    rows.append(
        {
            **score_strategy("family table, measured (train)", {i: tbl[T[i]["family"]] for i in test}, ans, test),
            "table": tbl,
        }
    )

    # online routers
    for name in [
        "classifier-micro",
        "classifier-lite",
        "classifier-llama8b",
        "classifier-micro-hard",
        "classifier-lite-hard",
    ]:
        rr = {r["id"]: r for r in jsonl(RES / f"routes-{name}.jsonl")}
        if all(i in rr for i in test):
            rows.append(score_strategy(name, {i: rr[i]["tier"] for i in test}, ans, test, router=rr))

    br = {r["id"]: r for r in jsonl(RES / "routes-bedrock.jsonl")}
    if all(i in br for i in test):
        own = {
            i: {"score": br[i]["score"], "cost": br[i]["answer_cost"], "latency_s": br[i]["latency_s"]} for i in test
        }
        rows.append(
            score_strategy(
                "bedrock intelligent prompt routing (lite/pro)",
                {i: br[i]["tier"] for i in test},
                ans,
                test,
                own_answers=own,
            )
        )

    dr = {r["id"]: r for r in jsonl(RES / "routes-decider.jsonl")}
    if all(i in dr for i in test + train):
        rows.append(score_strategy("decider (argmax)", {i: dr[i]["tier"] for i in test}, ans, test, router=dr))
        curve = []
        for k in range(0, 101, 2):
            tau = k / 100
            s_tr = score_strategy("", {i: decider_policy(dr[i]["probs"], tau) for i in train}, ans, train, router=dr)
            s_te = score_strategy("", {i: decider_policy(dr[i]["probs"], tau) for i in test}, ans, test, router=dr)
            curve.append({"tau": tau, "train": s_tr, "test": s_te})
        (RES / "curve-decider.json").write_text(json.dumps(curve, indent=1))
        # tau chosen on train: the cheapest setting that keeps quality within 2 points of always-pro
        pro_q_train = st.mean(ans[i]["pro"]["score"] for i in train)
        ok = [c for c in curve if c["train"]["quality"] >= pro_q_train - 0.02] or curve[-1:]
        best = min(ok, key=lambda c: c["train"]["cost_per_1k"])
        rows.append(
            {**best["test"], "strategy": f"decider (threshold {best['tau']:.2f}, tuned on train)", "tau": best["tau"]}
        )

    dh = {r["id"]: r for r in jsonl(RES / "routes-decider-hard.jsonl")}
    if all(i in dh for i in test + train):
        curve_h = []
        for k in range(0, 101, 2):
            tau = k / 100  # escalate to pro when P(needs multi-step reasoning) >= tau
            pick = lambda i, t=tau: "pro" if dh[i]["p_hard"] >= t else "micro"  # noqa: E731
            curve_h.append(
                {
                    "tau": tau,
                    "train": score_strategy("", {i: pick(i) for i in train}, ans, train, router=dh),
                    "test": score_strategy("", {i: pick(i) for i in test}, ans, test, router=dh),
                }
            )
        (RES / "curve-decider-hard.json").write_text(json.dumps(curve_h, indent=1))
        rows.append(
            {**score_strategy("decider-hard (P >= 0.50)", {i: dh[i]["tier"] for i in test}, ans, test, router=dh)}
        )
        pro_q_train = st.mean(ans[i]["pro"]["score"] for i in train)
        ok = [c for c in curve_h if c["train"]["quality"] >= pro_q_train - 0.02] or curve_h[:1]
        best = min(ok, key=lambda c: c["train"]["cost_per_1k"])
        rows.append(
            {
                **best["test"],
                "strategy": f"decider-hard (threshold {best['tau']:.2f}, tuned on train)",
                "tau": best["tau"],
            }
        )

    apo_scores = RES / "apo_scores.json"
    if apo_scores.exists():
        sc = json.loads(apo_scores.read_text())  # {family: {model: score}} from the APO job (optimized prompts)
        apo_tbl = family_table(
            {(f, m): sc[f][m] for f in FAMILIES for m in MODELS if m in sc[f]},
            [m for m in MODELS if all(m in sc[f] for f in FAMILIES)],
        )
        opt = load_answers("answers-optimized.jsonl")
        if all(apo_tbl[T[i]["family"]] in opt.get(i, {}) for i in test):
            own = {i: opt[i][apo_tbl[T[i]["family"]]] for i in test}
            rows.append(
                {
                    **score_strategy(
                        "APO model selection + optimized prompts",
                        {i: apo_tbl[T[i]["family"]] for i in test},
                        ans,
                        test,
                        own_answers=own,
                    ),
                    "table": apo_tbl,
                }
            )
        rows.append(
            {
                **score_strategy(
                    "APO model selection, original prompts", {i: apo_tbl[T[i]["family"]] for i in test}, ans, test
                ),
                "table": apo_tbl,
            }
        )

    pro = next(r for r in rows if r["strategy"] == "always pro")
    for r in rows:
        r["quality_vs_pro"] = r["quality"] / pro["quality"]
        r["savings_vs_pro"] = 1 - r["cost_per_1k"] / pro["cost_per_1k"]

    # price sensitivity: what if the top tier cost k times Nova Pro? Same tokens, same routing decisions.
    def at_price(r: dict, k: float) -> float:
        ac = r["_answer_costs"]
        return sum(c * (k if m == "pro" else 1.0) for m, c in ac.values()) / len(ac) * 1000 + r["router_cost_per_1k"]

    multipliers = [1, 2, 4, 8, 16, 32, 64]
    sensitivity = {r["strategy"]: [round(at_price(r, k), 5) for k in multipliers] for r in rows}
    for r in rows:
        always = sensitivity["always pro"]
        r["break_even_multiplier"] = next(
            (k for k, c, a in zip(multipliers, sensitivity[r["strategy"]], always, strict=True) if c <= a), None
        )

    per_family = {
        f: {m: st.mean(ans[i][m]["score"] for i in test if T[i]["family"] == f) for m in MODELS} for f in FAMILIES
    }
    for r in rows:
        r.pop("_answer_costs", None)
    (RES / "summary.json").write_text(
        json.dumps(
            {
                "n_test": len(test),
                "strategies": rows,
                "per_family_test": per_family,
                "price_multipliers": multipliers,
                "sensitivity": sensitivity,
            },
            indent=1,
        )
    )
    print(
        f"\n{'strategy':52} {'quality':>8} {'vs pro':>7} {'$/1k req':>9} {'saved':>7} "
        f"{'p50 s':>6} {'p95 s':>6} {'oracle':>7}"
    )
    for r in rows:
        print(
            f"{r['strategy'][:52]:52} {r['quality']:8.3f} {r['quality_vs_pro']:7.1%} {r['cost_per_1k']:9.4f} "
            f"{r['savings_vs_pro']:7.1%} {r['latency_p50']:6.2f} {r['latency_p95']:6.2f} {r['oracle_match']:7.1%}"
            f"  breakeven x{r['break_even_multiplier']}"
        )


if __name__ == "__main__":
    main()
