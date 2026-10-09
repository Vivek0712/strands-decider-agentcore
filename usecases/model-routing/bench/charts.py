"""Charts for the analysis, from results/summary.json and the decider threshold curves.

python bench/charts.py      # writes docs/img/*.png
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES, IMG = ROOT / "results", ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"
C = {
    "decider": "#FF9900",
    "classifier": "#2563EB",
    "bedrock": "#7C3AED",
    "APO": "#059669",
    "oracle": "#9CA3AF",
    "always": "#4B5563",
    "family": "#0891B2",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def color(name: str) -> str:
    for k, v in C.items():
        if name.startswith(k):
            return v
    return INK


def short(name: str) -> str:
    return (
        name.replace("bedrock intelligent prompt routing (lite/pro)", "Bedrock prompt router")
        .replace("classifier-", "classifier: ")
        .replace(", tuned on train", "")
        .replace("always ", "always ")
        .replace("APO model selection + optimized prompts", "APO selection + prompts")
        .replace("APO model selection, original prompts", "APO selection")
        .replace("family table, measured (train)", "family table (measured)")
    )


KEY = [
    "classifier-haiku45",
    "always micro",
    "always pro",
    "always scout",
    "oracle (micro/lite/pro)",
    "bedrock intelligent prompt routing (lite/pro)",
    "classifier-micro",
    "classifier-micro-hard",
    "decider (argmax)",
    "APO model selection + optimized prompts",
    "APO model selection, original prompts",
]
MARK = {
    "always micro": "s",
    "always pro": "s",
    "always scout": "s",
    "oracle (micro/lite/pro)": "*",
    "classifier-micro": "o",
    "classifier-micro-hard": "D",
    "decider (argmax)": "o",
}
STYLE = {
    "classifier-haiku45": ("#D97757", "-"),
    "always pro": ("#111827", "-"),
    "always micro": ("#6B7280", ":"),
    "always scout": ("#D97706", ":"),
    "oracle (micro/lite/pro)": ("#9CA3AF", "--"),
    "bedrock intelligent prompt routing (lite/pro)": ("#7C3AED", "-"),
    "classifier-micro": ("#2563EB", "-"),
    "classifier-micro-hard": ("#60A5FA", "-."),
    "decider (argmax)": ("#FF9900", "-"),
    "APO model selection + optimized prompts": ("#059669", "-"),
    "APO model selection, original prompts": ("#34D399", "-."),
}


def main() -> None:
    s = json.loads((RES / "summary.json").read_text())
    rows = s["strategies"]
    tuned = [r["strategy"] for r in rows if r["strategy"].startswith("decider-hard (threshold")]
    key = KEY + tuned
    STYLE.update({t: ("#B45309", "--") for t in tuned})

    # 1. quality against cost, with the decider threshold curves
    fig, ax = plt.subplots(figsize=(11, 7), dpi=130)
    for name, fname, ls in (
        ("decider (tier question)", "curve-decider.json", "-"),
        ("decider (multi-step question)", "curve-decider-hard.json", "--"),
    ):
        p = RES / fname
        if p.exists():
            cv = json.loads(p.read_text())
            ax.plot(
                [c["test"]["cost_per_1k"] for c in cv],
                [100 * c["test"]["quality"] for c in cv],
                ls,
                color=C["decider"],
                lw=1.6,
                alpha=0.7,
                label=f"{name}: every threshold",
            )
    for r in [r for r in rows if r["strategy"] in key]:
        col = STYLE.get(r["strategy"], (color(r["strategy"]), "-"))[0]
        ax.scatter(
            r["cost_per_1k"],
            100 * r["quality"],
            s=150 if r["strategy"].startswith("oracle") else 95,
            marker=MARK.get(r["strategy"], "o"),
            color=col,
            zorder=3,
            edgecolor="white",
            label=f"{short(r['strategy'])}: {100 * r['quality']:.1f}, ${r['cost_per_1k']:.3f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("USD per 1,000 requests, router included (log scale)")
    ax.set_ylabel("quality: mean grader score x 100")
    ax.set_title(f"Quality against cost on {s['n_test']} held-out requests", loc="left")
    ax.grid(color=GRID)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG / "quality-vs-cost.png", bbox_inches="tight")
    plt.close(fig)

    # 2. price sensitivity: the top tier at k times the Nova Pro price
    ks = s["price_multipliers"]
    pick = [k for k in key if k in s["sensitivity"] and k not in ("always micro", "always scout")]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)
    for name in pick:
        ys = s["sensitivity"][name]
        col, ls = STYLE.get(name, (color(name), "-"))
        ax.plot(
            ks,
            ys,
            marker="o",
            ms=4,
            lw=2.6 if name == "always pro" else 1.8,
            color=col,
            linestyle=ls,
            label=short(name),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks, [f"{k}x" for k in ks])
    ax.set_xlabel("top-tier price as a multiple of Nova Pro (1x = $0.80 / $3.20 per million tokens)")
    ax.set_ylabel("USD per 1,000 requests (log)")
    ax.set_title("When does a router pay for itself? Same tokens, same decisions, pricier top tier", loc="left")
    ax.grid(color=GRID, which="both", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG / "price-sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    # 3. latency the router adds before the answer starts
    lat = [r for r in rows if r.get("router_latency_p50", 0) > 0 or r["strategy"].startswith("bedrock")]
    lat.sort(key=lambda r: r.get("router_latency_p50", 0))
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(lat) + 1.5), dpi=130)
    ax.barh(
        [short(r["strategy"]) for r in lat],
        [r.get("router_latency_p50", 0) for r in lat],
        color=[color(r["strategy"]) for r in lat],
    )
    for i, r in enumerate(lat):
        v = r.get("router_latency_p50", 0)
        ax.text(
            v + 0.1,
            i,
            "routes inside the answer call" if r["strategy"].startswith("bedrock") else f"{v:.2f} s",
            va="center",
            fontsize=11,
        )
    ax.set_xlabel("median seconds added before the answer starts")
    ax.set_title("Latency added by the router", loc="left")
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    fig.savefig(IMG / "router-latency.png")
    plt.close(fig)

    # 4. per-family quality of each model (test split)
    fam = s["per_family_test"]
    models = ["micro", "lite", "pro", "scout", "llama8b"]
    labels = {
        "micro": "Nova Micro",
        "lite": "Nova Lite",
        "pro": "Nova Pro",
        "scout": "Llama 4 Scout",
        "llama8b": "Llama 3.1 8B",
    }
    fams = list(fam)
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
    data = [[100 * fam[f][m] for m in models] for f in fams]
    im = ax.imshow(data, cmap="YlGn", vmin=40, vmax=100, aspect="auto")
    ax.set_xticks(range(len(models)), [labels[m] for m in models])
    ax.set_yticks(range(len(fams)), fams)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=12, color=INK)
    ax.set_title("Quality by task family and model (test split, original prompts)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG / "family-quality.png")
    plt.close(fig)

    # 5. where each strategy sends traffic
    mix_rows = [r for r in rows if r.get("mix") and not r["strategy"].startswith("always")]
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(mix_rows) + 1.5), dpi=130)
    left = [0.0] * len(mix_rows)
    for tier, col in (
        ("micro", "#059669"),
        ("lite", "#2563EB"),
        ("pro", "#A3418F"),
        ("scout", "#D97706"),
        ("llama8b", "#9CA3AF"),
    ):
        vals = [100 * r["mix"].get(tier, 0) for r in mix_rows]
        if any(vals):
            ax.barh([short(r["strategy"]) for r in mix_rows], vals, left=left, color=col, label=labels[tier])
            left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_xlabel("% of requests")
    ax.set_title("Where each strategy sends the requests", loc="left")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG / "routing-mix.png")
    plt.close(fig)
    # 6. ladder B: Nova Micro or Claude Sonnet 4.6, real prices
    fl = s.get("frontier_ladder", {}).get("strategies", [])
    if fl:
        top = s["frontier_ladder"]["top"]
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)
        p = RES / "curve-decider-frontier.json"
        if p.exists():
            cv = json.loads(p.read_text())
            ax.plot(
                [c["test"]["cost_per_1k"] for c in cv],
                [100 * c["test"]["quality"] for c in cv],
                "-",
                color="#FF9900",
                lw=1.6,
                alpha=0.7,
                label="decider (multi-step question): every threshold",
            )
        cols = {
            "always micro": "#6B7280",
            f"always {top}": "#111827",
            f"oracle (micro/{top})": "#9CA3AF",
            "classifier-micro-hard": "#2563EB",
            "classifier-lite-hard": "#60A5FA",
            "classifier-haiku45-hard": "#D97757",
            "classifier-sonnet46-hard": "#7C3AED",
        }
        for r in fl:
            col = cols.get(r["strategy"], "#B45309" if r["strategy"].startswith("decider") else INK)
            ax.scatter(
                r["cost_per_1k"],
                100 * r["quality"],
                s=110,
                color=col,
                edgecolor="white",
                zorder=3,
                marker="*" if r["strategy"].startswith("oracle") else "o",
                label=f"{short(r['strategy']).replace('-hard', '')}: {100 * r['quality']:.1f}, ${r['cost_per_1k']:.2f}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("USD per 1,000 requests, router included (log scale)")
        ax.set_ylabel("quality: mean grader score x 100")
        ax.set_title("Ladder B: Nova Micro for easy requests, Claude Sonnet 4.6 for hard ones", loc="left")
        ax.grid(color=GRID)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
        fig.tight_layout()
        fig.savefig(IMG / "ladder-b.png", bbox_inches="tight")
        plt.close(fig)
    # 7. Advanced Prompt Optimization: original against optimized prompt, held-out test, per (family, model)
    pp = RES / "apo_pairs.json"
    if pp.exists():
        pairs = sorted(json.loads(pp.read_text()), key=lambda p: (p["family"], p["model"]))
        names = {
            "micro": "Nova Micro",
            "lite": "Nova Lite",
            "pro": "Nova Pro",
            "haiku45": "Haiku 4.5",
            "sonnet46": "Sonnet 4.6",
        }
        fig, ax = plt.subplots(figsize=(11, 0.5 * len(pairs) + 1.8), dpi=130)
        for k, p_ in enumerate(pairs):
            a_, b_ = 100 * p_["test_original"], 100 * p_["test_optimized"]
            col = "#059669" if b_ > a_ + 0.5 else "#DC2626" if b_ < a_ - 0.5 else "#6B7280"
            ax.plot([a_, b_], [k, k], color=col, lw=2.5, zorder=2)
            ax.scatter([a_], [k], color="#9CA3AF", s=60, zorder=3)
            ax.scatter([b_], [k], color=col, s=80, zorder=3)
            ax.text(
                101.5,
                k,
                f"{p_['tokens_in_original']:.0f} -> {p_['tokens_in_optimized']:.0f} input tokens",
                va="center",
                fontsize=10,
                color=MUTED,
            )
        ax.set_yticks(range(len(pairs)), [f"{p_['family']} / {names.get(p_['model'], p_['model'])}" for p_ in pairs])
        ax.set_xlim(min(40, min(100 * min(p_["test_original"], p_["test_optimized"]) for p_ in pairs) - 5), 125)
        ax.set_xticks(range(40, 101, 10))
        ax.set_xlabel("held-out test quality (grey: original prompt, coloured: APO-optimized prompt)")
        ax.set_title("Bedrock Advanced Prompt Optimization, measured on the test split", loc="left")
        ax.grid(axis="x", color=GRID)
        fig.tight_layout()
        fig.savefig(IMG / "apo-pairs.png", bbox_inches="tight")
        plt.close(fig)
    print("charts ->", IMG)


if __name__ == "__main__":
    main()
