"""Build the routing benchmark: 6 task families, 36 requests each, every one graded automatically.

    python bench/build_dataset.py            # writes data/tasks.jsonl

| family  | source                                   | licence    | typical difficulty |
|---------|------------------------------------------|------------|--------------------|
| extract | synthetic order notes (seeded)           | this repo  | easy               |
| sports  | BIG-Bench Hard sports_understanding      | MIT        | easy to medium     |
| json    | synthetic contact notes to JSON (seeded) | this repo  | medium             |
| math    | GSM8K test                               | MIT        | medium to hard     |
| code    | MBPP sanitized test (run against tests)  | CC-BY-4.0  | medium to hard     |
| logic   | BIG-Bench Hard: logical deduction (5),   | MIT        | hard               |
|         | shuffled objects (5), date understanding |            |                    |

Each row: {id, family, split, prompt, input, answer, grader} where `prompt` is the full request a
user would send, `input` the variable part (for prompt templates), and `answer` what the grader needs.
Half of each family is `train` (used to tune routers and by prompt optimization), half `test`.
"""

from __future__ import annotations

import json
import pathlib
import random

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parent.parent
N = 36
rng = random.Random(20261008)

# One instruction per family. {input} is the variable part; templates.py exposes the same text
# with {{input}} for Bedrock prompt optimization.
TEMPLATES = {
    "extract": "Read the order note and answer the question with only the value, nothing else.\n\n{input}",
    "sports": "Is the following sentence plausible? Answer with only yes or no.\n\n{input}",
    "json": (
        "Convert the contact note into JSON with exactly these keys: name, email, phone, city, company. "
        "Use null for anything missing. Output only the JSON object.\n\n{input}"
    ),
    "math": "Solve the problem. Show brief working, then end with a final line 'Answer: <number>'.\n\n{input}",
    "code": (
        "Write a Python function for the task below. Output only one ```python code block with the function "
        "and any imports, no tests and no explanation.\n\n{input}"
    ),
    "logic": (
        "Answer the multiple-choice question. Think step by step, then end with a final line "
        "'Answer: (X)' where X is the option letter.\n\n{input}"
    ),
}


def parquet(repo: str, path: str) -> pd.DataFrame:
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset"))


# ---- synthetic families ------------------------------------------------------------------------

FIRST = ["Priya", "Marco", "Aisha", "Tom", "Lena", "Diego", "Hannah", "Kenji", "Sofia", "Ravi", "Emma", "Omar"]
LAST = ["Nair", "Rossi", "Khan", "Walsh", "Schmidt", "Fernandez", "Brooks", "Ono", "Costa", "Iyer", "Lund", "Haddad"]
CITY = ["Madurai", "Lisbon", "Austin", "Leeds", "Munich", "Osaka", "Toronto", "Nairobi", "Seville", "Perth"]
COMPANY = ["Acme Robotics", "Bluefin Labs", "Northwind", "Kestrel Foods", "Orbit Freight", "Tandem Health"]
ITEMS = ["USB-C hubs", "desk lamps", "label printers", "ergonomic chairs", "4K monitors", "standing desks"]


def extract_rows() -> list[dict]:
    rows = []
    for _ in range(N):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        qty, item = rng.randint(2, 40), rng.choice(ITEMS)
        price = rng.choice([19.5, 24.99, 39.0, 129.0, 249.99, 399.0])
        day, month = rng.randint(1, 28), rng.choice(["March", "April", "May", "June", "July"])
        order = f"ORD-{rng.randint(10000, 99999)}"
        note = (
            f"Order {order} was placed by {name} on {day} {month} 2026 for {qty} {item} at "
            f"${price:.2f} each, shipping to {rng.choice(CITY)}."
        )
        q, a = rng.choice(
            [
                ("How many units were ordered?", str(qty)),
                ("What is the order number?", order),
                ("Who placed the order?", name),
                ("What was the unit price in dollars?", f"{price:.2f}"),
            ]
        )
        rows.append({"input": f"Order note: {note}\nQuestion: {q}", "answer": a, "grader": "extract"})
    return rows


def json_rows() -> list[dict]:
    rows = []
    for _ in range(N):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        rec = {
            "name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}@{rng.choice(['example.com', 'mail.test', 'corp.example'])}",
            "phone": f"+{rng.choice([1, 44, 91, 49])} {rng.randint(200, 999)} {rng.randint(100, 999)} "
            f"{rng.randint(1000, 9999)}",
            "city": rng.choice(CITY),
            "company": rng.choice(COMPANY),
        }
        for k in rng.sample(["email", "phone", "city", "company"], rng.choice([0, 1, 1, 2])):
            rec[k] = None
        bits = [f"spoke with {rec['name']}"]
        if rec["company"]:
            bits.append(rng.choice([f"who works at {rec['company']}", f"({rec['company']})"]))
        if rec["city"]:
            bits.append(rng.choice([f"based in {rec['city']}", f"now living in {rec['city']}"]))
        if rec["email"]:
            bits.append(rng.choice([f"email is {rec['email']}", f"reach her/him at {rec['email']}"]))
        if rec["phone"]:
            bits.append(rng.choice([f"mobile {rec['phone']}", f"call on {rec['phone']}"]))
        head, tail = bits[0], bits[1:]
        rng.shuffle(tail)
        note = "Met at the expo - " + head + ", " + ", ".join(tail) + ". Follow up next week about the pilot."
        rows.append({"input": f"Contact note: {note}", "answer": json.dumps(rec), "grader": "json"})
    return rows


# ---- public families ---------------------------------------------------------------------------


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def math_rows() -> list[dict]:
    df = sample(parquet("openai/gsm8k", "main/test-00000-of-00001.parquet"), N, 1)
    return [
        {"input": r.question, "answer": r.answer.split("####")[-1].strip().replace(",", ""), "grader": "math"}
        for r in df.itertuples()
    ]


def code_rows() -> list[dict]:
    df = sample(parquet("google-research-datasets/mbpp", "sanitized/test-00000-of-00001.parquet"), N, 2)
    rows = []
    for r in df.itertuples():
        tests = list(r.test_list)
        # the function name and signature come from the first test, as in the MBPP paper's prompt
        task = f"{r.prompt}\nYour function must pass this example: {tests[0]}"
        rows.append(
            {
                "input": task,
                "answer": json.dumps({"tests": tests, "imports": list(r.test_imports)}),
                "grader": "code",
                "reference": r.code,
            }
        )
    return rows


def bbh(task: str, n: int, seed: int) -> list[dict]:
    df = sample(parquet("lukaemon/bbh", f"{task}/test-00000-of-00001.parquet"), n, seed)
    return [
        {"input": r.input, "answer": r.target.strip(), "grader": "choice", "subtask": task} for r in df.itertuples()
    ]


def sports_rows() -> list[dict]:
    rows = bbh("sports_understanding", N, 3)
    for r in rows:
        r["input"] = r["input"].replace("Is the following sentence plausible? ", "").strip('"')
        r["grader"] = "yesno"
    return rows


def logic_rows() -> list[dict]:
    return (
        bbh("logical_deduction_five_objects", 12, 4)
        + bbh("tracking_shuffled_objects_five_objects", 12, 5)
        + bbh("date_understanding", 12, 6)
    )


def main() -> None:
    fams = {
        "extract": extract_rows(),
        "sports": sports_rows(),
        "json": json_rows(),
        "math": math_rows(),
        "code": code_rows(),
        "logic": logic_rows(),
    }
    out = ROOT / "data" / "tasks.jsonl"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with out.open("w") as f:
        for fam, rows in fams.items():
            idx = list(range(len(rows)))
            random.Random(fam).shuffle(idx)
            for k, i in enumerate(idx):
                r = rows[i]
                row = {
                    "id": f"{fam}-{i:02d}",
                    "family": fam,
                    "split": "train" if k % 2 == 0 else "test",
                    "prompt": TEMPLATES[fam].format(input=r["input"]),
                    **r,
                }
                f.write(json.dumps(row) + "\n")
                n += 1
    (ROOT / "data" / "templates.json").write_text(
        json.dumps({k: v.replace("{input}", "{{input}}") for k, v in TEMPLATES.items()}, indent=2) + "\n"
    )
    print(f"{n} tasks -> {out}")


if __name__ == "__main__":
    main()
