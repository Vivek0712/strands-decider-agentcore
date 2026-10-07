# Contributing

Issues and pull requests are welcome.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,examples]"
pytest && ruff check .
```

- Unit tests use a fake engine and a fake runtime: they need no AWS account and no model download.
- Changes to `runtime/` should be checked in a local container (`docker run --cpus 2 -m 8g ...`, see the README) and, when they can change answers, with `scripts/collect_answers.py` and `scripts/compare_answers.py` against a reference run.
- Report measured numbers with the hardware, model commit and strands-decider commit they came from.
- Never commit `agentcore/aws-targets.json`, `agentcore/.cli/`, account IDs or credentials.

By contributing, you agree that your contributions are licensed under the [MIT-0 license](LICENSE).
