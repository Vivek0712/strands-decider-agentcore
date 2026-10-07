"""Runtime settings, read once from environment variables (set per runtime in agentcore.json)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

QUANT_MODES = ("int8", "bf16", "fp32")


@dataclass(frozen=True)
class Settings:
    model: str = "StrandsAgents/strands-decider-2B-hobson-v21"  # Hugging Face repo id or a local directory
    revision: str | None = None  # a commit hash pins the exact weights
    quant: str = "int8"  # int8: fits and runs fast on the 2 vCPU / 8 GB microVM; bf16 / fp32 for comparison
    model_name: str = ""  # reported in every response; defaults to the repo name
    threads: int = 0  # torch threads; 0 = one per vCPU
    qengine: str = "qnnpack"  # int8 kernel backend on arm64: qnnpack (lean) or onednn
    max_tokens: int = 4096  # prompt window; inputs longer than this are shortened by strands-decider
    max_questions: int = 16  # per request, guards latency (each question is one forward pass on CPU)
    max_batch: int = 8  # requests per batch call
    load_timeout_s: float = 900.0  # how long an invocation waits for a cold model
    warmup: bool = True  # answer one question after loading so the first real call is not the slowest
    hf_home: str = field(default_factory=lambda: os.environ.get("HF_HOME", "/tmp/hf"))

    @property
    def display_name(self) -> str:
        if self.model_name:
            return self.model_name
        base = self.model.rstrip("/").split("/")[-1]
        return f"{base}@{self.revision[:7]}" if self.revision else base

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        quant = e.get("DECIDER_QUANT", "int8").lower()
        if quant not in QUANT_MODES:
            raise ValueError(f"DECIDER_QUANT must be one of {QUANT_MODES}, got {quant!r}")
        return cls(
            model=e.get("DECIDER_MODEL", cls.model),
            revision=e.get("DECIDER_REVISION") or None,
            quant=quant,
            model_name=e.get("DECIDER_MODEL_NAME", ""),
            threads=int(e.get("DECIDER_THREADS", "0")),
            qengine=e.get("DECIDER_QENGINE", "qnnpack"),
            max_tokens=int(e.get("DECIDER_MAX_TOKENS", "4096")),
            max_questions=int(e.get("DECIDER_MAX_QUESTIONS", "16")),
            max_batch=int(e.get("DECIDER_MAX_BATCH", "8")),
            load_timeout_s=float(e.get("DECIDER_LOAD_TIMEOUT_S", "900")),
            warmup=e.get("DECIDER_WARMUP", "1") not in ("0", "false", "no"),
            hf_home=e.get("HF_HOME", "/tmp/hf"),
        )
