"""Load a Strands Decider checkpoint for CPU inference on an AgentCore microVM.

An AgentCore Runtime session is a 2 vCPU / 8 GB arm64 microVM (Graviton2 class: int8 dot-product
instructions, no bf16 ones). strands-decider's CPU path upcasts the torso to fp32, about 9 GB for a
2B model, which does not fit. So this loader:

  1. downloads the decider (LoRA adapter + head, ~90 MB) at a pinned revision; strands-decider
     then fetches the base model at the revision the adapter was trained on (~4.6 GB, ~20 s);
  2. merges the LoRA adapter into the base weights (same function, no adapter overhead per call);
  3. int8 mode: replaces every nn.Linear with a dynamically quantized int8 Linear, one layer at a
     time so memory never holds a second full copy, and keeps the remaining small parameters in
     fp32 (CPU-friendly). bf16 mode keeps bf16 weights; fp32 mode upcasts (needs >9 GB).

The engine itself is strands-decider's SystemOneEngine, unchanged.
"""

from __future__ import annotations

import gc
import os
import resource
import threading
import time
from typing import Any

from .config import Settings


def rss_gb() -> float:
    """Peak resident memory of this process, in GB (Linux reports KiB, macOS bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e9 if os.uname().sysname == "Darwin" else 1e6), 2)


def memory_now() -> dict[str, float]:
    """Current anonymous (heap) and file-backed (mmap) resident memory in GB, from /proc on Linux."""
    out: dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                key, _, val = line.partition(":")
                if key in ("VmRSS", "RssAnon", "RssFile"):
                    out[key] = round(int(val.split()[0]) / 1e6, 2)
    except OSError:
        pass
    return out


def release_memory() -> None:
    """Collect garbage and hand freed heap pages back to the OS (glibc keeps them otherwise, so a
    load that briefly needed 2x the weights would look like it still does)."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # not glibc (macOS dev runs)
        pass


def weight_bytes(module: Any) -> dict[str, float]:
    """GB held by the model's tensors, by dtype, counting int8 packed weights too."""

    out: dict[str, float] = {}
    seen: set[int] = set()
    for t in list(module.parameters()) + list(module.buffers()):
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        key = str(t.dtype).replace("torch.", "")
        out[key] = out.get(key, 0) + t.numel() * t.element_size()
    for m in module.modules():
        if type(m).__name__ == "Linear" and hasattr(m, "_weight_bias"):
            w, b = m._weight_bias()
            size = w.numel() * w.element_size() + (b.numel() * b.element_size() if b is not None else 0)
            out["int8"] = out.get("int8", 0) + size
    return {k: round(v / 1e9, 2) for k, v in out.items()}


def cpu_limit() -> int:
    """CPUs this process may really use: the cgroup quota when set (docker --cpus, ECS), else the
    affinity mask (AgentCore microVMs expose exactly their vCPUs)."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            quota, period = fh.read().split()
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 2


def _float32_output_embedding(embedding: Any) -> Any:
    """Wrap the bf16 embedding so its output is fp32 for the int8 layers after it. Keeping the
    ~0.5B-parameter embedding table in bf16 saves about 1 GB over upcasting it."""
    import torch

    class Float32Embedding(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        @property
        def weight(self) -> torch.Tensor:  # transformers reads embed_tokens.weight (dtype, tying)
            return self.inner.weight

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.inner(input_ids).float()

    return Float32Embedding(embedding)


def _quantize_linears_int8(module: Any, qconfig: Any = None) -> int:
    """Swap each nn.Linear under `module` for a dynamically quantized int8 Linear, layer by layer."""
    import torch
    from torch.ao.nn.quantized.dynamic import Linear as QLinear

    qconfig = qconfig or torch.ao.quantization.per_channel_dynamic_qconfig
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            child.float()
            child.qconfig = qconfig
            setattr(module, name, QLinear.from_float(child))
            child.weight = None  # drop the fp32 copy now, not when the walk ends
            del child
            count += 1
        else:
            count += _quantize_linears_int8(child, qconfig)
    return count


class DeciderEngine:
    """Owns the model; thread-safe (one forward pass at a time on the shared model)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = threading.Event()
        self.error: str | None = None
        self.info: dict[str, Any] = {"model": settings.display_name, "source": settings.model,
                                     "revision": settings.revision, "quant": settings.quant, "state": "starting"}
        self._engine: Any = None
        self._lock = threading.Lock()

    # ---- loading ------------------------------------------------------------------------------

    def load(self) -> None:
        t0 = time.time()
        try:
            import json

            import torch
            from huggingface_hub import snapshot_download
            from strands_decider.infer import EngineConfig, SystemOneEngine
            from strands_decider.modeling import StrandsDeciderModel

            torch.set_num_threads(self.settings.threads or cpu_limit())
            stages: dict[str, float] = {}
            mark = time.time()

            def stage(name: str) -> None:
                nonlocal mark
                stages[name] = round(time.time() - mark, 1)
                mark = time.time()

            self.info["state"] = "downloading"
            hub = os.path.join(self.settings.hf_home, "hub")
            path = self.settings.model
            if not os.path.isdir(path):
                path = snapshot_download(self.settings.model, revision=self.settings.revision, cache_dir=hub)
                self.info["resolved_commit"] = os.path.basename(path.rstrip("/"))
            stage("download_decider")
            # fetch the base model explicitly, at the revision the adapter was trained on, so its
            # download time is measured on its own (strands-decider would otherwise do it inside load)
            cfg_path = next((os.path.join(path, f) for f in ("strands_decider_config.json", "hobson_config.json")
                             if os.path.exists(os.path.join(path, f))), None)
            if cfg_path:
                with open(cfg_path, encoding="utf-8") as fh:
                    dcfg = json.load(fh)
                if dcfg.get("base_model") and not os.path.isdir(dcfg["base_model"]):
                    snapshot_download(dcfg["base_model"], revision=dcfg.get("base_revision"),
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
            stage("download_base")
            self.info["state"] = "loading"
            with torch.inference_mode():
                # sdpa: memory-efficient attention; the default eager kernel materialises n x n score
                # matrices, which runs a 2 vCPU / 8 GB session out of memory on long inputs
                model = StrandsDeciderModel.load(path, attn_implementation="sdpa")
                if self.settings.max_tokens:  # the prompt window on CPU (longer inputs are shortened)
                    model.config.max_length = min(int(model.config.max_length or self.settings.max_tokens),
                                                  self.settings.max_tokens)
                stage("load")
                torso = model.torso
                if hasattr(torso, "merge_and_unload"):  # fold the LoRA adapter into the base weights
                    model.torso = torso.merge_and_unload()
                release_memory()
                stage("merge_lora")
                if self.settings.quant == "int8":
                    self.info["state"] = "quantizing"
                    # qnnpack keeps about 2x the int8 bytes once weights are packed on first use;
                    # onednn (the arm64 default) keeps about 3x, which costs ~1.4 GB on a 2B model
                    torch.backends.quantized.engine = self.settings.qengine
                    self.info["qengine"] = self.settings.qengine
                    self.info["int8_linears"] = _quantize_linears_int8(model.torso)
                    emb = model.torso.get_input_embeddings()
                    # The weights are memory-mapped from the safetensors file; merging the adapter
                    # wrote into those pages (private copies, ~3 GB). They are freed only when nothing
                    # references the mapping, and the embedding still does: give it its own storage.
                    emb.weight.data = emb.weight.data.clone()
                    for p in model.torso.parameters():  # norms, convs, biases: small, fp32 on CPU
                        if p.data_ptr() != emb.weight.data_ptr():
                            p.data = p.data.float()
                    for b in model.torso.buffers():
                        if b.is_floating_point():
                            b.data = b.data.float()
                    model.torso.set_input_embeddings(_float32_output_embedding(emb))
                    stage("quantize_int8")
                elif self.settings.quant == "fp32":
                    model.torso.float()
                release_memory()
                if self.settings.quant in ("int8", "bf16"):
                    # strands-decider upcasts a half-precision torso to fp32 on CPU (~9 GB for 2B), which
                    # would undo the memory plan above; keep the dtypes chosen here
                    SystemOneEngine._upcast_torso_for_cpu = lambda _self: None  # type: ignore[method-assign]
                engine = SystemOneEngine(model, EngineConfig(device="cpu", model_name=self.settings.display_name))
            self._engine = engine
            self.info["stages_s"] = stages
            release_memory()
            self.info["weights_gb"] = weight_bytes(model)
            self.info["memory_after_load_gb"] = memory_now()
            t_dl = t0 + stages.get("download_decider", 0) + stages.get("download_base", 0)
            t_load = time.time()
            release_memory()
            if self.settings.warmup:
                self.info["state"] = "warming up"
                self._evaluate({"state": "The sky is clear today.",
                                "questions": {"q": {"type": "noul", "instructions": "Is this about the weather?"}}})
            self.info.update(state="ready", download_s=round(t_dl - t0, 1), load_s=round(t_load - t_dl, 1),
                             warmup_s=round(time.time() - t_load, 1), total_s=round(time.time() - t0, 1),
                             peak_rss_gb=rss_gb(), memory_gb=memory_now(), threads=torch.get_num_threads(),
                             base_model=getattr(model.config, "base_model", None),
                             max_length=getattr(model.config, "max_length", None))
            self.ready.set()
        except Exception as e:  # reported on /ping-backed health calls and every invocation
            self.error = f"{type(e).__name__}: {e}"
            self.info.update(state="failed", error=self.error, total_s=round(time.time() - t0, 1))
            self.ready.set()
            raise

    def start_background_load(self) -> threading.Thread:
        t = threading.Thread(target=self._load_logged, name="decider-load", daemon=True)
        t.start()
        return t

    def _load_logged(self) -> None:
        try:
            self.load()
        except Exception:  # noqa: BLE001 - already recorded in self.error
            import traceback

            traceback.print_exc()

    # ---- inference ----------------------------------------------------------------------------

    def wait_ready(self, timeout: float | None = None) -> None:
        if not self.ready.wait(timeout if timeout is not None else self.settings.load_timeout_s):
            raise TimeoutError(f"the model is still loading ({self.info['state']}); try again shortly")
        if self.error:
            raise RuntimeError(f"the model failed to load: {self.error}")

    def _evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        from strands_decider.schema import SystemOneRequest

        req = SystemOneRequest.model_validate(request)
        with self._lock:  # one forward pass at a time: the model and its caches are shared
            resp = self._engine.evaluate(req)
        return resp.model_dump()

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.wait_ready()
        return self._evaluate(request)
