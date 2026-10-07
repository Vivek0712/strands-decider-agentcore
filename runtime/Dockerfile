# Strands Decider on Amazon Bedrock AgentCore Runtime (linux/arm64, CPU).
FROM public.ecr.aws/docker/library/python:3.12-slim-trixie

# git: pip installs strands-decider from GitHub at a pinned commit (PyPI 0.1.0 predates the
# current inference code)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=1048576 \
    MALLOC_TRIM_THRESHOLD_=1048576

# CPU-only torch first, pinned, so nothing later pulls a CUDA build into the image
ARG TORCH_VERSION=2.8.0
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

ARG STRANDS_DECIDER_REF=3e94e9d84c620ed5a95f1a3310c3decb971e261c
COPY requirements.txt /tmp/requirements.txt
RUN echo "torch==${TORCH_VERSION}" > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/requirements.txt \
       "strands-decider @ git+https://github.com/strands-labs/strands-decider@${STRANDS_DECIDER_REF}"

RUN useradd -m -u 1000 bedrock_agentcore && mkdir -p /tmp/hf && chown bedrock_agentcore:bedrock_agentcore /tmp/hf
WORKDIR /app
COPY --chown=bedrock_agentcore:bedrock_agentcore main.py ./
COPY --chown=bedrock_agentcore:bedrock_agentcore decider_runtime/ ./decider_runtime/
USER bedrock_agentcore

# AgentCore Runtime HTTP contract: /invocations and /ping on 8080
EXPOSE 8080
CMD ["python", "main.py"]
