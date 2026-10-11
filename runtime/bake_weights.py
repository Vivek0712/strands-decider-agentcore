"""Download a decider and its base model into the image at build time (cold starts then read local files).

    HF_HOME=/opt/hf python bake_weights.py StrandsAgents/strands-decider-2B-hobson-v19 <commit>
"""
import json
import os
import sys

from huggingface_hub import snapshot_download

repo, rev = sys.argv[1], sys.argv[2]
hub = os.path.join(os.environ["HF_HOME"], "hub")
path = snapshot_download(repo, revision=rev, cache_dir=hub)
for name in ("strands_decider_config.json", "hobson_config.json"):
    cfg = os.path.join(path, name)
    if os.path.exists(cfg):
        d = json.load(open(cfg))
        if d.get("base_model"):
            snapshot_download(d["base_model"], revision=d.get("base_revision"), cache_dir=hub,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        break
print("baked", repo, rev, "->", path)
