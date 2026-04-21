"""
05 · Export (RunPod-Version) — merge LoRA v3 ins BF16-Base, push nach HF, ~40 min.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap, apply_gpu_optims, patch_unsloth_telemetry

bootstrap(install=True)

import torch
assert torch.cuda.is_available(), "CUDA required"
apply_gpu_optims()

patch_unsloth_telemetry()

from unsloth import FastLanguageModel
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

TOK = os.environ["HF_TOKEN"]
OWNER = os.environ.get("CODERLLM_HF_OWNER", "zurd46")
BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
ADAPTER = f"{OWNER}/EliCoder-30B-A3B-LoRA-LongCtx"
MERGED_REPO = f"{OWNER}/EliCoder-30B-A3B"
OUT = Path("/workspace/merged")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=4096,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    token=TOK,
)

model = PeftModel.from_pretrained(model, ADAPTER, token=TOK)
model = model.merge_and_unload()

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(OUT)
print("merge done, uploading …")

create_repo(MERGED_REPO, repo_type="model", private=True, exist_ok=True, token=TOK)
api = HfApi(token=TOK)
api.upload_folder(
    folder_path=str(OUT),
    repo_id=MERGED_REPO,
    repo_type="model",
    commit_message="merged SFT+DPO+LongCtx adapters into BF16 base",
)
print(f"merged model → https://huggingface.co/{MERGED_REPO}")

Path("/workspace/.phase05_done").touch()
print("\nPIPELINE COMPLETE — pod kann gestoppt werden")
