# %% [markdown]
# # 05 · Export — Merge + Push Full-Precision Safetensors
#
# Input:  LoRA v3 aus 04_longctx
# Output: gemergtes BF16-Modell auf HF (~60 GB)
#
# Lokaler Schritt danach:
#
#   git clone zurd46/coder-16b-dyn-base-fp16
#   cd build_pipeline && coderllm-build convert-gguf && coderllm-build convert-mlx
#   coderllm-build upload

# %%
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
#                "transformers>=4.46" "peft>=0.13" "accelerate>=1.0" huggingface_hub

# %%
import os, torch, shutil
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

TOK = os.environ["HF_TOKEN"]
OWNER = "zurd46"
BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
ADAPTER = f"{OWNER}/coder-16b-dyn-lora-longctx"
MERGED_REPO = f"{OWNER}/coder-16b-dyn-base-fp16"
OUT = Path("/content/merged")

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=4096,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    token=TOK,
)

model = PeftModel.from_pretrained(model, ADAPTER, token=TOK)
model = model.merge_and_unload()

# %%
OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(OUT)
print("merge done, uploading …")

# %%
create_repo(MERGED_REPO, repo_type="model", private=True, exist_ok=True, token=TOK)
api = HfApi(token=TOK)
api.upload_folder(
    folder_path=str(OUT),
    repo_id=MERGED_REPO,
    repo_type="model",
    commit_message="merged SFT+DPO+LongCtx adapters into BF16 base",
)
print(f"merged model → https://huggingface.co/{MERGED_REPO}")

# %% [markdown]
# ## Nächster Schritt (lokal, nicht in Colab)
#
# ```bash
# cd build_pipeline
# # temporär configs/quants.yaml anpassen:
# #   base.hf_repo: zurd46/coder-16b-dyn-base-fp16
# coderllm-build download
# coderllm-build convert-mlx
# coderllm-build convert-gguf
# coderllm-build package --kind all
# coderllm-build upload
# ```
