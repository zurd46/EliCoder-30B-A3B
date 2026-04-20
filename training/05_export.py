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

# %% [markdown]
# ## Colab Bootstrap

# %%
def _bootstrap(pip_extras):
    import os, subprocess, sys
    from pathlib import Path
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except Exception:
        in_colab = False

    repo_url = os.environ.get("CODERLLM_REPO_URL", "https://github.com/zurd46/CoderLLM.git")
    repo_dir = Path(os.environ.get("CODERLLM_DIR", "/content/CoderLLM"))

    if in_colab:
        try:
            from google.colab import userdata
            for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                v = userdata.get(key)
                if v:
                    os.environ[key] = v
            if os.environ.get("HF_TOKEN"):
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
                print("HF_TOKEN loaded from Colab Secrets")
            else:
                print("WARN: HF_TOKEN not found in Colab Secrets — add it before proceeding")
        except Exception:
            pass

        if not repo_dir.exists():
            print(f"cloning {repo_url} -> {repo_dir}")
            res = subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])
            if res.returncode != 0:
                print("=" * 70)
                print(f"git clone FAILED for {repo_url}")
                print()
                print("Wahrscheinliche Ursachen:")
                print("  1. Repo existiert noch nicht auf GitHub")
                print("  2. Repo ist privat (braucht Token im URL)")
                print()
                print("Fixes:")
                print("  A) Repo pushen — lokal auf deinem Mac:")
                print("       cd CoderLLM")
                print("       gh repo create zurd46/CoderLLM --public --source=. --push")
                print()
                print("  B) Privates Repo: setze CODERLLM_REPO_URL vor dem Run, z.B.:")
                print("       import os")
                print("       os.environ['CODERLLM_REPO_URL'] = 'https://TOKEN@github.com/zurd46/CoderLLM.git'")
                print()
                print("  C) Anderer Namespace: setze CODERLLM_REPO_URL auf dein Repo")
                print("=" * 70)
                raise RuntimeError("repo clone failed — see instructions above")

        os.chdir(repo_dir / "training")
        print(f"cwd = {repo_dir / 'training'}")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pip_extras], check=False)

_bootstrap(["unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git", "transformers>=4.46", "peft>=0.13", "accelerate>=1.0", "huggingface_hub>=0.25"])
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
