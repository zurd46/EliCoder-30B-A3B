# %% [markdown]
# # 03 · DPO mit Unsloth (Phase B)
#
# Input:  LoRA v1 aus 02_sft
# Output: LoRA v2 (Senior-Dev-Verhalten geprägt)
# Dauer:  ~4 h H100.

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
            for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
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

        gh_tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        effective_url = repo_url
        if gh_tok and repo_url.startswith("https://github.com/"):
            effective_url = repo_url.replace("https://", f"https://x-access-token:{gh_tok}@", 1)
            print("using GitHub token for private repo clone")

        if not repo_dir.exists():
            print(f"cloning {repo_url} -> {repo_dir}")
            res = subprocess.run(["git", "clone", "--depth", "1", effective_url, str(repo_dir)])
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

_bootstrap(["unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git", "trl>=0.12", "transformers>=4.46", "datasets>=3.0", "peft>=0.13", "accelerate>=1.0", "bitsandbytes", "wandb", "pyyaml"])
# %%
import os, yaml, torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig

PatchDPOTrainer()

TOK = os.environ["HF_TOKEN"]
CFG = yaml.safe_load(Path("configs/dpo.yaml").read_text())

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=CFG["training"]["max_seq_length"],
    load_in_4bit=True,
    token=TOK,
)

from peft import PeftModel
model = PeftModel.from_pretrained(model, CFG["adapter_repo"], is_trainable=True, token=TOK)

# %%
ds = load_dataset(CFG["training"]["dataset"], split=CFG["training"]["split"], token=TOK)

# %%
args = DPOConfig(
    output_dir=CFG["output"]["local_dir"],
    beta=CFG["training"]["beta"],
    num_train_epochs=CFG["training"]["num_train_epochs"],
    per_device_train_batch_size=CFG["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=CFG["training"]["gradient_accumulation_steps"],
    learning_rate=CFG["training"]["learning_rate"],
    lr_scheduler_type=CFG["training"]["lr_scheduler_type"],
    warmup_ratio=CFG["training"]["warmup_ratio"],
    bf16=CFG["training"]["bf16"],
    optim=CFG["training"]["optim"],
    gradient_checkpointing=CFG["training"]["gradient_checkpointing"],
    logging_steps=CFG["training"]["logging_steps"],
    save_steps=CFG["training"]["save_steps"],
    seed=CFG["training"]["seed"],
    max_length=CFG["training"]["max_seq_length"],
    max_prompt_length=CFG["training"]["max_prompt_length"],
    report_to="wandb",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=ds,
    tokenizer=tokenizer,
)
trainer.train()

# %%
out = CFG["output"]["local_dir"]
model.save_pretrained(out)
tokenizer.save_pretrained(out)
model.push_to_hub(CFG["output"]["hf_repo"], token=TOK, private=True)
tokenizer.push_to_hub(CFG["output"]["hf_repo"], token=TOK, private=True)
print(f"DPO done → {CFG['output']['hf_repo']}")
