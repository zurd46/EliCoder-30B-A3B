# %% [markdown]
# # 02 · SFT mit Unsloth (Phase A)
#
# Runtime: Colab H100 80 GB.
# Dauer: ~12 h (Unsloth macht ~2× vanilla).

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
# %% [markdown]
# ## Unsloth Telemetry-Patch
# Unsloth 2026.4.6 ruft beim Model-Load einen HF-Stats-Endpoint, der in Colab
# regelmäßig 120s hängt. Dieses Patch no-op't die Calls *vor* dem Unsloth-Import.

# %%
import unsloth.models._utils as _u
_u._get_statistics = lambda *a, **kw: None
_u.time_limited_stats_check = lambda *a, **kw: None
print("unsloth stats patched")

# %%
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"] = "CoderLLM"
os.environ["WANDB_NAME"] = "sft-phase-a"

import yaml, torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

TOK = os.environ["HF_TOKEN"]

CFG = yaml.safe_load(Path("configs/sft.yaml").read_text())

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=CFG["training"]["max_seq_length"],
    dtype=None,
    load_in_4bit=CFG["load_in_4bit"],
    token=TOK,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CFG["lora"]["r"],
    lora_alpha=CFG["lora"]["alpha"],
    lora_dropout=CFG["lora"]["dropout"],
    target_modules=CFG["lora"]["target_modules"],
    use_rslora=CFG["lora"]["use_rslora"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# %%
ds = load_dataset(CFG["training"]["dataset"], split=CFG["training"]["split"], token=TOK)

def fmt(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )}

ds = ds.map(fmt, remove_columns=ds.column_names)

# %%
args = SFTConfig(
    output_dir=CFG["output"]["local_dir"],
    per_device_train_batch_size=CFG["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=CFG["training"]["gradient_accumulation_steps"],
    warmup_steps=CFG["training"]["warmup_steps"],
    num_train_epochs=CFG["training"]["num_train_epochs"],
    learning_rate=CFG["training"]["learning_rate"],
    bf16=CFG["training"]["bf16"],
    optim=CFG["training"]["optim"],
    weight_decay=CFG["training"]["weight_decay"],
    lr_scheduler_type=CFG["training"]["lr_scheduler_type"],
    logging_steps=CFG["training"]["logging_steps"],
    save_strategy=CFG["training"]["save_strategy"],
    save_steps=CFG["training"]["save_steps"],
    save_total_limit=CFG["training"]["save_total_limit"],
    seed=CFG["training"]["seed"],
    max_seq_length=CFG["training"]["max_seq_length"],
    packing=True,
    report_to="wandb",
    dataset_text_field="text",
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=args)
trainer.train()

# %%
out = CFG["output"]["local_dir"]
model.save_pretrained(out)
tokenizer.save_pretrained(out)
model.push_to_hub(CFG["output"]["hf_repo"], token=TOK, private=True)
tokenizer.push_to_hub(CFG["output"]["hf_repo"], token=TOK, private=True)
print(f"SFT done → {CFG['output']['hf_repo']}")
