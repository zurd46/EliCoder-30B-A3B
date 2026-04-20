# %% [markdown]
# # 03 · DPO mit Unsloth (Phase B)
#
# Input:  LoRA v1 aus 02_sft
# Output: LoRA v2 (Senior-Dev-Verhalten geprägt)
# Dauer:  ~4 h H100.

# %%
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
#                "trl>=0.12" "transformers>=4.46" "datasets>=3.0" "peft>=0.13" \
#                "accelerate>=1.0" bitsandbytes wandb pyyaml

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
