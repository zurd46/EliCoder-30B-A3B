"""
03 · DPO (RunPod-Version) — Input: SFT-LoRA, Output: DPO-LoRA, ~4 h auf H100.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap, apply_gpu_optims, checkpoint_dir, patch_unsloth_telemetry

bootstrap(install=True)

import torch
assert torch.cuda.is_available(), "CUDA required"
apply_gpu_optims()

os.environ.setdefault("WANDB_PROJECT", "CoderLLM")
os.environ.setdefault("WANDB_NAME", "dpo-phase-b-runpod")

patch_unsloth_telemetry()

import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from transformers.trainer_utils import get_last_checkpoint

PatchDPOTrainer()

TOK = os.environ["HF_TOKEN"]
CFG = yaml.safe_load(Path("configs/dpo.yaml").read_text())
CKPT = checkpoint_dir("dpo-phase-b")
print(f"checkpoints → {CKPT}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=CFG["training"]["max_seq_length"],
    load_in_4bit=True,
    device_map={"": 0},
    token=TOK,
)
model = PeftModel.from_pretrained(model, CFG["adapter_repo"], is_trainable=True, token=TOK)

ds = load_dataset(CFG["training"]["dataset"], split=CFG["training"]["split"], token=TOK)

args = DPOConfig(
    output_dir=str(CKPT),
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
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    push_to_hub=True,
    hub_model_id=CFG["output"]["hf_repo"],
    hub_strategy="checkpoint",
    hub_token=TOK,
    hub_private_repo=True,
)

trainer = DPOTrainer(
    model=model, ref_model=None, args=args,
    train_dataset=ds, tokenizer=tokenizer,
)

last_ckpt = get_last_checkpoint(str(CKPT))
if last_ckpt:
    print(f"RESUME von {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("kein Checkpoint gefunden — fresh run")
    trainer.train()

model.save_pretrained(str(CKPT))
tokenizer.save_pretrained(str(CKPT))
model.push_to_hub(CFG["output"]["hf_repo"], token=TOK, private=True)
tokenizer.push_to_hub(CFG["output"]["hf_repo"], token=TOK, private=True)
print(f"DPO done → {CFG['output']['hf_repo']}")

Path("/workspace/.phase03_done").touch()
