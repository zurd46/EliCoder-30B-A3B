"""
02 · SFT (RunPod-Version) — Qwen3-Coder-30B 4-bit + LoRA, ~12 h auf H100.

GPU-Optimierungen gegenüber Colab:
  - TF32 matmul (Hopper beschleunigt)
  - per_device_train_batch_size 1 → 2 (nutzt 94GB H100-NVL VRAM)
  - gradient_accumulation_steps 64 → 32 (gleicher effektiver Batch, halber Overhead)
  - Checkpoint-Mirror auf HF Hub (auto-resume bei Spot-Interrupt)
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
os.environ.setdefault("WANDB_NAME", "sft-phase-a-runpod")

patch_unsloth_telemetry()

import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

TOK = os.environ["HF_TOKEN"]
CFG = yaml.safe_load(Path("configs/sft.yaml").read_text())
CKPT = checkpoint_dir("sft-phase-a")
print(f"checkpoints → {CKPT}")

# H100-NVL (94GB) Overrides — doppelter Batch, halbe Accum
TRAIN = dict(CFG["training"])
gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
if gpu_total >= 88:
    TRAIN["per_device_train_batch_size"] = 2
    TRAIN["gradient_accumulation_steps"] = max(1, TRAIN["gradient_accumulation_steps"] // 2)
    print(f"H100-NVL detected ({gpu_total:.0f}GB) — bsz=2, grad_accum={TRAIN['gradient_accumulation_steps']}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=TRAIN["max_seq_length"],
    dtype=None,
    load_in_4bit=CFG["load_in_4bit"],
    token=TOK,
    device_map={"": 0},
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

ds = load_dataset(TRAIN["dataset"], split=TRAIN["split"], token=TOK)

def fmt(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )}

ds = ds.map(fmt, remove_columns=ds.column_names)

args = SFTConfig(
    output_dir=str(CKPT),
    per_device_train_batch_size=TRAIN["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAIN["gradient_accumulation_steps"],
    warmup_steps=TRAIN["warmup_steps"],
    num_train_epochs=TRAIN["num_train_epochs"],
    learning_rate=TRAIN["learning_rate"],
    bf16=TRAIN["bf16"],
    optim=TRAIN["optim"],
    weight_decay=TRAIN["weight_decay"],
    lr_scheduler_type=TRAIN["lr_scheduler_type"],
    logging_steps=TRAIN["logging_steps"],
    save_strategy=TRAIN["save_strategy"],
    save_steps=TRAIN["save_steps"],
    save_total_limit=TRAIN["save_total_limit"],
    seed=TRAIN["seed"],
    max_seq_length=TRAIN["max_seq_length"],
    packing=TRAIN.get("packing", True),
    dataset_num_proc=TRAIN.get("dataset_num_proc", 4),
    remove_unused_columns=TRAIN.get("remove_unused_columns", False),
    eval_strategy=TRAIN.get("eval_strategy", "no"),
    dataloader_pin_memory=TRAIN.get("dataloader_pin_memory", False),
    gradient_checkpointing=TRAIN.get("gradient_checkpointing", True),
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=CFG["output"]["hf_repo"],
    hub_strategy="checkpoint",
    hub_token=TOK,
    hub_private_repo=True,
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=args)

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
print(f"SFT done → {CFG['output']['hf_repo']}")

Path("/workspace/.phase02_done").touch()
