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

from _agent_eval import AgentEvalCallback

PatchDPOTrainer()

TOK = os.environ["HF_TOKEN"]
CFG = yaml.safe_load(Path("configs/dpo.yaml").read_text())
CKPT = checkpoint_dir("dpo-phase-b")
print(f"checkpoints → {CKPT}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=CFG["training"]["max_seq_length"],
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
    token=TOK,
)
model = PeftModel.from_pretrained(model, CFG["adapter_repo"], is_trainable=True, token=TOK)

ds = load_dataset(CFG["training"]["dataset"], split=CFG["training"]["split"], token=TOK)


T = CFG["training"]
eval_frac = float(T.get("eval_fraction", 0) or 0)
eval_ds = None
if eval_frac > 0:
    split = ds.train_test_split(test_size=eval_frac, seed=T["seed"])
    ds, eval_ds = split["train"], split["test"]
    print(f"eval split: {len(ds):,} train / {len(eval_ds):,} eval ({eval_frac:.2%})")

args = DPOConfig(
    output_dir=str(CKPT),
    beta=T["beta"],
    num_train_epochs=T["num_train_epochs"],
    per_device_train_batch_size=T["per_device_train_batch_size"],
    gradient_accumulation_steps=T["gradient_accumulation_steps"],
    learning_rate=T["learning_rate"],
    lr_scheduler_type=T["lr_scheduler_type"],
    warmup_ratio=T["warmup_ratio"],
    bf16=T["bf16"],
    optim=T["optim"],
    gradient_checkpointing=T["gradient_checkpointing"],
    logging_steps=T["logging_steps"],
    save_steps=T["save_steps"],
    save_total_limit=T.get("save_total_limit", 2),
    seed=T["seed"],
    max_length=T["max_seq_length"],
    max_prompt_length=T["max_prompt_length"],
    eval_strategy=T.get("eval_strategy", "no") if eval_ds is not None else "no",
    eval_steps=T.get("eval_steps", 200),
    per_device_eval_batch_size=T.get("per_device_eval_batch_size", 1),
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    push_to_hub=True,
    hub_model_id=CFG["output"]["hf_repo"],
    hub_strategy="end",
    hub_token=TOK,
    hub_private_repo=True,
)

trainer = DPOTrainer(
    model=model, ref_model=None, args=args,
    train_dataset=ds, eval_dataset=eval_ds,
    tokenizer=tokenizer,
    callbacks=[AgentEvalCallback(tokenizer)],
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
