"""
04 · Long-Context (RunPod-Version) — YaRN auf 262k, nur Layer 30-47, ~2 h auf H100.
Nutzt DPO-LoRA als Start, gefriert untere Layer.
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
os.environ.setdefault("WANDB_NAME", "longctx-phase-c-runpod")

patch_unsloth_telemetry()

import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
from transformers.trainer_utils import get_last_checkpoint

TOK = os.environ["HF_TOKEN"]
CFG = yaml.safe_load(Path("configs/longctx.yaml").read_text())
CKPT = checkpoint_dir("longctx-phase-c")
print(f"checkpoints → {CKPT}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=CFG["training"]["max_seq_length"],
    device_map={"": 0},
    load_in_4bit=True,
    dtype=torch.bfloat16,
    rope_scaling={
        "type": CFG["rope"]["rope_scaling_type"],
        "factor": CFG["rope"]["rope_scaling_factor"],
        "original_max_position_embeddings": CFG["rope"]["original_max_position_embeddings"],
    },
    token=TOK,
)

model = PeftModel.from_pretrained(model, CFG["adapter_repo"], is_trainable=True, token=TOK)

freeze_until = CFG["layer_freeze"]["freeze_until_layer"]
for name, p in model.named_parameters():
    frozen = False
    if ".layers." in name:
        try:
            idx = int(name.split(".layers.")[1].split(".")[0])
            if idx < freeze_until:
                p.requires_grad = False
                frozen = True
        except ValueError:
            pass
    if not frozen and "lora" not in name.lower():
        p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {trainable/1e6:.1f} M")

ds = load_dataset(CFG["training"]["dataset"], split=CFG["training"]["split"], token=TOK)

# Dataset tokenization caching
import hashlib
cache_key = hashlib.md5(f"{CFG['training']['dataset']}_{CFG['training']['max_seq_length']}_{tokenizer.name_or_path}".encode()).hexdigest()
cache_file = f"/workspace/cache/longctx_{cache_key}.arrow"

def fmt(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

ds = ds.map(fmt, remove_columns=ds.column_names, cache_file_name=cache_file)

T = CFG["training"]
eval_frac = float(T.get("eval_fraction", 0) or 0)
eval_ds = None
if eval_frac > 0:
    split = ds.train_test_split(test_size=eval_frac, seed=T.get("seed", 42))
    ds, eval_ds = split["train"], split["test"]
    print(f"eval split: {len(ds):,} train / {len(eval_ds):,} eval ({eval_frac:.1%})")

args = SFTConfig(
    output_dir=str(CKPT),
    per_device_train_batch_size=T["per_device_train_batch_size"],
    gradient_accumulation_steps=T["gradient_accumulation_steps"],
    num_train_epochs=T["num_train_epochs"],
    learning_rate=T["learning_rate"],
    lr_scheduler_type=T["lr_scheduler_type"],
    warmup_ratio=T["warmup_ratio"],
    bf16=T["bf16"],
    optim=T["optim"],
    gradient_checkpointing=T["gradient_checkpointing"],
    max_seq_length=T["max_seq_length"],
    packing=False,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    dataset_text_field="text",
    save_steps=T.get("save_steps", 50),
    save_total_limit=T.get("save_total_limit", 2),
    seed=T.get("seed", 42),
    eval_strategy=T.get("eval_strategy", "no") if eval_ds is not None else "no",
    per_device_eval_batch_size=T.get("per_device_eval_batch_size", 1),
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    push_to_hub=True,
    hub_model_id=CFG["output"]["hf_repo"],
    hub_strategy="end",
    hub_token=TOK,
    hub_private_repo=True,
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=ds, eval_dataset=eval_ds,
    args=args,
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
print(f"LongCtx done → {CFG['output']['hf_repo']}")

Path("/workspace/.phase04_done").touch()
