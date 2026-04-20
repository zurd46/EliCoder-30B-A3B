# %% [markdown]
# # 04 · Long-Context Stretch (Phase C)
#
# Ziel: stabile Retention bis 200k Token.
# Strategie: YaRN-Rescaling + Fine-Tune nur Layer 30-47 auf synthetischen
# Needle-in-Haystack-Samples (32k–200k).
# Dauer: ~2 h H100.

# %%
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
#                "trl>=0.12" "transformers>=4.46" "datasets>=3.0" "peft>=0.13" \
#                "accelerate>=1.0" bitsandbytes wandb pyyaml

# %%
import os, yaml, torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from peft import PeftModel

TOK = os.environ["HF_TOKEN"]
CFG = yaml.safe_load(Path("configs/longctx.yaml").read_text())

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=CFG["training"]["max_seq_length"],
    load_in_4bit=True,
    rope_scaling={
        "type": CFG["rope"]["rope_scaling_type"],
        "factor": CFG["rope"]["rope_scaling_factor"],
        "original_max_position_embeddings": CFG["rope"]["original_max_position_embeddings"],
    },
    token=TOK,
)

model = PeftModel.from_pretrained(model, CFG["adapter_repo"], is_trainable=True, token=TOK)

# %%
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

# %%
ds = load_dataset(CFG["training"]["dataset"], split=CFG["training"]["split"], token=TOK)

def fmt(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

ds = ds.map(fmt, remove_columns=ds.column_names)

# %%
args = SFTConfig(
    output_dir=CFG["output"]["local_dir"],
    per_device_train_batch_size=CFG["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=CFG["training"]["gradient_accumulation_steps"],
    num_train_epochs=CFG["training"]["num_train_epochs"],
    learning_rate=CFG["training"]["learning_rate"],
    lr_scheduler_type=CFG["training"]["lr_scheduler_type"],
    warmup_ratio=CFG["training"]["warmup_ratio"],
    bf16=CFG["training"]["bf16"],
    optim=CFG["training"]["optim"],
    gradient_checkpointing=CFG["training"]["gradient_checkpointing"],
    max_seq_length=CFG["training"]["max_seq_length"],
    packing=False,
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
print(f"LongCtx done → {CFG['output']['hf_repo']}")
