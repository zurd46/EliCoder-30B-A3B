# %% [markdown]
# # 02 · SFT mit Unsloth (Phase A)
#
# Runtime: Colab H100 80 GB.
# Dauer: ~12 h (Unsloth macht ~2× vanilla).

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

os.environ["WANDB_PROJECT"] = "coder-16b-dyn"
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
    warmup_ratio=CFG["training"]["warmup_ratio"],
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
