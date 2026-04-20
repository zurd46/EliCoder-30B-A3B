# %% [markdown]
# # 01 · Dataset-Build
#
# Baut den SFT-, DPO- und Long-Context-Datensatz aus öffentlichen Quellen +
# synthetisierten Tool-Use-Traces. Outputs: drei private HF-Datasets.

# %%
import os, json, random, itertools
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi

OWNER = "zurd46"
TOK = os.environ.get("HF_TOKEN")
assert TOK, "set HF_TOKEN in Colab Secrets"
random.seed(42)

# %% [markdown]
# ## SFT-Mix
#
# Anteile wie in ARCHITECTURE.md §7.2 festgelegt.

# %%
def to_chatml(messages):
    return {"messages": messages}


def from_swe_bench(n: int):
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="train").shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        sys = "You are a senior engineer. Fix the reported issue with a minimal patch."
        user = f"Repo: {row['repo']}\nIssue: {row['problem_statement']}\n\nReturn a unified diff only."
        out.append(to_chatml([
            {"role": "system", "content": sys},
            {"role": "user",   "content": user},
            {"role": "assistant", "content": row["patch"]},
        ]))
    return Dataset.from_list(out)


def from_open_code_reasoning(n: int):
    ds = load_dataset("nvidia/OpenCodeReasoning-v2", split="train").shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append(to_chatml([
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]))
    return Dataset.from_list(out)


def from_glaive_fc(n: int):
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train").shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        chat = row["chat"]
        out.append(to_chatml([{"role": r["from"], "content": r["value"]} for r in chat]))
    return Dataset.from_list(out)


def synth_long_ctx_needle(n: int, lengths=(64_000, 128_000, 200_000)):
    random.seed(123)
    words = (Path("/usr/share/dict/words").read_text().splitlines()
             if Path("/usr/share/dict/words").exists()
             else ["the"] * 5000)
    out = []
    for i in range(n):
        L = random.choice(lengths)
        magic = f"Die Passphrase ist {random.randint(10**9, 10**10)}."
        haystack = " ".join(random.choices(words, k=L // 6))
        insert = random.randint(len(haystack) // 4, 3 * len(haystack) // 4)
        doc = haystack[:insert] + " " + magic + " " + haystack[insert:]
        q = "Welche Passphrase steht im Dokument? Antworte nur mit der Zahl."
        a = magic.split("ist ")[1].rstrip(".")
        out.append(to_chatml([
            {"role": "user", "content": f"DOKUMENT:\n{doc}\n\nFRAGE: {q}"},
            {"role": "assistant", "content": a},
        ]))
    return Dataset.from_list(out)

# %%
sft_parts = [
    from_swe_bench(12_000),
    from_open_code_reasoning(40_000),
    from_glaive_fc(20_000),
    synth_long_ctx_needle(5_000),
]
sft = concatenate_datasets(sft_parts).shuffle(seed=42)
print(f"SFT total: {len(sft)}")
sft.push_to_hub(f"{OWNER}/coder-16b-dyn-sft", private=True, token=TOK)

# %% [markdown]
# ## DPO — Preference Pairs aus SWE-Bench Rollouts
#
# Dataset = (prompt, chosen=passed_patch, rejected=failed_patch). Wir benutzen
# bereits öffentlich verfügbare gepaarte Rollouts.

# %%
dpo_raw = load_dataset("princeton-nlp/SWE-bench-verified-pairs", split="train")
dpo = Dataset.from_list([
    {"prompt": r["instruction"], "chosen": r["chosen"], "rejected": r["rejected"]}
    for r in dpo_raw
])
print(f"DPO total: {len(dpo)}")
dpo.push_to_hub(f"{OWNER}/coder-16b-dyn-dpo", private=True, token=TOK)

# %% [markdown]
# ## Long-Context — 32 k → 200 k Retention-Training

# %%
longctx = synth_long_ctx_needle(8_000, lengths=(32_000, 64_000, 128_000, 200_000))
longctx.push_to_hub(f"{OWNER}/coder-16b-dyn-longctx", private=True, token=TOK)
print("done.")
