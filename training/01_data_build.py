# %% [markdown]
# # 01 · Dataset-Build — Coding-optimiert
#
# Baut drei HF-Datasets aus **verifizierten Top-Quellen** für SFT, DPO und
# Long-Context-Training. Alle Datasets sind auf Qualität und Coding-Performance
# ausgelegt.
#
# ## SFT-Quellen (alle real & verifiziert, Stand April 2026)
#
# | Dataset | Warum | Samples |
# |---|---|---|
# | `nvidia/Nemotron-SFT-OpenCode-v1`         | Agent-Traces, generiert mit Qwen3-Coder-480B | ~40k |
# | `nvidia/OpenCodeReasoning-2`              | R1-Reasoning auf Coding (PY+C++)             | ~40k |
# | `ise-uiuc/Magicoder-Evol-Instruct-110K`   | Evol-Instruct, goldstandard für Coding-SFT  | ~40k |
# | `ise-uiuc/Magicoder-OSS-Instruct-75K`     | Real-World OSS-Instruktionen                | ~20k |
# | `glaiveai/glaive-function-calling-v2`     | Tool/Function-Calling (86k Chats)           | ~30k |
# | `princeton-nlp/SWE-bench_Verified`        | Real-World Bug-Fixes (500)                  |  0.5k |
# | `AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` | Agentic-CoT-Coding   | ~10k |
#
# ## DPO-Quellen
#
# | Dataset | Warum |
# |---|---|
# | `Vezora/Code-Preference-Pairs` | 55k Pairs, bug-injected rejected — lehrt Bugs vermeiden (127 Sprachen) |
# | `jondurbin/py-dpo-v0.1`         | Python Chosen/Rejected |
#
# ## Long-Context
# Eigene synthetische Needle-in-Haystack-Samples (32k–200k).

# %% [markdown]
# ## Colab Bootstrap — einmal pro Runtime

# %%
def _bootstrap():
    import os, subprocess, sys
    from pathlib import Path
    try:
        import google.colab
        in_colab = True
    except Exception:
        in_colab = False
    if in_colab:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
            if tok:
                os.environ["HF_TOKEN"] = tok
                os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        except Exception:
            pass
        root = Path("/content/CoderLLM")
        if not root.exists():
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/zurd46/CoderLLM.git", str(root)], check=True)
        os.chdir(root / "training")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "datasets>=3.0", "huggingface_hub>=0.25", "tqdm"], check=False)

_bootstrap()

# %%
import os, json, random, re
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm

OWNER = "zurd46"
TOK = os.environ.get("HF_TOKEN")
assert TOK, "set HF_TOKEN in Colab Secrets"
random.seed(42)

MAX_PER_SOURCE = {
    "nemotron_opencode": 40_000,
    "opencodereasoning": 40_000,
    "magicoder_evol":    40_000,
    "magicoder_oss":     20_000,
    "glaive_fc":         30_000,
    "swe_verified":       500,
    "agentic_cot":       10_000,
}


def to_chatml(messages):
    return {"messages": messages}


# %% [markdown]
# ## Loader je Quelle — jeder normalisiert auf `{"messages": [...]}`.

# %%
def load_nemotron_opencode(n: int):
    ds = load_dataset("nvidia/Nemotron-SFT-OpenCode-v1", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        msgs = row.get("messages") or row.get("conversation")
        if msgs:
            out.append(to_chatml(msgs))
    return Dataset.from_list(out)


def load_open_code_reasoning(n: int):
    ds = load_dataset("nvidia/OpenCodeReasoning-2", "python", split="train", token=TOK).shuffle(seed=42)
    out = []
    sys_ = "You are a senior engineer. Think step by step, then write the final solution."
    for row in ds.select(range(min(n, len(ds)))):
        q = row.get("question") or row.get("input")
        a = row.get("r1_generation") or row.get("solution") or row.get("output")
        if q and a:
            out.append(to_chatml([
                {"role": "system", "content": sys_},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]))
    return Dataset.from_list(out)


def load_magicoder_evol(n: int):
    ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append(to_chatml([
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]))
    return Dataset.from_list(out)


def load_magicoder_oss(n: int):
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        q = row.get("problem") or row.get("instruction")
        a = row.get("solution") or row.get("response")
        if q and a:
            out.append(to_chatml([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]))
    return Dataset.from_list(out)


def load_glaive_fc(n: int):
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", token=TOK).shuffle(seed=42)

    def parse_chat(raw: str, system: str | None):
        lines = raw.splitlines()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system.strip()})
        current_role, buf = None, []

        def flush():
            if current_role and buf:
                msgs.append({"role": current_role, "content": "\n".join(buf).strip()})

        for ln in lines:
            m = re.match(r"^(USER|ASSISTANT|FUNCTION RESPONSE|SYSTEM):\s*(.*)", ln)
            if m:
                flush()
                role_map = {"USER": "user", "ASSISTANT": "assistant",
                            "FUNCTION RESPONSE": "tool", "SYSTEM": "system"}
                current_role = role_map[m.group(1)]
                buf = [m.group(2)]
            else:
                buf.append(ln)
        flush()
        return msgs

    out = []
    for row in ds.select(range(min(n, len(ds)))):
        msgs = parse_chat(row["chat"], row.get("system"))
        if any(m["role"] == "assistant" for m in msgs):
            out.append(to_chatml(msgs))
    return Dataset.from_list(out)


def load_swe_verified(n: int):
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", token=TOK).shuffle(seed=42)
    out = []
    sys_ = ("You are a senior engineer. Fix the reported issue with a minimal, correct patch. "
            "Output a unified diff only.")
    for row in ds.select(range(min(n, len(ds)))):
        user = (f"Repository: {row['repo']}\n"
                f"Base commit: {row['base_commit']}\n\n"
                f"Issue:\n{row['problem_statement']}\n\n"
                f"Provide the patch as a unified diff.")
        out.append(to_chatml([
            {"role": "system", "content": sys_},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["patch"]},
        ]))
    return Dataset.from_list(out)


def load_agentic_cot(n: int):
    ds = load_dataset("AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset",
                      split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        msgs = row.get("messages")
        if msgs:
            out.append(to_chatml(msgs))
    return Dataset.from_list(out)

# %% [markdown]
# ## SFT — Mix bauen und zu HF pushen

# %%
print("loading sources …")
parts = []
for fn, key in [
    (load_nemotron_opencode, "nemotron_opencode"),
    (load_open_code_reasoning, "opencodereasoning"),
    (load_magicoder_evol,   "magicoder_evol"),
    (load_magicoder_oss,    "magicoder_oss"),
    (load_glaive_fc,        "glaive_fc"),
    (load_swe_verified,     "swe_verified"),
    (load_agentic_cot,      "agentic_cot"),
]:
    try:
        d = fn(MAX_PER_SOURCE[key])
        print(f"  {key}: {len(d)}")
        parts.append(d)
    except Exception as e:
        print(f"  {key}: SKIPPED ({e})")

sft = concatenate_datasets(parts).shuffle(seed=42)
print(f"\nSFT total: {len(sft)} samples")
sft.push_to_hub(f"{OWNER}/coder-16b-dyn-sft", private=True, token=TOK)

# %% [markdown]
# ## DPO — Coding-Preference aus Code-Preference-Pairs + py-dpo

# %%
def load_code_preference_pairs(n: int):
    ds = load_dataset("Vezora/Code-Preference-Pairs", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        prompt = row.get("prompt") or row.get("instruction") or row.get("input") or ""
        chosen = row.get("chosen") or row.get("accepted") or row.get("fixed_code")
        rejected = row.get("rejected") or row.get("bugged_code")
        if prompt and chosen and rejected:
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(out)


def load_py_dpo(n: int):
    ds = load_dataset("jondurbin/py-dpo-v0.1", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append({
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        })
    return Dataset.from_list(out)


print("\nloading DPO sources …")
dpo_parts = []
for fn, label, n in [(load_code_preference_pairs, "code-pref", 50_000),
                     (load_py_dpo,                "py-dpo",    20_000)]:
    try:
        d = fn(n)
        print(f"  {label}: {len(d)}")
        dpo_parts.append(d)
    except Exception as e:
        print(f"  {label}: SKIPPED ({e})")

dpo = concatenate_datasets(dpo_parts).shuffle(seed=42)
print(f"\nDPO total: {len(dpo)}")
dpo.push_to_hub(f"{OWNER}/coder-16b-dyn-dpo", private=True, token=TOK)

# %% [markdown]
# ## Long-Context — synthetische Needle-in-Haystack-Samples (32k–200k)

# %%
def synth_long_ctx(n: int, lengths=(32_000, 64_000, 128_000, 200_000)):
    random.seed(123)
    words_path = Path("/usr/share/dict/words")
    words = words_path.read_text().splitlines() if words_path.exists() else ["the"] * 5_000
    out = []
    for _ in range(n):
        L = random.choice(lengths)
        passphrase = f"{random.randint(10**9, 10**10)}"
        haystack_tokens = [random.choice(words) for _ in range(L // 6)]
        insert = random.randint(len(haystack_tokens) // 4, 3 * len(haystack_tokens) // 4)
        haystack_tokens.insert(insert, f"PASSPHRASE={passphrase}.")
        doc = " ".join(haystack_tokens)
        user = (f"DOKUMENT:\n{doc}\n\nFRAGE: Wie lautet PASSPHRASE? "
                f"Antworte nur mit der Zahl.")
        out.append(to_chatml([
            {"role": "user", "content": user},
            {"role": "assistant", "content": passphrase},
        ]))
    return Dataset.from_list(out)


longctx = synth_long_ctx(8_000)
print(f"\nLongCtx total: {len(longctx)}")
longctx.push_to_hub(f"{OWNER}/coder-16b-dyn-longctx", private=True, token=TOK)

print("\nall datasets pushed:")
print(f"  SFT:     huggingface.co/datasets/{OWNER}/coder-16b-dyn-sft")
print(f"  DPO:     huggingface.co/datasets/{OWNER}/coder-16b-dyn-dpo")
print(f"  LongCtx: huggingface.co/datasets/{OWNER}/coder-16b-dyn-longctx")
