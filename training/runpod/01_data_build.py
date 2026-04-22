"""
01 · Dataset-Build (RunPod-Version) — CPU-Phase, ~25 min.

SFT-Quellen (Priorität: einzigartig + verifiziert):
  - nvidia/Nemotron-SFT-OpenCode-v1        (15k)
  - nvidia/OpenCodeReasoning-2             (30k) — Chain-of-Thought Python
  - m-a-p/CodeFeedback-Filtered-Instruction (30k) — execution-verified, 2024
  - bigcode/self-oss-instruct-sc2-exec-filter-50k (25k) — OSS-inspired + verified
  - bigcode/commitpackft                   (25k) — echte GitHub-Commits, einzigartig
  - ise-uiuc/Magicoder-Evol-Instruct-110K  (15k)
  - princeton-nlp/SWE-bench_Verified        (500) — real repo patches
  - AlicanKiraz0/Agentic-Chain-of-Thought  (5k)

DPO-Quellen:
  - Vezora/Code-Preference-Pairs           (50k) — buggy vs. fixed
  - jondurbin/py-dpo-v0.1                  (20k)

LongCtx:
  - Synthetisch mit Python-Code als Haystack (statt Zufallswörter)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap(install=True)

import os, random, re
from datasets import load_dataset, Dataset, concatenate_datasets

OWNER = os.environ.get("CODERLLM_HF_OWNER", "zurd46")
TOK = os.environ["HF_TOKEN"]
random.seed(42)

MAX_PER_SOURCE = {
    "nemotron_opencode":   15_000,
    "opencodereasoning":   30_000,
    "codefeedback":        30_000,
    "self_oss_instruct":   35_000,
    "magicoder_evol":      15_000,
    "swe_verified":           500,
    "magicoder_evol":      20_000,
}


def to_chatml(messages):
    return {"messages": messages}


def load_nemotron_opencode(n: int):
    ds = load_dataset("nvidia/Nemotron-SFT-OpenCode-v1", split="general", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        msgs = row.get("messages") or row.get("conversation")
        if msgs:
            out.append(to_chatml(msgs))
    return Dataset.from_list(out)


def load_open_code_reasoning(n: int):
    ds = load_dataset("nvidia/OpenCodeReasoning-2", split="train", token=TOK).shuffle(seed=42)
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


def load_codefeedback(n: int):
    """Execution-verified code instructions — 2024, unlikely in Qwen pretraining."""
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        q = row.get("query") or ""
        a = row.get("answer") or ""
        if q and a:
            out.append(to_chatml([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]))
    return Dataset.from_list(out)


def load_self_oss_instruct(n: int):
    """OSS-inspired instructions with execution verification."""
    ds = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        inst = row.get("instruction") or ""
        resp = row.get("response") or ""
        if inst and resp:
            out.append(to_chatml([
                {"role": "user", "content": inst},
                {"role": "assistant", "content": resp},
            ]))
    return Dataset.from_list(out)


def load_commitpackft(n: int):
    """Real GitHub commits — highly unique, teaches code patching."""
    ds = load_dataset("bigcode/commitpackft", "python", split="train", token=TOK, trust_remote_code=True).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        old = row.get("old_contents") or ""
        new = row.get("new_contents") or ""
        msg = row.get("subject") or row.get("message") or ""
        # skip trivial or very long diffs
        if not (old and new and msg) or len(old) > 3000 or len(new) > 3000:
            continue
        out.append(to_chatml([
            {"role": "user",
             "content": f"Apply this change to the Python code:\n\n{msg}\n\n```python\n{old}\n```"},
            {"role": "assistant",
             "content": f"```python\n{new}\n```"},
        ]))
        if len(out) >= n:
            break
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


print("loading SFT sources …")
loaders = [
    (load_nemotron_opencode,   "nemotron_opencode"),
    (load_open_code_reasoning, "opencodereasoning"),
    (load_codefeedback,        "codefeedback"),
    (load_self_oss_instruct,   "self_oss_instruct"),

    (load_magicoder_evol,      "magicoder_evol"),
    (load_swe_verified,        "swe_verified"),

]

from concurrent.futures import ThreadPoolExecutor

def _load_worker(fn_key):
    fn, key = fn_key
    try:
        d = fn(MAX_PER_SOURCE[key])
        print(f"  {key}: {len(d)}")
        return d
    except Exception as e:
        print(f"  {key}: SKIPPED ({e})")
        return None

parts = []
with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(_load_worker, loaders))

parts = [r for r in results if r is not None]

sft = concatenate_datasets(parts).shuffle(seed=42)
print(f"\nSFT total: {len(sft)} samples")
sft.push_to_hub(f"{OWNER}/EliCoder-Dataset-SFT", private=True, token=TOK)


# ── DPO ──────────────────────────────────────────────────────────────────────

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
        out.append({"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]})
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
dpo.push_to_hub(f"{OWNER}/EliCoder-Dataset-DPO", private=True, token=TOK)


# ── LongCtx — Python-Code als Haystack (statt Zufallswörter) ─────────────────

def synth_long_ctx_code(n: int, lengths=(16_000, 32_000, 64_000, 128_000)):
    """Needle-in-haystack mit Python-Code — realistischer als Zufallswörter."""
    random.seed(123)

    # Lade Python-Snippets als Haystack-Material
    snippets = []
    try:
        snip_ds = load_dataset("nampdn-ai/tiny-codes", split="train", token=TOK)
        snippets = [r.get("response", "")[:600] for r in snip_ds if r.get("response")][:8000]
        print(f"  longctx filler: {len(snippets)} snippets from tiny-codes")
    except Exception as e:
        print(f"  longctx filler: tiny-codes SKIPPED ({e}), using fallback")

    if not snippets:
        snippets = [
            "def helper(x):\n    return x * 2\n",
            "import os\nimport sys\nfrom pathlib import Path\n",
            "class MyClass:\n    def __init__(self, val):\n        self.val = val\n",
            "for i in range(10):\n    print(i)\n",
            "def process(data: list) -> dict:\n    return {str(i): v for i, v in enumerate(data)}\n",
            "if __name__ == '__main__':\n    main()\n",
        ] * 2000

    out = []
    for _ in range(n):
        L = random.choice(lengths)
        fn_name = f"target_{random.randint(10**6, 10**7)}"
        return_val = str(random.randint(10**9, 10**10))

        parts, total = [], 0
        while total < L:
            s = random.choice(snippets)
            parts.append(s)
            total += len(s)

        needle = (f"\n\ndef {fn_name}() -> int:\n"
                  f"    \"\"\"Returns the secret target value.\"\"\"\n"
                  f"    return {return_val}\n\n")
        idx = random.randint(max(1, len(parts) // 4), 3 * len(parts) // 4)
        parts.insert(idx, needle)

        code = "\n".join(parts)
        out.append(to_chatml([
            {"role": "user",
             "content": (f"CODEBASE:\n```python\n{code}\n```\n\n"
                         f"What integer does `{fn_name}()` return? Answer with just the number.")},
            {"role": "assistant", "content": return_val},
        ]))
    return Dataset.from_list(out)


longctx = synth_long_ctx_code(6_000)
print(f"\nLongCtx total: {len(longctx)}")
longctx.push_to_hub(f"{OWNER}/EliCoder-Dataset-LongCtx", private=True, token=TOK)

print("\nall datasets pushed:")
print(f"  SFT:     huggingface.co/datasets/{OWNER}/EliCoder-Dataset-SFT")
print(f"  DPO:     huggingface.co/datasets/{OWNER}/EliCoder-Dataset-DPO")
print(f"  LongCtx: huggingface.co/datasets/{OWNER}/EliCoder-Dataset-LongCtx")

Path("/workspace/.phase01_done").touch()
