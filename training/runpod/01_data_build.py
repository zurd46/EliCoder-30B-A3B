"""
01 · Dataset-Build (RunPod-Version) — CPU-Phase, ~25 min.

SFT-Quellen — reiner Agentik-Fokus (~115k Samples, nur Tool-Call-Signal):
  - Salesforce/xlam-function-calling-60k    (60k) — BFCL-Style single-turn Tool-Calls
  - glaiveai/glaive-function-calling-v2     (25k) — Diverse Tool-Schemas + Dialog-Kontext
  - Team-ACE/ToolACE                        (15k) — Multi-turn Agent-Loops
  - NousResearch/hermes-function-calling-v1 (15k) — OpenAI-kompatible Function-Calls
  - princeton-nlp/SWE-bench_Verified         (500) — real repo patches (Struktur)

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
    "xlam_tool_calls":     60_000,  # BFCL-Style single-turn Function-Calls (max)
    "glaive_fc_v2":        25_000,  # Diverse Tool-Schemas + Dialog-Kontext
    "toolace":             15_000,  # Multi-turn Agent-Loops
    "hermes_fc":           15_000,  # OpenAI-kompatible Function-Calls
    "swe_verified":           500,  # Real repo patches (Struktur)
}


def to_chatml(messages):
    return {"messages": messages}


def load_xlam_tool_calls(n: int):
    """BFCL-Style Function-Calling Traces — Kern für Agent-Tool-Use.

    xLAM-60k Schema: {query, tools (JSON-list), answers (JSON-list von tool_calls)}.
    Wir formatieren als Qwen-ChatML: system mit Tool-Schema, user-Query,
    assistant antwortet DIREKT mit <tool_call>-Block (keine Preamble → weniger
    Output-Tokens pro Call → schnellere Agent-Loops auf M-Series Macs).
    """
    import json as _json
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        query = row.get("query") or ""
        tools_raw = row.get("tools") or "[]"
        answers_raw = row.get("answers") or "[]"
        if not (query and tools_raw and answers_raw):
            continue
        try:
            tools = _json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
            calls = _json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except Exception:
            continue
        if not calls:
            continue
        # Qwen-Format: <tools>…</tools> im System, <tool_call>…</tool_call> im Assistant.
        sys_ = (
            "You are a function-calling agent. Respond with tool calls only, no prose.\n"
            "<tools>\n" + _json.dumps(tools, ensure_ascii=False) + "\n</tools>"
        )
        # Ein <tool_call>-Block pro Aufruf; mehrere Calls → mehrere Blöcke hintereinander.
        tc_blocks = "\n".join(
            f"<tool_call>\n{_json.dumps(c, ensure_ascii=False)}\n</tool_call>" for c in calls
        )
        out.append(to_chatml([
            {"role": "system", "content": sys_},
            {"role": "user", "content": query},
            {"role": "assistant", "content": tc_blocks},
        ]))
    return Dataset.from_list(out)


def load_glaive_fc_v2(n: int):
    """Glaive Function-Calling v2 — diverse Tool-Schemas + Dialog-Kontext.

    Schema: `system` (mit Tool-Schema) + `chat` (String mit USER:/ASSISTANT:/FUNCTION RESPONSE:-Markern).
    Wir parsen den Chat-String in ChatML-Messages um.
    """
    import re as _re
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", token=TOK).shuffle(seed=42)
    out = []
    # Glaive nutzt diese Rollen-Marker im chat-String.
    pattern = _re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*", _re.IGNORECASE)
    role_map = {"user": "user", "assistant": "assistant", "function response": "tool"}
    for row in ds.select(range(min(n, len(ds)))):
        sys_ = (row.get("system") or "").strip()
        chat = (row.get("chat") or "").strip()
        if not chat:
            continue
        parts_ = pattern.split(chat)
        # parts_ = ["", "USER", "...", "ASSISTANT", "...", ...]
        messages = []
        if sys_:
            messages.append({"role": "system", "content": sys_})
        for i in range(1, len(parts_) - 1, 2):
            role = role_map.get(parts_[i].strip().lower())
            content = parts_[i + 1].strip().rstrip("<|endoftext|>").strip()
            if role and content:
                messages.append({"role": role, "content": content})
        if len(messages) >= 2:
            out.append(to_chatml(messages))
    return Dataset.from_list(out)


def load_toolace(n: int):
    """Multi-turn Agent-Dialoge — echte Tool-Loops (Call → Result → Follow-up).

    ToolACE hat Konversationen mit mehreren Tool-Calls hintereinander,
    genau das Muster das der Agent auf dem Mac lernen muss.
    """
    import json as _json
    ds = load_dataset("Team-ACE/ToolACE", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        convs = row.get("conversations") or row.get("messages")
        tools = row.get("system") or row.get("tools") or ""
        if not convs:
            continue
        # ToolACE-Rollen normalisieren → ChatML
        messages = []
        if tools:
            tools_str = tools if isinstance(tools, str) else _json.dumps(tools, ensure_ascii=False)
            messages.append({"role": "system", "content": tools_str})
        role_map = {"user": "user", "human": "user",
                    "assistant": "assistant", "gpt": "assistant",
                    "tool": "tool", "function": "tool",
                    "system": "system"}
        for turn in convs:
            role = role_map.get(turn.get("from") or turn.get("role") or "", "user")
            content = turn.get("value") or turn.get("content") or ""
            if content:
                messages.append({"role": role, "content": content})
        if len(messages) >= 2:
            out.append(to_chatml(messages))
    return Dataset.from_list(out)


def load_hermes_fc(n: int):
    """OpenAI-kompatible Function-Call-Traces — Nous Research.

    Hermes-FC nutzt das Standard-`tools`-Schema und `tool_calls`-Antwortformat.
    Wir übernehmen die messages direkt (sie sind bereits ChatML-strukturiert).
    """
    import json as _json
    # func_calling_singleturn ist der kleinere, verifizierte Split.
    ds = load_dataset("NousResearch/hermes-function-calling-v1",
                      "func_calling_singleturn", split="train", token=TOK).shuffle(seed=42)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        msgs = row.get("conversations") or row.get("messages")
        if not msgs:
            continue
        role_map = {"system": "system", "human": "user", "user": "user",
                    "gpt": "assistant", "assistant": "assistant",
                    "tool": "tool", "function": "tool"}
        normalized = []
        for turn in msgs:
            role = role_map.get(turn.get("from") or turn.get("role") or "", "user")
            content = turn.get("value") or turn.get("content") or ""
            if content:
                normalized.append({"role": role, "content": content})
        if len(normalized) >= 2:
            out.append(to_chatml(normalized))
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


print("loading SFT sources (Agentik-Fokus) …")
loaders = [
    (load_xlam_tool_calls, "xlam_tool_calls"),
    (load_glaive_fc_v2,    "glaive_fc_v2"),
    (load_toolace,         "toolace"),
    (load_hermes_fc,       "hermes_fc"),
    (load_swe_verified,    "swe_verified"),
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
