"""
01 · Dataset-Build (RunPod-Version) — CPU-Phase, ~25 min.

SFT-Quellen — reiner Agentik-Fokus (~115k Samples, nur Tool-Call-Signal):
  - Salesforce/xlam-function-calling-60k    (60k) — BFCL-Style single-turn Tool-Calls
  - glaiveai/glaive-function-calling-v2     (25k) — Diverse Tool-Schemas + Dialog-Kontext
  - Team-ACE/ToolACE                        (15k) — Multi-turn Agent-Loops
  - NousResearch/hermes-function-calling-v1 (15k) — OpenAI-kompatible Function-Calls
  - princeton-nlp/SWE-bench_Verified         (500) — real repo patches (Struktur)

DPO-Quellen — Agent-Conciseness (Mac-Speed via weniger Output-Tokens):
  - Synthetisch aus xLAM (50k) — chosen: direkter Tool-Call,
    rejected: Preamble-Prosa + Tool-Call oder falsche Tool-Argumente

LongCtx — Agent-Needle-in-Haystack:
  - Synthetisch: 200+ Tool-Schemas als Haystack, Needle = Ziel-Tool,
    Query verlangt korrekten Call mit dem Needle-Tool
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


# ── DPO — Synthetic Agent-Conciseness ────────────────────────────────────────
# Ziel: Modell lernt DIREKT mit <tool_call>-Block zu antworten, ohne Preamble.
# Jedes Output-Token weniger = schnellere Agent-Loops auf Mac (M-Series Decode).

VERBOSE_PREAMBLES = [
    "Of course! I'd be happy to help you with that. Let me think about which tool is best here.\n\n",
    "Sure, I can help! Based on your request, I think the right approach is to call the following tool:\n\n",
    "Great question! To accomplish this, I need to use a function call. Here is what I'll do:\n\n",
    "I understand what you're asking. Let me walk you through my reasoning before making the call.\n"
    "First, I identify the relevant function, then I prepare the arguments carefully.\n\n",
    "Let me help you with that! I'll use the appropriate tool to handle your request:\n\n",
    "Absolutely, I can assist. Let me break this down step-by-step before I invoke the function.\n"
    "Step 1: Understand the goal.\nStep 2: Pick the right tool.\nStep 3: Fill in arguments.\n\n",
    "Okay, so looking at your query, I believe the best way forward is to leverage a function call.\n"
    "Here is my plan:\n\n",
]


def build_agent_dpo(n: int):
    """Synthetic Agent-DPO aus xLAM — Concise-vs-Verbose Tool-Calls.

    Prompt: Tool-Schema-Kontext + User-Query
    chosen: direkter <tool_call>-Block (keine Prosa)
    rejected: geschwätziger Preamble + exakt derselbe Tool-Call

    Damit lernt DPO *nur* den Unterschied zwischen kurz und geschwätzig —
    Korrektheit des Tool-Calls bleibt konstant. Der Conciseness-Effekt
    wird isoliert trainiert.
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

        tc_blocks = "\n".join(
            f"<tool_call>\n{_json.dumps(c, ensure_ascii=False)}\n</tool_call>" for c in calls
        )
        preamble = random.choice(VERBOSE_PREAMBLES)

        prompt = (
            "You are a function-calling agent. Respond with tool calls only, no prose.\n"
            "<tools>\n" + _json.dumps(tools, ensure_ascii=False) + "\n</tools>\n\n"
            f"User: {query}\nAssistant:"
        )

        out.append({
            "prompt":   prompt,
            "chosen":   tc_blocks,
            "rejected": preamble + tc_blocks,
        })
    return Dataset.from_list(out)


print("\nbuilding Agent-DPO (synthetic concise-vs-verbose) …")
dpo = build_agent_dpo(50_000).shuffle(seed=42)
print(f"DPO total: {len(dpo)}")
dpo.push_to_hub(f"{OWNER}/EliCoder-Dataset-DPO", private=True, token=TOK)


# ── LongCtx — Agentisches Needle-in-Haystack über Tool-Schemas ───────────────
# Realistischer Use-Case: Agent bekommt 200+ Tools (z.B. MCP mit vielen Servern)
# und muss das richtige auswählen und korrekt aufrufen. Needle = gesuchtes Tool.

def synth_long_ctx_agentic(n: int, lengths=(16_000, 32_000, 64_000, 128_000)):
    """Needle-in-haystack mit Tool-Schemas — Agent-Navigation durch große Tool-Kataloge.

    Kontext: Riesiges <tools>-Block mit N Schemas (Haystack).
    Needle: Ein eindeutig benanntes Ziel-Tool irgendwo dazwischen.
    Query: Fordert Aufruf des Needle-Tools mit gegebenen Argumenten.
    Expected: Direkter <tool_call>-Block mit korrektem Namen + Argumenten.
    """
    import json as _json
    random.seed(123)

    # Sammle reale Tool-Schemas aus xLAM als Haystack-Material.
    print("  longctx: sammle Tool-Schemas aus xLAM …")
    pool_ds = load_dataset("Salesforce/xlam-function-calling-60k",
                           split="train", token=TOK).shuffle(seed=123)
    tool_pool = []
    for row in pool_ds:
        try:
            tools = _json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
            for t in tools:
                if isinstance(t, dict) and t.get("name"):
                    tool_pool.append(t)
        except Exception:
            continue
        if len(tool_pool) >= 20_000:
            break
    # Deduplizieren nach Tool-Name (behalte erstes Vorkommen).
    seen = set()
    unique_pool = []
    for t in tool_pool:
        if t["name"] not in seen:
            seen.add(t["name"])
            unique_pool.append(t)
    print(f"  longctx: {len(unique_pool)} unique Tool-Schemas im Pool")

    if len(unique_pool) < 50:
        raise RuntimeError("Nicht genügend Tool-Schemas für LongCtx-Needle-Build")

    out = []
    for _ in range(n):
        L_chars = random.choice(lengths) * 4  # ≈4 chars per token als grobe Heuristik

        # Needle: zufälliges Tool mit eindeutigem Namen aussuchen.
        needle = random.choice(unique_pool)
        needle_name = needle["name"]

        # Haystack aufbauen: Tool-Schemas bis char-budget erreicht.
        haystack, total = [], 0
        while total < L_chars:
            t = random.choice(unique_pool)
            if t["name"] == needle_name:
                continue  # Kollision mit Needle vermeiden
            s = _json.dumps(t, ensure_ascii=False)
            haystack.append(t)
            total += len(s) + 2  # + trennung

        # Needle irgendwo in die Mitte einfügen.
        idx = random.randint(len(haystack) // 4, 3 * len(haystack) // 4)
        haystack.insert(idx, needle)

        # Synthetische Argumente aus Needle-Schema.
        args = {}
        params = needle.get("parameters", {}) or {}
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        for pname, pspec in list(props.items())[:3]:  # max 3 Args
            ptype = (pspec or {}).get("type", "string")
            if ptype == "integer":
                args[pname] = random.randint(1, 100)
            elif ptype == "number":
                args[pname] = round(random.uniform(1.0, 100.0), 2)
            elif ptype == "boolean":
                args[pname] = random.choice([True, False])
            elif ptype == "array":
                args[pname] = []
            else:
                args[pname] = f"value_{random.randint(1, 999)}"

        tools_json = _json.dumps(haystack, ensure_ascii=False)
        sys_msg = (
            "You are a function-calling agent. Respond with tool calls only, no prose.\n"
            f"<tools>\n{tools_json}\n</tools>"
        )
        user_msg = (
            f"Call the `{needle_name}` function"
            + (f" with these arguments: {_json.dumps(args, ensure_ascii=False)}." if args
               else " (no arguments required).")
        )
        tool_call = {"name": needle_name, "arguments": args}
        assistant_msg = f"<tool_call>\n{_json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"

        out.append(to_chatml([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]))
    return Dataset.from_list(out)


longctx = synth_long_ctx_agentic(6_000)
print(f"\nLongCtx total: {len(longctx)}")
longctx.push_to_hub(f"{OWNER}/EliCoder-Dataset-LongCtx", private=True, token=TOK)

print("\nall datasets pushed:")
print(f"  SFT:     huggingface.co/datasets/{OWNER}/EliCoder-Dataset-SFT")
print(f"  DPO:     huggingface.co/datasets/{OWNER}/EliCoder-Dataset-DPO")
print(f"  LongCtx: huggingface.co/datasets/{OWNER}/EliCoder-Dataset-LongCtx")

Path("/workspace/.phase01_done").touch()
