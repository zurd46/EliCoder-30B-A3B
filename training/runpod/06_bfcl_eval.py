"""
06 · BFCL-lite Endabnahme — evaluiert das fertig gemergte Modell auf Tool-Calls.

Input:  zurd46/EliCoder-30B-A3B (merged BF16 aus Phase 05)
Output: W&B + JSON-Report unter /workspace/bfcl_report.json

Evaluiert gegen drei Sub-Benchmarks (approximiert, nicht offizielles BFCL-Scoring):
  - simple   : 1 Tool, 1 Call, einfache Args
  - multiple : viele verfügbare Tools, Modell muss das richtige wählen
  - parallel : 1 Query, muss mehrere Tool-Calls gleichzeitig produzieren

Metriken pro Subset:
  - parse_rate       % valide <tool_call>-Blöcke
  - name_accuracy    % korrekte Tool-Namen
  - args_accuracy    % exakt passende Argumente (nach JSON-Normalisierung)
  - full_accuracy    % name_match UND args_match
  - avg_out_tokens   Decode-Länge (Mac-Speed-Proxy)
  - decode_toks_per_sec  Raw-Throughput auf der H100

Braucht nur read-access zum merged repo. Läuft ~15–25 min.
"""
from __future__ import annotations

import json as _json
import os
import re as _re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap, apply_gpu_optims, patch_unsloth_telemetry

bootstrap(install=True)

import torch
assert torch.cuda.is_available(), "CUDA required"
apply_gpu_optims()

patch_unsloth_telemetry()

from datasets import load_dataset
from unsloth import FastLanguageModel

TOK = os.environ["HF_TOKEN"]
OWNER = os.environ.get("CODERLLM_HF_OWNER", "zurd46")
MERGED_REPO = f"{OWNER}/EliCoder-30B-A3B"

os.environ.setdefault("WANDB_PROJECT", "CoderLLM")
os.environ.setdefault("WANDB_NAME", "bfcl-eval-runpod")

_TOOL_CALL_RE = _re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", _re.DOTALL)

# Muss identisch zum Trainings-Prompt in 01_data_build.py sein — sonst widerspricht
# der Eval-Prompt dem gelernten Signal.
AGENT_SYS = (
    "You are a function-calling agent. Respond with tool calls only, no prose.\n"
    "Emit one <tool_call>…</tool_call> block per call, JSON on a single line, "
    "compact (no spaces). End immediately after the last </tool_call>."
)


def _parse_calls(text: str):
    """Extrahiere alle <tool_call>…</tool_call>-Blöcke als dict-Liste."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            calls.append(_json.loads(m.group(1)))
        except Exception:
            pass
    return calls


def _normalize_args(args) -> str:
    """JSON-kanonisch für Vergleich — Key-Reihenfolge egal, Whitespace egal."""
    try:
        return _json.dumps(args, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(args)


# ── BFCL-Testset aus gorilla-llm laden ──────────────────────────────────────
# Das offizielle Berkeley-Function-Calling-Leaderboard-Dataset.
# Wir nehmen drei repräsentative Splits für simple / multiple / parallel.

def _load_bfcl(subset: str, n: int):
    """Lädt BFCL-Testset. Fallback auf xLAM-held-out, falls BFCL nicht verfügbar."""
    try:
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                          split=subset, token=TOK)
        print(f"  {subset}: {len(ds)} samples from gorilla-llm BFCL")
        return list(ds.select(range(min(n, len(ds)))))
    except Exception as e:
        print(f"  {subset}: BFCL unavailable ({e}) — falling back to xLAM held-out")
        seed_map = {"simple": 7001, "multiple": 7002, "parallel": 7003}
        ds = load_dataset("Salesforce/xlam-function-calling-60k",
                          split="train", token=TOK).shuffle(seed=seed_map.get(subset, 7000))
        return list(ds.select(range(n)))


def _row_to_prompt_and_gold(row: dict):
    """Normalisiert BFCL-Rows ODER xLAM-Rows auf (prompt_messages, gold_calls).

    Gold-Calls: Liste von {"name": str, "arguments": dict}.
    """
    # BFCL-Format
    if "question" in row and "function" in row:
        funcs = row["function"] if isinstance(row["function"], list) else [row["function"]]
        q = row["question"]
        if isinstance(q, list) and q and isinstance(q[0], list):
            q = q[0]
        if isinstance(q, list) and q and isinstance(q[0], dict):
            user_content = q[-1].get("content", "")
        else:
            user_content = str(q)
        gold_calls = []
        ground = row.get("ground_truth") or row.get("answers") or []
        if isinstance(ground, str):
            try:
                ground = _json.loads(ground)
            except Exception:
                ground = []
        for g in ground if isinstance(ground, list) else [ground]:
            if isinstance(g, dict) and "name" in g:
                gold_calls.append({"name": g["name"], "arguments": g.get("arguments", {})})
        tools = funcs
        return tools, user_content, gold_calls

    # xLAM-Format
    try:
        tools = _json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        gold_calls = _json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
    except Exception:
        return None, None, None
    return tools, row.get("query", ""), gold_calls


def _eval_subset(model, tokenizer, rows: list, label: str, max_new_tokens: int = 128):
    parse_ok = 0
    name_ok = 0
    args_ok = 0
    full_ok = 0
    total_out = 0
    total_secs = 0.0
    n_scored = 0

    device = next(model.parameters()).device

    for row in rows:
        tools, query, gold = _row_to_prompt_and_gold(row)
        if not tools or not query or not gold:
            continue

        sys_msg = (
            AGENT_SYS + "\n<tools>\n"
            + _json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
            + "\n</tools>"
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": query},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                stop_strings=["</tool_call>"],
                tokenizer=tokenizer,
            )
        dt = time.perf_counter() - t0

        gen = out[0, inputs["input_ids"].shape[1]:]
        n_tokens = int(gen.shape[0])
        total_out += n_tokens
        total_secs += dt
        text = tokenizer.decode(gen, skip_special_tokens=True)

        parsed = _parse_calls(text)
        if parsed:
            parse_ok += 1

        # Name-Match + Args-Match — set-basiert, Reihenfolge egal.
        gold_names = {g["name"] for g in gold if isinstance(g, dict) and "name" in g}
        pred_names = {p["name"] for p in parsed if isinstance(p, dict) and "name" in p}
        name_match = bool(gold_names) and gold_names == pred_names

        gold_sig = {(g["name"], _normalize_args(g.get("arguments", {})))
                    for g in gold if isinstance(g, dict) and "name" in g}
        pred_sig = {(p["name"], _normalize_args(p.get("arguments", {})))
                    for p in parsed if isinstance(p, dict) and "name" in p}
        args_match = bool(gold_sig) and gold_sig == pred_sig

        if name_match:
            name_ok += 1
        if args_match:
            args_ok += 1
        if name_match and args_match:
            full_ok += 1

        n_scored += 1

    if n_scored == 0:
        return {"error": "no scorable samples"}

    metrics = {
        f"bfcl/{label}/parse_rate":       round(parse_ok / n_scored, 4),
        f"bfcl/{label}/name_accuracy":    round(name_ok / n_scored, 4),
        f"bfcl/{label}/args_accuracy":    round(args_ok / n_scored, 4),
        f"bfcl/{label}/full_accuracy":    round(full_ok / n_scored, 4),
        f"bfcl/{label}/avg_out_tokens":   round(total_out / n_scored, 2),
        f"bfcl/{label}/decode_toks_sec":  round(total_out / total_secs, 2) if total_secs > 0 else 0.0,
        f"bfcl/{label}/n":                n_scored,
    }
    print(f"[bfcl] {label}: {metrics}")
    return metrics


print(f"loading merged model → {MERGED_REPO}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MERGED_REPO,
    max_seq_length=8192,
    load_in_4bit=True,  # Eval in 4-bit spart VRAM; ändert Ergebnisse nur marginal
    dtype=torch.bfloat16,
    device_map={"": 0},
    token=TOK,
)
FastLanguageModel.for_inference(model)
model.eval()

SUBSETS = [
    ("simple",   100),
    ("multiple", 100),
    ("parallel", 100),
]

all_metrics: dict = {"model": MERGED_REPO}
for name, n in SUBSETS:
    print(f"\n=== BFCL subset: {name} (n={n}) ===")
    rows = _load_bfcl(name, n)
    all_metrics.update(_eval_subset(model, tokenizer, rows, name))

# Aggregat über alle Subsets (ungewichtet)
parse_vals = [v for k, v in all_metrics.items() if k.endswith("/parse_rate")]
full_vals  = [v for k, v in all_metrics.items() if k.endswith("/full_accuracy")]
tok_vals   = [v for k, v in all_metrics.items() if k.endswith("/avg_out_tokens")]
sec_vals   = [v for k, v in all_metrics.items() if k.endswith("/decode_toks_sec")]

if full_vals:
    all_metrics["bfcl/overall/parse_rate"]      = round(sum(parse_vals) / len(parse_vals), 4)
    all_metrics["bfcl/overall/full_accuracy"]   = round(sum(full_vals) / len(full_vals), 4)
    all_metrics["bfcl/overall/avg_out_tokens"]  = round(sum(tok_vals) / len(tok_vals), 2)
    all_metrics["bfcl/overall/decode_toks_sec"] = round(sum(sec_vals) / len(sec_vals), 2)

print("\n=== OVERALL ===")
for k in sorted(all_metrics):
    if k.startswith("bfcl/overall/"):
        print(f"  {k}: {all_metrics[k]}")

report_path = Path("/workspace/bfcl_report.json")
report_path.write_text(_json.dumps(all_metrics, indent=2))
print(f"\nreport → {report_path}")

try:
    import wandb
    wandb.init(project=os.environ.get("WANDB_PROJECT", "CoderLLM"),
               name=os.environ.get("WANDB_NAME", "bfcl-eval-runpod"))
    wandb.log({k: v for k, v in all_metrics.items() if isinstance(v, (int, float))})
    wandb.finish()
except Exception as e:
    print(f"(wandb skipped: {e})")

Path("/workspace/.phase06_done").touch()
print("\nBFCL-lite eval done.")
