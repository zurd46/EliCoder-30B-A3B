"""
Streamt SFT/DPO/LongCtx-Datasets und berechnet Token-Längen-Statistiken
mit dem echten Qwen3-Coder-Tokenizer (inkl. Chat-Template für SFT).

Nutzung:
    HF_TOKEN=... python training/analyze_seqlen.py
"""
from __future__ import annotations
import os, sys, statistics
from datasets import load_dataset
from transformers import AutoTokenizer

TOK_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")

SAMPLE_N = int(os.environ.get("SAMPLE_N", "2000"))

tok = AutoTokenizer.from_pretrained(TOK_NAME, token=HF_TOKEN, trust_remote_code=True)


def percentiles(xs: list[int]) -> dict:
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> int:
        if n == 0:
            return 0
        k = min(n - 1, int(round(p / 100 * (n - 1))))
        return xs_sorted[k]

    return {
        "n": n,
        "min": xs_sorted[0],
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": xs_sorted[-1],
        "mean": int(statistics.mean(xs_sorted)),
    }


def analyze(name: str, repo: str, kind: str) -> None:
    print(f"\n=== {name} — {repo} ===", flush=True)
    ds = load_dataset(repo, split="train", streaming=True, token=HF_TOKEN)
    lens: list[int] = []
    extra: dict[str, list[int]] = {}

    for i, row in enumerate(ds):
        if i >= SAMPLE_N:
            break
        try:
            if kind == "sft":
                txt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
                lens.append(len(tok(txt, add_special_tokens=False)["input_ids"]))
            elif kind == "dpo":
                prompt = row.get("prompt") or ""
                chosen = row.get("chosen") or ""
                rejected = row.get("rejected") or ""
                p = len(tok(prompt, add_special_tokens=False)["input_ids"])
                c = len(tok(chosen, add_special_tokens=False)["input_ids"])
                r = len(tok(rejected, add_special_tokens=False)["input_ids"])
                lens.append(p + max(c, r))
                extra.setdefault("prompt", []).append(p)
                extra.setdefault("chosen", []).append(c)
                extra.setdefault("rejected", []).append(r)
            elif kind == "longctx":
                txt = row.get("text") or row.get("content") or ""
                if not txt and "messages" in row:
                    txt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
                lens.append(len(tok(txt, add_special_tokens=False)["input_ids"]))
        except Exception as e:
            print(f"  row {i} err: {e}", flush=True)
            continue

        if (i + 1) % 200 == 0:
            print(f"  …{i+1} rows", flush=True)

    if not lens:
        print("  NO ROWS!")
        return

    stats = percentiles(lens)
    print(f"  total tokens/row: n={stats['n']} min={stats['min']} "
          f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
          f"p99={stats['p99']} max={stats['max']} mean={stats['mean']}")
    for k, xs in extra.items():
        s = percentiles(xs)
        print(f"  {k:<10} p50={s['p50']} p95={s['p95']} p99={s['p99']} max={s['max']}")

    # kosten/nutzen tabelle
    print("  truncation-cost bei verschiedenen max_seq_length:")
    for ms in (2048, 4096, 6144, 8192, 16384):
        over = sum(1 for x in lens if x > ms)
        print(f"    max_seq={ms:>6}: {over:>4}/{len(lens)} ({100*over/len(lens):.1f}%) truncated")


if __name__ == "__main__":
    targets = [
        ("SFT", "zurd46/coder-16b-dyn-sft", "sft"),
        ("DPO", "zurd46/coder-16b-dyn-dpo", "dpo"),
        ("LongCtx", "zurd46/coder-16b-dyn-longctx", "longctx"),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, repo, kind in targets:
        if only and only.lower() not in name.lower():
            continue
        analyze(name, repo, kind)
