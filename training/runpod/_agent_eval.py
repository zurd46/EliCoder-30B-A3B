"""
Agent-Eval Callback — live Metriken während Training.

Misst auf einem 50-Sample Held-out Set (xLAM mit anderem Shuffle-Seed als Training):
  - agent/parse_rate       % valide <tool_call>…</tool_call>-JSON-Blöcke
  - agent/name_match       % wo Tool-Name mit Gold-Answer übereinstimmt
  - agent/avg_output_tokens Mac-Speed-Proxy: Decode-Länge pro Antwort

Wird nach jedem trainer.evaluate() aufgerufen (via on_evaluate-Hook).
Alle Metriken gehen in stdout und — falls aktiv — W&B.
"""
from __future__ import annotations

import json as _json
import os
import re as _re
from typing import Optional

import torch
from transformers import TrainerCallback

_TOOL_CALL_RE = _re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", _re.DOTALL)

# Muss identisch zum Trainings-Prompt in 01_data_build.py sein — sonst
# widerspricht der Eval-Prompt dem gelernten Signal.
AGENT_SYS = (
    "You are a function-calling agent. Respond with tool calls only, no prose.\n"
    "Emit one <tool_call>…</tool_call> block per call, JSON on a single line, "
    "compact (no spaces). End immediately after the last </tool_call>."
)


def _parse_tool_call(text: str) -> Optional[dict]:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return _json.loads(m.group(1))
    except Exception:
        return None


class AgentEvalCallback(TrainerCallback):
    """Periodische Agent-Qualitäts-Eval auf xLAM-Held-out-Samples."""

    def __init__(self, tokenizer, n_samples: int = 20, seed: int = 999,
                 max_new_tokens: int = 96, max_prompt_tokens: int = 2048):
        # Defaults bewusst knapp gehalten: wir laufen WÄHREND Training,
        # wo VRAM bereits ~90% belegt ist (weights + optimizer + grads).
        # 50 Samples × 128 max_new_tokens × 3k prompt = OOM auf H100-NVL.
        # 20×96×2k passt mit ~2 GB Luftpolster nach torch.cuda.empty_cache().
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self._seed = seed
        self._samples: Optional[list] = None

    def _load_samples(self):
        if self._samples is not None:
            return self._samples
        from datasets import load_dataset
        tok = os.environ.get("HF_TOKEN")
        ds = load_dataset(
            "Salesforce/xlam-function-calling-60k", split="train", token=tok
        ).shuffle(seed=self._seed)
        out = []
        # 4× puffer weil einige Samples beim Parsen ausfallen können.
        for row in ds.select(range(min(self.n_samples * 4, len(ds)))):
            try:
                tools = _json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
                calls = _json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
                if not calls:
                    continue
                out.append({"query": row["query"], "tools": tools, "gold": calls[0]})
                if len(out) >= self.n_samples:
                    break
            except Exception:
                continue
        self._samples = out
        print(f"[agent-eval] loaded {len(out)} held-out xLAM samples (seed={self._seed})")
        return out

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        samples = self._load_samples()
        if not samples:
            print("[agent-eval] no samples available — skipping")
            return

        parse_ok = 0
        name_ok = 0
        total_out_tokens = 0

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        # Vor Eval so viel VRAM wie möglich freigeben — das Training hält
        # KV-Caches, Gradienten-Buffer etc. die während generate() nicht
        # gebraucht werden. Ohne das OOM'ed generate() zuverlässig
        # (256 MB extra-Allokation schlägt auf 86 MB freien VRAM fehl).
        torch.cuda.empty_cache()

        with torch.no_grad():
            for s in samples:
                sys_msg = (
                    AGENT_SYS + "\n<tools>\n"
                    + _json.dumps(s["tools"], ensure_ascii=False, separators=(",", ":"))
                    + "\n</tools>"
                )
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": s["query"]},
                ]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = sys_msg + "\n\nUser: " + s["query"] + "\nAssistant:"

                inputs = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=self.max_prompt_tokens,
                ).to(device)

                try:
                    out = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        stop_strings=["</tool_call>"],
                        tokenizer=self.tokenizer,
                    )
                except Exception as e:
                    # Generate kann bei 4-bit + Checkpointing stolpern — skip Sample.
                    print(f"[agent-eval] generate() failed on sample: {e}")
                    continue

                gen = out[0, inputs["input_ids"].shape[1]:]
                total_out_tokens += int(gen.shape[0])
                text = self.tokenizer.decode(gen, skip_special_tokens=True)
                parsed = _parse_tool_call(text)
                if parsed is not None:
                    parse_ok += 1
                    gold_name = s["gold"].get("name") if isinstance(s["gold"], dict) else None
                    if gold_name and parsed.get("name") == gold_name:
                        name_ok += 1

                # Per-sample cleanup — generate() allokiert KV-cache on-the-fly,
                # wenn wir den nicht freigeben, wächst VRAM über 20 samples hinweg.
                del out, gen, inputs
                torch.cuda.empty_cache()

        n = len(samples)
        metrics = {
            "agent/parse_rate":        round(parse_ok / n, 4),
            "agent/name_match":        round(name_ok / n, 4),
            "agent/avg_output_tokens": round(total_out_tokens / n, 2),
        }
        print(f"[agent-eval] step={state.global_step} {metrics}")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except Exception:
            pass

        if was_training:
            model.train()
