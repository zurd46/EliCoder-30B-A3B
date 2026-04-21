"""
Model card templates for HF Hub uploads.

Two variants:
  - MERGED_CARD  — for the final BF16 merged model repo
  - ADAPTER_CARD — for the intermediate LoRA adapter repos (SFT / DPO / LongCtx)

Used by 05_export.py (merged card) and push_modelcards.py (all repos).
"""
from __future__ import annotations


MERGED_CARD = """\
---
license: apache-2.0
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
language:
- en
- de
library_name: transformers
pipeline_tag: text-generation
tags:
- code
- coding-assistant
- qwen3
- qwen3-coder
- moe
- mixture-of-experts
- lora
- sft
- dpo
- long-context
- yarn
---

# CoderLLM 16B-dyn (BF16 base)

A fine-tuned version of [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) built as the in-house coding assistant for [impulsai.ch](https://impulsai.ch).

**Developer:** Daniel Zurmühle · [impulsai.ch](https://impulsai.ch)

This checkpoint is the **merged BF16 base model** produced at the end of a three-phase LoRA training pipeline (SFT → DPO → Long-Context). It is intended as the source for downstream quantization (MLX, GGUF) and not primarily as a direct inference target.

## Goal

Build a coding-first assistant optimized for bilingual (English / German) developer workflows at impulsai.ch, focused on:

1. Day-to-day code generation, refactoring, and review (Python, TypeScript/JavaScript, Go, Rust, Swift)
2. Tool-use / function-calling orchestration for internal automation
3. Agentic, chain-of-thought coding tasks (multi-step problem decomposition)
4. Understanding long repository contexts (up to 262k tokens via YaRN)
5. Running locally on Apple Silicon through MLX / GGUF quantizations (M-series Macs used daily at impulsai.ch)

## Model details

| | |
|---|---|
| Architecture | Qwen3-MoE (128 experts, top-8 routing) |
| Total parameters | ~30B |
| Active parameters / token | ~3B |
| Context length | up to 262,144 tokens (YaRN, scaling factor 4.0) |
| Precision | BF16 |
| Base model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Fine-tuning method | LoRA (r=64, α=128) on attention projections |
| License | Apache-2.0 (inherited from base) |

## Training pipeline

Five-phase pipeline on a single NVIDIA H100 NVL (94 GB):

### Phase 1 — Dataset construction (CPU)
All training datasets are consolidated from public HuggingFace sources and pushed to private HF repos for reproducibility.

**SFT dataset** → `zurd46/coder-16b-dyn-sft` (~180k samples)

| Source | Samples (max) | Purpose |
|---|---|---|
| `nvidia/Nemotron-SFT-OpenCode-v1` | 40,000 | NVIDIA-curated high-quality open code SFT |
| `nvidia/OpenCodeReasoning-2` (python split) | 40,000 | Chain-of-thought reasoning with step-by-step solutions |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | 40,000 | Evolved coding instructions (WizardCoder-style) |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 20,000 | OSS-inspired coding problems |
| `glaiveai/glaive-function-calling-v2` | 30,000 | Function-calling and tool-use conversations |
| `princeton-nlp/SWE-bench_Verified` | 500 | Real GitHub issues with verified patches |
| `AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` | 10,000 | Agentic multi-step coding reasoning |

**DPO dataset** → `zurd46/coder-16b-dyn-dpo` (~70k preference pairs)

| Source | Pairs (max) | Purpose |
|---|---|---|
| `Vezora/Code-Preference-Pairs` | 50,000 | Buggy vs. fixed code pairs (quality signal) |
| `jondurbin/py-dpo-v0.1` | 20,000 | General Python preference pairs |

**Long-Context dataset** → `zurd46/coder-16b-dyn-longctx` (8,000 samples)

- Synthetic needle-in-haystack: randomly generated passphrases embedded in word-based haystacks of 32k / 64k / 128k / 200k tokens
- Purpose: stabilize positional encodings after YaRN extension to 262k, not to teach new semantic skills

### Phase 2 — Supervised Fine-Tuning (SFT)
- LoRA r=64, α=128 on `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `use_rslora=True` (rank-stabilized scaling)
- Sequence packing at `max_seq_length=8192`
- Per-device batch 2 × gradient accumulation 32 → effective batch 64
- Learning rate 2e-5, cosine schedule, 100 warmup steps, 1 epoch
- Optimizer: 8-bit Paged AdamW, weight decay 0.01
- Gradient checkpointing enabled (Unsloth kernels)
- Output adapter → `zurd46/coder-16b-dyn-lora-sft`

### Phase 3 — Direct Preference Optimization (DPO)
- Built on top of the SFT LoRA adapter
- β = 0.1, learning rate 5e-6, cosine schedule
- `max_seq_length=8192`, `max_prompt_length=8192`
- Effective batch 32, 600 training steps
- Output adapter → `zurd46/coder-16b-dyn-lora-dpo`

### Phase 4 — Long-Context extension (YaRN)
- Built on top of the DPO LoRA adapter
- YaRN rope scaling: factor 4.0, original 65,536 → target 262,144 positions
- Layer-freeze: only transformer layers ≥ 30 are trainable
- Learning rate 2e-5, 1 epoch on the synthetic LongCtx dataset
- Output adapter → `zurd46/coder-16b-dyn-lora-longctx`

### Phase 5 — Merge & export
- SFT + DPO + LongCtx LoRA deltas merged into the BF16 base weights
- Shard size 5 GB, `safetensors` format
- Uploaded as this repository

## How to use

### With `transformers`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "zurd46/coder-16b-dyn-base-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Write a Python function that computes Fibonacci numbers iteratively."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
out = model.generate(inputs, max_new_tokens=512, temperature=0.2)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### With MLX (Apple Silicon, recommended for Mac)
Quantized MLX builds are published as sibling repos:

| Repo | Bits | Size | Use case |
|---|---|---|---|
| [`zurd46/coder-16b-dyn-mlx-4bit`](https://huggingface.co/zurd46/coder-16b-dyn-mlx-4bit) | 4 | ~16.5 GB | Default — fast on M-series Macs |
| [`zurd46/coder-16b-dyn-mlx-3bit`](https://huggingface.co/zurd46/coder-16b-dyn-mlx-3bit) | 3 | ~12.8 GB | Tighter memory budget |

```bash
pip install mlx-lm
python -m mlx_lm.generate --model zurd46/coder-16b-dyn-mlx-4bit --prompt "Fix this function: ..."
```

### With llama.cpp / LM Studio
GGUF quants (Unsloth Dynamic 2.0, imatrix-calibrated) are published as sibling repos:

| Repo | Type | Size | Use case |
|---|---|---|---|
| [`zurd46/coder-16b-dyn-gguf-UD-Q4_K_XL`](https://huggingface.co/zurd46/coder-16b-dyn-gguf-UD-Q4_K_XL) | Q4_K_M | ~17.7 GB | Default quality profile |
| [`zurd46/coder-16b-dyn-gguf-UD-Q3_K_XL`](https://huggingface.co/zurd46/coder-16b-dyn-gguf-UD-Q3_K_XL) | Q3_K_L | ~13.8 GB | Long-context profile (more KV budget) |
| [`zurd46/coder-16b-dyn-gguf-UD-IQ2_M`](https://huggingface.co/zurd46/coder-16b-dyn-gguf-UD-IQ2_M) | IQ2_M | ~10.8 GB | Fits 16 GB hardware |

All GGUF repos ship a `model.yaml` for one-click LM Studio import.

## Limitations

- **Training-data overlap with pre-training.** Several SFT sources (Magicoder, Nemotron) are standard open-source corpora and likely overlap with Qwen3-Coder's pre-training. Expect modest gains on generic coding benchmarks; the clearest gains come from the DPO preference signal (buggy vs. fixed code).
- **Long-context training is synthetic.** The YaRN fine-tune uses random-word needle-in-haystack data, not real long-form code repositories. Real long-range code reasoning may still be weaker than the 262k nominal context suggests.
- **No safety-specific RLHF.** The DPO phase targets code-quality signal, not harmlessness.
- **LoRA scope.** Only attention projections were fine-tuned; MoE expert weights are unchanged from the base model.
- **Not benchmarked.** No public evaluation runs (HumanEval, MBPP, SWE-bench) have been reported for this checkpoint.

## Acknowledgments

- [Qwen team (Alibaba)](https://huggingface.co/Qwen) for the Qwen3-Coder-30B-A3B-Instruct base model
- [Unsloth](https://github.com/unslothai/unsloth) for the LoRA training stack
- Dataset authors: NVIDIA (Nemotron / OpenCodeReasoning), ise-uiuc (Magicoder), Glaive AI, Princeton NLP (SWE-bench), Vezora, Jon Durbin, AlicanKiraz0

## Citation

```bibtex
@misc{coderllm-16b-dyn,
  title  = {CoderLLM 16B-dyn: a LoRA-tuned Qwen3-Coder for bilingual coding assistance},
  author = {Zurm{\\"u}hle, Daniel},
  year   = {2026},
  url    = {https://huggingface.co/zurd46/coder-16b-dyn-base-fp16},
  note   = {impulsai.ch}
}
```
"""


ADAPTER_CARD_TEMPLATE = """\
---
license: apache-2.0
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
language:
- en
- de
library_name: peft
pipeline_tag: text-generation
tags:
- code
- coding-assistant
- qwen3-coder
- moe
- lora
- peft
- {phase_tag}
---

# CoderLLM 16B-dyn — {phase_name} LoRA adapter

{phase_summary}

**Developer:** Daniel Zurmühle · [impulsai.ch](https://impulsai.ch)

This is an intermediate LoRA adapter in a multi-phase training pipeline. The final merged BF16 model is published at [`zurd46/coder-16b-dyn-base-fp16`](https://huggingface.co/zurd46/coder-16b-dyn-base-fp16) — use that for inference.

## Adapter details

| | |
|---|---|
| Base model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Starting point | {starting_point} |
| Training phase | {phase_name} |
| LoRA rank / alpha | r=64 / α=128 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | ~53M (0.17%) |
| Training dataset | {dataset_ref} |
| License | Apache-2.0 (inherited) |

## Training data

{dataset_details}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
adapter = "{adapter_repo}"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Pipeline context

{pipeline_context}

## Limitations

- This is an **intermediate adapter** — the fully trained and merged model is [`zurd46/coder-16b-dyn-base-fp16`](https://huggingface.co/zurd46/coder-16b-dyn-base-fp16).
- Only attention projections were fine-tuned; MoE expert weights are unchanged.
- No safety-specific RLHF and no public benchmark results.
"""


def adapter_card(phase: str) -> str:
    """Render the adapter card for a given phase ('sft', 'dpo', or 'longctx')."""
    variants = {
        "sft": {
            "phase_tag": "sft",
            "phase_name": "SFT",
            "phase_summary": (
                "Supervised fine-tuning (SFT) LoRA adapter built on top of `Qwen/Qwen3-Coder-30B-A3B-Instruct` "
                "for the [impulsai.ch](https://impulsai.ch) coding assistant."
            ),
            "starting_point": "`Qwen/Qwen3-Coder-30B-A3B-Instruct` (base)",
            "dataset_ref": "[`zurd46/coder-16b-dyn-sft`](https://huggingface.co/datasets/zurd46/coder-16b-dyn-sft) (~180k samples)",
            "adapter_repo": "zurd46/coder-16b-dyn-lora-sft",
            "dataset_details": (
                "Aggregated from seven public HuggingFace sources:\n\n"
                "- `nvidia/Nemotron-SFT-OpenCode-v1` (40k) — NVIDIA-curated high-quality code SFT\n"
                "- `nvidia/OpenCodeReasoning-2` (40k, python split) — chain-of-thought coding reasoning\n"
                "- `ise-uiuc/Magicoder-Evol-Instruct-110K` (40k) — evolved coding instructions\n"
                "- `ise-uiuc/Magicoder-OSS-Instruct-75K` (20k) — OSS-inspired problems\n"
                "- `glaiveai/glaive-function-calling-v2` (30k) — function-calling / tool-use\n"
                "- `princeton-nlp/SWE-bench_Verified` (500) — real GitHub issues + patches\n"
                "- `AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` (10k) — agentic reasoning"
            ),
            "pipeline_context": (
                "Phase 2 of 5. The next phase (DPO) is applied on top of this adapter and is published at "
                "[`zurd46/coder-16b-dyn-lora-dpo`](https://huggingface.co/zurd46/coder-16b-dyn-lora-dpo)."
            ),
        },
        "dpo": {
            "phase_tag": "dpo",
            "phase_name": "DPO",
            "phase_summary": (
                "Direct Preference Optimization (DPO) LoRA adapter built on top of the SFT adapter. "
                "Aligns the model towards bug-free, higher-quality code for the [impulsai.ch](https://impulsai.ch) coding assistant."
            ),
            "starting_point": "[`zurd46/coder-16b-dyn-lora-sft`](https://huggingface.co/zurd46/coder-16b-dyn-lora-sft)",
            "dataset_ref": "[`zurd46/coder-16b-dyn-dpo`](https://huggingface.co/datasets/zurd46/coder-16b-dyn-dpo) (~70k preference pairs)",
            "adapter_repo": "zurd46/coder-16b-dyn-lora-dpo",
            "dataset_details": (
                "Aggregated from two public HuggingFace preference datasets:\n\n"
                "- `Vezora/Code-Preference-Pairs` (50k) — buggy vs. fixed code pairs\n"
                "- `jondurbin/py-dpo-v0.1` (20k) — Python preference pairs"
            ),
            "pipeline_context": (
                "Phase 3 of 5. The next phase (Long-Context / YaRN) is applied on top of this adapter and is published at "
                "[`zurd46/coder-16b-dyn-lora-longctx`](https://huggingface.co/zurd46/coder-16b-dyn-lora-longctx)."
            ),
        },
        "longctx": {
            "phase_tag": "long-context",
            "phase_name": "Long-Context (YaRN)",
            "phase_summary": (
                "Long-context extension LoRA adapter built on top of the DPO adapter. "
                "Uses YaRN rope scaling (factor 4.0) to extend the usable context from 65,536 to 262,144 tokens."
            ),
            "starting_point": "[`zurd46/coder-16b-dyn-lora-dpo`](https://huggingface.co/zurd46/coder-16b-dyn-lora-dpo)",
            "dataset_ref": "[`zurd46/coder-16b-dyn-longctx`](https://huggingface.co/datasets/zurd46/coder-16b-dyn-longctx) (8k synthetic samples)",
            "adapter_repo": "zurd46/coder-16b-dyn-lora-longctx",
            "dataset_details": (
                "Synthetic needle-in-haystack dataset: 8,000 samples with passphrases embedded in word-based "
                "haystacks of 32k / 64k / 128k / 200k tokens. The goal is to stabilize positional encodings "
                "after YaRN extension, not to teach new semantic skills. Only transformer layers ≥ 30 are trainable."
            ),
            "pipeline_context": (
                "Phase 4 of 5. Phase 5 merges all three LoRA adapters (SFT + DPO + LongCtx) into the BF16 base weights, "
                "published at [`zurd46/coder-16b-dyn-base-fp16`](https://huggingface.co/zurd46/coder-16b-dyn-base-fp16)."
            ),
        },
    }
    return ADAPTER_CARD_TEMPLATE.format(**variants[phase])


def write_card(path, content: str = MERGED_CARD) -> None:
    """Write the model card to `path/README.md`."""
    from pathlib import Path
    p = Path(path) / "README.md"
    p.write_text(content, encoding="utf-8")
    print(f"model card → {p}")
