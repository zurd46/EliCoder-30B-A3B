"""
Model card templates for HF Hub uploads.

Used by 05_export.py to push a professional README.md alongside the merged model.
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

# CoderLLM 16B-dyn (BF16)

Fine-tuned version of [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) — a 30B-parameter Mixture-of-Experts model with ~3B parameters active per token, optimized as a coding assistant.

This checkpoint is the **merged BF16 base model** produced at the end of a three-phase LoRA training pipeline (SFT → DPO → Long-Context). It is intended as the source for downstream quantization (MLX, GGUF, AWQ, GPTQ) rather than as an inference target itself.

## Model details

| | |
|---|---|
| Architecture | Qwen3-MoE (128 experts, top-8 routing) |
| Total parameters | ~30B |
| Active parameters / token | ~3B |
| Context length | up to 262,144 tokens (YaRN scaling) |
| Precision | BF16 (dense export) |
| Base model | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| Fine-tuning method | LoRA (r=64, α=128) on attention projections |
| License | Apache-2.0 (inherited from base) |

## Intended use

- Code generation, refactoring, and review across multiple languages (primary: Python, JavaScript, TypeScript, Go, Rust)
- Tool-use / function-calling workflows
- Agentic coding tasks with long context (entire repositories / multi-file edits)
- Bilingual prompts (English / German)

Not intended for: safety-critical decisions, medical/legal advice, production use without independent evaluation.

## Training pipeline

The model was trained in four phases on a single NVIDIA H100 NVL:

### 1. Dataset build (CPU)
Consolidated datasets pushed to private HF Hub repos — see the training repository for the exact build recipe.

| Phase | Dataset | Samples |
|---|---|---|
| SFT | `zurd46/coder-16b-dyn-sft` | ~180k |
| DPO | `zurd46/coder-16b-dyn-dpo` | ~70k |
| Long-Context | `zurd46/coder-16b-dyn-longctx` | ~8k synthetic |

SFT sources include Nemotron-SFT-OpenCode, OpenCodeReasoning-2, Magicoder-Evol / OSS-Instruct, Glaive Function-Calling, SWE-bench_Verified, and Agentic-CoT-Coding.
DPO sources include Vezora/Code-Preference-Pairs and jondurbin/py-dpo-v0.1.

### 2. SFT (Supervised Fine-Tuning)
- LoRA r=64, α=128 on `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Sequence packing at 8192 tokens
- Effective batch size 64, learning rate 2e-5 (cosine), 1 epoch

### 3. DPO (Direct Preference Optimization)
- Built on the SFT LoRA adapter
- β = 0.05, learning rate 2e-6
- Sequence length 8192, effective batch size 32

### 4. Long-Context extension
- YaRN rope scaling (factor 4.0) to 262k positions
- Layer-freeze: only layers 30+ trained
- Small learning rate (2e-5) for stability

### 5. Merge
The SFT + DPO + LongCtx LoRA adapters are merged into the BF16 base weights to produce this checkpoint.

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
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, temperature=0.2)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### With MLX (Apple Silicon)
Quantized MLX builds are published as sibling repos (4-bit / 6-bit / 8-bit). Use `mlx-lm`:
```bash
pip install mlx-lm
python -m mlx_lm.generate --model zurd46/coder-16b-dyn-mlx-4bit --prompt "Fix this function: ..."
```

### With llama.cpp / LM Studio
GGUF quants (Q4_K_M, Q5_K_M, Q6_K, Q8_0) are published as sibling repos and are directly loadable in LM Studio.

## Limitations

- **Training-data overlap with pre-training.** Several SFT sources (Magicoder, Nemotron) are widely used and likely overlap with Qwen3-Coder's pre-training corpus. Expect modest gains on generic coding benchmarks; larger gains on tasks matching the specific DPO preference signal (buggy vs. fixed code).
- **Long-context training is synthetic.** The YaRN fine-tune uses synthetic needle-in-haystack data, not real long-form code repositories. Genuine long-range reasoning may still be weaker than the model card's 262k context length suggests.
- **No safety-specific RLHF.** The DPO phase targets code quality, not harmlessness.
- **LoRA scope.** Only attention projections were fine-tuned; MoE expert weights are unchanged from the base model.

## Evaluation

No standardized benchmark runs have been published for this checkpoint. Internal evaluation focuses on held-out samples from the SFT and DPO datasets.

## Acknowledgments

- [Qwen team](https://huggingface.co/Qwen) for the Qwen3-Coder-30B-A3B-Instruct base model
- [Unsloth](https://github.com/unslothai/unsloth) for the LoRA training stack
- Dataset authors: NVIDIA (Nemotron / OpenCodeReasoning), ise-uiuc (Magicoder), Glaive AI, Princeton NLP (SWE-bench), Vezora, Jon Durbin, AlicanKiraz0

## Citation

```bibtex
@misc{coderllm-16b-dyn,
  title  = {CoderLLM 16B-dyn: a LoRA-tuned Qwen3-Coder for coding assistance},
  author = {Zurmühle, Daniel},
  year   = {2026},
  howpublished = {\\url{https://huggingface.co/zurd46/coder-16b-dyn-base-fp16}}
}
```
"""


def write_card(path, content: str = MERGED_CARD) -> None:
    """Write the model card to `path/README.md`."""
    from pathlib import Path
    p = Path(path) / "README.md"
    p.write_text(content, encoding="utf-8")
    print(f"model card → {p}")
