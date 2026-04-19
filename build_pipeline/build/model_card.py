from pathlib import Path
from rich.console import Console

from .config import load

console = Console()


def _card(cfg, variant_kind: str, variant_id: str, description: str, size_gb: float) -> str:
    tags_yaml = "\n".join(f"- {t}" for t in cfg.tags)
    base = cfg.base_repo
    repo_full = f"{cfg.hf_owner}/{cfg.hf_repo_prefix}-{variant_id}"

    return f"""---
license: {cfg.license}
base_model:
  - {base}
library_name: transformers
pipeline_tag: text-generation
tags:
{tags_yaml}
- {variant_kind}
language:
- en
- de
---

# {cfg.display_name} — {variant_id}

{description}

- **Kind:** {variant_kind.upper()}
- **Base Model:** [{base}](https://huggingface.co/{base})
- **Expected size on disk:** ~{size_gb:.1f} GB
- **Recommended runtime:** [LM Studio ≥ 0.3.6](https://lmstudio.ai)
- **License:** {cfg.license}

## Quickstart — LM Studio

```bash
# 1. Install LM Studio (>= 0.3.6) from https://lmstudio.ai
# 2. Download this variant (UI) or via CLI:
lms get {repo_full}
# 3. Start OpenAI-compatible server
lms server start
```

Then point any OpenAI client at `http://localhost:1234/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
resp = client.chat.completions.create(
    model="{cfg.hf_repo_prefix}-{variant_id}",
    messages=[{{"role": "user", "content": "Scaffolde eine FastAPI app mit Tests."}}],
    tools=[...],
)
```

## Quickstart — llama.cpp ({variant_kind} only if GGUF)

```bash
llama-cli -m ./{cfg.model_name}-{variant_id}.gguf -p "Hallo" --n-gpu-layers 99
```

## Quickstart — MLX ({variant_kind} only if MLX)

```python
from mlx_lm import load, generate
model, tok = load("{repo_full}")
print(generate(model, tok, prompt="def quicksort(", max_tokens=256))
```

## Features

- Tool/Function-Calling (OpenAI-kompatibel)
- 256k nativer Kontext (empfohlen: 131k in LM Studio default)
- Zweisprachig (DE/EN) Tool-Use & Code
- Optimiert für Senior-Dev-Workflow: plan → act → verify

## Hardware-Empfehlung

| Variant | Minimum RAM/VRAM | Optimal |
|---|---|---|
| UD-IQ2_M  | 16 GB | 24 GB |
| UD-Q3_K_XL | 20 GB | 32 GB |
| UD-Q4_K_XL | 24 GB | 32 GB |
| mlx-4bit  | M2/M3/M4/M5 mit 24+ GB | M-Max/Ultra |

## Fine-tuning

Dieses Modell wurde via Unsloth QLoRA auf einem kuratierten Code-+-Tool-Use-Dataset
post-getrained (SFT + DPO + Long-Context-Stretch). Trainings-Notebooks im Haupt-Repo.

## Citation

Base: `{base}`. Please cite the upstream Qwen team for the pretrained backbone.

## Author

Built by **{cfg.hf_owner}**. Released under {cfg.license}.
"""


def write_card(pkg_dir: Path, variant_id: str) -> Path:
    cfg = load()
    all_q = [(q.id, "gguf", q.description, q.expected_size_gb) for q in cfg.gguf] \
          + [(q.id, "mlx",  q.description, q.expected_size_gb) for q in cfg.mlx]
    entry = next((e for e in all_q if e[0] == variant_id), None)
    if entry is None:
        raise ValueError(f"Unknown variant {variant_id}")
    _, kind, desc, size = entry
    md = _card(cfg, kind, variant_id, desc, size)
    target = pkg_dir / "README.md"
    target.write_text(md)
    console.print(f"[green]Model card written[/] \u2192 {target}")
    return target
