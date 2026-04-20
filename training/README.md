# Training (Colab H100)

Fünf Notebooks. Reihenfolge strikt:

1. **01_data_build.py** — kuratiert Mix aus SWE-Bench-Train, OpenCodeReasoning-v2, Glaive-Function-Calling und synthetischen Long-Context-Needles. Outputs nach HF-Hub als Private-Dataset.
2. **02_sft_unsloth.py** — Phase A: Supervised Fine-Tuning mit Unsloth QLoRA (Rank 64), 3 Epochen, ~12 h H100.
3. **03_dpo_unsloth.py** — Phase B: Direct Preference Optimization auf passed/failed SWE-Paaren, ~4 h.
4. **04_longctx.py** — Phase C: nur Layer 30-47 fine-tunen für 128k→256k Stretch, ~2 h.
5. **05_export.py** — merge LoRA + push Safetensors zu HF; lokal dann mit `build_pipeline` in MLX+GGUF konvertieren.

Alle Dateien sind `.py` mit `# %%` Cell-Markern — direkt in Colab als Notebook öffenbar oder `jupytext --to ipynb`.

## Setup in Colab

```python
# Cell 1
!pip install -q unsloth "trl>=0.12" "transformers>=4.46" "datasets>=3.0" \
               "peft>=0.13" "accelerate>=1.0" bitsandbytes wandb
```

Runtime: **H100 80 GB** (Colab Pro Compute Units).

## Konfigurationen

Siehe `configs/` — alle Hyperparameter liegen dort, nicht in den Notebooks.

## HF-Tokens

Colab → `Secrets` → `HF_TOKEN` mit write-scope. Wird von allen Notebooks gelesen.

## Reihenfolge der Ausführung

```
01_data_build.py  (einmalig, produziert Dataset)
      ↓
02_sft_unsloth.py  (Phase A, LoRA-Adapter v1)
      ↓
03_dpo_unsloth.py  (Phase B, liest LoRA v1, schreibt v2)
      ↓
04_longctx.py      (Phase C, liest LoRA v2, schreibt v3)
      ↓
05_export.py       (merged full-precision safetensors → HF)
      ↓
# lokal:
cd ../build_pipeline && coderllm-build all --upload
```
