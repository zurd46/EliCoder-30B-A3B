# CoderLLM

Ein **Open-Source Coder-Modell** für Consumer-Hardware (Mac M-Series oder Windows/Linux mit CUDA/Vulkan) — gebaut, um in **LM Studio** zu laufen, mit Tool-Use, 256 k Kontext und Senior-Dev-Verhalten.

Ziel: schnellstes **und** bestes Open-Source-Coder-Modell für Macbook (32 GB) und Desktop-GPUs.

```
Base (Qwen3-Coder-30B-A3B)
         │
         ▼  (Colab H100, Unsloth)
   SFT → DPO → Long-Ctx    ←── training/ notebooks
         │
         ▼  (Mac/Linux/Windows)
   MLX + GGUF (imatrix)    ←── build_pipeline/
         │
         ▼
   Hugging Face Hub
         │
         ▼
   LM Studio (localhost:1234/v1)
         │
         ▼
   Agent Runtime (Tools, Context-Mgr)  ←── coder/ (Phase 3)
```

---

## Repo-Struktur

| Ordner | Zweck | Status |
|---|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kompletter Architektur-Plan | ✅ |
| [build_pipeline/](build_pipeline/) | Download → MLX+GGUF → HF-Upload | ✅ |
| [training/](training/) | Colab-Notebooks (SFT + DPO + Long-Ctx) | 🛠️ |
| [eval/](eval/) | SWE-Bench + HumanEval+ + BFCL Harness | 🛠️ |
| [coder/](coder/) | Agent-Runtime (LM-Studio-Client + Tools) | 🛠️ |

---

## Drei Phasen

### Phase 1 — Build-Pipeline (fertig)

Siehe [build_pipeline/README.md](build_pipeline/README.md).

```bash
cd build_pipeline
pip install -e .
export HF_TOKEN=hf_xxx
bash scripts/build_all.sh
```

Produziert auf einem Mac:
- `zurd46/coder-16b-dyn-mlx-4bit`
- `zurd46/coder-16b-dyn-mlx-3bit`
- `zurd46/coder-16b-dyn-UD-Q4_K_XL`
- `zurd46/coder-16b-dyn-UD-Q3_K_XL`
- `zurd46/coder-16b-dyn-UD-IQ2_M`

Auf Windows/Linux: nur die GGUFs (MLX wird automatisch übersprungen).

### Phase 2 — Fine-Tuning (Colab H100)

Siehe [training/](training/). 5 Notebooks:

1. **Dataset-Build** — SWE-Bench + Tool-Traces + Long-Ctx Mix
2. **SFT** (Unsloth QLoRA, ~12 h)
3. **DPO** auf SWE-Bench-Paaren (~4 h)
4. **Long-Ctx-Stretch** Layer 30-47 (~2 h)
5. **Export** → merge → safetensors → via build_pipeline → HF

Gesamt ~18 h H100, ~35 CHF Colab Pro Compute Units.

### Phase 3 — Agent-Runtime

Siehe [coder/](coder/). Python-Client, der auf `localhost:1234/v1` spricht:

```bash
pip install -e ./coder
coder-agent "Scaffolde eine FastAPI app mit Tests und pushe zu GitHub"
```

Volle Tool-Suite (§6 in ARCHITECTURE.md): Filesystem CRUD, Projekt-Scaffolding,
Package-Manager, Git+GitHub, Docker, LSP, Testing, Debug — alles cross-platform.

---

## Ziele (messbar)

| Bench | Ziel |
|---|---|
| SWE-Bench Verified | ≥ 60 % |
| HumanEval+ | ≥ 92 % |
| BFCL v3 Tool-Use | ≥ 88 % |
| Decode-Speed M5 @ 8k | ≥ 140 tok/s (mit Speculative) |
| Decode-Speed RTX 4090 @ 8k | ≥ 220 tok/s |
| Kontext | 262 k (Needle ≥ 95 % @ 200 k) |

Detaillierte Ziele + Architektur in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Schnellstart: Modell **ohne** Fine-Tuning konvertieren & nutzen

Wenn du nur den Base in LM Studio testen willst:

```bash
cd build_pipeline
pip install -e .
coderllm-build download
coderllm-build convert-gguf --quant UD-Q4_K_XL
coderllm-build package --kind gguf --quant UD-Q4_K_XL
# LM Studio > Import Model > work/packages/coder-16b-dyn-UD-Q4_K_XL/
```

---

## Lizenz

Apache 2.0. Base-Attribution an [Qwen Team](https://huggingface.co/Qwen).
