# EliCoder-30B-A3B

Ein **Open-Source Coder-Modell + Agent-Runtime** für Consumer-Hardware (Mac M-Series oder Windows/Linux mit CUDA/Vulkan). Das Modell läuft in **LM Studio**, der mitgelieferte Agent [`coder`](coder/) spricht OpenAI-kompatibel gegen `localhost:1234/v1` und bringt eine vollständige Tool-Suite (Filesystem, Git, GitHub, Shell, Tests, Projekt-Scaffolding, LSP, Dev-Server, Semantic-Search, Memory) sowie Autonomie-Stufen (`safe` / `standard` / `yolo`), Streaming, parallele Tool-Calls und Context-Compaction mit.

Modell auf HF: **[zurd46/EliCoder-30B-A3B](https://huggingface.co/zurd46/EliCoder-30B-A3B)**. 256 k Kontext, Tool-Use, Senior-Dev-Verhalten.

Ziel: schnellstes **und** bestes Open-Source-Coder-Modell für Macbook (32 GB) und Desktop-GPUs.

```
Base (Qwen3-Coder-30B-A3B)
         │
         ▼  (RunPod/Colab H100, Unsloth)
   SFT → DPO → Long-Ctx    ←── training/ notebooks
         │
         ▼
   EliCoder-30B-A3B (merged)
         │
         ▼  (Mac/Linux/Windows)
   MLX + GGUF (imatrix)    ←── build_pipeline/
         │
         ▼
   Hugging Face Hub  (zurd46/EliCoder-30B-A3B-*)
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
- `zurd46/EliCoder-30B-A3B-mlx-4bit`
- `zurd46/EliCoder-30B-A3B-mlx-3bit`
- `zurd46/EliCoder-30B-A3B-UD-Q4_K_XL`
- `zurd46/EliCoder-30B-A3B-UD-Q3_K_XL`
- `zurd46/EliCoder-30B-A3B-UD-IQ2_M`

Auf Windows/Linux: nur die GGUFs (MLX wird automatisch übersprungen).

### Phase 2 — Fine-Tuning (Colab H100)

Siehe [training/](training/). 5 Notebooks:

1. **Dataset-Build** — SWE-Bench + Tool-Traces + Long-Ctx Mix
2. **SFT** (Unsloth QLoRA, ~12 h)
3. **DPO** auf SWE-Bench-Paaren (~4 h)
4. **Long-Ctx-Stretch** Layer 30-47 (~2 h)
5. **Export** → merge → safetensors → via build_pipeline → HF

Gesamt ~18 h H100, ~35 CHF Colab Pro Compute Units.

### Phase 3 — Agent-Runtime ([coder/](coder/))

Senior-Dev-Agent vor LM Studio. Python-CLI, OpenAI-kompatibel gegen `localhost:1234/v1`:

```bash
cd coder && pip install -e .
coder-agent health
coder-agent run --autonomy standard "Scaffolde eine FastAPI-App mit Tests und pushe als GitHub-Repo"
coder-agent repl --cwd ~/projects/mein-projekt   # interaktiv mit /plan, /budget, /clear-cache
```

**Tool-Suite:** Filesystem CRUD, Projekt-Scaffolding, Package-Manager, Git + GitHub (inkl. PR/Release),
Shell (optional sandboxed), Docker, LSP, Testing, Dev-Server, Semantic-Search, Memory — cross-platform.

**Runtime-Features:** Token-Streaming, parallele Tool-Calls, Context-Compaction bei > 70 % Füllung,
LRU-Cache für Read-Tools, Reflection-Retry, dynamische Temperatur (Plan/Exec/Reflect),
Model-Router zu kleinerem Seiten-Modell, Sandbox-Shell (`sandbox-exec` / `bwrap`), Wall-Clock-Budget.

**Autonomie-Stufen:**

| Stufe | Lesen | Schreiben | Löschen | git push / PR-Merge |
|---|---|---|---|---|
| `safe` | ✅ | Bestätigung | Bestätigung | ❌ |
| `standard` | ✅ | ✅ | Bestätigung | Bestätigung |
| `yolo` | ✅ | ✅ | ✅ | ✅ |

Details: [coder/README.md](coder/README.md), Architektur-Referenz: §6 in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
# LM Studio > Import Model > work/packages/EliCoder-30B-A3B-UD-Q4_K_XL/
```

---

## Lizenz

Apache 2.0. Base-Attribution an [Qwen Team](https://huggingface.co/Qwen).
