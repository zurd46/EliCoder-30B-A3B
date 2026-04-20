# CoderLLM Build Pipeline

End-to-end pipeline: **Qwen3-Coder-30B-A3B → MLX + GGUF (imatrix-calibrated) → LM-Studio-packages → Hugging Face**.

Ein Befehl (`coderllm-build all`) lädt das Basismodell, erzeugt alle Quant-Varianten (MLX 4-bit/3-bit + GGUF UD-Q4_K_XL / UD-Q3_K_XL / UD-IQ2_M), packt jeweils `model.yaml` + `README.md` dazu und pusht alles in dein HF-Namespace.

---

## Inhalt

| Pfad | Zweck |
|---|---|
| `build/cli.py` | Typer-CLI · Subcommands |
| `build/download.py` | HF snapshot_download der Base |
| `build/convert_mlx.py` | `mlx_lm.convert` für 4-bit / 3-bit |
| `build/convert_gguf.py` | llama.cpp `convert_hf_to_gguf` + `llama-imatrix` + `llama-quantize` |
| `build/lm_studio.py` | Paketiert jede Variante mit `model.yaml` für LM Studio |
| `build/model_card.py` | Generiert HF README je Variante (base-model-tag, LM-Studio-Quickstart) |
| `build/upload.py` | `create_repo` + `upload_folder` pro Variante |
| `configs/quants.yaml` | Definiert alle Quant-Profile + HF-Ziel |
| `configs/model_yaml_template.yaml` | LM-Studio-Preset (Tool-Use, 131k Ctx, q8_0 KV) |
| `scripts/build_all.sh` | One-Shot Orchestrator |

---

## Voraussetzungen

**Alle Plattformen**
- Python ≥ 3.10
- `git`, `cmake`, C++-Compiler (für llama.cpp build)
- 80+ GB freier Disk (Base ~60 GB + F16 GGUF ~60 GB + Quants ~50 GB)
- HuggingFace-Token (Write-Scope) → `export HF_TOKEN=hf_xxx`

**macOS (MLX-Build)**
- Apple Silicon (M1/M2/M3/M4/M5)
- Xcode Command Line Tools

**Linux / Windows**
- MLX-Konvertierung wird automatisch übersprungen — GGUF läuft trotzdem

---

## Installation

```bash
cd build_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env           # dann HF_TOKEN eintragen
```

Optional: eigenen Arbeitsordner außerhalb des Repos (wegen Dateigröße):

```bash
export CODERLLM_WORK=/Volumes/ExternalSSD/coderllm-work
```

---

## Benutzung

### Alles in einem Rutsch

```bash
export HF_TOKEN=hf_xxx
bash scripts/build_all.sh
```

Läuft in ~2-6 h (abhängig von CPU/GPU und Disk I/O). Upload am Ende schaltet sich automatisch an, wenn `HF_TOKEN` gesetzt ist.

### Einzelne Schritte (empfohlen beim Iterieren)

```bash
# Info zur Konfiguration
coderllm-build info

# 1. Basis-Gewichte von HF laden (~60 GB)
coderllm-build download

# 2. MLX-Varianten (nur Mac) — ~15 min pro Variante
coderllm-build convert-mlx                  # alle
coderllm-build convert-mlx --quant mlx-4bit  # nur eine

# 3. GGUF-Varianten (alle Plattformen)
coderllm-build convert-gguf                  # F16 + imatrix + alle quants
coderllm-build convert-gguf --quant UD-Q4_K_XL

# 4. LM-Studio-Packaging (fügt model.yaml hinzu)
coderllm-build package --kind all

# 5. HuggingFace Upload
coderllm-build whoami
coderllm-build upload --variant UD-Q4_K_XL
coderllm-build upload --kind gguf            # alle GGUFs
coderllm-build upload                        # alles

# Aufräumen (F16-Zwischenprodukt ~60 GB)
coderllm-build cleanup
```

### Dry-Run für Upload

```bash
coderllm-build upload --variant UD-Q4_K_XL --dry-run
```

---

## Ausgabe-Struktur

```
work/
├── base/Qwen__Qwen3-Coder-30B-A3B-Instruct/     # gedownloadete Safetensors
├── mlx/
│   ├── coder-16b-dyn-mlx-4bit/
│   └── coder-16b-dyn-mlx-3bit/
├── gguf/
│   ├── coder-16b-dyn-f16.gguf                   # Zwischenprodukt, löschbar
│   ├── coder-16b-dyn-f16.imatrix                # importance-matrix
│   ├── coder-16b-dyn-UD-Q4_K_XL.gguf
│   ├── coder-16b-dyn-UD-Q3_K_XL.gguf
│   └── coder-16b-dyn-UD-IQ2_M.gguf
├── packages/                                     # Upload-ready (model.yaml + README)
│   ├── coder-16b-dyn-mlx-4bit/
│   ├── coder-16b-dyn-UD-Q4_K_XL/
│   └── ...
└── llama.cpp/                                    # auto-cloned + gebaut
```

Jedes `packages/<variant>/`-Verzeichnis enthält genau was HuggingFace braucht und LM Studio direkt lesen kann.

---

## LM Studio nach Upload

```bash
# UI: "Search" → deinen Repo-Namen → Download
# CLI:
lms get zurd46/coder-16b-dyn-UD-Q4_K_XL
lms server start
```

Dann zeigt jeder OpenAI-kompatible Client (Cursor, Continue, Aider, unser Agent) auf `http://localhost:1234/v1`.

---

## Konfiguration anpassen

Alle Entscheidungen leben in `configs/quants.yaml`:

- **Base-Modell wechseln:** `base.hf_repo`
- **Quants hinzufügen/entfernen:** `gguf_quants[]`, `mlx_quants[]`
- **HF-Owner ändern:** `hf_upload.owner`
- **Private Repos:** `hf_upload.private: true`

`configs/model_yaml_template.yaml` definiert das LM-Studio-Preset (System-Prompt, Context-Length, KV-Cache-Type, Sampling).

---

## Troubleshooting

- **`llama-quantize` not found** → `scripts/build_all.sh` cloned+builded llama.cpp automatisch. Wenn cmake fehlt: `brew install cmake` / `choco install cmake`.
- **`mlx_lm` Import-Error auf Linux/Windows** → erwartet, MLX ist Apple-only. Der Step wird übersprungen.
- **HuggingFace 403** → Token-Scope: muss `write` haben; Organization-Access falls `hf_owner` ein Org ist.
- **Zu wenig Disk** → `CODERLLM_WORK` auf externe SSD setzen.
- **imatrix dauert ewig** → Korpus-Größe reduzieren (`--chunks 64`) in `convert_gguf.py`.

---

## Nächster Schritt

Diese Pipeline konvertiert + released Qwen3-Coder-30B-A3B unverändert. Der Fine-Tuning-Teil (der das Modell tatsächlich **unser** macht) läuft in Colab und produziert gemergte Safetensors, die dann exakt mit dieser Pipeline durchgeschickt werden → siehe `../training/` im Haupt-Repo.
