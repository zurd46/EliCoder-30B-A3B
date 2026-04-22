# CoderLLM — Agent Instructions

> **Language:** Project documentation and comments are primarily in **German**. Code identifiers, docstrings, and CLI output are a mix of German and English. All new documentation should be written in German to stay consistent with the existing READMEs and docs.

---

## Project Overview

CoderLLM (Model name: **EliCoder-30B-A3B**) is an open-source code model plus agent runtime targeting consumer hardware (Mac M-Series and Windows/Linux with CUDA/Vulkan). The project is organized into three phases:

1. **Build Pipeline** (`build_pipeline/`) — Downloads the base model (`Qwen/Qwen3-Coder-30B-A3B-Instruct`), quantizes it to GGUF (Unsloth Dynamic 2.0) and MLX, packages presets for LM Studio, and uploads to Hugging Face Hub.
2. **Training** (`training/`) — Fine-tunes the base model via Unsloth QLoRA on Google Colab or RunPod H100: SFT → DPO → Long-Context stretch → Export merged weights.
3. **Agent Runtime** (`coder/`) — A Python CLI (`coder-agent`) that speaks OpenAI-compatibly to LM Studio (`localhost:1234/v1`) and provides ~80 tools (filesystem, git, GitHub, shell, LSP, semantic search, memory, project scaffolding).

HF Hub namespace: `zurd46/EliCoder-30B-A3B-*`  
License: Apache 2.0 (base attribution to Qwen Team).

---

## Repository Structure

```
.
├── build_pipeline/          # Phase 1: Download → quantize → package → upload
│   ├── build/               # Python package (cli, download, convert, upload)
│   ├── configs/             # quants.yaml, model_yaml_template.yaml
│   ├── scripts/
│   ├── work/                # Working dir (base, gguf, mlx, packages, llama.cpp)
│   ├── pyproject.toml       # Package: coderllm-build
│   └── README.md
├── coder/                   # Phase 3: Agent runtime
│   ├── coder/               # Main package
│   │   ├── tools/           # ~20 tool modules
│   │   ├── agent.py         # Agent FSM / orchestration
│   │   ├── client.py        # LM Studio client
│   │   ├── cli.py           # Typer CLI entrypoint
│   │   ├── context.py       # Context manager
│   │   └── settings.py      # Pydantic settings
│   ├── tests/               # Currently empty (reserved)
│   ├── pyproject.toml       # Package: coderllm-agent
│   └── README.md
├── eval/                    # Phase 2/4: Benchmark harness
│   ├── run_*.py             # Individual benchmark scripts
│   ├── aggregate.py         # Results aggregation + regression guard
│   ├── requirements.txt
│   └── README.md
├── training/                # Phase 2: Fine-tuning
│   ├── configs/             # sft.yaml, dpo.yaml, longctx.yaml
│   ├── runpod/              # RunPod-specific optimized pipeline
│   ├── 01_data_build.py     # Source-of-truth scripts
│   ├── 02_sft_unsloth.py
│   ├── 03_dpo_unsloth.py
│   ├── 04_longctx.py
│   ├── 05_export.py
│   ├── make_notebooks.py    # Generates .ipynb from .py
│   └── README.md, RUNPOD.md
└── docs/
    ├── ARCHITECTURE.md      # Full 847-line architecture spec
    ├── FINETUNING_AUDIT_PLAN.md
    └── FINETUNING_SPEEDUP_PLAN.md
```

There is **no root-level `pyproject.toml`**. Each major component (`build_pipeline/`, `coder/`) is an independent pip-installable package.

---

## Technology Stack

- **Language:** Python ≥ 3.10
- **CLI Framework:** Typer (`typer>=0.12`)
- **TUI / Logging:** Rich (`rich>=13.7`)
- **Settings / Validation:** Pydantic v2 (`pydantic>=2.9`)
- **HTTP Client:** OpenAI Python SDK + httpx
- **ML Base:** HuggingFace `transformers`, `safetensors`, `torch`
- **Fine-Tuning:** Unsloth (QLoRA, FlashAttention-2), TRL (`SFTTrainer`, `DPOTrainer`), `bitsandbytes`
- **Quantization:** llama.cpp (GGUF with imatrix), `mlx-lm` (macOS only)
- **Inference Server:** LM Studio ≥ 0.3.6 (OpenAI-compatible on `localhost:1234/v1`)
- **Code Intelligence:** tree-sitter, tree-sitter-languages
- **Semantic Search:** sentence-transformers + faiss-cpu
- **GitHub Integration:** PyGithub
- **Git Automation:** GitPython
- **File Watching:** watchdog

---

## Build, Install and Run Commands

### Build Pipeline (`build_pipeline/`)

```bash
cd build_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # edit HF_TOKEN
```

Commands (entrypoint: `coderllm-build`):

| Command | Purpose |
|---------|---------|
| `coderllm-build info` | Show quant profiles and HF target |
| `coderllm-build download` | Download base model (~60 GB) |
| `coderllm-build convert-mlx [--quant mlx-4bit]` | MLX quant (macOS only) |
| `coderllm-build convert-gguf [--quant UD-Q4_K_XL]` | GGUF F16 + imatrix + quant |
| `coderllm-build package --kind all` | Package with LM Studio `model.yaml` |
| `coderllm-build upload [--variant ...]` | Upload to HuggingFace Hub |
| `coderllm-build cleanup` | Remove F16 intermediate (~60 GB) |
| `coderllm-build auto` | Auto-detect and run full pipeline |

One-shot orchestrator:
```bash
export HF_TOKEN=hf_xxx
bash scripts/build_all.sh
```

### Agent Runtime (`coder/`)

```bash
cd coder
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# optional semantic search
pip install -e ".[embeddings]"
```

Commands (entrypoint: `coder-agent`):

| Command | Purpose |
|---------|---------|
| `coder-agent health` | Check LM Studio connectivity |
| `coder-agent run --autonomy standard "PROMPT"` | One-shot task |
| `coder-agent repl --cwd ~/project` | Interactive REPL (`/plan`, `/budget`, `/clear-cache`, `/exit`) |
| `coder-agent tools --autonomy yolo` | List available tools |
| `coder-agent index --cwd ~/project` | Build semantic search index |

### Evaluation Harness (`eval/`)

```bash
cd eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires LM Studio running with a model loaded.

```bash
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=lm-studio
export MODEL_ID=...
python run_humaneval.py
python run_mbpp.py
python run_bfcl.py
python run_ruler.py
python run_livecodebench.py
python run_swebench.py     # expensive, Docker required
python aggregate.py        # writes results/summary.md; exits != 0 on >2% regression
```

### Training (`training/`)

**Colab (H100):** Open `.ipynb` notebooks in strict order (01 → 05). Each notebook auto-bootstraps (`_bootstrap()` clones repo, installs deps, reads `HF_TOKEN` from Colab secrets).

**RunPod (H100 NVL):**
```bash
cd training
bash runpod/pipeline.sh 2>&1 | tee -a /workspace/pipeline.log
```

Idempotent: writes `/workspace/.phaseNN_done` markers; auto-resumes from checkpoints; auto-shuts down pod at end if `RUNPOD_API_KEY` and `RUNPOD_POD_ID` are set.

**Source of truth:** `.py` files are the source; `.ipynb` files are generated via:
```bash
cd training
python make_notebooks.py
```

---

## Code Organization and Module Divisions

### `build_pipeline/build/`
- `cli.py` — Typer commands
- `config.py` — Loads `configs/quants.yaml`
- `download.py` — HF `snapshot_download`
- `convert_mlx.py` — `mlx_lm.convert` wrapper
- `convert_gguf.py` — llama.cpp `convert_hf_to_gguf` + `llama-imatrix` + `llama-quantize`
- `lm_studio.py` — Packages each variant with `model.yaml` preset
- `model_card.py` — Generates per-variant HF README
- `upload.py` — `create_repo` + `upload_folder`
- `auto.py` / `prebuilt.py` — Auto-detection and prebuilt fetching

### `coder/coder/`
- `agent.py` — Main agent loop: streaming, parallel fan-out, context compaction, retry with reflection
- `client.py` — OpenAI-compatible LM Studio client; dynamic temperature; model routing
- `cli.py` — Typer CLI (`run`, `repl`, `health`, `tools`, `index`)
- `context.py` — Project context summarizer
- `settings.py` — Single Pydantic `Settings` class with env-var fallbacks
- `tools/` — Modular tool suite:
  - `registry.py` — Dispatch with LRU cache, per-tool timeout, autonomy gate
  - `fs.py` — Filesystem CRUD
  - `shell.py` — `run_shell` + sandbox (`sandbox-exec` / `bwrap` / `firejail`)
  - `execute.py` — `run_tests`, `run_python`, `run_node`, `run_lint`, `run_typecheck`
  - `git_tool.py` — Git operations
  - `github_tool.py` — GitHub PR/issue/release operations
  - `patch.py` — `apply_patch`, `multi_edit`, `diff_files`
  - `lsp.py` — Diagnostics, goto-definition, find-references
  - `devserver.py` — Long-running dev server management + log capture
  - `semantic.py` — MiniLM index + cosine search
  - `planning.py` — `todo_write/read/update`, `budget_status`
  - `subagent.py` — `spawn_subagent` (focused tool subset + small model)
  - `think.py` — Scratchpad reasoning
  - `project.py` — Scaffolding templates
  - `code_intel.py` — AST symbols
  - `memory.py` — Persistent memory
  - `web.py` — `fetch_url`, `web_search`

### `eval/`
- `run_*.py` — Per-benchmark OpenAI-compatible harness
- `_common.py` — Shared utilities
- `aggregate.py` — Compares against `results/baseline.json`; non-zero exit on regression > 2 %

### `training/`
- `01_data_build.py` — Dataset construction
- `02_sft_unsloth.py` — Supervised fine-tuning (Unsloth QLoRA)
- `03_dpo_unsloth.py` — Direct preference optimization
- `04_longctx.py` — Long-context stretch (YaRN, freeze layers 0-29, train 30-47)
- `05_export.py` — Merge LoRA + export safetensors
- `configs/*.yaml` — Hyperparameters (LoRA r=64/α=128, max_seq_length=4096, adamw_8bit, etc.)
- `runpod/` — RunPod-optimized copies with `pipeline.sh` orchestrator

---

## Development Conventions and Code Style

- **Python version:** ≥ 3.10; use `from __future__ import annotations` at the top of modules.
- **Type hints:** Mandatory. Use `str | None`, `list[dict]`, etc. (PEP 604 union syntax).
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for module-level constants.
- **Imports:** Standard library first, then third-party, then local (relative imports within packages).
- **CLI pattern:** Every CLI lives in a `cli.py` using `typer.Typer(add_completion=False)` and `rich.console.Console` for output.
- **Settings pattern:** All tunables are centralized in a single Pydantic `BaseModel` (`Settings`) with `os.environ.get(...)` defaults.
- **Comments / docstrings:** Write in German where possible, especially user-facing strings and documentation. Code-internal comments may be in English.
- **No formatter/linter config files** (no `.flake8`, `ruff.toml`, `pyproject.toml` `[tool.ruff]`, etc.) are present in the repo. If you add one, keep it minimal and consistent with the existing style.

---

## Testing Strategy

- **Unit tests:** `coder/tests/` exists but is currently empty. There is no pytest configuration or CI pipeline yet.
- **Integration / regression tests:** The `eval/` harness serves as the primary quality gate. `aggregate.py` compares results to `results/baseline.json` and exits non-zero if any metric drops > 2 %.
- **Manual testing:**
  - Build pipeline: `coderllm-build info` + `coderllm-build download` + single-quant conversion.
  - Agent: `coder-agent health` + `coder-agent run "list files"`.
  - Eval: Run `run_humaneval.py` against a loaded LM Studio model.

---

## Deployment and Release Process

1. **Training:** Run Colab/RunPod notebooks in order (01–05). Output is a merged BF16 checkpoint on HF Hub.
2. **Build:** Point `build_pipeline/configs/quants.yaml` `base.hf_repo` to the merged checkpoint, then run `bash scripts/build_all.sh`.
3. **Upload:** Build pipeline auto-uploads all variants to `zurd46/EliCoder-30B-A3B-<variant>`.
4. **LM Studio:** Users install via `lms get zurd46/EliCoder-30B-A3B-UD-Q4_K_XL` or UI search.
5. **Agent:** End users install `coderllm-agent` locally and point it to `localhost:1234/v1`.

---

## Security Considerations

- **Autonomy gating:** The agent has three levels (`safe`, `standard`, `yolo`). Destructive tools (`delete_file`, `git_push`, `gh_merge_pr`, `gh_create_release`, etc.) require user confirmation in `standard` mode and are blocked in `safe` mode.
- **Sandbox shell:** Optional `sandbox-exec` (macOS) or `bwrap`/`firejail` (Linux) for `run_shell`. Disabled by default.
- **Secrets:** `HF_TOKEN`, `GITHUB_TOKEN`, and API keys are passed via environment variables or `.env` files. Never commit them.
- **Network:** Agent fetches URLs and searches the web. In `yolo` mode it can also create GitHub repos, open PRs, and trigger workflows.
- **Tool timeout:** Hard `ThreadPoolExecutor` timeout per tool call (default 120 s) to prevent runaway shell commands.
- **Context compaction:** Old conversation turns are summarized when context exceeds 70 % to avoid leaking stale sensitive data into the prompt.

---

## Environment Variables Cheat Sheet

| Variable | Used In | Default |
|----------|---------|---------|
| `HF_TOKEN` | build_pipeline, training | — |
| `OPENAI_BASE_URL` | coder, eval | `http://localhost:1234/v1` |
| `OPENAI_API_KEY` | coder, eval | `lm-studio` |
| `CODER_MODEL` | coder | `EliCoder-30B-A3B` |
| `CODER_SMALL_MODEL` | coder | — |
| `CODER_WORKDIR` | coder | `cwd` |
| `CODER_STREAM` | coder | `1` |
| `CODER_PARALLEL` | coder | `1` |
| `CODER_SANDBOX` | coder | `0` |
| `CODER_STEP_BUDGET` | coder | `30` |
| `CODER_TIME_BUDGET` | coder | `1800` |
| `CODER_CONTEXT_TOKENS` | coder | `32768` |
| `GITHUB_TOKEN` | coder | — |
| `CODERLLM_WORK` | build_pipeline | `build_pipeline/work` |
| `RUNPOD_API_KEY` / `RUNPOD_POD_ID` | training/runpod | — |

---

## Common Pitfalls

- **MLX on Linux/Windows:** `mlx-lm` is macOS-only. The build pipeline gracefully skips MLX conversion on non-Darwin platforms.
- **Disk space:** Build pipeline needs ~80+ GB free. Use `CODERLLM_WORK` on an external drive if needed.
- **Training OOM on H100:** Reduce `max_seq_length` to 16384 or `per_device_train_batch_size` to 1 in `configs/sft.yaml`.
- **Notebook vs. script source of truth:** Always edit `.py` files in `training/`, then run `make_notebooks.py`. Colab reads `.ipynb`.
- **LLM Studio must be running** before using `coder-agent` or `eval/` scripts.
- **Empty `coder/tests/`:** There is no automated test suite for the agent yet. Verify manually with `coder-agent health` and small tasks.
