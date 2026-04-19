# CoderLLM – Architektur & Plan

**Codename:** `coder-16b-dyn`
**Mission:** Schnellstes *und* bestes Open-Source-Coder-Modell für Consumer-Hardware via LM Studio.
**Ziel-Hardware (Primary):** MacBook M5, 32 GB Unified Memory, macOS 26+
**Ziel-Hardware (Secondary):** Windows 11 PC mit ≥ 16 GB VRAM (RTX 4070 Ti / 4080 / 4090 / 5080 / 5090) oder ≥ 32 GB RAM + CUDA/Vulkan
**Training-Hardware:** Google Colab Pro, H100 80 GB
**Runtime / Server:** **LM Studio ≥ 0.3.6** (Mac + Windows + Linux · OpenAI-kompatibel auf `localhost:1234/v1`)
**Model-Formate:**
- **GGUF (Unsloth UD-*)** — cross-platform, läuft auf Mac/Windows/Linux via llama.cpp
- **MLX 4-bit** — Mac-only Speed-Variante (Metal-nativ)
**Agent-Layer:** Python-Client gegen LM Studio API (kein eigener Inference-Server)
**Fine-Tune-Stack:** **Unsloth** (2× Speed, 70 % VRAM-Ersparnis, H100) + **Unsloth Studio** optional
**Lizenz-Ziel:** Apache 2.0 (für Base + Weights)

---

## 1. Ziele (Messbar) — "bestes **und** schnellstes Open-Source-Coder-Modell für Consumer-HW"

### 1.1 Qualität (Open-Source SOTA <16B aktiv)

| Dimension | Ziel | Messung |
|---|---|---|
| **SWE-Bench Verified** | **≥ 60 % resolved** | offizieller Harness, 500 Tasks |
| **HumanEval+** | **≥ 92 % pass@1** | EvalPlus |
| **MBPP+** | ≥ 85 % pass@1 | EvalPlus |
| **LiveCodeBench** (Jan–Apr 2026 Slice) | ≥ 50 % | kontaminationsfrei |
| **BFCL v3 Tool-Use** | ≥ 88 % gesamt · ≥ 95 % valid JSON | Gorilla Harness |
| **RULER @ 128k** | ≥ 85 % | NVIDIA long-ctx bench |

### 1.2 Speed (Apple M5, 32 GB · Windows RTX 4090, 24 GB VRAM)

| Dimension | Ziel M5 | Ziel RTX 4090 |
|---|---|---|
| **Time-to-first-token** (4k Prompt) | ≤ 250 ms | ≤ 100 ms |
| **Decode-Throughput** (8k Ctx, batch=1) | **≥ 140 tok/s** mit Speculative | **≥ 220 tok/s** |
| **Decode-Throughput** (128k Ctx) | ≥ 60 tok/s | ≥ 110 tok/s |
| **Prefill-Throughput** (Code-Ingest) | ≥ 3 500 tok/s | ≥ 8 000 tok/s |

Basis: Qwen3-Coder-30B-A3B MLX-4bit auf M4 Max erreicht bereits 68 tok/s ohne Spec-Decoding. Mit Draft-Model + LM-Studio-Optimierungen sind 140+ tok/s auf M5 realistisch.

### 1.3 Ressourcen

| Dimension | Ziel |
|---|---|
| **Kontextfenster** | 262 k Token nutzbar (Needle ≥ 95 % @ 200 k) |
| **RAM/VRAM (Inferenz)** | ≤ 20 GB peak bei 32 k aktivem KV · ≤ 28 GB bei 128k |
| **Disk-Footprint** | 13.8–17.7 GB (UD-Q3/Q4) |

### 1.4 Agent-Capability

| Fähigkeit | Ziel |
|---|---|
| **Datei erstellen** (Code, JSON, CSV, YAML, MD, TOML, …) | 100 % valide Syntax im Output |
| **Datei lesen/editieren** | diff-korrekt, keine Kollateral-Edits |
| **Mehrstufige Tool-Chains** | ≥ 20 Tool-Calls ohne Halluzination |
| **Parallele Tool-Calls** | unterstützt (LM Studio 0.3.6+ nativ) |
| **Ganze Projekte scaffoldieren** | `npm create`, `cargo new`, etc. + alle Files |
| **End-to-end Workflow**: „Scaffolde React-App + Tests + GitHub-Repo + erster PR" | 1-Prompt-Ziel |

**Nicht-Ziele:** Multimodalität, Chat-Small-Talk, Creative Writing.

---

## 2. Modell-Entscheidung

### 2.1 Base-Modell-Wahl (Stand April 2026)

Kandidaten (max 16 B **aktiv**):

| Base | Aktiv | Total | Ctx nativ | Unsloth UD-2.0 | SWE-Bench Base | Lizenz |
|---|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B-Instruct** | 3 B | 30.5 B | 256 k | ✅ | ~51 % | Apache 2.0 |
| Qwen3.5-9B (Coding-tuned) | 9 B | 9 B | 128 k | ✅ | ~46 % | Apache 2.0 |
| Qwen3.6-35B-A3B | 3 B | 35 B | 256 k | ✅ (preview) | tbd | Apache 2.0 |
| Qwen3-Coder-14B-Dense | 14 B | 14 B | 128 k | ✅ | ~47 % | Apache 2.0 |
| DeepSeek-Coder-V3-16B-Lite | 2.4 B | 16 B | 128 k | ⚠️ | ~46 % | DS License |

**Entscheidung: `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` (UD-Q4_K_XL)** — siehe §2.3 für Details.

- **A3B = nur 3 B aktiv pro Forward-Pass** → MoE lädt 8 von 128 Experten je Token
- **Unsloth Dynamic 2.0** per-layer-quantisiert → spürbar höhere MMLU bei *kleinerer* Datei als Vanilla Q4 (Gemma3-Referenz: +1 % MMLU bei −2 GB)
- **Kalibrierung mit 1.5 M Chat-Tokens** → Tool-Calling-Reliabilität höher als Standard-GGUFs
- 256 k Kontext **nativ** (kein YaRN-Patch nötig)
- Auf M4 Max gemessen: **MLX 4-bit ≈ 68 tok/s · Q4_K_M GGUF ≈ 40 tok/s** → Dual-Runtime sinnvoll

**Fallbacks:**
- Zu groß? → `UD-Q3_K_XL` (13.8 GB) oder `UD-IQ2_M` (10.8 GB, Dynamic-2.0 erhält Qualität)
- Zu langsam? → MLX-Variante statt GGUF (1.7× decode)
- Passt nicht? → `Qwen3.5-9B` Dense (kleineres Footprint, stabile Alternative)

### 2.2 Warum Unsloth Dynamic 2.0?

Zitat aus Unsloth-Docs: *"selectively quantizes layers much more intelligently, dynamically adjusting the quantization type of every possible layer, with combinations differing for each layer and model"*.

Konkrete Vorteile für unseren Use-Case:
- **Per-Layer-Adaptiv:** Sensitive Layer (Router-Gate, Attention-QK) bleiben hoch-präzise, wenig sensitive FFN-Experten aggressiv quantisiert
- **Custom-Calibration** auf Chat/Code-Mix → weniger Tool-Call-JSON-Fehler als Standard-GPTQ
- **KL-Divergenz 4-8 % niedriger** vs. Standard-Q4 → näher am BF16-Verhalten
- **Beide Formate verfügbar:** GGUF für llama.cpp & LM-Studio **und** direkt-quantisierbar zu MLX

### 2.3 Konkrete Quant-Auswahl (Unsloth-GGUFs, gemessen)

| Variante | Größe | RAM Budget bei 32 GB | Empfehlung |
|---|---|---|---|
| UD-IQ1_M | 9.6 GB | ~21 GB frei für KV → 256k easy | Nur wenn max-Ctx > Qualität |
| UD-IQ2_M | 10.8 GB | ~20 GB frei | Aggressiv, noch brauchbar |
| UD-Q2_K_XL | 11.8 GB | ~19 GB frei | Starkes Dynamic-Profil |
| **UD-Q3_K_XL** | **13.8 GB** | **~17 GB frei** | **Sweet-Spot Speed/Qual** |
| IQ4_XS | 16.4 GB | ~14 GB frei | Klein, hohe Qualität |
| **UD-Q4_K_XL** | **17.7 GB** | **~13 GB frei** | **Default — max Qualität < 20 GB** |
| UD-Q5_K_XL | 21.7 GB | ~9 GB frei | Nur bei kleinerem Ctx |

**Default-Empfehlung:** `UD-Q4_K_XL` für Qualität, `UD-Q3_K_XL` wenn KV-Cache für 128k+ Kontext priorisiert wird.

> Raptor-Mini (proprietäres Cursor-Modell) ist **nicht open-weights**. Wir replizieren den *Style* (Speed + Tool-Native + Long-Ctx) mit Unsloth-Dynamic-2.0 Gewichten.

### 2.4 Finales Architektur-Profil

```yaml
base: unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
architecture:
  type: MoE (Mixture of Experts)
  total_params: 30.5B
  active_params: 3.3B
  layers: 48
  hidden_size: 2048
  num_experts: 128
  experts_per_token: 8
  attention_heads: 32
  kv_heads: 4                    # GQA 8:1
  head_dim: 128
  rope_theta: 10_000_000         # long-ctx
  context_length: 262_144        # 256k native
  vocab_size: 151_936
  activation: SiLU
  norm: RMSNorm

quantization:
  primary:   UD-Q4_K_XL           # Unsloth Dynamic 2.0, 17.7 GB, GGUF
  speed:     MLX-4bit             # group_size=64, bits=4, 16-17 GB
  lowmem:    UD-Q3_K_XL           # 13.8 GB, für 128k+ Ctx mit KV

size_on_disk:
  gguf_ud_q4: 17.7 GB
  mlx_q4:     ~16.5 GB
```

---

## 3. Gesamt-Architektur (Layer-Diagramm)

```
┌─────────────────────────────────────────────────────────────────┐
│  USER  (CLI · VS-Code-Ext · Web-UI · LM-Studio-Chat)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  CODER-AGENT (unser Python-Paket, sitzt vor LM Studio)          │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐   │
│  │ Context-Mgr  │ │ Tool-Suite   │ │ Agent-FSM              │   │
│  │   (§5)       │ │   (§6)       │ │ think→act→verify       │   │
│  └──────────────┘ └──────────────┘ └────────────────────────┘   │
│                  │ baut finalen Prompt                          │
└──────────────────┬──────────────────────────────────────────────┘
                   │ HTTP · OpenAI-kompatibel
                   │ POST http://localhost:1234/v1/chat/completions
                   │   tools=[...]  stream=True
┌──────────────────▼──────────────────────────────────────────────┐
│  LM STUDIO  (Inference-Server, Metal-nativ)                     │
│  ┌─────────────────┐  ┌──────────────────┐                      │
│  │ llama.cpp Engine│  │  MLX Engine      │   Runtime-Auswahl    │
│  │ GGUF UD-Q4_K_XL │  │  mlx-4bit 64 GS  │   via model.yaml     │
│  └─────────────────┘  └──────────────────┘                      │
│  • Tool-Use Injection in System-Prompt (Jinja Chat-Template)    │
│  • KV-Cache: 8-bit, Paged, Prefix-Cache                         │
│  • JIT-Loader, Speculative Decoding, Concurrent Batching        │
└─────────────────────────────────────────────────────────────────┘
```

**Kernentscheidung:** Wir bauen keinen eigenen Inference-Server. LM Studio **ist** der Server (inkl. KV-Cache, Prefix-Cache, Speculative Decoding, Metal-Opts, OpenAI-API, Tool-Call-Parsing). Unser Code lebt eine Ebene darüber und fokussiert sich auf **Context-Automatisierung + Agent-Loop + Tools**.

---

## 4. Inferenz-Engine (LM Studio hostet · Dual-Backend verfügbar)

### 4.1 Warum LM Studio

| Feature | Nutzen für uns |
|---|---|
| **Beide Engines nativ** | GGUF (llama.cpp) + MLX — wir packagen beide, LM Studio wählt |
| **OpenAI-kompatibel** | Agent nutzt `openai` Python-SDK mit `base_url=http://localhost:1234/v1` |
| **Tool-Use (≥0.3.6)** | `/v1/chat/completions` akzeptiert `tools=[...]` nativ |
| **`model.yaml`** | Presets für Quality/Speed/LowMem in Modell einbacken |
| **Auto Prompt-Template** | liest Jinja-Template aus GGUF-Metadata |
| **KV/Prefix-Cache eingebaut** | keine Eigenentwicklung nötig |
| **Metal-Opts** | Flash-Attn, Speculative Decoding, Batching → frei mitgeliefert |
| **GUI + Headless** | `lms server start` für Daemon-Mode |

### 4.2 Modell-Varianten & LM-Studio-Profile

Wir pushen **drei Preset-Varianten** in `~/.lmstudio/models/coder-llm/`:

```
coder-llm/
├── coder-16b-dyn-q4/               ← GGUF · Default (Quality)
│   ├── coder-16b-dyn-UD-Q4_K_XL.gguf
│   └── model.yaml
├── coder-16b-dyn-q3/               ← GGUF · Long-Context-Profile
│   ├── coder-16b-dyn-UD-Q3_K_XL.gguf
│   └── model.yaml
└── coder-16b-mlx-4bit/             ← MLX · Speed-Profile
    ├── weights.safetensors
    ├── config.json · tokenizer.json
    └── model.yaml
```

**LM Studio Preset-Auswahl im UI** → User wählt "quality / speed / lowmem" → unser Agent schickt Request an den entsprechenden `model_id`.

### 4.3 Tool-Calling-Setup (kritisch!)

Tool-Use in LM Studio funktioniert **nur wenn das Chat-Template Tool-Calls kennt**.

**Sicherstellen:**
1. GGUF-Metadata enthält Jinja-Template mit `{{ tools }}` Branch (Qwen3-Coder hat das nativ)
2. `model.yaml` setzt:
   ```yaml
   metadata:
     tool_use: true
     system_prompt_inject: "You are a senior software engineer..."
   load:
     context_length: 131072       # 128k default; 262144 für max
     gpu_offload: max
     kv_cache_type: q8_0          # 8-bit KV → 50% RAM
     flash_attention: true
   inference:
     temperature: 0.2
     top_p: 0.95
     repeat_penalty: 1.05
   ```
3. Test-Call mit `curl` gegen `/v1/chat/completions` mit `tools=[...]` → muss `tool_calls` zurückgeben, nicht Text.

### 4.4 Backend-Wahl-Logik (Agent-Side)

```
                Agent-Request
                     │
                     ▼
         ┌─ kurz & Autocomplete? ──► mlx-4bit    (Speed-Modell)
         ├─ Tool-Call + Code? ─────► UD-Q4_K_XL  (Quality-Modell)
         └─ >64k Kontext? ─────────► UD-Q3_K_XL  (LowMem-Modell)
```

Der Agent-Code wählt `model="coder-16b-dyn-q4"` etc. — LM Studio lädt JIT den passenden Snapshot.

### 4.2 Quantisierung — per-Layer-Strategie (Unsloth Dynamic 2.0)

| Komponente | GGUF UD-Q4_K_XL | MLX 4-bit | Grund |
|---|---|---|---|
| Attention Q/K-Proj | 6-bit adaptiv | 4-bit GS=64 | QK sensitiv für Long-Ctx |
| Attention V/O-Proj | 4-bit | 4-bit GS=64 | weniger sensitiv |
| MoE Router-Gate | **8-bit** | **8-bit** | Routing entscheidet Expert-Auswahl |
| MoE Expert-FFN-Up | 4-bit | 4-bit GS=64 | 90% der Params |
| MoE Expert-FFN-Down | 4-bit | 4-bit GS=64 | ditto |
| Shared-Expert | 5-6 bit | 4-bit | aktiv bei jedem Token |
| Embedding / LM-Head (tied) | 6-bit | 4-bit | Logit-Präzision |
| RMSNorm / Bias | FP16 | FP16 | klein, kritisch |
| KV-Cache | 8-bit / 4-bit cold | 8-bit / 4-bit cold | §4.4 |

**Verfahren:**
- **GGUF:** Unsloth liefert UD-Q4_K_XL fertig — Kalibrierung mit >1.5 M Token Chat/Code-Mix durch Unsloth-Team
- **MLX:** `mlx_lm.convert --q-bits 4 --q-group-size 64` mit 512 Kalibrier-Samples aus `the-stack-v2`. Falls Zeit: **AWQ** statt naive Quant → bis zu 2 Punkte HumanEval+ mehr

### 4.3 Speed-Optimierungen (Metal-nativ)

| Technik | Gewinn | Kosten |
|---|---|---|
| **Flash-Attention-2 (Metal)** | 1.8× Prefill | 0 Qualität |
| **Speculative Decoding** mit Qwen3-Coder-0.5B Draft | 2.3× Decode | 0 Qualität (bei accept_rate>60%) |
| **KV-Cache Quantisierung (8-bit)** | 50 % RAM | <0.3 % Quality |
| **Paged Attention** (vLLM-Style) | Fragmentierung weg | keine |
| **Prefix Caching** | Re-use System-Prompt + Codebase | keine |
| **MoE Expert-Prefetch** | 15 % Decode | keine |
| **CUDA-Graph-Äquivalent** (MLX `mx.compile`) | 10-20 % | 0 |
| **Continuous Batching** | multi-request | keine |
| **Draft-Model Parallelism** auf Efficiency-Cores | +8 % | keine |

**Ziel-Budget:** First-Token ≤ 300 ms @ 4 k Prompt · Decode ≥ 110 tok/s.

### 4.4 KV-Cache — das Herzstück

```
┌─────────────────────────────────────────────────────────────┐
│  PAGED KV-CACHE (16 MB Pages, Ring-Allocator)               │
│                                                             │
│  [System]─[Project-Code-Map]─[Chat-History]─[Current]       │
│     │          │                  │            │            │
│     └─ fixed   └─ project-scoped  └─ sliding   └─ active    │
│                                                             │
│  Precision: FP16 (hot 8k) · INT8 (8k–128k) · INT4 (>128k)   │
└─────────────────────────────────────────────────────────────┘
```

- **Tri-Precision Cache:** hot=FP16, warm=INT8, cold=INT4 (auto-demote per Token-Alter) → 256 k ctx in ~6 GB
- **Prefix-Hash-Index:** SHA-256 über Token-Prefixes → Treffer spart 100 % Prefill
- **Snapshot/Restore:** Cache in `~/.coderllm/cache/<project-hash>.mlx` persistieren → beim nächsten Start des Projekts sofort warm
- **Evict-Policy:** LRU auf *Chunk-Level* (nicht Token), Chunks = semantische Code-Blöcke (Section 5.2)

---

## 5. Context-Manager (Projekt-Automatisierung)

Kernidee: Das Modell bekommt **nicht das ganze Repo** in den Prompt, sondern einen **live-aktualisierten, geranked Kontext** + KV-Cache der relevanten Teile.

### 5.1 Drei-Stufen-Kontext

```
Stufe 1: ALWAYS-HOT          (immer im KV-Cache)
         • README.md  • package.json/pyproject.toml  • Dateitree (gzip)
         • CLAUDE.md / .cursorrules / AGENTS.md
         • ~2-4k Token, FP16

Stufe 2: PROJECT-WARM        (on-demand gepaged)
         • Alle Dateien embedded (bge-code-v1, 1024-dim)
         • FAISS HNSW-Index auf Disk + mmap
         • Top-k per Query, quantisiert INT8

Stufe 3: ACTIVE-SESSION      (diese Konversation)
         • Aktuelle Datei(en), Fehlermeldungen, letzte Edits
         • FP16, sliding window 32k
```

### 5.2 Automatische Kontext-Pipeline

```
Repo-Änderung  (fswatch / inotify)
   │
   ▼
AST-Parser (tree-sitter)  ──► Chunk-Level Split
   │                           • Funktion/Klasse/Block
   ▼                           • max 512 Tokens je Chunk
Code-Embedder (bge-code-v1)    • Overlap via Import-Graph
   │
   ▼
FAISS-HNSW Index  +  BM25 (Hybrid)
   │
   ▼
Reranker (Jina-Reranker-v2-Code, 0.3B, MLX)
   │
   ▼
Kontext-Builder (Token-Budget-aware)
   │
   ├─ Immer: Imports, Typen, Interfaces der relevanten Chunks
   ├─ Top-N Chunks bis 80 % des freien Ctx
   └─ Auto-Expand bei Call-Graph-Nachbarn
```

### 5.3 Retrieval-Strategien (Agent wählt automatisch)

| Query-Typ | Strategie |
|---|---|
| "Fix bug in X" | Symbol-Resolve → File + Callers + Tests |
| "Add feature Y" | Similarity Search + Dir-Siblings + Konvention-Beispiele |
| "Refactor Z" | Full Call-Graph Closure |
| "Why is this slow?" | Datei + Hotpath-Trace + Benchmarks-Ordner |
| "Review PR" | Diff + modifizierte Dateien + direkte Abhängige |

### 5.4 KV-Cache-Sparmechanismen

| Mechanismus | RAM-Ersparnis |
|---|---|
| Tri-Precision Cache (§4.3) | 50-70 % |
| Chunk-Deduplication (same file in mehreren Turns) | 20 % |
| Speculative Eviction (Embedding-Similarität zum aktuellen Query) | 15 % |
| On-Disk Cache-Spill (.mlx-Format, mmap) | unbegrenzt |
| Shared Prefix über Sessions | 30-90 % bei Re-Open |

---

## 6. Tool-Suite (vollständig · cross-platform)

**Leitprinzip:** Der Agent muss **alles** können, was ein Senior-Dev in VS-Code + Terminal + Browser macht — erstellen, bearbeiten, löschen, testen, ausrollen, konfigurieren. Jedes Tool ist cross-platform (Mac/Linux/Windows).

### 6.1 Filesystem — komplette CRUD

| Tool | Funktion |
|---|---|
| `read_file(path, range?)` | Datei lesen, optional Zeilen-Range |
| `write_file(path, content)` | neue Datei erstellen oder überschreiben |
| `create_file(path, content)` | atomic create-only (fehler wenn existiert) |
| `edit_file(path, old, new, replace_all)` | exakte String-Ersetzung |
| `patch_file(path, diff)` | Unified-Diff anwenden (für mehrere Änderungen) |
| `append_file(path, content)` | Append-Mode |
| `delete_file(path)` | Datei löschen (mit Recycle-Bin auf Win/Mac) |
| `move_file(src, dst)` / `copy_file` / `rename_file` | |
| `create_dir(path)` | mkdir -p |
| `delete_dir(path, recursive)` | rmdir / rm -rf (mit Safety-Check) |
| `list_dir(path, depth?)` | tree-artige Auflistung |
| `glob(pattern)` | `**/*.ts` etc. |
| `grep(pattern, path?, regex?)` | ripgrep-backed |
| `file_info(path)` | size, mtime, mode, is_symlink |
| `chmod(path, mode)` | nur Unix (no-op Windows) |
| `watch_path(path)` | Live-Watcher, streamt Änderungen zurück |

### 6.2 Projekt-Scaffolding — "ganze Projekte aufsetzen"

| Tool | Funktion |
|---|---|
| `scaffold_project(template, target_dir, options)` | Templates: `next-app`, `vite-react`, `sveltekit`, `nuxt`, `fastapi`, `django`, `flask`, `express`, `nestjs`, `go-mod`, `cargo-bin`, `cargo-lib`, `rails`, `astro`, `expo`, `tauri`, `electron` |
| `init_git_repo(path, initial_branch)` | `git init` + `.gitignore` passend zum Template |
| `init_venv(path, python_version)` | `python -m venv` cross-platform |
| `init_node(path, package_manager)` | npm/pnpm/yarn/bun auto-detect, `package.json` |
| `init_pyproject(path, deps)` | `pyproject.toml` (uv/poetry/pip) |
| `init_cargo(path, type)` | `Cargo.toml` |
| `create_env_file(path, vars)` | `.env` + `.env.example` + gitignore-Eintrag |
| `create_dockerfile(path, stack)` | passend zum Detect-Stack |
| `create_ci_config(path, provider)` | `.github/workflows/ci.yml`, `.gitlab-ci.yml`, … |
| `create_readme(path, project_info)` | Initial-README-Generation |

### 6.3 Package- & Dependency-Management

| Tool | Funktion |
|---|---|
| `package_install(manager, pkgs, dev?)` | npm/pnpm/yarn/bun/pip/uv/poetry/cargo/go/gem — auto-detect aus Projekt |
| `package_remove(manager, pkgs)` | |
| `package_update(manager, pkgs?)` | |
| `package_list(manager)` | |
| `package_audit(manager)` | Security-Audit (npm audit, pip-audit, cargo audit) |
| `lock_file_sync(manager)` | `npm ci`, `pip sync`, `cargo lock` |

### 6.4 Code-Intelligence (LSP-basiert)

| Tool | Funktion |
|---|---|
| `ast_query(path, query)` | tree-sitter Queries (py, ts, go, rust, java, …) |
| `find_definition(symbol, path?)` | LSP |
| `find_references(symbol)` | LSP |
| `call_hierarchy(symbol)` | incoming/outgoing |
| `rename_symbol(symbol, new_name)` | projektweit, refactor-safe |
| `type_of(symbol)` / `hover(symbol)` | LSP |
| `get_diagnostics(path?)` | Errors/Warnings aller offenen Files |
| `organize_imports(path)` | |
| `format_file(path)` | prettier/black/ruff/gofmt/rustfmt auto |

### 6.5 Execution — "wirklich ausführen"

| Tool | Funktion |
|---|---|
| `run_shell(cmd, cwd?, env?, timeout?)` | **Plattform-bewusst:** bash auf Mac/Linux, PowerShell + cmd-Fallback auf Windows |
| `run_python(code, kernel_id?)` | persistente Jupyter-Kernel pro Session |
| `run_node(code)` | Ephemeral Node-Eval |
| `run_tests(path?, framework?)` | auto-detect pytest/jest/vitest/go test/cargo test/phpunit/rspec/mocha |
| `run_benchmark(path)` | auto: pytest-benchmark/criterion/go bench |
| `run_coverage(path)` | coverage.py/c8/tarpaulin |
| `run_lint(path, tool?)` | ruff/eslint/clippy/golangci-lint |
| `run_typecheck(path, tool?)` | mypy/pyright/tsc/flow |
| `debug_start(path, breakpoints)` | DAP-basiert |
| `debug_step` / `debug_vars` / `debug_continue` | |
| `run_server(cmd, port)` | startet Dev-Server im Hintergrund |
| `stop_server(id)` | |
| `curl(url, method, body, headers)` | HTTP-Request für Smoke-Test |

### 6.6 Database-Operationen

| Tool | Funktion |
|---|---|
| `db_query(connection, sql)` | SQLite/Postgres/MySQL |
| `db_migrate(dir)` | Alembic/Prisma/Knex/Goose auto-detect |
| `db_seed(file)` | |
| `db_dump` / `db_restore` | |

### 6.7 Git — lokale VCS-Kontrolle

| Tool | Funktion |
|---|---|
| `git_init` / `git_clone(url)` | |
| `git_status` · `git_diff(ref?)` · `git_log` · `git_blame(path)` | |
| `git_add(paths)` · `git_commit(message)` | |
| `git_branch_create` / `git_checkout` / `git_merge` / `git_rebase` | |
| `git_stash` / `git_pop` | |
| `git_tag(name, ref?)` | |
| `git_remote_add(name, url)` · `git_push(remote, branch)` · `git_pull` | |
| `git_cherry_pick(hash)` · `git_revert(hash)` | |

### 6.8 GitHub — Full-Access (PyGithub)

| Tool | Funktion |
|---|---|
| `gh_repo_create(name, private?, description?)` | aus dem Nichts neues Repo |
| `gh_repo_delete` · `gh_repo_fork` · `gh_repo_archive` | |
| `gh_create_pr(head, base, title, body)` | |
| `gh_list_prs` · `gh_get_pr` · `gh_merge_pr` · `gh_close_pr` | |
| `gh_review_pr(id, body, event)` | |
| `gh_create_issue` · `gh_list_issues` · `gh_close_issue` · `gh_comment(id, body)` | |
| `gh_files_changed(pr_id)` · `gh_diff_pr` | |
| `gh_create_release(tag, title, body, assets)` | |
| `gh_workflow_trigger(name)` · `gh_workflow_status(run_id)` · `gh_workflow_logs` | |
| `gh_secrets_set(name, value)` | Repo-Secrets für CI |
| `gh_gist_create(files, public?)` | |

### 6.9 Docker / Container

| Tool | Funktion |
|---|---|
| `docker_build(dockerfile, tag)` · `docker_run(image, env, ports)` | |
| `docker_compose(action)` | up/down/ps/logs |
| `docker_exec(container, cmd)` | |

### 6.10 Web-Zugriff

| Tool | Funktion |
|---|---|
| `fetch_url(url, method?, headers?)` | HTML/JSON/Plaintext abrufen |
| `web_search(query)` | Tavily / Brave API |
| `scrape(url, selector?)` | BeautifulSoup / Playwright-light |

### 6.11 Systemeinrichtung (Setup-Agent)

| Tool | Funktion |
|---|---|
| `which(binary)` | prüft ob Tool installiert |
| `install_tool(name)` | brew/choco/winget/apt je Plattform |
| `set_env_persistent(key, val, scope)` | user oder machine (Windows-aware) |
| `create_symlink(target, link)` | |
| `open_port(port)` / `check_port(port)` | |

### 6.12 Memory & Meta

| Tool | Funktion |
|---|---|
| `remember(key, value)` · `recall(key)` · `forget(key)` | persistente Projekt-Notizen (SQLite) |
| `plan(steps)` · `sub_task(goal)` · `reflect(on)` | Agent-Meta-Kontrolle |
| `ask_user(question)` | nur bei echten Blockern (sonst autonom) |

### 6.13 Sicherheits-Gates

Agent hat **drei Autonomie-Stufen** (via `--autonomy={safe,standard,yolo}`):

| Aktion | safe | standard | yolo |
|---|---|---|---|
| read_file, grep, ast_query | ✅ | ✅ | ✅ |
| write/edit/create Dateien im cwd | ❓ ask | ✅ | ✅ |
| delete_file / delete_dir | ❓ ask | ❓ ask | ✅ |
| run_shell / run_tests | ❓ ask | ✅ | ✅ |
| package_install | ❓ ask | ✅ | ✅ |
| git push / gh_create_pr | ❓ ask | ❓ ask | ✅ |
| gh_repo_create / gh_repo_delete | ❌ | ❓ ask | ✅ |
| rm außerhalb cwd · sudo | ❌ | ❌ | ❓ ask |

### 6.14 Tool-Call-Format

JSON-Function-Calling (Qwen3-Coder nativ, LM Studio leitet `tools=[...]` korrekt durch). Parser ist **grammar-constrained** (llguidance) → 0 JSON-Syntaxfehler.

```json
{"name": "edit_file", "arguments": {
  "path": "src/api.py",
  "old": "def get_user(",
  "new": "def get_user_by_id(",
  "replace_all": true
}}
```

Parallele Tool-Calls werden unterstützt (LM Studio 0.3.6+) — Agent kann z.B. gleichzeitig `read_file × 5` ausführen.

### 6.15 Agent-Loop (Senior-Dev-Verhalten)

```
1. UNDERSTAND   → Ziel zerlegen, nur bei echter Ambiguität fragen
2. INVESTIGATE  → parallele Tool-Calls (read_file × N, grep, ast_query)
3. PLAN         → explizite Schritt-Liste (in scratchpad)
4. EXECUTE      → ein Schritt, kleinster Diff
5. VERIFY       → run_tests, get_diagnostics, self-review
6. ITERATE      → bei rot: root-cause, keine Quick-Fixes
7. REPORT       → diff-summary + next-steps
```

Implementiert als **Finite-State-Machine** mit Budget-Guards (max 30 Tool-Calls / Task default, via ENV override).

### 6.2 Tool-Call-Format

JSON-Function-Calling (Qwen3-Coder nativ). Parser ist **grammar-constrained** (llguidance) → 0 Syntax-Fehler möglich.

```json
{"name": "edit_file", "arguments": {
  "path": "src/api.py",
  "old": "def get_user(",
  "new": "def get_user_by_id(",
  "replace_all": true
}}
```

### 6.3 Agent-Loop (Senior-Dev-Verhalten)

```
1. UNDERSTAND   → decompose goal, ask only if critical ambiguity
2. INVESTIGATE  → parallel tool-calls (read_file × N, grep, ast_query)
3. PLAN         → explicit step-list (written to scratchpad)
4. EXECUTE      → one step at a time, smallest diff
5. VERIFY       → run_tests, get_diagnostics, self-review
6. ITERATE      → wenn VERIFY rot: root-cause, keine Quick-Fixes
7. REPORT       → diff-summary + next-steps
```

Implementiert als **Finite-State-Machine** mit Budget-Guards (max 20 Tool-Calls / Task default, via ENV override).

---

## 7. Training / Fine-Tuning (Colab Pro H100)

### 7.1 Trainings-Ziel

Base-Modell ist bereits stark. Wir **Post-Trainen** auf drei Achsen:

1. **Tool-Use-Verstärkung** → 100 % valide JSON, deutscher & englischer Prompt-Style
2. **Agent-Loop-Verhalten** → Plan → Act → Verify Muster
3. **Lange-Kontext-Retention** → 256 k Needle-in-Haystack

### 7.2 Daten-Mix

| Dataset | Samples | Gewicht |
|---|---|---|
| `swe-bench-extra` (trainable subset) | 12 k | 25 % |
| `open-o1-sft-coding` | 30 k | 20 % |
| `nvidia/OpenCodeReasoning-v2` | 40 k | 15 % |
| Synthetic Tool-Traces (eigenes GPT-5-codex Pipeline) | 50 k | 20 % |
| `glaive-function-calling-v3` | 20 k | 10 % |
| Long-Ctx Needle (synth, 64k–256k) | 5 k | 10 % |

Gesamt ~157 k Samples, avg 6 k Tokens = **~940 M Tokens**.

### 7.3 Method: Unsloth QLoRA (H100-budget-freundlich)

**Warum Unsloth:** Laut Unsloth-Docs **2× Trainings-Speed und 70 % weniger VRAM** vs. Vanilla-HF. Das macht 30B-MoE auf Single-H100 überhaupt erst praktikabel. Unsloth patched Rotary-Embeddings, Flash-Attention, und Cross-Entropy in Triton-Kernels.

**Alternative (No-Code):** **Unsloth Studio** — lokale Web-UI für Training + Export zu GGUF in einem Flow. Für Iteration auf Datasets sinnvoll, für Production-Training verwenden wir Notebooks (Reproduzierbarkeit).

| Parameter | Wert | Notiz |
|---|---|---|
| Framework | **Unsloth** (Triton-Kernels) | 2× Speed, 70 % VRAM-Ersparnis |
| Base-Präzision | NF4 (bitsandbytes 4-bit) | Unsloth-nativ |
| LoRA-Rank | 64 | sweet-spot für MoE |
| LoRA-Alpha | 128 | 2× Rank |
| Target-Module | `q_proj,k_proj,v_proj,o_proj, gate_proj,up_proj,down_proj, router` | Experten + Router **beide** fine-tunen |
| Expert-LoRA-Strategy | `all_experts` | sonst lernen nur aktive Experts |
| Sequence-Length (Training) | 32 768 (dynamic, grad-checkpointed) | Unsloth Long-Context-Patch |
| Batch (effective) | 64 | via Grad-Accum |
| Grad-Accum | 32 | micro-batch 2 |
| LR | 1e-4 cosine, 3 % warmup | standard |
| Epochen | 3 | |
| Optimizer | Paged AdamW 8-bit | Unsloth-Default |
| Precision (compute) | BF16 | H100 nativ |
| Duration (H100 80GB) | **~14 h (mit Unsloth)** statt 28 h vanilla | halbiert durch Unsloth |
| Kosten-Schätzung | **~30 CHF** Colab Pro Compute Units | |

**Export-Flow:** `unsloth.save_pretrained_merged()` → FP16 safetensors → Unsloth-native `save_to_gguf()` erzeugt direkt Dynamic-2.0-quantisierte GGUFs (UD-Q4_K_XL, UD-Q3_K_XL etc.) **ohne Extra-Kalibrierung** — benutzt Unslothis 1.5 M-Token Calibration-Set automatisch.

**Pipeline:** TRL SFT → DPO auf SWE-Bench-Präferenz-Paaren → Long-Ctx-Stretch (alles in Unsloth-Framework).

### 7.4 Training-Pipeline (3 Phasen)

```
Phase A — SFT (20 h)
  • Standard Instruct + Tool-Calls
  • Ziel: Format-Compliance, Deutschkompetenz

Phase B — DPO (6 h)
  • Preference-Paare: (passed_SWE_task, failed_SWE_task)
  • Ziel: Senior-Dev-Verhalten (root-cause > quick-fix)

Phase C — Long-Context-Stretch (2 h)
  • RULER + Needle Training, 128k → 256k
  • Nur bestimmte Layer finegetuned (Layer 30-47)
```

### 7.5 Deployment-Pipeline (Colab → Mac)

Unsloth liefert **beide Formate in einem Flow**:

```
                Colab H100  +  Unsloth
                     │
                     ▼
         unsloth.save_pretrained_merged()
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
  save_to_gguf(            mlx_lm.convert(
    "UD-Q4_K_XL",            "--q-bits 4"
    "UD-Q3_K_XL",            "--q-group-size 64"
    "UD-IQ2_M"             )
  )                             │
           │                    │
           ▼                    ▼
    coder-16b-dyn-Q4.gguf    coder-16b-mlx-4bit/
    coder-16b-dyn-Q3.gguf    (~16 GB)
    coder-16b-dyn-IQ2.gguf
    (10.8 / 13.8 / 17.7 GB)
           │                    │
           └──────────┬─────────┘
                      ▼
                rsync to Mac
                      │
                      ▼
  ~/.coderllm/models/
    ├── gguf/coder-16b-dyn-Q4.gguf
    ├── gguf/coder-16b-dyn-Q3.gguf
    └── mlx/coder-16b-dyn-4bit/
```

**Runtime-Launcher** entscheidet anhand `--backend={gguf,mlx}` und `--profile={quality,speed,lowmem}` welche Datei geladen wird. Default `gguf + quality = UD-Q4_K_XL`.

---

## 8. Evaluation-Pipeline

### 8.1 Continuous-Eval (nach jedem Training)

| Benchmark | Tool | Zeit |
|---|---|---|
| HumanEval+ | EvalPlus | 5 min |
| MBPP+ | EvalPlus | 8 min |
| BFCL v3 (Tool-Use) | Gorilla-Repo | 20 min |
| LiveCodeBench (recent) | offiziell | 45 min |
| SWE-Bench Verified | SWE-agent harness | 6–10 h |
| RULER (32k/128k/256k) | NVIDIA | 90 min |

Alle Evals laufen **auf Mac M5 am quantisierten Model** (nicht nur Colab-Float16).

### 8.2 Regression-Guard

Git-Hook: kein Release wenn irgendein Benchmark > 2 % unter letzter Baseline.

---

## 9. Repository-Struktur

```
CoderLLM/
├── ARCHITECTURE.md              # dieses Dokument
├── README.md
├── pyproject.toml
├── justfile                     # task-runner
│
├── coder/                       # Runtime (Mac)
│   ├── __init__.py
│   ├── server.py                # FastAPI OpenAI-kompatibel
│   ├── inference/
│   │   ├── engine.py            # Backend-Router (gguf|mlx)
│   │   ├── backend_gguf.py      # llama-cpp-python + Metal
│   │   ├── backend_mlx.py       # mlx_lm
│   │   ├── sampler.py           # Temp, TopP, MinP, DRY
│   │   ├── speculative.py       # Draft-Model Coordinator
│   │   ├── kv_cache.py          # Paged + Tri-Precision
│   │   └── compile.py           # mx.compile Wrappers
│   ├── context/
│   │   ├── watcher.py           # fswatch
│   │   ├── indexer.py           # tree-sitter + embed
│   │   ├── faiss_store.py
│   │   ├── reranker.py
│   │   └── builder.py           # Token-budget Packer
│   ├── tools/
│   │   ├── fs.py · git.py · github.py · shell.py · lsp.py · web.py
│   │   └── registry.py
│   ├── agent/
│   │   ├── loop.py              # FSM
│   │   ├── plan.py
│   │   └── critic.py
│   └── util/
│       ├── grammar.py           # llguidance schemas
│       └── logging.py
│
├── training/                    # Colab (Unsloth)
│   ├── 01_data_build.ipynb
│   ├── 02_sft_unsloth.ipynb     # Unsloth SFT (2× speed)
│   ├── 03_dpo_unsloth.ipynb     # Unsloth DPO
│   ├── 04_longctx.ipynb
│   ├── 05_export_gguf.ipynb     # save_to_gguf UD-*
│   ├── 06_export_mlx.ipynb      # mlx_lm.convert
│   ├── data/
│   │   ├── build_swe.py
│   │   ├── synthesize_tool_traces.py
│   │   └── schemas.py
│   └── configs/
│       ├── sft.yaml · dpo.yaml · longctx.yaml
│
├── eval/
│   ├── run_humaneval.py
│   ├── run_mbpp.py
│   ├── run_bfcl.py
│   ├── run_livecodebench.py
│   ├── run_swebench.py
│   └── run_ruler.py
│
├── bench/                       # Speed-Benchmarks Mac
│   ├── latency.py  throughput.py  memory.py
│
└── scripts/
    ├── install.sh
    ├── download_base.sh
    └── convert_to_mlx.sh
```

---

## 10. Roadmap

| Phase | Dauer | Output |
|---|---|---|
| **0. Setup** | 1 Tag | Repo-Skeleton, MLX installiert, Base gedownloadet & getestet |
| **1. MVP-Runtime** | 3 Tage | MLX-Inferenz + KV-Cache + FastAPI-Server + 5 Tools |
| **2. Context-Engine** | 4 Tage | tree-sitter + FAISS + Reranker + Builder |
| **3. Agent-Loop** | 3 Tage | FSM + alle Tools + Grammar-constrained JSON |
| **4. Baseline-Eval** | 2 Tage | alle Benchmarks auf Base (ohne FT) laufen |
| **5. Daten & SFT** | 5 Tage | Colab Phase A |
| **6. DPO** | 2 Tage | Colab Phase B |
| **7. Long-Ctx** | 1 Tag | Colab Phase C |
| **8. MLX-Convert + Re-Eval** | 1 Tag | Ziel-Metriken in §1 |
| **9. GitHub-Integration + Polish** | 3 Tage | CLI, VS-Code-Extension stub |
| **Gesamt** | **~25 Tage** | v1.0 |

---

## 11. Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Mitigation |
|---|---|---|
| 30B MoE sprengt 32 GB bei 256k Ctx | mittel | Fallback 14B-Dense, Tri-Precision-Cache |
| Speculative Decoding bringt < 50 % Accept | niedrig | Draft-Model auf Coder-Data re-tunen |
| SWE-Bench Ziel 55 % nicht erreicht | mittel | Scaffold verbessern (SWE-Agent Tools) statt Modell |
| Colab-Timeout bei 28h Training | hoch | Checkpointing alle 30 min, auto-resume |
| License-Konflikt bei Daten | niedrig | Nur Apache/MIT/CC-BY Quellen |

---

## 12. Entscheidungs-Checkpoints (bevor Code geschrieben wird)

Bitte bestätigen:

1. ✅ / ❌  **Base: Qwen3-Coder-30B-A3B** (3B aktiv, MoE) via Unsloth-Dynamic-2.0 — oder lieber dichtes 14B/9B?
2. ✅ / ❌  **Dual-Runtime GGUF + MLX** — oder nur eins (GGUF=Qualität, MLX=Speed)?
3. ✅ / ❌  **Default-Quant UD-Q4_K_XL** (17.7 GB, 13 GB frei für KV) — oder UD-Q3_K_XL für mehr Ctx-Budget?
4. ✅ / ❌  **Training mit Unsloth-Framework** (halbiert H100-Zeit auf ~14 h, ~30 CHF)?
5. ✅ / ❌  **Kontextfenster 256 k** (vs. 128 k = schneller)?
6. ✅ / ❌  **SWE-Bench-Ziel 55 %** realistisch — oder höher/niedriger?
7. ✅ / ❌  **Repo-Struktur in §9** ok?

Nach Go implementiere ich Phase 0 + 1 (Runtime-MVP mit beiden Backends).
