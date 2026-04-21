# coder — Agent Runtime

Senior-Dev-Agent vor LM Studio. Ein Python-CLI, das OpenAI-kompatibel gegen `localhost:1234/v1` spricht und eine vollständige Tool-Suite (Filesystem, Git, GitHub, Shell, Tests, Projekt-Scaffolding, LSP, Dev-Server, Semantic-Search, Memory) bereitstellt.

## Install

```bash
cd coder
pip install -e .
# optional: semantic search + screenshots
pip install -e ".[embeddings]"
pip install playwright && playwright install chromium
```

## Voraussetzungen

1. **LM Studio** (≥ 0.3.6) läuft → Modell geladen → Server gestartet (`lms server start` oder UI)
2. Endpoint: `http://localhost:1234/v1` (Standard)
3. Optional: `GITHUB_TOKEN`, `TAVILY_API_KEY` / `BRAVE_API_KEY`, `CODER_SMALL_MODEL` (für Model-Router)

## Benutzung

```bash
# Health-Check
coder-agent health

# Einmalprompt
coder-agent run "Scaffolde eine FastAPI-App mit Tests und pushe als GitHub-Repo"

# Interaktiver REPL (mit /plan, /budget, /clear-cache)
coder-agent repl --cwd ~/projects/mein-projekt

# Tool-Liste
coder-agent tools --autonomy yolo

# Autonomie-Stufen: safe | standard | yolo
coder-agent run --autonomy yolo "Fix den failenden Test und committe"

# Semantic-Search-Index bauen (einmalig)
coder-agent index --cwd ~/projects/mein-projekt

# Alle Runtime-Flags
coder-agent run --stream --parallel --sandbox --router \
  --small-model qwen2.5-coder-3b \
  --autonomy standard --max-steps 30 \
  "Migriere von requests zu httpx, aktualisiere Tests"
```

## Runtime-Features

| Feature | Flag / Env | Default | Beschreibung |
|---|---|---|---|
| Token-Streaming (Live-Panel) | `--stream` / `CODER_STREAM` | on | Rendert Assistant-Output live, schrumpft gefühlte Latenz |
| Parallele Tool-Calls | `--parallel` / `CODER_PARALLEL` | on | Fan-out unabhängiger Tool-Calls über `ThreadPoolExecutor` |
| Context-Compaction | `compaction_enabled` | on | Summarisiert alte Turns automatisch bei > 70 % Context-Füllung |
| Read-Tool-Cache | `cache_enabled` | on | LRU-Cache für `read_file`, `grep`, `glob`, `list_dir`, `file_info`, `find_references`, … |
| Retry mit Reflection | `reflect_on_error` | on | Bei Tool-Errors injiziert der Agent eine Reflection-Nachricht |
| Dynamic Temperature | `dynamic_temperature` | on | Planning 0.4 / Execution 0.1 / Reflection 0.25 |
| Model-Router | `--router` / `CODER_SMALL_MODEL` | off | Routet Compaction & Sub-Agents zu kleinerem Modell |
| Sandbox-Shell | `--sandbox` / `CODER_SANDBOX` | off | `sandbox-exec` (macOS) / `bwrap`/`firejail` (Linux) |
| Wall-Clock-Budget | `CODER_TIME_BUDGET` | 1800 s | Beendet den Loop vor dem Step-Limit |
| Per-Tool Timeout | `tool_call_default_timeout` | 120 s | Harter Thread-Timeout pro Tool-Call |

## Autonomie-Stufen

| Stufe | Lesen | Schreiben | Löschen | git push / gh merge |
|---|---|---|---|---|
| `safe`     | ✅ | Bestätigung | Bestätigung | ❌ |
| `standard` | ✅ | ✅ | Bestätigung | Bestätigung |
| `yolo`     | ✅ | ✅ | ✅ | ✅ |

Destruktive Tools (`delete_file`, `delete_dir`, `git_push`, `gh_create_pr`,
`gh_merge_pr`, `gh_repo_create`, `gh_create_release`, `gh_workflow_trigger`)
fragen im `standard`-Modus pro Call nach Bestätigung.

## Environment Variablen

```bash
export OPENAI_BASE_URL=http://localhost:1234/v1   # LM Studio
export CODER_MODEL=EliCoder-30B-A3B                # geladenes Haupt-Modell
export CODER_SMALL_MODEL=qwen2.5-coder-3b          # optional: Router-Ziel für einfache Tasks
export CODER_WORKDIR=/pfad/zum/projekt             # default: cwd
export CODER_STREAM=1                              # 0 = streaming off
export CODER_PARALLEL=1                            # 0 = sequentielle Tool-Calls
export CODER_SANDBOX=0                             # 1 = run_shell sandboxed
export CODER_STEP_BUDGET=30
export CODER_TIME_BUDGET=1800
export CODER_CONTEXT_TOKENS=32768                  # Compaction-Referenzgröße
export CODER_PARALLEL_WORKERS=6
export GITHUB_TOKEN=ghp_xxx
export TAVILY_API_KEY=tvly_xxx                     # optional
```

## Tool-Überblick (~80 Tools)

- **Filesystem**: `read_file`, `write_file`, `create_file`, `edit_file`, `append_file`, `delete_file`, `move_file`, `copy_file`, `create_dir`, `delete_dir`, `list_dir`, `glob`, `grep`, `file_info`
- **Patching / Multi-Edit**: `apply_patch` (unified diff), `multi_edit` (atomic multi-site edits), `diff_files`
- **Shell & Execution**: `run_shell` (`sandbox=true` supported), `run_tests`, `run_python`, `run_node`, `run_lint`, `run_typecheck`, `package_install`
- **Dev-Server**: `dev_server_start`, `dev_server_logs`, `dev_server_stop`, `dev_server_list`, `http_check`, `wait_for_http`, `browser_screenshot` (Playwright)
- **Git**: `git_init`, `git_clone`, `git_status`, `git_diff`, `git_log`, `git_add`, `git_commit`, `git_checkout`, `git_merge`, `git_branch`, `git_push`, `git_pull`, `git_stash`
- **GitHub**: `gh_whoami`, `gh_repo_create`, `gh_list_prs`, `gh_get_pr`, `gh_create_pr`, `gh_merge_pr`, `gh_comment_pr`, `gh_review_pr`, `gh_list_issues`, `gh_create_issue`, `gh_close_issue`, `gh_create_release`, `gh_workflow_trigger`
- **Code Intelligence**: `ast_symbols`, `find_symbol`, `goto_definition`, `find_references`, `get_diagnostics` (tsc/pyright/ruff/clippy)
- **Semantic Search**: `semantic_index_build`, `semantic_search`, `semantic_index_status`
- **Project Scaffolding**: `scaffold_project` (next-app, vite-react, fastapi, express, django, cargo-bin, go-mod, tauri, expo, …), `init_venv`, `create_env_file`, `create_dockerfile`, `create_github_workflow`
- **Web**: `fetch_url`, `web_search`
- **Memory**: `remember`, `recall`, `forget`, `list_memory`
- **Planning & Reasoning**: `todo_write`, `todo_read`, `todo_update`, `think`, `budget_status`, `spawn_subagent`

## Architektur

```
agent.py        ──>  orchestriert: streaming + parallel fan-out + compaction + retry
client.py       ──>  LM-Studio chat (stream_assembled, dynamic temperature, model routing)
settings.py     ──>  alle Tunables in einem Ort
tools/
  registry.py   ──>  dispatch mit LRU-cache, per-tool timeout, autonomy gate
  planning.py   ──>  todo_write/read/update, budget_status
  subagent.py   ──>  spawn_subagent (focused tool subset + small model)
  think.py      ──>  scratchpad
  patch.py      ──>  apply_patch / multi_edit / diff_files
  lsp.py        ──>  get_diagnostics / goto_definition / find_references
  devserver.py  ──>  long-running server mit log-capture + http_check
  semantic.py   ──>  MiniLM-Index + Cosine-Search
  shell.py      ──>  run_shell + sandbox (sandbox-exec / bwrap / firejail)
  fs.py, git_tool.py, github_tool.py, execute.py, code_intel.py, project.py, web.py, memory.py
```

## Beispiel-Tasks

```bash
coder-agent run "Lies alle Python-Dateien unter src/ und erkläre die Architektur"
coder-agent run --autonomy yolo "Migriere von requests zu httpx, aktualisiere Tests, committe"
coder-agent run "Scaffolde Next.js-App 'marketing', installiere Tailwind, pushe zu gh:zurd46/marketing"
coder-agent run "Starte den dev-Server, warte auf http://localhost:3000, screenshotte /pricing"
coder-agent run "Suche semantisch nach 'rate limiting middleware' und öffne den Hot-Path"
```

## REPL-Slash-Commands

| Command | Effekt |
|---|---|
| `/plan` | Druckt aktuellen Task-Plan |
| `/budget` | Tool-Calls, Laufzeit, Step-Limit |
| `/clear-cache` | LRU-Cache leeren |
| `/exit` oder `/quit` | Beenden |
