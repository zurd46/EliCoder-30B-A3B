# coder — Agent Runtime

Senior-Dev-Agent vor LM Studio. Ein Python-CLI, das OpenAI-kompatibel gegen `localhost:1234/v1` spricht und eine vollständige Tool-Suite (Filesystem, Git, GitHub, Shell, Tests, Projekt-Scaffolding, Memory) bereitstellt.

## Install

```bash
cd coder
pip install -e .
```

## Voraussetzungen

1. **LM Studio** (≥ 0.3.6) läuft → Modell geladen → Server gestartet (`lms server start` oder UI)
2. Endpoint: `http://localhost:1234/v1` (Standard)
3. Optional: `GITHUB_TOKEN`, `TAVILY_API_KEY` / `BRAVE_API_KEY`

## Benutzung

```bash
# Health-Check
coder-agent health

# Einmalprompt
coder-agent run "Scaffolde eine FastAPI-App mit Tests und pushe als GitHub-Repo"

# Interaktiver REPL
coder-agent repl --cwd ~/projects/mein-projekt

# Tool-Liste
coder-agent tools --autonomy yolo

# Autonomie-Stufen: safe | standard | yolo
coder-agent run --autonomy yolo "Fix den failenden Test und committe"
```

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
export CODER_MODEL=coder-16b-dyn-UD-Q4_K_XL       # geladenes Modell
export CODER_WORKDIR=/pfad/zum/projekt            # default: cwd
export GITHUB_TOKEN=ghp_xxx
export TAVILY_API_KEY=tvly_xxx                    # optional
```

## Tool-Überblick (~60 Tools)

- **Filesystem**: `read_file`, `write_file`, `create_file`, `edit_file`, `append_file`, `delete_file`, `move_file`, `copy_file`, `create_dir`, `delete_dir`, `list_dir`, `glob`, `grep`, `file_info`
- **Shell & Execution**: `run_shell`, `run_server`, `stop_server`, `run_tests`, `run_python`, `run_node`, `run_lint`, `run_typecheck`, `package_install`
- **Git**: `git_init`, `git_clone`, `git_status`, `git_diff`, `git_log`, `git_add`, `git_commit`, `git_checkout`, `git_merge`, `git_branch`, `git_push`, `git_pull`, `git_stash`
- **GitHub**: `gh_whoami`, `gh_repo_create`, `gh_list_prs`, `gh_get_pr`, `gh_create_pr`, `gh_merge_pr`, `gh_comment_pr`, `gh_review_pr`, `gh_list_issues`, `gh_create_issue`, `gh_close_issue`, `gh_create_release`, `gh_workflow_trigger`
- **Code Intelligence**: `ast_symbols`, `find_symbol`
- **Project Scaffolding**: `scaffold_project` (next-app, vite-react, fastapi, express, django, cargo-bin, go-mod, tauri, expo, …), `init_venv`, `create_env_file`, `create_dockerfile`, `create_github_workflow`
- **Web**: `fetch_url`, `web_search`
- **Memory**: `remember`, `recall`, `forget`, `list_memory`

## Beispiel-Tasks

```bash
coder-agent run "Lies alle Python-Dateien unter src/ und erkläre die Architektur"
coder-agent run --autonomy yolo "Migriere von requests zu httpx, aktualisiere Tests, committe"
coder-agent run "Scaffolde Next.js-App 'marketing', installiere Tailwind, pushe zu gh:zurd46/marketing"
coder-agent run "Was hat der letzte Test-Run gemacht? Debugge den Fail."
```
