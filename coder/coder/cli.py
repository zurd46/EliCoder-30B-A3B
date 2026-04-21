from __future__ import annotations
import sys, os
import typer
from pathlib import Path
from rich.console import Console

from .agent import Agent
from .settings import Settings

app = typer.Typer(add_completion=False, help="Coder — senior-dev agent running on top of LM Studio.")
console = Console()


def _settings_from_cli(
    workdir: Path,
    model: str | None,
    small_model: str | None,
    base_url: str | None,
    autonomy: str,
    max_steps: int | None,
    stream: bool | None,
    parallel: bool | None,
    sandbox: bool | None,
    router: bool | None,
) -> Settings:
    kwargs: dict = {"workdir": workdir.resolve(), "autonomy": autonomy}
    if max_steps is not None: kwargs["max_tool_steps"] = max_steps
    if stream is not None: kwargs["stream"] = stream
    if parallel is not None: kwargs["parallel_tools"] = parallel
    if sandbox is not None: kwargs["sandbox_shell"] = sandbox
    if router is not None: kwargs["model_router_enabled"] = router
    s = Settings(**kwargs)
    if model: s.model = model
    if small_model: s.small_model = small_model
    if base_url: s.base_url = base_url
    return s


@app.command()
def run(
    prompt: str = typer.Argument(..., help="What to do."),
    workdir: Path = typer.Option(Path.cwd(), "--cwd", help="Project root"),
    model: str = typer.Option(None, "--model", envvar="CODER_MODEL"),
    small_model: str = typer.Option(None, "--small-model", envvar="CODER_SMALL_MODEL"),
    base_url: str = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    autonomy: str = typer.Option("standard", "--autonomy", help="safe | standard | yolo"),
    max_steps: int = typer.Option(30, "--max-steps"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    parallel: bool = typer.Option(True, "--parallel/--no-parallel"),
    sandbox: bool = typer.Option(False, "--sandbox/--no-sandbox"),
    router: bool = typer.Option(False, "--router/--no-router", help="Route simple tasks to small_model."),
):
    s = _settings_from_cli(workdir, model, small_model, base_url, autonomy, max_steps,
                           stream, parallel, sandbox, router)
    Agent(s).run(prompt)


@app.command()
def repl(
    workdir: Path = typer.Option(Path.cwd(), "--cwd"),
    model: str = typer.Option(None, "--model", envvar="CODER_MODEL"),
    small_model: str = typer.Option(None, "--small-model", envvar="CODER_SMALL_MODEL"),
    base_url: str = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    autonomy: str = typer.Option("standard", "--autonomy"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    parallel: bool = typer.Option(True, "--parallel/--no-parallel"),
    sandbox: bool = typer.Option(False, "--sandbox/--no-sandbox"),
    router: bool = typer.Option(False, "--router/--no-router"),
):
    s = _settings_from_cli(workdir, model, small_model, base_url, autonomy, None,
                           stream, parallel, sandbox, router)
    agent = Agent(s)
    console.print(f"[bold]coder[/] @ {workdir}  ([cyan]{s.model}[/])  type /exit to quit  · /plan · /budget · /clear-cache")
    while True:
        try:
            q = console.input("[bold magenta]›[/] ")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = q.strip()
        if cmd in ("/exit", "/quit"): break
        if not cmd: continue
        if cmd == "/plan":
            for t in agent.get_plan():
                console.print(f"  [{t.get('status','?')}] {t.get('id','?')}: {t.get('content','')}")
            continue
        if cmd == "/budget":
            console.print(agent.budget_status())
            continue
        if cmd == "/clear-cache":
            agent.registry._cache.invalidate_all()
            console.print("[green]cache cleared[/]")
            continue
        agent.run(q)


@app.command()
def tools(
    autonomy: str = typer.Option("standard", "--autonomy"),
):
    s = Settings(autonomy=autonomy)
    agent = Agent(s)
    for t in agent.registry.to_openai():
        f = t["function"]
        console.print(f"[bold cyan]{f['name']}[/]  {f['description']}")
    console.print(f"\n[dim]{len(agent.registry.to_openai())} tools available at autonomy={autonomy}[/]")


@app.command()
def health(
    base_url: str = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    model: str = typer.Option(None, "--model", envvar="CODER_MODEL"),
):
    import httpx
    s = Settings()
    if base_url: s.base_url = base_url
    if model: s.model = model
    url = s.base_url.rstrip("/") + "/models"
    try:
        r = httpx.get(url, timeout=5)
        console.print(f"[green]{url}[/] → {r.status_code}")
        console.print(r.json())
    except Exception as e:
        console.print(f"[red]unreachable:[/] {e}")
        raise typer.Exit(1)


@app.command("index")
def index_cmd(
    workdir: Path = typer.Option(Path.cwd(), "--cwd"),
    max_files: int = typer.Option(2000, "--max-files"),
):
    """Build the semantic search index for the project."""
    s = Settings(workdir=workdir.resolve())
    agent = Agent(s)
    result = agent.registry.dispatch("semantic_index_build", f'{{"max_files": {max_files}}}')
    console.print(result)


if __name__ == "__main__":
    app()
