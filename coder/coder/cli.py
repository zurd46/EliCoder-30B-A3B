from __future__ import annotations
import sys, os
import typer
from pathlib import Path
from rich.console import Console

from .agent import Agent
from .settings import Settings

app = typer.Typer(add_completion=False, help="Coder — senior-dev agent running on top of LM Studio.")
console = Console()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="What to do."),
    workdir: Path = typer.Option(Path.cwd(), "--cwd", help="Project root"),
    model: str = typer.Option(None, "--model", envvar="CODER_MODEL"),
    base_url: str = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    autonomy: str = typer.Option("standard", "--autonomy", help="safe | standard | yolo"),
    max_steps: int = typer.Option(30, "--max-steps"),
):
    s = Settings(workdir=workdir.resolve(), autonomy=autonomy, max_tool_steps=max_steps)
    if model:
        s.model = model
    if base_url:
        s.base_url = base_url
    Agent(s).run(prompt)


@app.command()
def repl(
    workdir: Path = typer.Option(Path.cwd(), "--cwd"),
    model: str = typer.Option(None, "--model", envvar="CODER_MODEL"),
    base_url: str = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    autonomy: str = typer.Option("standard", "--autonomy"),
):
    s = Settings(workdir=workdir.resolve(), autonomy=autonomy)
    if model: s.model = model
    if base_url: s.base_url = base_url
    agent = Agent(s)
    console.print(f"[bold]coder[/] @ {workdir}  ([cyan]{s.model}[/])  type /exit to quit")
    while True:
        try:
            q = console.input("[bold magenta]›[/] ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip() in ("/exit", "/quit"): break
        if not q.strip(): continue
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


if __name__ == "__main__":
    app()
