from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable
from rich.console import Console

from ..settings import Settings, Autonomy

console = Console()


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]
    min_autonomy: Autonomy = "standard"
    needs_confirmation: bool = False

    def to_openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }}


AUTONOMY_ORDER = {"safe": 0, "standard": 1, "yolo": 2}


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    settings: Settings | None = None

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def to_openai(self) -> list[dict]:
        if self.settings is None:
            return [t.to_openai() for t in self.tools.values()]
        auto_level = AUTONOMY_ORDER[self.settings.autonomy]
        return [
            t.to_openai() for t in self.tools.values()
            if AUTONOMY_ORDER[t.min_autonomy] <= auto_level
        ]

    def dispatch(self, name: str, args_json: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid json args: {e}"})

        if self.settings and AUTONOMY_ORDER[tool.min_autonomy] > AUTONOMY_ORDER[self.settings.autonomy]:
            return json.dumps({"error": f"tool {name} blocked by autonomy={self.settings.autonomy}"})

        if tool.needs_confirmation and self.settings and self.settings.autonomy != "yolo":
            console.print(f"[yellow]Tool[/] [bold]{name}[/] [yellow]wants to run with[/] {args}")
            ok = console.input("approve? [y/N] ").strip().lower() == "y"
            if not ok:
                return json.dumps({"error": "user rejected tool call"})

        try:
            result = tool.func(**args)
        except TypeError as e:
            return json.dumps({"error": f"bad args: {e}"})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

        return result if isinstance(result, str) else json.dumps(result, default=str)


def build_registry(settings: Settings) -> Registry:
    r = Registry(settings=settings)
    from . import fs, git_tool, github_tool, shell, execute, code_intel, project, web, memory
    fs.register(r, settings)
    git_tool.register(r, settings)
    github_tool.register(r, settings)
    shell.register(r, settings)
    execute.register(r, settings)
    code_intel.register(r, settings)
    project.register(r, settings)
    web.register(r, settings)
    memory.register(r, settings)
    return r
