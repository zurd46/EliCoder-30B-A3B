from __future__ import annotations
import json
from typing import Iterable
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .client import LMClient
from .settings import Settings
from .tools import Registry, build_registry
from .context import ProjectContext

SYSTEM_PROMPT = """You are **Coder**, a senior software engineer working directly on the user's machine.

You have tools to read, write, edit, delete files; scaffold projects; run tests, linters, shell commands; manage git and GitHub. Use them. Do not just describe what you would do — do it, verify it, and report the diff.

Principles:
- Plan briefly, then execute. Small, verified steps beat big speculative ones.
- Always check before you change (read_file, grep, ast_symbols, run_tests).
- After each edit, verify (run_tests, run_typecheck, run_lint). Fix what you broke.
- When the user asks for a whole project, scaffold it end-to-end: init repo, deps, sample code, tests, CI, README.
- Never invent APIs. If unsure, read the source.
- Only ask the user for clarification when a true ambiguity blocks you. Otherwise decide and proceed.
- Respond in the user's language.

Tool usage is encouraged. Prefer parallel tool calls when the information is independent.
"""


class Agent:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()
        self.client = LMClient(self.s)
        self.registry: Registry = build_registry(self.s)
        self.ctx = ProjectContext(self.s)
        self.console = Console()

    def _bootstrap_messages(self, user_msg: str) -> list[dict]:
        project_summary = self.ctx.summarize()
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Current working directory: {self.s.workdir}\nAutonomy level: {self.s.autonomy}\n\n{project_summary}"},
            {"role": "user", "content": user_msg},
        ]

    def run(self, user_msg: str) -> str:
        messages = self._bootstrap_messages(user_msg)
        tools = self.registry.to_openai()

        final_text = ""
        for step in range(self.s.max_tool_steps):
            resp = self.client.chat(messages, tools=tools)
            msg = resp.choices[0].message

            if getattr(msg, "tool_calls", None):
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    self.console.print(Panel.fit(
                        f"[bold cyan]{tc.function.name}[/]\n{tc.function.arguments[:500]}",
                        title=f"tool call [{step+1}/{self.s.max_tool_steps}]",
                    ))
                    result = self.registry.dispatch(tc.function.name, tc.function.arguments)
                    self.console.print(Panel.fit(
                        result[:1200] + ("…" if len(result) > 1200 else ""),
                        title="result",
                        border_style="green",
                    ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            final_text = msg.content or ""
            break
        else:
            final_text = "[tool step limit reached]"

        self.console.print(Panel(Markdown(final_text), title="coder", border_style="magenta"))
        return final_text
