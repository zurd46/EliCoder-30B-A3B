from __future__ import annotations
import json
from .registry import Registry, Tool
from ..settings import Settings


SUB_SYSTEM = """You are a focused sub-agent spawned by Coder for a narrow task.
Execute the task, use only the allowed tools, then return a tight summary of findings/results.
Do not ask the user questions. Keep the final answer under 600 tokens."""


def register(r: Registry, s: Settings) -> None:

    def spawn_subagent(task: str, focused_tools: list[str] | None = None, max_steps: int = 12,
                       prefer_small: bool = True) -> str:
        # Lazy import to avoid circular import.
        from ..agent import Agent
        from ..settings import Settings as S

        sub_settings = S(**{**s.model_dump(), "max_tool_steps": max_steps})
        sub = Agent(sub_settings)

        # Restrict tool surface if requested.
        if focused_tools:
            allowed = set(focused_tools)
            sub.registry.tools = {k: v for k, v in sub.registry.tools.items() if k in allowed}

        # Override model preference.
        original_chat = sub.client.chat
        def chat_override(messages, tools=None, stream=False, phase=None, prefer_small=prefer_small):
            return original_chat(messages, tools=tools, stream=stream, phase=phase, prefer_small=prefer_small)
        sub.client.chat = chat_override  # type: ignore[assignment]

        sub.s.stream = False  # keep sub-agent output quiet
        result = sub.run(f"{SUB_SYSTEM}\n\nTASK: {task}")
        return json.dumps({
            "ok": True,
            "summary": result,
            "tool_calls": sub._tool_calls_made,
        })

    for t in [
        Tool("spawn_subagent",
             "Spawn a focused sub-agent for read-heavy or exploratory tasks. Only the summary returns to the main context.",
             {"type": "object", "properties": {
                "task": {"type": "string"},
                "focused_tools": {"type": "array", "items": {"type": "string"},
                                  "description": "Restrict the sub-agent to these tool names."},
                "max_steps": {"type": "integer", "default": 12},
                "prefer_small": {"type": "boolean", "default": True},
             }, "required": ["task"]},
             spawn_subagent, "standard"),
    ]:
        r.register(t)
