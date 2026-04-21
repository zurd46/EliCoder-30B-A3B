from __future__ import annotations
import json
from .registry import Registry, Tool
from ..settings import Settings


def register(r: Registry, s: Settings) -> None:

    def think(thought: str) -> str:
        """Free-form reasoning. Side-effect-free; the thought stays in the tool trace
        so the model can chain deeper reasoning without a write tool."""
        agent = getattr(r, "_agent", None)
        if agent is not None:
            agent.add_thought(thought)
        return json.dumps({"logged": True, "chars": len(thought)})

    for t in [
        Tool("think",
             "Log a chain-of-thought step. No side effects. Use for planning, debugging hypotheses, tradeoffs.",
             {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]},
             think, "safe"),
    ]:
        r.register(t)
