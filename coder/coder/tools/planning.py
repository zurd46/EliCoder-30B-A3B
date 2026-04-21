from __future__ import annotations
import json
from .registry import Registry, Tool
from ..settings import Settings


def register(r: Registry, s: Settings) -> None:

    def todo_write(tasks: list[dict]) -> str:
        cleaned: list[dict] = []
        for i, t in enumerate(tasks or []):
            cleaned.append({
                "id": str(t.get("id") or i + 1),
                "content": str(t.get("content", "")).strip(),
                "status": t.get("status", "pending") if t.get("status") in ("pending", "in_progress", "completed") else "pending",
            })
        agent = getattr(r, "_agent", None)
        if agent is not None:
            agent.set_plan(cleaned)
        return json.dumps({"ok": True, "plan": cleaned})

    def todo_read() -> str:
        agent = getattr(r, "_agent", None)
        plan = agent.get_plan() if agent is not None else []
        return json.dumps({"plan": plan})

    def todo_update(id: str, status: str) -> str:
        if status not in ("pending", "in_progress", "completed"):
            return json.dumps({"ok": False, "error": "invalid status"})
        agent = getattr(r, "_agent", None)
        if agent is None:
            return json.dumps({"ok": False, "error": "no agent bound"})
        plan = agent.get_plan()
        for t in plan:
            if str(t.get("id")) == str(id):
                t["status"] = status
                agent.set_plan(plan)
                return json.dumps({"ok": True, "plan": plan})
        return json.dumps({"ok": False, "error": f"id {id} not found"})

    def budget_status() -> str:
        agent = getattr(r, "_agent", None)
        if agent is None:
            return json.dumps({"tool_calls": 0, "elapsed_sec": 0})
        return json.dumps(agent.budget_status())

    for t in [
        Tool("todo_write", "Set or replace the task plan. tasks = [{id, content, status}].",
             {"type": "object", "properties": {
                "tasks": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                }}}
             }, "required": ["tasks"]},
             todo_write, "safe"),
        Tool("todo_read", "Read the current plan.",
             {"type": "object", "properties": {}}, todo_read, "safe"),
        Tool("todo_update", "Update the status of a single task.",
             {"type": "object", "properties": {"id": {"type": "string"}, "status": {"type": "string"}}, "required": ["id", "status"]},
             todo_update, "safe"),
        Tool("budget_status", "Report step count and elapsed wall-clock time.",
             {"type": "object", "properties": {}}, budget_status, "safe"),
    ]:
        r.register(t)
