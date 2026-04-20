from __future__ import annotations
import json, sqlite3
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


def register(r: Registry, s: Settings) -> None:
    db_path = s.workdir / ".coder" / "memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("CREATE TABLE IF NOT EXISTS memory (k TEXT PRIMARY KEY, v TEXT, ts REAL)")

    def remember(key: str, value: str) -> str:
        conn.execute(
            "INSERT OR REPLACE INTO memory (k, v, ts) VALUES (?, ?, strftime('%s','now'))",
            (key, value),
        )
        return json.dumps({"ok": True})

    def recall(key: str) -> str:
        cur = conn.execute("SELECT v, ts FROM memory WHERE k = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return json.dumps({"ok": False, "error": "not found"})
        return json.dumps({"ok": True, "value": row[0], "ts": row[1]})

    def forget(key: str) -> str:
        conn.execute("DELETE FROM memory WHERE k = ?", (key,))
        return json.dumps({"ok": True})

    def list_memory() -> str:
        rows = conn.execute("SELECT k, v FROM memory ORDER BY ts DESC LIMIT 200").fetchall()
        return json.dumps([{"key": k, "value": v[:200]} for k, v in rows])

    for t in [
        Tool("remember", "Persist a fact for later recall (per-project).",
             {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]},
             remember, "safe"),
        Tool("recall", "Retrieve a previously remembered value by key.",
             {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
             recall, "safe"),
        Tool("forget", "Delete a remembered key.",
             {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
             forget, "safe"),
        Tool("list_memory", "List remembered keys.",
             {"type": "object", "properties": {}}, list_memory, "safe"),
    ]:
        r.register(t)
