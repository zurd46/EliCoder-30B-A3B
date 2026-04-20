from __future__ import annotations
import json, os, subprocess, sys, shlex
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


def _run(cmd: str, cwd: Path, env: dict | None, timeout: int) -> dict:
    is_win = os.name == "nt"
    shell = True if is_win else False
    args = cmd if is_win else shlex.split(cmd)
    try:
        proc = subprocess.run(
            args, cwd=cwd, env={**os.environ, **(env or {})},
            capture_output=True, text=True, timeout=timeout, shell=shell,
        )
        return {
            "exit": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"exit": -1, "error": f"timeout after {timeout}s"}


def register(r: Registry, s: Settings) -> None:

    def run_shell(cmd: str, cwd: str | None = None, timeout: int = 120, env: dict | None = None) -> str:
        working = s.workdir if cwd is None else (s.workdir / cwd).resolve()
        return json.dumps(_run(cmd, working, env, timeout))

    _servers: dict[str, subprocess.Popen] = {}

    def run_server(cmd: str, name: str, cwd: str | None = None) -> str:
        working = s.workdir if cwd is None else (s.workdir / cwd).resolve()
        is_win = os.name == "nt"
        args = cmd if is_win else shlex.split(cmd)
        proc = subprocess.Popen(
            args, cwd=working,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            shell=is_win,
        )
        _servers[name] = proc
        return json.dumps({"ok": True, "pid": proc.pid, "name": name})

    def stop_server(name: str) -> str:
        p = _servers.pop(name, None)
        if p is None:
            return json.dumps({"ok": False, "error": "unknown server"})
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        return json.dumps({"ok": True})

    for t in [
        Tool("run_shell", "Execute a shell command (bash on Unix, cmd on Windows).",
             {"type": "object", "properties": {"cmd": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}, "env": {"type": "object"}}, "required": ["cmd"]},
             run_shell, "standard"),
        Tool("run_server", "Start a long-running background process (dev server).",
             {"type": "object", "properties": {"cmd": {"type": "string"}, "name": {"type": "string"}, "cwd": {"type": "string"}}, "required": ["cmd", "name"]},
             run_server, "standard"),
        Tool("stop_server", "Stop a background process started with run_server.",
             {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
             stop_server, "standard"),
    ]:
        r.register(t)
