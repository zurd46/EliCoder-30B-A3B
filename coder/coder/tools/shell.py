from __future__ import annotations
import json, os, subprocess, sys, shlex, shutil, platform
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


def _sandbox_prefix() -> list[str] | None:
    """Return a command prefix that sandboxes the child, or None if unsupported."""
    if platform.system() == "Darwin":
        # sandbox-exec is deprecated but still present on macOS.
        if shutil.which("sandbox-exec"):
            profile = (
                "(version 1)(allow default)"
                "(deny network*)"
                "(allow network-bind (local ip))"
                "(deny file-write* (subpath \"/\"))"
            )
            return ["sandbox-exec", "-p", profile]
    elif platform.system() == "Linux":
        if shutil.which("bwrap"):
            return ["bwrap", "--unshare-all", "--share-net", "--dev-bind", "/", "/"]
        if shutil.which("firejail"):
            return ["firejail", "--quiet", "--net=none", "--private-tmp"]
    return None


def _run(cmd: str, cwd: Path, env: dict | None, timeout: int, sandbox: bool) -> dict:
    is_win = os.name == "nt"
    shell_mode = True if is_win else False
    args = cmd if is_win else shlex.split(cmd)
    if sandbox and not is_win:
        prefix = _sandbox_prefix()
        if prefix is None:
            return {"exit": -1, "error": "sandbox requested but no sandbox tool available"}
        args = [*prefix, *(args if isinstance(args, list) else shlex.split(cmd))]
    try:
        proc = subprocess.run(
            args, cwd=cwd, env={**os.environ, **(env or {})},
            capture_output=True, text=True, timeout=timeout, shell=shell_mode,
        )
        return {
            "exit": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
            "sandbox": sandbox,
        }
    except subprocess.TimeoutExpired:
        return {"exit": -1, "error": f"timeout after {timeout}s"}


def register(r: Registry, s: Settings) -> None:

    def run_shell(cmd: str, cwd: str | None = None, timeout: int = 120,
                  env: dict | None = None, sandbox: bool | None = None) -> str:
        working = s.workdir if cwd is None else (s.workdir / cwd).resolve()
        use_sandbox = s.sandbox_shell if sandbox is None else bool(sandbox)
        return json.dumps(_run(cmd, working, env, timeout, use_sandbox))

    # Legacy simple server management kept for compatibility; prefer dev_server_* for log access.
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
        Tool("run_shell",
             "Execute a shell command (bash on Unix, cmd on Windows). Set sandbox=true on Linux/macOS for a network/FS-restricted run.",
             {"type": "object", "properties": {
                 "cmd": {"type": "string"}, "cwd": {"type": "string"},
                 "timeout": {"type": "integer"}, "env": {"type": "object"},
                 "sandbox": {"type": "boolean"},
             }, "required": ["cmd"]},
             run_shell, "standard"),
        Tool("run_server", "Start a long-running background process (dev server). Use dev_server_start if you want logs.",
             {"type": "object", "properties": {"cmd": {"type": "string"}, "name": {"type": "string"}, "cwd": {"type": "string"}}, "required": ["cmd", "name"]},
             run_server, "standard"),
        Tool("stop_server", "Stop a background process started with run_server.",
             {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
             stop_server, "standard"),
    ]:
        r.register(t)
