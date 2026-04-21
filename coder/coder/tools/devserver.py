from __future__ import annotations
import json, os, shlex, subprocess, threading, time, httpx
from collections import deque
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


class _BgProcess:
    def __init__(self, proc: subprocess.Popen, name: str, cmd: str, cwd: Path):
        self.proc = proc
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.log: deque[str] = deque(maxlen=2000)
        self.started_at = time.time()
        self._stop_flag = threading.Event()
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self._stop_flag.is_set():
                break
            self.log.append(line.rstrip("\n"))

    def stop(self) -> None:
        self._stop_flag.set()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


_servers: dict[str, _BgProcess] = {}
_lock = threading.Lock()


def register(r: Registry, s: Settings) -> None:

    def dev_server_start(cmd: str, name: str, cwd: str | None = None, env: dict | None = None) -> str:
        with _lock:
            if name in _servers and _servers[name].proc.poll() is None:
                return json.dumps({"ok": False, "error": f"server {name} already running"})
            working = s.workdir if cwd is None else (s.workdir / cwd).resolve()
            is_win = os.name == "nt"
            args = cmd if is_win else shlex.split(cmd)
            proc = subprocess.Popen(
                args, cwd=working,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, shell=is_win,
                env={**os.environ, **(env or {})},
            )
            _servers[name] = _BgProcess(proc, name, cmd, working)
            return json.dumps({"ok": True, "name": name, "pid": proc.pid, "cmd": cmd})

    def dev_server_logs(name: str, tail: int = 200) -> str:
        bg = _servers.get(name)
        if bg is None:
            return json.dumps({"ok": False, "error": "unknown server"})
        lines = list(bg.log)[-tail:]
        return json.dumps({
            "ok": True, "name": name, "running": bg.proc.poll() is None,
            "uptime_sec": round(time.time() - bg.started_at, 1),
            "lines": lines,
        })

    def dev_server_stop(name: str) -> str:
        with _lock:
            bg = _servers.pop(name, None)
            if bg is None:
                return json.dumps({"ok": False, "error": "unknown server"})
            bg.stop()
            return json.dumps({"ok": True, "exit": bg.proc.returncode})

    def dev_server_list() -> str:
        out = []
        for name, bg in _servers.items():
            out.append({
                "name": name, "cmd": bg.cmd,
                "running": bg.proc.poll() is None,
                "uptime_sec": round(time.time() - bg.started_at, 1),
                "pid": bg.proc.pid,
            })
        return json.dumps({"servers": out})

    def http_check(url: str, expect_status: int = 200, timeout: int = 10) -> str:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as c:
                resp = c.get(url)
            return json.dumps({
                "ok": resp.status_code == expect_status,
                "status": resp.status_code,
                "elapsed_ms": round(resp.elapsed.total_seconds() * 1000, 1),
                "content_type": resp.headers.get("content-type"),
                "body_preview": resp.text[:2000],
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def wait_for_http(url: str, expect_status: int = 200, timeout: int = 60,
                      interval: float = 1.0) -> str:
        deadline = time.time() + timeout
        last: dict = {}
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=interval + 2, follow_redirects=True) as c:
                    resp = c.get(url)
                last = {"status": resp.status_code, "ok": resp.status_code == expect_status}
                if last["ok"]:
                    return json.dumps({"ok": True, "status": resp.status_code,
                                       "waited_sec": round(time.time() - (deadline - timeout), 1)})
            except Exception as e:
                last = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            time.sleep(interval)
        return json.dumps({"ok": False, "error": "timeout", "last": last})

    def browser_screenshot(url: str, output: str, width: int = 1280, height: int = 720) -> str:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            return json.dumps({"ok": False, "error": "playwright not installed (`pip install playwright && playwright install chromium`)"})
        out_path = (s.workdir / output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.screenshot(path=str(out_path), full_page=True)
            browser.close()
        return json.dumps({"ok": True, "path": str(out_path)})

    for t in [
        Tool("dev_server_start",
             "Start a long-running dev server in the background and capture its stdout.",
             {"type": "object", "properties": {
                 "cmd": {"type": "string"}, "name": {"type": "string"},
                 "cwd": {"type": "string"}, "env": {"type": "object"},
             }, "required": ["cmd", "name"]},
             dev_server_start, "standard"),
        Tool("dev_server_logs", "Read the last N log lines of a running dev server.",
             {"type": "object", "properties": {"name": {"type": "string"}, "tail": {"type": "integer"}}, "required": ["name"]},
             dev_server_logs, "safe"),
        Tool("dev_server_stop", "Stop a running dev server.",
             {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
             dev_server_stop, "standard"),
        Tool("dev_server_list", "List background dev servers.",
             {"type": "object", "properties": {}}, dev_server_list, "safe"),
        Tool("http_check", "HTTP GET a URL and return status + preview.",
             {"type": "object", "properties": {"url": {"type": "string"}, "expect_status": {"type": "integer"}, "timeout": {"type": "integer"}}, "required": ["url"]},
             http_check, "safe"),
        Tool("wait_for_http", "Poll a URL until it returns the expected status or timeout.",
             {"type": "object", "properties": {"url": {"type": "string"}, "expect_status": {"type": "integer"}, "timeout": {"type": "integer"}, "interval": {"type": "number"}}, "required": ["url"]},
             wait_for_http, "safe"),
        Tool("browser_screenshot",
             "Navigate to URL with headless Chromium and save a full-page screenshot (needs `playwright`).",
             {"type": "object", "properties": {"url": {"type": "string"}, "output": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["url", "output"]},
             browser_screenshot, "standard"),
    ]:
        r.register(t)
