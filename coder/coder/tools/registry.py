from __future__ import annotations
import hashlib, json, signal, threading, time
from collections import OrderedDict
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
    cacheable: bool = False
    timeout_sec: int | None = None

    def to_openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }}


AUTONOMY_ORDER = {"safe": 0, "standard": 1, "yolo": 2}


class _LRU:
    def __init__(self, cap: int):
        self.cap = cap
        self._d: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, k: str) -> str | None:
        with self._lock:
            if k in self._d:
                self._d.move_to_end(k)
                return self._d[k]
            return None

    def put(self, k: str, v: str) -> None:
        with self._lock:
            self._d[k] = v
            self._d.move_to_end(k)
            while len(self._d) > self.cap:
                self._d.popitem(last=False)

    def invalidate_all(self) -> None:
        with self._lock:
            self._d.clear()


def _run_with_timeout(fn: Callable[..., Any], args: dict, timeout: int | None) -> Any:
    """Run a callable with a soft wall-clock timeout using a worker thread."""
    if not timeout or timeout <= 0:
        return fn(**args)
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = fn(**args)
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"tool exceeded {timeout}s timeout")
    if "error" in result:
        raise result["error"]
    return result.get("value")


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    settings: Settings | None = None
    _cache: _LRU = field(default_factory=lambda: _LRU(512))
    _agent: Any = None  # backref, set by Agent

    def attach_agent(self, agent: Any) -> None:
        self._agent = agent
        if self.settings:
            self._cache = _LRU(self.settings.cache_max_entries)

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

    @staticmethod
    def _cache_key(name: str, args_json: str) -> str:
        return hashlib.sha256(f"{name}|{args_json}".encode()).hexdigest()

    def _invalidate_write_cache(self, name: str) -> None:
        # Any file-writing / shell / git / scaffold tool invalidates all cached reads.
        if name.startswith(("write_", "create_", "edit_", "append_", "delete_", "move_", "copy_",
                            "apply_", "multi_", "run_", "git_", "gh_", "scaffold_", "init_", "package_")):
            self._cache.invalidate_all()

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

        use_cache = bool(self.settings and self.settings.cache_enabled and tool.cacheable)
        cache_key = self._cache_key(name, args_json) if use_cache else ""
        if use_cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit

        timeout = tool.timeout_sec or (self.settings.tool_call_default_timeout if self.settings else None)

        try:
            start = time.time()
            result = _run_with_timeout(tool.func, args, timeout)
            dur = time.time() - start
        except TimeoutError as e:
            return json.dumps({"error": f"timeout: {e}"})
        except TypeError as e:
            return json.dumps({"error": f"bad args: {e}"})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

        out = result if isinstance(result, str) else json.dumps(result, default=str)

        if use_cache:
            self._cache.put(cache_key, out)
        else:
            self._invalidate_write_cache(name)

        return out


def build_registry(settings: Settings) -> Registry:
    r = Registry(settings=settings, _cache=_LRU(settings.cache_max_entries))
    from . import fs, git_tool, github_tool, shell, execute, code_intel, project, web, memory
    from . import planning, subagent, think, patch, lsp, devserver, semantic
    fs.register(r, settings)
    git_tool.register(r, settings)
    github_tool.register(r, settings)
    shell.register(r, settings)
    execute.register(r, settings)
    code_intel.register(r, settings)
    project.register(r, settings)
    web.register(r, settings)
    memory.register(r, settings)
    planning.register(r, settings)
    subagent.register(r, settings)
    think.register(r, settings)
    patch.register(r, settings)
    lsp.register(r, settings)
    devserver.register(r, settings)
    semantic.register(r, settings)
    return r
