from __future__ import annotations
import json, subprocess, shutil, sys
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


def register(r: Registry, s: Settings) -> None:

    def get_diagnostics(path: str = ".", tool: str | None = None) -> str:
        """Pull diagnostics via ruff/pyright for Python, tsc for TS, clippy for Rust."""
        cwd = (s.workdir / path).resolve()
        if tool is None:
            if (cwd / "tsconfig.json").exists() and shutil.which("npx"):
                tool = "tsc"
            elif (cwd / "pyproject.toml").exists() and shutil.which("pyright"):
                tool = "pyright"
            elif (cwd / "pyproject.toml").exists() and shutil.which("ruff"):
                tool = "ruff"
            elif (cwd / "Cargo.toml").exists():
                tool = "clippy"
        if tool is None:
            return json.dumps({"ok": False, "error": "no diagnostics tool detected"})
        cmds = {
            "tsc": ["npx", "tsc", "--noEmit", "--pretty", "false"],
            "pyright": ["pyright", "--outputjson"],
            "ruff": ["ruff", "check", "--output-format", "json", "."],
            "clippy": ["cargo", "clippy", "--message-format", "json", "--all-targets"],
        }
        cmd = cmds.get(tool)
        if cmd is None:
            return json.dumps({"ok": False, "error": f"unknown tool: {tool}"})
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)

        parsed: object = None
        if tool in ("pyright", "ruff"):
            try:
                parsed = json.loads(out.stdout)
            except Exception:
                parsed = None

        return json.dumps({
            "ok": out.returncode == 0, "tool": tool, "exit": out.returncode,
            "parsed": parsed,
            "stdout": None if parsed is not None else out.stdout[-8000:],
            "stderr": out.stderr[-2000:],
        })

    def goto_definition(symbol: str, path: str = ".") -> str:
        """Heuristic goto-definition via ripgrep: find declaration-like matches."""
        rg = shutil.which("rg")
        if rg is None:
            return json.dumps({"ok": False, "error": "ripgrep (rg) not installed"})
        patterns = [
            rf"\b(def|class|function|const|let|var|interface|type|struct|enum|impl|fn|fun|record)\s+{symbol}\b",
            rf"^{symbol}\s*=",
            rf"\b{symbol}\s*:\s*(function|\()",
        ]
        hits = []
        for pat in patterns:
            proc = subprocess.run(
                ["rg", "-n", "--json", pat, str((s.workdir / path).resolve())],
                capture_output=True, text=True, timeout=30,
            )
            for line in proc.stdout.splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") == "match":
                    d = obj["data"]
                    hits.append({
                        "file": d["path"]["text"],
                        "line": d["line_number"],
                        "text": d["lines"]["text"].strip()[:200],
                    })
        return json.dumps({"symbol": symbol, "hits": hits[:60]})

    def find_references(symbol: str, path: str = ".") -> str:
        rg = shutil.which("rg")
        if rg is None:
            return json.dumps({"ok": False, "error": "ripgrep (rg) not installed"})
        proc = subprocess.run(
            ["rg", "-n", "-w", symbol, str((s.workdir / path).resolve())],
            capture_output=True, text=True, timeout=30,
        )
        refs = []
        for line in proc.stdout.splitlines()[:500]:
            parts = line.split(":", 2)
            if len(parts) == 3:
                refs.append({"file": parts[0], "line": int(parts[1]) if parts[1].isdigit() else 0,
                             "text": parts[2].strip()[:200]})
        return json.dumps({"symbol": symbol, "references": refs})

    for t in [
        Tool("get_diagnostics",
             "Run type/lint diagnostics (tsc/pyright/ruff/clippy) and return structured results.",
             {"type": "object", "properties": {"path": {"type": "string"}, "tool": {"type": "string"}}, "required": []},
             get_diagnostics, "safe"),
        Tool("goto_definition",
             "Heuristic goto-definition for a symbol (ripgrep-backed).",
             {"type": "object", "properties": {"symbol": {"type": "string"}, "path": {"type": "string"}}, "required": ["symbol"]},
             goto_definition, "safe", cacheable=True),
        Tool("find_references",
             "Find all word-boundary references to a symbol in the project.",
             {"type": "object", "properties": {"symbol": {"type": "string"}, "path": {"type": "string"}}, "required": ["symbol"]},
             find_references, "safe", cacheable=True),
    ]:
        r.register(t)
