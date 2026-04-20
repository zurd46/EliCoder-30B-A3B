from __future__ import annotations
import json, subprocess, os, shlex, sys, shutil
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


def _detect_test_runner(cwd: Path) -> list[str] | None:
    if (cwd / "package.json").exists():
        pj = json.loads((cwd / "package.json").read_text())
        scripts = pj.get("scripts", {})
        for k in ("test", "test:unit"):
            if k in scripts:
                mgr = "pnpm" if (cwd / "pnpm-lock.yaml").exists() else \
                      "yarn" if (cwd / "yarn.lock").exists() else \
                      "bun" if (cwd / "bun.lockb").exists() else "npm"
                return [mgr, "test"] if mgr != "npm" else ["npm", "test"]
    if (cwd / "pyproject.toml").exists() or (cwd / "pytest.ini").exists() or (cwd / "tests").exists():
        return [sys.executable, "-m", "pytest", "-x", "--tb=short"]
    if (cwd / "Cargo.toml").exists():
        return ["cargo", "test"]
    if (cwd / "go.mod").exists():
        return ["go", "test", "./..."]
    if (cwd / "Gemfile").exists():
        return ["bundle", "exec", "rspec"]
    return None


def register(r: Registry, s: Settings) -> None:

    def run_tests(path: str = ".", framework: str | None = None) -> str:
        cwd = (s.workdir / path).resolve()
        if framework:
            runners = {
                "pytest": [sys.executable, "-m", "pytest", "-x", "--tb=short"],
                "jest": ["npx", "jest", "--silent"],
                "vitest": ["npx", "vitest", "run"],
                "go": ["go", "test", "./..."],
                "cargo": ["cargo", "test"],
            }
            cmd = runners.get(framework)
        else:
            cmd = _detect_test_runner(cwd)
        if cmd is None:
            return json.dumps({"ok": False, "error": "no test runner detected"})
        try:
            out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
            return json.dumps({"ok": out.returncode == 0, "exit": out.returncode,
                               "stdout": out.stdout[-8000:], "stderr": out.stderr[-4000:],
                               "cmd": " ".join(cmd)})
        except subprocess.TimeoutExpired:
            return json.dumps({"ok": False, "error": "timeout"})

    def run_python(code: str, timeout: int = 60) -> str:
        try:
            out = subprocess.run(
                [sys.executable, "-c", code], cwd=s.workdir,
                capture_output=True, text=True, timeout=timeout,
            )
            return json.dumps({"ok": out.returncode == 0, "stdout": out.stdout[-8000:], "stderr": out.stderr[-4000:]})
        except subprocess.TimeoutExpired:
            return json.dumps({"ok": False, "error": "timeout"})

    def run_node(code: str, timeout: int = 60) -> str:
        if not shutil.which("node"):
            return json.dumps({"ok": False, "error": "node not installed"})
        try:
            out = subprocess.run(
                ["node", "-e", code], cwd=s.workdir,
                capture_output=True, text=True, timeout=timeout,
            )
            return json.dumps({"ok": out.returncode == 0, "stdout": out.stdout[-8000:], "stderr": out.stderr[-4000:]})
        except subprocess.TimeoutExpired:
            return json.dumps({"ok": False, "error": "timeout"})

    def run_lint(path: str = ".", tool: str | None = None) -> str:
        cwd = (s.workdir / path).resolve()
        if tool is None:
            if (cwd / "pyproject.toml").exists() and shutil.which("ruff"):
                tool = "ruff"
            elif (cwd / "package.json").exists() and shutil.which("npx"):
                tool = "eslint"
            elif (cwd / "Cargo.toml").exists():
                tool = "clippy"
        cmds = {
            "ruff": ["ruff", "check", "."],
            "eslint": ["npx", "eslint", "."],
            "clippy": ["cargo", "clippy", "--all-targets"],
        }
        cmd = cmds.get(tool)
        if cmd is None:
            return json.dumps({"ok": False, "error": "no linter detected/configured"})
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return json.dumps({"ok": out.returncode == 0, "exit": out.returncode,
                           "stdout": out.stdout[-8000:], "stderr": out.stderr[-4000:]})

    def run_typecheck(path: str = ".", tool: str | None = None) -> str:
        cwd = (s.workdir / path).resolve()
        if tool is None:
            if (cwd / "tsconfig.json").exists():
                tool = "tsc"
            elif (cwd / "pyproject.toml").exists():
                tool = "mypy"
        cmds = {
            "tsc": ["npx", "tsc", "--noEmit"],
            "mypy": [sys.executable, "-m", "mypy", "."],
            "pyright": ["pyright"],
        }
        cmd = cmds.get(tool)
        if cmd is None:
            return json.dumps({"ok": False, "error": "no typechecker configured"})
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return json.dumps({"ok": out.returncode == 0, "exit": out.returncode,
                           "stdout": out.stdout[-8000:], "stderr": out.stderr[-4000:]})

    def package_install(manager: str, packages: list[str], dev: bool = False, cwd: str = ".") -> str:
        working = (s.workdir / cwd).resolve()
        cmd = _install_cmd(manager, packages, dev)
        if cmd is None:
            return json.dumps({"ok": False, "error": f"unknown manager: {manager}"})
        out = subprocess.run(cmd, cwd=working, capture_output=True, text=True, timeout=600)
        return json.dumps({"ok": out.returncode == 0, "stdout": out.stdout[-4000:], "stderr": out.stderr[-2000:]})

    for t in [
        Tool("run_tests", "Auto-detect test runner and execute.",
             {"type": "object", "properties": {"path": {"type": "string"}, "framework": {"type": "string"}}, "required": []},
             run_tests, "standard"),
        Tool("run_python", "Execute a Python snippet.",
             {"type": "object", "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["code"]},
             run_python, "standard"),
        Tool("run_node", "Execute a Node.js snippet.",
             {"type": "object", "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["code"]},
             run_node, "standard"),
        Tool("run_lint", "Run the project's linter.",
             {"type": "object", "properties": {"path": {"type": "string"}, "tool": {"type": "string"}}, "required": []},
             run_lint, "standard"),
        Tool("run_typecheck", "Run the project's type checker.",
             {"type": "object", "properties": {"path": {"type": "string"}, "tool": {"type": "string"}}, "required": []},
             run_typecheck, "standard"),
        Tool("package_install", "Install packages via npm/pnpm/yarn/bun/pip/uv/poetry/cargo.",
             {"type": "object", "properties": {"manager": {"type": "string"}, "packages": {"type": "array", "items": {"type": "string"}}, "dev": {"type": "boolean"}, "cwd": {"type": "string"}}, "required": ["manager", "packages"]},
             package_install, "standard"),
    ]:
        r.register(t)


def _install_cmd(manager: str, pkgs: list[str], dev: bool) -> list[str] | None:
    m = manager.lower()
    if m == "npm":    return ["npm", "install", "--save-dev" if dev else "--save", *pkgs]
    if m == "pnpm":   return ["pnpm", "add", "-D" if dev else "", *pkgs]
    if m == "yarn":   return ["yarn", "add", "-D" if dev else "", *pkgs]
    if m == "bun":    return ["bun", "add", "-d" if dev else "", *pkgs]
    if m == "pip":    return [sys.executable, "-m", "pip", "install", *pkgs]
    if m == "uv":     return ["uv", "pip", "install", *pkgs]
    if m == "poetry": return ["poetry", "add", *(["--group", "dev"] if dev else []), *pkgs]
    if m == "cargo":  return ["cargo", "add", *pkgs]
    if m == "go":     return ["go", "get", *pkgs]
    return None
