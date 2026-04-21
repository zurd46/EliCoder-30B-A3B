from __future__ import annotations
import json, subprocess, tempfile, os
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


def _safe(settings: Settings, path: str) -> Path:
    p = (settings.workdir / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        p.relative_to(settings.workdir)
    except ValueError:
        raise PermissionError(f"path outside workdir: {p}")
    return p


def register(r: Registry, s: Settings) -> None:

    def apply_patch(diff: str, strip: int = 1, check_only: bool = False) -> str:
        """Apply a unified diff via `git apply` (preferred) or `patch` fallback."""
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
            f.write(diff)
            patch_path = f.name
        try:
            cmd = ["git", "apply", f"-p{strip}"]
            if check_only:
                cmd.append("--check")
            cmd.append(patch_path)
            proc = subprocess.run(cmd, cwd=s.workdir, capture_output=True, text=True, timeout=60)
            if proc.returncode == 0:
                return json.dumps({"ok": True, "tool": "git-apply", "check_only": check_only,
                                   "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]})
            # Fallback to `patch` if git apply fails.
            alt = subprocess.run(
                ["patch", f"-p{strip}", "-i", patch_path, *(["--dry-run"] if check_only else [])],
                cwd=s.workdir, capture_output=True, text=True, timeout=60,
            )
            return json.dumps({
                "ok": alt.returncode == 0, "tool": "patch",
                "stdout": alt.stdout[-2000:], "stderr": alt.stderr[-2000:],
                "git_stderr": proc.stderr[-1000:],
            })
        except FileNotFoundError as e:
            return json.dumps({"ok": False, "error": f"tool missing: {e}"})
        except subprocess.TimeoutExpired:
            return json.dumps({"ok": False, "error": "timeout"})
        finally:
            try:
                os.unlink(patch_path)
            except OSError:
                pass

    def multi_edit(path: str, edits: list[dict]) -> str:
        """Apply multiple {old,new,replace_all?} edits atomically.
        All edits must match; on any failure, the file is left unchanged."""
        p = _safe(s, path)
        text = p.read_text()
        original = text
        applied: list[dict] = []
        for i, e in enumerate(edits):
            old = e.get("old", "")
            new = e.get("new", "")
            replace_all = bool(e.get("replace_all", False))
            if old == "":
                return json.dumps({"ok": False, "error": f"edit[{i}] has empty 'old'"})
            if old not in text:
                return json.dumps({"ok": False, "error": f"edit[{i}] old not found", "applied": applied})
            count = text.count(old)
            if count > 1 and not replace_all:
                return json.dumps({"ok": False, "error": f"edit[{i}] old appears {count}× (set replace_all)",
                                   "applied": applied})
            text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            applied.append({"i": i, "count": count if replace_all else 1})

        if text == original:
            return json.dumps({"ok": True, "path": str(p), "applied": applied, "noop": True})
        p.write_text(text)
        return json.dumps({"ok": True, "path": str(p), "applied": applied,
                           "bytes_before": len(original), "bytes_after": len(text)})

    def write_patch_from_files(old_path: str, new_path: str) -> str:
        """Compute a unified diff between two files in the workdir."""
        a = _safe(s, old_path)
        b = _safe(s, new_path)
        proc = subprocess.run(["diff", "-u", str(a), str(b)], capture_output=True, text=True)
        return json.dumps({"diff": proc.stdout[-20000:], "exit": proc.returncode})

    for t in [
        Tool("apply_patch",
             "Apply a unified diff to the workdir (uses `git apply`, falls back to `patch`).",
             {"type": "object", "properties": {
                 "diff": {"type": "string"},
                 "strip": {"type": "integer", "default": 1},
                 "check_only": {"type": "boolean", "default": False},
             }, "required": ["diff"]},
             apply_patch, "standard"),
        Tool("multi_edit",
             "Apply multiple string replacements to one file atomically. edits = [{old, new, replace_all?}].",
             {"type": "object", "properties": {
                 "path": {"type": "string"},
                 "edits": {"type": "array", "items": {"type": "object", "properties": {
                     "old": {"type": "string"},
                     "new": {"type": "string"},
                     "replace_all": {"type": "boolean"},
                 }, "required": ["old", "new"]}},
             }, "required": ["path", "edits"]},
             multi_edit, "standard"),
        Tool("diff_files", "Compute a unified diff between two files in the workdir.",
             {"type": "object", "properties": {
                 "old_path": {"type": "string"}, "new_path": {"type": "string"}
             }, "required": ["old_path", "new_path"]},
             write_patch_from_files, "safe", cacheable=True),
    ]:
        r.register(t)
