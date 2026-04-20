from __future__ import annotations
import os, shutil, fnmatch, re, json
from pathlib import Path
from pathspec import PathSpec
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

    def read_file(path: str, start: int = 0, end: int | None = None) -> str:
        p = _safe(s, path)
        lines = p.read_text(errors="replace").splitlines()
        chunk = lines[start:end] if end else lines[start:]
        return json.dumps({"path": str(p), "lines": len(lines), "content": "\n".join(chunk)})

    def write_file(path: str, content: str) -> str:
        p = _safe(s, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return json.dumps({"ok": True, "path": str(p), "bytes": len(content)})

    def create_file(path: str, content: str = "") -> str:
        p = _safe(s, path)
        if p.exists():
            return json.dumps({"ok": False, "error": "exists"})
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return json.dumps({"ok": True, "path": str(p)})

    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        p = _safe(s, path)
        text = p.read_text()
        if old not in text:
            return json.dumps({"ok": False, "error": "old not found"})
        count = text.count(old)
        if count > 1 and not replace_all:
            return json.dumps({"ok": False, "error": f"old appears {count} times; set replace_all=true or add context"})
        new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        p.write_text(new_text)
        return json.dumps({"ok": True, "path": str(p), "replacements": count if replace_all else 1})

    def append_file(path: str, content: str) -> str:
        p = _safe(s, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content)
        return json.dumps({"ok": True})

    def delete_file(path: str) -> str:
        p = _safe(s, path)
        if p.is_dir():
            return json.dumps({"ok": False, "error": "use delete_dir for directories"})
        p.unlink()
        return json.dumps({"ok": True})

    def move_file(src: str, dst: str) -> str:
        a, b = _safe(s, src), _safe(s, dst)
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(a, b)
        return json.dumps({"ok": True})

    def copy_file(src: str, dst: str) -> str:
        a, b = _safe(s, src), _safe(s, dst)
        b.parent.mkdir(parents=True, exist_ok=True)
        if a.is_dir():
            shutil.copytree(a, b)
        else:
            shutil.copy2(a, b)
        return json.dumps({"ok": True})

    def create_dir(path: str) -> str:
        p = _safe(s, path)
        p.mkdir(parents=True, exist_ok=True)
        return json.dumps({"ok": True})

    def delete_dir(path: str, recursive: bool = False) -> str:
        p = _safe(s, path)
        if recursive:
            shutil.rmtree(p)
        else:
            p.rmdir()
        return json.dumps({"ok": True})

    def list_dir(path: str = ".", depth: int = 1) -> str:
        p = _safe(s, path)
        spec = PathSpec.from_lines("gitwildmatch", s.ignore_patterns)
        entries = []
        for root, dirs, files in os.walk(p):
            rel = os.path.relpath(root, p)
            d = 0 if rel == "." else rel.count(os.sep) + 1
            if d >= depth:
                dirs[:] = []
            dirs[:] = [x for x in dirs if not spec.match_file(x)]
            for f in files:
                if spec.match_file(f):
                    continue
                entries.append(os.path.join(rel, f) if rel != "." else f)
        return json.dumps({"path": str(p), "entries": entries})

    def glob_search(pattern: str) -> str:
        matches = [str(p.relative_to(s.workdir)) for p in s.workdir.glob(pattern)]
        return json.dumps({"pattern": pattern, "matches": matches[:1000]})

    def grep(pattern: str, path: str = ".", is_regex: bool = True, max_results: int = 200) -> str:
        rx = re.compile(pattern) if is_regex else re.compile(re.escape(pattern))
        spec = PathSpec.from_lines("gitwildmatch", s.ignore_patterns)
        p = _safe(s, path)
        hits = []
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if not spec.match_file(d)]
            for f in files:
                if spec.match_file(f):
                    continue
                fp = Path(root) / f
                try:
                    for i, line in enumerate(fp.read_text(errors="ignore").splitlines(), 1):
                        if rx.search(line):
                            hits.append({"file": str(fp.relative_to(s.workdir)), "line": i, "text": line[:300]})
                            if len(hits) >= max_results:
                                return json.dumps({"hits": hits, "truncated": True})
                except Exception:
                    continue
        return json.dumps({"hits": hits, "truncated": False})

    def file_info(path: str) -> str:
        p = _safe(s, path)
        st = p.stat()
        return json.dumps({
            "path": str(p), "size": st.st_size, "mtime": st.st_mtime,
            "is_dir": p.is_dir(), "is_file": p.is_file(),
        })

    for tool in [
        Tool("read_file", "Read a text file; optional line range.",
             {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["path"]},
             read_file, "safe"),
        Tool("write_file", "Create or overwrite a file.",
             {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
             write_file, "standard"),
        Tool("create_file", "Create a new file; fails if it exists.",
             {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path"]},
             create_file, "standard"),
        Tool("edit_file", "Exact string replacement in a file.",
             {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["path", "old", "new"]},
             edit_file, "standard"),
        Tool("append_file", "Append content to file.",
             {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
             append_file, "standard"),
        Tool("delete_file", "Delete a file.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
             delete_file, "standard", needs_confirmation=True),
        Tool("move_file", "Move / rename a file.",
             {"type": "object", "properties": {"src": {"type": "string"}, "dst": {"type": "string"}}, "required": ["src", "dst"]},
             move_file, "standard"),
        Tool("copy_file", "Copy a file or directory.",
             {"type": "object", "properties": {"src": {"type": "string"}, "dst": {"type": "string"}}, "required": ["src", "dst"]},
             copy_file, "standard"),
        Tool("create_dir", "Create a directory (mkdir -p).",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
             create_dir, "standard"),
        Tool("delete_dir", "Delete a directory.",
             {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]},
             delete_dir, "standard", needs_confirmation=True),
        Tool("list_dir", "List a directory tree up to given depth.",
             {"type": "object", "properties": {"path": {"type": "string"}, "depth": {"type": "integer"}}, "required": []},
             list_dir, "safe"),
        Tool("glob", "Glob files with a pattern (e.g. **/*.ts).",
             {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
             glob_search, "safe"),
        Tool("grep", "Search file contents.",
             {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "is_regex": {"type": "boolean"}, "max_results": {"type": "integer"}}, "required": ["pattern"]},
             grep, "safe"),
        Tool("file_info", "Stat a file.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
             file_info, "safe"),
    ]:
        r.register(tool)
