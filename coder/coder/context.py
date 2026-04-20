from __future__ import annotations
import os, json
from pathlib import Path
from typing import Iterable
from pathspec import PathSpec

from .settings import Settings


HOT_FILES = [
    "README.md", "README.txt", "AGENTS.md", "CLAUDE.md", ".cursorrules",
    "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "Gemfile",
    "pom.xml", "build.gradle", "composer.json",
]


class ProjectContext:
    def __init__(self, settings: Settings):
        self.s = settings
        self._spec = PathSpec.from_lines("gitwildmatch", settings.ignore_patterns)

    def file_tree(self, max_entries: int = 300) -> str:
        entries: list[str] = []
        for root, dirs, files in os.walk(self.s.workdir):
            rel_root = os.path.relpath(root, self.s.workdir)
            dirs[:] = [d for d in dirs if not self._spec.match_file(d)]
            for f in files:
                if self._spec.match_file(f):
                    continue
                rel = f if rel_root == "." else os.path.join(rel_root, f)
                entries.append(rel)
                if len(entries) >= max_entries:
                    return "\n".join(entries) + "\n… (truncated)"
        return "\n".join(entries)

    def hot_snippets(self, max_chars: int = 6000) -> str:
        out: list[str] = []
        remaining = max_chars
        for name in HOT_FILES:
            p = self.s.workdir / name
            if not p.exists() or not p.is_file():
                continue
            text = p.read_text(errors="replace")
            snippet = text[: min(len(text), remaining)]
            out.append(f"## {name}\n```\n{snippet}\n```")
            remaining -= len(snippet)
            if remaining <= 200:
                break
        return "\n\n".join(out)

    def summarize(self) -> str:
        tree = self.file_tree()
        hot = self.hot_snippets()
        return (
            f"# Project at {self.s.workdir}\n\n"
            f"## File tree\n```\n{tree}\n```\n\n"
            f"{hot}"
        )
