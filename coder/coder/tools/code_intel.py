from __future__ import annotations
import json
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


LANG_BY_EXT = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".rb": "ruby", ".php": "php", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".cs": "c_sharp", ".swift": "swift", ".kt": "kotlin",
}


def register(r: Registry, s: Settings) -> None:

    def ast_symbols(path: str) -> str:
        from tree_sitter_languages import get_parser
        p = (s.workdir / path).resolve()
        lang = LANG_BY_EXT.get(p.suffix)
        if lang is None:
            return json.dumps({"ok": False, "error": f"unsupported ext {p.suffix}"})
        parser = get_parser(lang)
        tree = parser.parse(p.read_bytes())
        symbols = []

        def walk(node, depth=0):
            if node.type in ("function_definition", "function_declaration", "method_definition",
                             "class_definition", "class_declaration", "class", "impl_item",
                             "struct_item", "enum_item", "interface_declaration"):
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "(anon)"
                symbols.append({"kind": node.type, "name": name,
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1})
            for child in node.children:
                walk(child, depth + 1)

        walk(tree.root_node)
        return json.dumps({"path": str(p), "language": lang, "symbols": symbols})

    def find_symbol(name: str, path: str = ".") -> str:
        from tree_sitter_languages import get_parser
        root = (s.workdir / path).resolve()
        hits = []
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            lang = LANG_BY_EXT.get(file.suffix)
            if lang is None:
                continue
            try:
                parser = get_parser(lang)
                tree = parser.parse(file.read_bytes())
            except Exception:
                continue

            def walk(node):
                if hasattr(node, "children"):
                    name_node = node.child_by_field_name("name") if hasattr(node, "child_by_field_name") else None
                    if name_node is not None:
                        txt = name_node.text.decode("utf-8", errors="replace")
                        if txt == name:
                            hits.append({"file": str(file.relative_to(s.workdir)), "kind": node.type,
                                         "line": node.start_point[0] + 1})
                    for c in node.children:
                        walk(c)
            walk(tree.root_node)
        return json.dumps({"name": name, "hits": hits[:200]})

    for t in [
        Tool("ast_symbols", "List top-level symbols (functions, classes) in a file via tree-sitter.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
             ast_symbols, "safe"),
        Tool("find_symbol", "Find definitions of a symbol across the project.",
             {"type": "object", "properties": {"name": {"type": "string"}, "path": {"type": "string"}}, "required": ["name"]},
             find_symbol, "safe"),
    ]:
        r.register(t)
