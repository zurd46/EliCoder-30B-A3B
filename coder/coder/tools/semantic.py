from __future__ import annotations
import json, hashlib, os
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
             ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
             ".kt", ".scala", ".md", ".mdx", ".yaml", ".yml", ".toml"}


def _chunk_file(text: str, chunk_lines: int = 40, overlap: int = 8) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        j = min(i + chunk_lines, len(lines))
        out.append((i + 1, j, "\n".join(lines[i:j])))
        if j >= len(lines):
            break
        i = j - overlap if j - overlap > i else j
    return out


class _SemIndex:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.cache = workdir / ".coder" / "semantic"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.embed_path = self.cache / "embeddings.npy"
        self.meta_path = self.cache / "meta.json"
        self._model = None
        self._emb = None
        self._meta: list[dict] = []

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def build(self, ignore_patterns: list[str], max_files: int = 2000) -> dict:
        import numpy as np  # type: ignore
        model = self._load_model()

        from pathspec import PathSpec
        spec = PathSpec.from_lines("gitwildmatch", ignore_patterns)

        chunks_text: list[str] = []
        meta: list[dict] = []
        files_done = 0
        for root, dirs, files in os.walk(self.workdir):
            dirs[:] = [d for d in dirs if not spec.match_file(d)]
            for f in files:
                if spec.match_file(f):
                    continue
                fp = Path(root) / f
                if fp.suffix not in CODE_EXTS:
                    continue
                try:
                    txt = fp.read_text(errors="ignore")
                except Exception:
                    continue
                for (a, b, chunk) in _chunk_file(txt):
                    chunks_text.append(chunk)
                    meta.append({
                        "file": str(fp.relative_to(self.workdir)),
                        "start": a, "end": b,
                    })
                files_done += 1
                if files_done >= max_files:
                    break
            if files_done >= max_files:
                break

        if not chunks_text:
            return {"ok": False, "error": "no files to index"}

        emb = model.encode(chunks_text, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
        np.save(self.embed_path, emb)
        self.meta_path.write_text(json.dumps(meta))
        self._emb = emb
        self._meta = meta
        return {"ok": True, "files": files_done, "chunks": len(meta)}

    def _ensure_loaded(self) -> bool:
        import numpy as np  # type: ignore
        if self._emb is None and self.embed_path.exists() and self.meta_path.exists():
            self._emb = np.load(self.embed_path)
            self._meta = json.loads(self.meta_path.read_text())
        return self._emb is not None

    def search(self, query: str, k: int = 10) -> dict:
        import numpy as np  # type: ignore
        if not self._ensure_loaded():
            return {"ok": False, "error": "index not built — call semantic_index_build first"}
        model = self._load_model()
        q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self._emb @ q
        top = np.argsort(-scores)[:k]
        hits = []
        for i in top:
            m = self._meta[int(i)]
            hits.append({"file": m["file"], "start": m["start"], "end": m["end"],
                         "score": float(scores[int(i)])})
        return {"ok": True, "hits": hits}


def register(r: Registry, s: Settings) -> None:
    index = _SemIndex(s.workdir)

    def semantic_index_build(max_files: int = 2000) -> str:
        try:
            return json.dumps(index.build(s.ignore_patterns, max_files=max_files))
        except ImportError as e:
            return json.dumps({"ok": False, "error": f"missing deps: {e} — install with `pip install coderllm-agent[embeddings]`"})

    def semantic_search(query: str, k: int = 10) -> str:
        try:
            return json.dumps(index.search(query, k=k))
        except ImportError as e:
            return json.dumps({"ok": False, "error": f"missing deps: {e} — install with `pip install coderllm-agent[embeddings]`"})

    def semantic_index_status() -> str:
        exists = index.embed_path.exists() and index.meta_path.exists()
        return json.dumps({
            "built": exists,
            "embed_path": str(index.embed_path),
            "meta_entries": len(json.loads(index.meta_path.read_text())) if index.meta_path.exists() else 0,
        })

    for t in [
        Tool("semantic_index_build",
             "Build a local embeddings index over the project for semantic search. Requires `[embeddings]` extra.",
             {"type": "object", "properties": {"max_files": {"type": "integer"}}, "required": []},
             semantic_index_build, "standard"),
        Tool("semantic_search",
             "Semantic search over the project codebase (after semantic_index_build).",
             {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]},
             semantic_search, "safe", cacheable=True),
        Tool("semantic_index_status", "Report whether the semantic index has been built.",
             {"type": "object", "properties": {}}, semantic_index_status, "safe"),
    ]:
        r.register(t)
