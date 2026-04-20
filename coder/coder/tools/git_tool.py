from __future__ import annotations
import json
from pathlib import Path
from .registry import Registry, Tool
from ..settings import Settings


def register(r: Registry, s: Settings) -> None:
    import git

    def _repo() -> "git.Repo":
        try:
            return git.Repo(s.workdir, search_parent_directories=True)
        except Exception as e:
            raise RuntimeError(f"not a git repo: {e}")

    def git_init(path: str = ".", initial_branch: str = "main") -> str:
        p = (s.workdir / path).resolve()
        repo = git.Repo.init(p, initial_branch=initial_branch)
        return json.dumps({"ok": True, "path": str(repo.working_dir)})

    def git_clone(url: str, path: str | None = None) -> str:
        target = (s.workdir / (path or url.split("/")[-1].removesuffix(".git"))).resolve()
        git.Repo.clone_from(url, target)
        return json.dumps({"ok": True, "path": str(target)})

    def git_status() -> str:
        repo = _repo()
        return json.dumps({
            "branch": repo.active_branch.name if not repo.head.is_detached else "(detached)",
            "dirty": repo.is_dirty(),
            "untracked": repo.untracked_files[:200],
            "modified": [i.a_path for i in repo.index.diff(None)][:200],
            "staged": [i.a_path for i in repo.index.diff("HEAD")][:200],
        })

    def git_diff(ref: str | None = None, staged: bool = False) -> str:
        repo = _repo()
        if staged:
            d = repo.git.diff("--cached")
        elif ref:
            d = repo.git.diff(ref)
        else:
            d = repo.git.diff()
        return json.dumps({"diff": d[-16000:]})

    def git_log(n: int = 20) -> str:
        repo = _repo()
        entries = []
        for c in list(repo.iter_commits())[:n]:
            entries.append({"hash": c.hexsha[:10], "author": c.author.name, "msg": c.message.splitlines()[0], "when": c.committed_datetime.isoformat()})
        return json.dumps({"commits": entries})

    def git_add(paths: list[str]) -> str:
        repo = _repo()
        repo.index.add(paths)
        return json.dumps({"ok": True, "added": paths})

    def git_commit(message: str) -> str:
        repo = _repo()
        c = repo.index.commit(message)
        return json.dumps({"ok": True, "hash": c.hexsha[:10]})

    def git_checkout(branch: str, create: bool = False) -> str:
        repo = _repo()
        if create:
            repo.git.checkout("-b", branch)
        else:
            repo.git.checkout(branch)
        return json.dumps({"ok": True, "branch": branch})

    def git_merge(branch: str) -> str:
        repo = _repo()
        out = repo.git.merge(branch)
        return json.dumps({"ok": True, "result": out})

    def git_branch(list_all: bool = False) -> str:
        repo = _repo()
        cur = repo.active_branch.name if not repo.head.is_detached else None
        branches = [h.name for h in repo.heads]
        if list_all:
            branches += [f"remotes/{r.name}/{ref.name}" for r in repo.remotes for ref in r.refs]
        return json.dumps({"current": cur, "branches": branches})

    def git_push(remote: str = "origin", branch: str | None = None) -> str:
        repo = _repo()
        b = branch or repo.active_branch.name
        info = repo.remote(remote).push(b)
        return json.dumps({"ok": True, "summary": [str(i) for i in info]})

    def git_pull(remote: str = "origin", branch: str | None = None) -> str:
        repo = _repo()
        b = branch or repo.active_branch.name
        out = repo.git.pull(remote, b)
        return json.dumps({"ok": True, "output": out})

    def git_stash(pop: bool = False, message: str | None = None) -> str:
        repo = _repo()
        if pop:
            out = repo.git.stash("pop")
        elif message:
            out = repo.git.stash("push", "-m", message)
        else:
            out = repo.git.stash()
        return json.dumps({"ok": True, "output": out})

    for t in [
        Tool("git_init", "Initialize a new git repo.",
             {"type": "object", "properties": {"path": {"type": "string"}, "initial_branch": {"type": "string"}}, "required": []},
             git_init, "standard"),
        Tool("git_clone", "Clone a remote repo.",
             {"type": "object", "properties": {"url": {"type": "string"}, "path": {"type": "string"}}, "required": ["url"]},
             git_clone, "standard"),
        Tool("git_status", "Show working tree status.",
             {"type": "object", "properties": {}}, git_status, "safe"),
        Tool("git_diff", "Show unstaged or staged diff.",
             {"type": "object", "properties": {"ref": {"type": "string"}, "staged": {"type": "boolean"}}, "required": []},
             git_diff, "safe"),
        Tool("git_log", "Show commit log.",
             {"type": "object", "properties": {"n": {"type": "integer"}}, "required": []},
             git_log, "safe"),
        Tool("git_add", "Stage files.",
             {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]},
             git_add, "standard"),
        Tool("git_commit", "Create a commit with the given message.",
             {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
             git_commit, "standard"),
        Tool("git_checkout", "Switch branches, optionally creating.",
             {"type": "object", "properties": {"branch": {"type": "string"}, "create": {"type": "boolean"}}, "required": ["branch"]},
             git_checkout, "standard"),
        Tool("git_merge", "Merge a branch into current.",
             {"type": "object", "properties": {"branch": {"type": "string"}}, "required": ["branch"]},
             git_merge, "standard"),
        Tool("git_branch", "List branches.",
             {"type": "object", "properties": {"list_all": {"type": "boolean"}}, "required": []},
             git_branch, "safe"),
        Tool("git_push", "Push to remote.",
             {"type": "object", "properties": {"remote": {"type": "string"}, "branch": {"type": "string"}}, "required": []},
             git_push, "standard", needs_confirmation=True),
        Tool("git_pull", "Pull from remote.",
             {"type": "object", "properties": {"remote": {"type": "string"}, "branch": {"type": "string"}}, "required": []},
             git_pull, "standard"),
        Tool("git_stash", "Stash or pop stashed changes.",
             {"type": "object", "properties": {"pop": {"type": "boolean"}, "message": {"type": "string"}}, "required": []},
             git_stash, "standard"),
    ]:
        r.register(t)
