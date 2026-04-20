from __future__ import annotations
import json
from .registry import Registry, Tool
from ..settings import Settings


def register(r: Registry, s: Settings) -> None:
    from github import Github, GithubException

    def _gh() -> Github:
        if not s.github_token:
            raise RuntimeError("GITHUB_TOKEN not set")
        return Github(s.github_token)

    def gh_whoami() -> str:
        return json.dumps({"user": _gh().get_user().login})

    def gh_repo_create(name: str, private: bool = False, description: str = "", org: str | None = None) -> str:
        gh = _gh()
        owner = gh.get_organization(org) if org else gh.get_user()
        repo = owner.create_repo(name=name, private=private, description=description)
        return json.dumps({"ok": True, "full_name": repo.full_name, "ssh_url": repo.ssh_url, "clone_url": repo.clone_url})

    def gh_list_prs(repo: str, state: str = "open") -> str:
        prs = list(_gh().get_repo(repo).get_pulls(state=state))
        return json.dumps([{"number": p.number, "title": p.title, "author": p.user.login, "draft": p.draft, "head": p.head.ref} for p in prs[:100]])

    def gh_get_pr(repo: str, number: int) -> str:
        p = _gh().get_repo(repo).get_pull(number)
        return json.dumps({
            "number": p.number, "title": p.title, "body": p.body, "state": p.state,
            "head": p.head.ref, "base": p.base.ref, "mergeable": p.mergeable,
            "url": p.html_url,
        })

    def gh_create_pr(repo: str, head: str, base: str, title: str, body: str = "", draft: bool = False) -> str:
        pr = _gh().get_repo(repo).create_pull(title=title, body=body, head=head, base=base, draft=draft)
        return json.dumps({"ok": True, "number": pr.number, "url": pr.html_url})

    def gh_merge_pr(repo: str, number: int, method: str = "squash") -> str:
        pr = _gh().get_repo(repo).get_pull(number)
        res = pr.merge(merge_method=method)
        return json.dumps({"ok": res.merged, "sha": res.sha})

    def gh_comment_pr(repo: str, number: int, body: str) -> str:
        pr = _gh().get_repo(repo).get_pull(number)
        c = pr.create_issue_comment(body)
        return json.dumps({"ok": True, "id": c.id})

    def gh_review_pr(repo: str, number: int, body: str, event: str = "COMMENT") -> str:
        pr = _gh().get_repo(repo).get_pull(number)
        rev = pr.create_review(body=body, event=event)
        return json.dumps({"ok": True, "id": rev.id, "state": rev.state})

    def gh_list_issues(repo: str, state: str = "open") -> str:
        issues = _gh().get_repo(repo).get_issues(state=state)
        return json.dumps([{"number": i.number, "title": i.title, "author": i.user.login} for i in issues[:100]])

    def gh_create_issue(repo: str, title: str, body: str = "", labels: list[str] | None = None) -> str:
        i = _gh().get_repo(repo).create_issue(title=title, body=body, labels=labels or [])
        return json.dumps({"ok": True, "number": i.number, "url": i.html_url})

    def gh_close_issue(repo: str, number: int) -> str:
        _gh().get_repo(repo).get_issue(number).edit(state="closed")
        return json.dumps({"ok": True})

    def gh_create_release(repo: str, tag: str, title: str, body: str = "", prerelease: bool = False) -> str:
        rel = _gh().get_repo(repo).create_git_release(tag=tag, name=title, message=body, prerelease=prerelease)
        return json.dumps({"ok": True, "url": rel.html_url})

    def gh_workflow_trigger(repo: str, workflow: str, ref: str = "main", inputs: dict | None = None) -> str:
        wf = _gh().get_repo(repo).get_workflow(workflow)
        ok = wf.create_dispatch(ref=ref, inputs=inputs or {})
        return json.dumps({"ok": bool(ok)})

    for t in [
        Tool("gh_whoami", "Show authenticated GitHub user.",
             {"type": "object", "properties": {}}, gh_whoami, "safe"),
        Tool("gh_repo_create", "Create a new GitHub repository.",
             {"type": "object", "properties": {"name": {"type": "string"}, "private": {"type": "boolean"}, "description": {"type": "string"}, "org": {"type": "string"}}, "required": ["name"]},
             gh_repo_create, "standard", needs_confirmation=True),
        Tool("gh_list_prs", "List PRs on a repository (owner/repo).",
             {"type": "object", "properties": {"repo": {"type": "string"}, "state": {"type": "string"}}, "required": ["repo"]},
             gh_list_prs, "safe"),
        Tool("gh_get_pr", "Get a single PR.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "number": {"type": "integer"}}, "required": ["repo", "number"]},
             gh_get_pr, "safe"),
        Tool("gh_create_pr", "Open a new pull request.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "head": {"type": "string"}, "base": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}, "draft": {"type": "boolean"}}, "required": ["repo", "head", "base", "title"]},
             gh_create_pr, "standard", needs_confirmation=True),
        Tool("gh_merge_pr", "Merge a pull request.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "number": {"type": "integer"}, "method": {"type": "string"}}, "required": ["repo", "number"]},
             gh_merge_pr, "standard", needs_confirmation=True),
        Tool("gh_comment_pr", "Comment on a PR.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "number": {"type": "integer"}, "body": {"type": "string"}}, "required": ["repo", "number", "body"]},
             gh_comment_pr, "standard"),
        Tool("gh_review_pr", "Review a PR (APPROVE | REQUEST_CHANGES | COMMENT).",
             {"type": "object", "properties": {"repo": {"type": "string"}, "number": {"type": "integer"}, "body": {"type": "string"}, "event": {"type": "string"}}, "required": ["repo", "number", "body"]},
             gh_review_pr, "standard"),
        Tool("gh_list_issues", "List issues.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "state": {"type": "string"}}, "required": ["repo"]},
             gh_list_issues, "safe"),
        Tool("gh_create_issue", "Open a new issue.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}}}, "required": ["repo", "title"]},
             gh_create_issue, "standard"),
        Tool("gh_close_issue", "Close an issue.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "number": {"type": "integer"}}, "required": ["repo", "number"]},
             gh_close_issue, "standard"),
        Tool("gh_create_release", "Create a GitHub release.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "tag": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}, "prerelease": {"type": "boolean"}}, "required": ["repo", "tag", "title"]},
             gh_create_release, "standard", needs_confirmation=True),
        Tool("gh_workflow_trigger", "Dispatch a workflow run.",
             {"type": "object", "properties": {"repo": {"type": "string"}, "workflow": {"type": "string"}, "ref": {"type": "string"}, "inputs": {"type": "object"}}, "required": ["repo", "workflow"]},
             gh_workflow_trigger, "standard", needs_confirmation=True),
    ]:
        r.register(t)
