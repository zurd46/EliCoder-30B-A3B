from __future__ import annotations
import json, os
import httpx
from .registry import Registry, Tool
from ..settings import Settings


def register(r: Registry, s: Settings) -> None:

    def fetch_url(url: str, method: str = "GET", headers: dict | None = None,
                  body: str | None = None, timeout: int = 30) -> str:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as c:
                resp = c.request(method, url, headers=headers, content=body)
            return json.dumps({
                "status": resp.status_code,
                "headers": dict(resp.headers),
                "body": resp.text[:20000],
            })
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def web_search(query: str, max_results: int = 5) -> str:
        tavily = os.environ.get("TAVILY_API_KEY")
        brave = os.environ.get("BRAVE_API_KEY")
        if tavily:
            r_ = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily, "query": query, "max_results": max_results},
                timeout=30,
            )
            return json.dumps(r_.json())
        if brave:
            r_ = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": brave, "Accept": "application/json"},
                timeout=30,
            )
            return json.dumps(r_.json())
        return json.dumps({"error": "no TAVILY_API_KEY or BRAVE_API_KEY set"})

    for t in [
        Tool("fetch_url", "HTTP request. Returns up to 20k chars of response body.",
             {"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string"}, "headers": {"type": "object"}, "body": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["url"]},
             fetch_url, "safe"),
        Tool("web_search", "Web search via Tavily or Brave (needs API key).",
             {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]},
             web_search, "safe"),
    ]:
        r.register(t)
