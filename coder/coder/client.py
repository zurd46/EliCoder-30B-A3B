from __future__ import annotations
from typing import Iterable, Any, Iterator
from openai import OpenAI
from .settings import Settings, Phase


class LMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._c = OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def _pick_model(self, phase: Phase | None, prefer_small: bool) -> str:
        if self.s.model_router_enabled and prefer_small and self.s.small_model:
            return self.s.small_model
        return self.s.model

    def _pick_temperature(self, phase: Phase | None) -> float:
        if not self.s.dynamic_temperature or phase is None:
            return self.s.temperature
        return {
            "planning": self.s.temperature_planning,
            "execution": self.s.temperature_execution,
            "reflection": self.s.temperature_reflection,
        }.get(phase, self.s.temperature)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
        phase: Phase | None = None,
        prefer_small: bool = False,
    ) -> Any:
        kwargs = dict(
            model=self._pick_model(phase, prefer_small),
            messages=messages,
            temperature=self._pick_temperature(phase),
            max_tokens=self.s.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
        return self._c.chat.completions.create(**kwargs)

    def stream_text(
        self, messages: list[dict], tools: list[dict] | None = None,
        phase: Phase | None = None, prefer_small: bool = False,
    ) -> Iterable[str]:
        for chunk in self.chat(messages, tools=tools, stream=True, phase=phase, prefer_small=prefer_small):
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def stream_assembled(
        self, messages: list[dict], tools: list[dict] | None = None,
        phase: Phase | None = None, prefer_small: bool = False,
        on_token: "callable | None" = None,
    ) -> dict:
        """Stream a chat completion and assemble content + tool_calls.

        Returns a dict shaped like {"content": str, "tool_calls": [...]} that matches
        the non-streaming `message` surface used by the agent.
        """
        content_buf: list[str] = []
        tool_acc: dict[int, dict] = {}
        for chunk in self.chat(messages, tools=tools, stream=True, phase=phase, prefer_small=prefer_small):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content_buf.append(delta.content)
                if on_token:
                    on_token(delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = tc.index
                slot = tool_acc.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments
        tool_calls = [
            {"id": v["id"] or f"call_{i}", "type": "function",
             "function": {"name": v["name"], "arguments": v["arguments"]}}
            for i, v in sorted(tool_acc.items())
            if v["name"]
        ]
        return {"content": "".join(content_buf), "tool_calls": tool_calls}

    def count_tokens_rough(self, messages: list[dict]) -> int:
        """Very rough token estimate (chars / 4). Good enough for compaction triggers."""
        total = 0
        for m in messages:
            total += len(str(m.get("content") or "")) // 4
            for tc in m.get("tool_calls") or []:
                total += len(str(tc.get("function", {}).get("arguments", ""))) // 4
        return total
