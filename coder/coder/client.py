from __future__ import annotations
from typing import Iterable, Any
from openai import OpenAI
from .settings import Settings


class LMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._c = OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, stream: bool = False) -> Any:
        kwargs = dict(
            model=self.s.model,
            messages=messages,
            temperature=self.s.temperature,
            max_tokens=self.s.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
        return self._c.chat.completions.create(**kwargs)

    def stream_text(self, messages: list[dict], tools: list[dict] | None = None) -> Iterable[str]:
        for chunk in self.chat(messages, tools=tools, stream=True):
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
