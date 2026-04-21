from __future__ import annotations
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

Autonomy = Literal["safe", "standard", "yolo"]
Phase = Literal["planning", "execution", "reflection"]


class Settings(BaseModel):
    base_url: str = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
    api_key: str = os.environ.get("OPENAI_API_KEY", "lm-studio")
    model: str = os.environ.get("CODER_MODEL", "EliCoder-30B-A3B")
    small_model: str | None = os.environ.get("CODER_SMALL_MODEL")
    temperature: float = 0.2
    temperature_planning: float = 0.4
    temperature_execution: float = 0.1
    temperature_reflection: float = 0.25
    max_tokens: int = 4096
    max_tool_steps: int = 30
    autonomy: Autonomy = "standard"
    workdir: Path = Path(os.environ.get("CODER_WORKDIR", Path.cwd())).resolve()
    github_token: str | None = os.environ.get("GITHUB_TOKEN")

    stream: bool = os.environ.get("CODER_STREAM", "1") != "0"
    parallel_tools: bool = os.environ.get("CODER_PARALLEL", "1") != "0"
    parallel_max_workers: int = int(os.environ.get("CODER_PARALLEL_WORKERS", "6"))

    compaction_enabled: bool = True
    compaction_trigger_ratio: float = 0.70
    compaction_max_context_tokens: int = int(os.environ.get("CODER_CONTEXT_TOKENS", "32768"))
    compaction_keep_recent_turns: int = 6

    cache_enabled: bool = True
    cache_max_entries: int = 512

    reflect_on_error: bool = True
    reflect_max_retries: int = 2

    step_budget: int = int(os.environ.get("CODER_STEP_BUDGET", "30"))
    wallclock_budget_sec: int = int(os.environ.get("CODER_TIME_BUDGET", "1800"))
    tool_call_default_timeout: int = 120

    sandbox_shell: bool = os.environ.get("CODER_SANDBOX", "0") == "1"

    dynamic_temperature: bool = True
    model_router_enabled: bool = False

    ignore_patterns: list[str] = [
        ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
        ".next", ".turbo", ".cache", "*.lock", "*.min.js", "*.min.css",
    ]


def load() -> Settings:
    return Settings()
