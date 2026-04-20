from __future__ import annotations
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

Autonomy = Literal["safe", "standard", "yolo"]


class Settings(BaseModel):
    base_url: str = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
    api_key: str = os.environ.get("OPENAI_API_KEY", "lm-studio")
    model: str = os.environ.get("CODER_MODEL", "coder-16b-dyn-UD-Q4_K_XL")
    temperature: float = 0.2
    max_tokens: int = 4096
    max_tool_steps: int = 30
    autonomy: Autonomy = "standard"
    workdir: Path = Path(os.environ.get("CODER_WORKDIR", Path.cwd())).resolve()
    github_token: str | None = os.environ.get("GITHUB_TOKEN")
    ignore_patterns: list[str] = [
        ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
        ".next", ".turbo", ".cache", "*.lock", "*.min.js", "*.min.css",
    ]


def load() -> Settings:
    return Settings()
