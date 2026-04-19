from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from .paths import CONFIGS


@dataclass
class GGUFQuant:
    id: str
    llama_cpp_type: str
    imatrix: bool
    priority: int
    expected_size_gb: float
    description: str


@dataclass
class MLXQuant:
    id: str
    bits: int
    group_size: int
    expected_size_gb: float
    description: str


@dataclass
class BuildConfig:
    model_name: str
    display_name: str
    base_repo: str
    base_revision: str
    license: str
    gguf: list[GGUFQuant]
    mlx: list[MLXQuant]
    hf_owner: str
    hf_repo_prefix: str
    hf_private: bool
    tags: list[str]
    raw: dict[str, Any]


def load() -> BuildConfig:
    data = yaml.safe_load((CONFIGS / "quants.yaml").read_text())
    return BuildConfig(
        model_name=data["model_name"],
        display_name=data["display_name"],
        base_repo=data["base"]["hf_repo"],
        base_revision=data["base"]["revision"],
        license=data["base"]["license"],
        gguf=[GGUFQuant(**q) for q in data["gguf_quants"]],
        mlx=[MLXQuant(**q) for q in data["mlx_quants"]],
        hf_owner=data["hf_upload"]["owner"],
        hf_repo_prefix=data["hf_upload"]["repo_prefix"],
        hf_private=data["hf_upload"]["private"],
        tags=data["hf_upload"]["default_tags"],
        raw=data,
    )


def template_path() -> Path:
    return CONFIGS / "model_yaml_template.yaml"
