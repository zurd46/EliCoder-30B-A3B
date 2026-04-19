from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
WORK = Path(os.environ.get("CODERLLM_WORK", ROOT / "work")).resolve()

BASE_DIR = WORK / "base"
MLX_DIR = WORK / "mlx"
GGUF_DIR = WORK / "gguf"
PACKAGES_DIR = WORK / "packages"
CACHE_DIR = WORK / "cache"

LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", WORK / "llama.cpp")).resolve()


def ensure_dirs() -> None:
    for p in (WORK, BASE_DIR, MLX_DIR, GGUF_DIR, PACKAGES_DIR, CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)
