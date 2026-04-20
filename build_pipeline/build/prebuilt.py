from __future__ import annotations
from pathlib import Path
from rich.console import Console
from huggingface_hub import snapshot_download, hf_hub_download

from .config import load
from .paths import CACHE_DIR, GGUF_DIR, MLX_DIR, ensure_dirs

console = Console()


def fetch_prebuilt_mlx(variant_id: str, force: bool = False) -> Path:
    ensure_dirs()
    cfg = load()
    mapping = (cfg.raw.get("prebuilt") or {}).get("mlx_repos") or {}
    repo = mapping.get(variant_id)
    if not repo:
        raise ValueError(f"No prebuilt MLX repo configured for {variant_id}. "
                         f"Available: {list(mapping)}")

    target = MLX_DIR / f"{cfg.model_name}-{variant_id}"
    if target.exists() and any(target.glob("*.safetensors")) and not force:
        console.print(f"[yellow]Prebuilt MLX {variant_id} already present at[/] {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]Downloading prebuilt MLX[/] {repo}")
    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        cache_dir=str(CACHE_DIR),
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "chat_template*",
                        "special_tokens_map*", "added_tokens.json", "README.md", "LICENSE*"],
        max_workers=4,
    )
    console.print(f"[green]Prebuilt MLX ready[/] \u2192 {target}")
    return target


def fetch_prebuilt_gguf(variant_id: str, force: bool = False) -> Path:
    ensure_dirs()
    cfg = load()
    q = next((q for q in cfg.gguf if q.id == variant_id), None)
    if q is None:
        raise ValueError(f"Unknown GGUF variant: {variant_id}")
    if not q.prebuilt_file:
        raise ValueError(f"No prebuilt_file mapped for {variant_id}; run `convert-gguf` instead.")

    repo = (cfg.raw.get("prebuilt") or {}).get("gguf_repo")
    if not repo:
        raise ValueError("No prebuilt.gguf_repo configured")

    target = GGUF_DIR / f"{cfg.model_name}-{variant_id}.gguf"
    if target.exists() and target.stat().st_size > 0 and not force:
        console.print(f"[yellow]Prebuilt GGUF {variant_id} already present at[/] {target}")
        return target

    console.print(f"[bold cyan]Downloading prebuilt GGUF[/] {repo}/{q.prebuilt_file}")
    path = hf_hub_download(
        repo_id=repo,
        filename=q.prebuilt_file,
        cache_dir=str(CACHE_DIR),
        local_dir=str(GGUF_DIR),
    )
    final = Path(path)
    if final != target:
        if target.exists():
            target.unlink()
        final.rename(target)
    console.print(f"[green]Prebuilt GGUF ready[/] \u2192 {target}")
    return target


def fetch_prebuilt(kind: str, variant_id: str, force: bool = False) -> Path:
    if kind == "mlx":
        return fetch_prebuilt_mlx(variant_id, force=force)
    if kind == "gguf":
        return fetch_prebuilt_gguf(variant_id, force=force)
    raise ValueError(f"Unknown kind: {kind}")
