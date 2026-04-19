from pathlib import Path
from rich.console import Console
from huggingface_hub import snapshot_download

from .config import load
from .paths import BASE_DIR, CACHE_DIR, ensure_dirs

console = Console()


def download_base(force: bool = False) -> Path:
    ensure_dirs()
    cfg = load()
    target = BASE_DIR / cfg.base_repo.replace("/", "__")

    if target.exists() and any(target.iterdir()) and not force:
        console.print(f"[yellow]Base already present at[/] {target}")
        return target

    console.print(f"[bold cyan]Downloading[/] {cfg.base_repo}@{cfg.base_revision}")
    snapshot_download(
        repo_id=cfg.base_repo,
        revision=cfg.base_revision,
        local_dir=str(target),
        cache_dir=str(CACHE_DIR),
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.model",
            "tokenizer*",
            "chat_template*",
            "generation_config*",
            "special_tokens_map*",
            "added_tokens.json",
            "README.md",
            "LICENSE*",
        ],
    )
    console.print(f"[green]Base ready at[/] {target}")
    return target


def base_dir() -> Path:
    cfg = load()
    return BASE_DIR / cfg.base_repo.replace("/", "__")
