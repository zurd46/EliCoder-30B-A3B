from pathlib import Path
from rich.console import Console
from huggingface_hub import snapshot_download

from .config import load
from .paths import BASE_DIR, CACHE_DIR, ensure_dirs

console = Console()


def _is_complete(target: Path) -> bool:
    if not target.exists():
        return False
    safetensors = list(target.glob("*.safetensors"))
    index = target / "model.safetensors.index.json"
    if not safetensors:
        return False
    if index.exists():
        import json
        try:
            manifest = json.loads(index.read_text())
            expected = set(manifest.get("weight_map", {}).values())
            present = {p.name for p in safetensors}
            if not expected.issubset(present):
                return False
        except Exception:
            return False
    return True


def download_base(force: bool = False) -> Path:
    ensure_dirs()
    cfg = load()
    target = BASE_DIR / cfg.base_repo.replace("/", "__")

    if _is_complete(target) and not force:
        console.print(f"[green]Base already complete at[/] {target}")
        return target

    console.print(f"[bold cyan]Downloading[/] {cfg.base_repo}@{cfg.base_revision}  (resume-capable)")
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
        max_workers=4,
    )

    if not _is_complete(target):
        raise RuntimeError(
            f"Download finished but base is still incomplete at {target}. "
            "Re-run `coderllm-build download` to resume."
        )

    console.print(f"[green]Base complete at[/] {target}")
    return target


def base_dir() -> Path:
    cfg = load()
    return BASE_DIR / cfg.base_repo.replace("/", "__")
