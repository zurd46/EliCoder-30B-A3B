from pathlib import Path
import os
from rich.console import Console
from huggingface_hub import HfApi, create_repo, upload_folder

from .config import load
from .paths import PACKAGES_DIR, ensure_dirs
from .model_card import write_card

console = Console()


def _token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        raise RuntimeError("Set HF_TOKEN env var (https://huggingface.co/settings/tokens)")
    return tok


def _variant_kind(cfg, variant_id: str) -> str:
    if any(q.id == variant_id for q in cfg.gguf):
        return "gguf"
    if any(q.id == variant_id for q in cfg.mlx):
        return "mlx"
    raise ValueError(f"Unknown variant {variant_id}")


def upload_variant(variant_id: str, private: bool | None = None, dry_run: bool = False) -> str:
    ensure_dirs()
    cfg = load()
    pkg = PACKAGES_DIR / f"{cfg.model_name}-{variant_id}"
    if not pkg.exists():
        raise FileNotFoundError(f"Package missing: {pkg}. Run `package` first.")

    write_card(pkg, variant_id)

    repo_id = f"{cfg.hf_owner}/{cfg.hf_repo_prefix}-{variant_id}"
    console.print(f"[bold cyan]Target repo:[/] {repo_id}")

    if dry_run:
        console.print("[yellow]Dry run — nothing uploaded[/]")
        return repo_id

    token = _token()
    is_private = cfg.hf_private if private is None else private

    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=is_private,
        exist_ok=True,
        token=token,
    )

    console.print(f"[bold cyan]Uploading[/] {pkg} \u2192 {repo_id}")
    upload_folder(
        folder_path=str(pkg),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Release {variant_id}",
        ignore_patterns=[".*", "*.tmp"],
    )
    console.print(f"[green]Uploaded[/] https://huggingface.co/{repo_id}")
    return repo_id


def upload_all(kind: str | None = None, private: bool | None = None, dry_run: bool = False) -> list[str]:
    cfg = load()
    variants: list[str] = []
    if kind in (None, "gguf"):
        variants += [q.id for q in cfg.gguf]
    if kind in (None, "mlx"):
        variants += [q.id for q in cfg.mlx]
    return [upload_variant(v, private=private, dry_run=dry_run) for v in variants]


def whoami() -> None:
    api = HfApi(token=_token())
    console.print(api.whoami())
