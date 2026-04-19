from pathlib import Path
import shutil
from rich.console import Console

from .config import load, template_path
from .paths import GGUF_DIR, MLX_DIR, PACKAGES_DIR, ensure_dirs

console = Console()


def _render_model_yaml(display_name: str, description: str, owner: str) -> str:
    tpl = template_path().read_text()
    return (
        tpl.replace("{{DISPLAY_NAME}}", display_name)
           .replace("{{DESCRIPTION}}", description)
           .replace("{{OWNER}}", owner)
    )


def package_gguf(variant_id: str | None = None, force: bool = False) -> list[Path]:
    ensure_dirs()
    cfg = load()
    ids = [q.id for q in cfg.gguf] if variant_id is None else [variant_id]
    out_dirs: list[Path] = []

    for qid in ids:
        q = next((x for x in cfg.gguf if x.id == qid), None)
        if q is None:
            raise ValueError(f"Unknown GGUF variant {qid}")
        src = GGUF_DIR / f"{cfg.model_name}-{qid}.gguf"
        if not src.exists():
            console.print(f"[yellow]Skip {qid}: GGUF missing ({src})[/]")
            continue

        out = PACKAGES_DIR / f"{cfg.model_name}-{qid}"
        if out.exists() and force:
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        dst_gguf = out / src.name
        if not dst_gguf.exists():
            shutil.copy2(src, dst_gguf)

        yml = _render_model_yaml(
            display_name=f"{cfg.display_name} ({qid})",
            description=q.description,
            owner=cfg.hf_owner,
        )
        (out / "model.yaml").write_text(yml)
        console.print(f"[green]Packaged GGUF[/] \u2192 {out}")
        out_dirs.append(out)

    return out_dirs


def package_mlx(variant_id: str | None = None, force: bool = False) -> list[Path]:
    ensure_dirs()
    cfg = load()
    ids = [q.id for q in cfg.mlx] if variant_id is None else [variant_id]
    out_dirs: list[Path] = []

    for qid in ids:
        q = next((x for x in cfg.mlx if x.id == qid), None)
        if q is None:
            raise ValueError(f"Unknown MLX variant {qid}")

        src = MLX_DIR / f"{cfg.model_name}-{qid}"
        if not src.exists():
            console.print(f"[yellow]Skip {qid}: MLX dir missing ({src})[/]")
            continue

        out = PACKAGES_DIR / f"{cfg.model_name}-{qid}"
        if out.exists() and force:
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            dst = out / item.name
            if dst.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        yml = _render_model_yaml(
            display_name=f"{cfg.display_name} ({qid})",
            description=q.description,
            owner=cfg.hf_owner,
        )
        (out / "model.yaml").write_text(yml)
        console.print(f"[green]Packaged MLX[/] \u2192 {out}")
        out_dirs.append(out)

    return out_dirs
