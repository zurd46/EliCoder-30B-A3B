from pathlib import Path
import sys
from rich.console import Console

from .config import MLXQuant, load
from .download import base_dir
from .paths import MLX_DIR, ensure_dirs

console = Console()


def _variant_dir(cfg_name: str, q: MLXQuant) -> Path:
    return MLX_DIR / f"{cfg_name}-{q.id}"


def convert_mlx(quant_id: str | None = None, force: bool = False) -> list[Path]:
    if sys.platform != "darwin":
        console.print("[red]MLX is Apple-Silicon only. Skip on this platform.[/]")
        return []

    from mlx_lm import convert as mlx_convert

    ensure_dirs()
    cfg = load()
    base = base_dir()
    if not base.exists():
        raise FileNotFoundError(f"Base model missing at {base}. Run `download` first.")

    targets = cfg.mlx if quant_id is None else [q for q in cfg.mlx if q.id == quant_id]
    if not targets:
        raise ValueError(f"Unknown MLX quant id: {quant_id}")

    produced: list[Path] = []
    for q in targets:
        out = _variant_dir(cfg.model_name, q)
        if out.exists() and any(out.iterdir()) and not force:
            console.print(f"[yellow]MLX {q.id} already built at[/] {out}")
            produced.append(out)
            continue

        out.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold cyan]Converting MLX {q.id}[/] ({q.bits}-bit, gs={q.group_size})")
        mlx_convert.convert(
            hf_path=str(base),
            mlx_path=str(out),
            quantize=True,
            q_bits=q.bits,
            q_group_size=q.group_size,
            dtype="float16",
        )
        console.print(f"[green]MLX {q.id} done[/] \u2192 {out}")
        produced.append(out)

    return produced
