from pathlib import Path
import shutil
import subprocess
import sys
from rich.console import Console

from .config import MLXQuant, load
from .download import base_dir
from .paths import MLX_DIR, ensure_dirs

console = Console()


def _variant_dir(cfg_name: str, q: MLXQuant) -> Path:
    return MLX_DIR / f"{cfg_name}-{q.id}"


def _run(cmd: list[str]) -> None:
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    subprocess.run(cmd, check=True)


def _has_weights(p: Path) -> bool:
    return p.exists() and any(p.glob("*.safetensors"))


def convert_mlx(quant_id: str | None = None, force: bool = False) -> list[Path]:
    if sys.platform != "darwin":
        console.print("[red]MLX is Apple-Silicon only. Skip on this platform.[/]")
        return []

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
        if _has_weights(out) and not force:
            console.print(f"[yellow]MLX {q.id} already built at[/] {out}")
            produced.append(out)
            continue
        if out.exists() and force:
            shutil.rmtree(out)

        console.print(f"[bold cyan]Converting MLX {q.id}[/] ({q.bits}-bit, gs={q.group_size})")
        cmd = [
            sys.executable, "-m", "mlx_lm", "convert",
            "--hf-path", str(base),
            "--mlx-path", str(out),
            "-q",
            "--q-bits", str(q.bits),
            "--q-group-size", str(q.group_size),
            "--dtype", "float16",
        ]
        try:
            _run(cmd)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[yellow]Retrying with legacy `mlx_lm.convert` entry point[/]")
            cmd[3] = "mlx_lm.convert"
            del cmd[2]
            _run(cmd)

        if not _has_weights(out):
            raise RuntimeError(f"MLX conversion produced no weights in {out}")

        console.print(f"[green]MLX {q.id} done[/] \u2192 {out}")
        produced.append(out)

    return produced
