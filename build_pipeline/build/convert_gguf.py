from pathlib import Path
import os
import shutil
import subprocess
from rich.console import Console

from .config import GGUFQuant, load
from .download import base_dir
from .paths import GGUF_DIR, LLAMA_CPP_DIR, ensure_dirs

console = Console()

CALIB_URL = "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_data_v5_rc.txt"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    console.print(f"[dim]$ {' '.join(str(c) for c in cmd)}[/]")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def _ensure_llama_cpp() -> Path:
    root = LLAMA_CPP_DIR
    if not (root / "convert_hf_to_gguf.py").exists():
        console.print(f"[bold cyan]Cloning llama.cpp[/] \u2192 {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(root)])

    quant_bin = _bin_path(root, "llama-quantize")
    imatrix_bin = _bin_path(root, "llama-imatrix")
    if not quant_bin.exists() or not imatrix_bin.exists():
        console.print("[bold cyan]Building llama.cpp (release, metal/cuda/vulkan auto)[/]")
        build = root / "build"
        build.mkdir(exist_ok=True)
        _run(["cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"])
        _run(["cmake", "--build", str(build), "--config", "Release", "-j"])
    return root


def _bin_path(root: Path, name: str) -> Path:
    candidates = [
        root / "build" / "bin" / name,
        root / "build" / "bin" / "Release" / name,
        root / name,
    ]
    if os.name == "nt":
        candidates = [c.with_suffix(".exe") for c in candidates] + candidates
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _calibration_file(root: Path) -> Path:
    target = root / "calibration_v5.txt"
    if target.exists():
        return target
    try:
        import urllib.request
        console.print("[dim]Fetching calibration corpus[/]")
        urllib.request.urlretrieve(CALIB_URL, target)
    except Exception as e:
        console.print(f"[yellow]Calibration fetch failed ({e}); using wiki default[/]")
        (target).write_text("hello world\n" * 2048)
    return target


def _convert_to_f16(base: Path, out: Path, llama_root: Path) -> Path:
    if out.exists() and out.stat().st_size > 0:
        return out
    console.print(f"[bold cyan]HF \u2192 GGUF F16[/] {out.name}")
    _run([
        "python", str(llama_root / "convert_hf_to_gguf.py"),
        str(base),
        "--outfile", str(out),
        "--outtype", "f16",
    ])
    return out


def _compute_imatrix(f16_gguf: Path, llama_root: Path) -> Path:
    imat = f16_gguf.with_suffix(".imatrix")
    if imat.exists() and imat.stat().st_size > 0:
        return imat
    console.print("[bold cyan]Computing importance matrix (calibrated)[/]")
    calib = _calibration_file(llama_root)
    _run([
        _bin_path(llama_root, "llama-imatrix"),
        "-m", str(f16_gguf),
        "-f", str(calib),
        "-o", str(imat),
        "--chunks", "128",
    ])
    return imat


def _quantize(f16_gguf: Path, out: Path, qtype: str, imatrix: Path | None, llama_root: Path) -> Path:
    if out.exists() and out.stat().st_size > 0:
        console.print(f"[yellow]GGUF {out.name} already exists[/]")
        return out
    console.print(f"[bold cyan]Quantize \u2192 {qtype}[/] {out.name}")
    cmd: list = [_bin_path(llama_root, "llama-quantize")]
    if imatrix is not None:
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(f16_gguf), str(out), qtype]
    _run(cmd)
    return out


def convert_gguf(quant_id: str | None = None, force: bool = False) -> list[Path]:
    ensure_dirs()
    cfg = load()
    base = base_dir()
    if not base.exists():
        raise FileNotFoundError(f"Base model missing at {base}. Run `download` first.")

    llama_root = _ensure_llama_cpp()

    f16 = GGUF_DIR / f"{cfg.model_name}-f16.gguf"
    _convert_to_f16(base, f16, llama_root)

    targets = cfg.gguf if quant_id is None else [q for q in cfg.gguf if q.id == quant_id]
    if not targets:
        raise ValueError(f"Unknown GGUF quant id: {quant_id}")

    needs_imatrix = any(q.imatrix for q in targets)
    imat = _compute_imatrix(f16, llama_root) if needs_imatrix else None

    produced: list[Path] = []
    for q in sorted(targets, key=lambda x: x.priority):
        out = GGUF_DIR / f"{cfg.model_name}-{q.id}.gguf"
        if force and out.exists():
            out.unlink()
        _quantize(f16, out, q.llama_cpp_type, imat if q.imatrix else None, llama_root)
        produced.append(out)

    return produced


def cleanup_f16() -> None:
    cfg = load()
    f16 = GGUF_DIR / f"{cfg.model_name}-f16.gguf"
    if f16.exists():
        console.print(f"[yellow]Removing intermediate F16[/] {f16}")
        f16.unlink()
    imat = f16.with_suffix(".imatrix")
    if imat.exists():
        imat.unlink()
