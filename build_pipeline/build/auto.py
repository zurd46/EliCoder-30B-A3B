from __future__ import annotations
import os, sys, subprocess, platform, shutil
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

from .config import BuildConfig, GGUFQuant, MLXQuant, load

console = Console()


@dataclass
class Host:
    platform: str
    arch: str
    total_ram_gb: float
    available_ram_gb: float
    apple_silicon: bool
    cpu_brand: str
    gpu_cores: int | None
    cuda_vram_gb: float | None
    has_metal: bool
    swap_used_gb: float

    def describe(self) -> str:
        vram = f"{self.cuda_vram_gb:.0f} GB CUDA" if self.cuda_vram_gb else "—"
        cpu = self.cpu_brand or self.arch
        gpu = f" · {self.gpu_cores}-core GPU" if self.gpu_cores else ""
        return (f"{cpu}{gpu} · Unified RAM {self.total_ram_gb:.1f} GB "
                f"(verfügbar jetzt: {self.available_ram_gb:.1f} GB, "
                f"Swap: {self.swap_used_gb:.1f} GB) · "
                f"Metal={self.has_metal} · VRAM={vram}")


KV_CACHE_GB_PER_K_TOKENS = 0.0011


def _system_reserve_gb(total_ram_gb: float, plat: str) -> float:
    if plat == "darwin":
        base = 5.0
        scaled = total_ram_gb * 0.18
        lm_studio_overhead = 1.0
        return max(base, min(scaled + lm_studio_overhead, 10.0))
    if plat.startswith("linux"):
        return max(2.0, min(total_ram_gb * 0.10, 6.0))
    return max(4.0, min(total_ram_gb * 0.15, 8.0))


def detect() -> Host:
    plat = sys.platform
    arch = platform.machine().lower()
    apple_silicon = plat == "darwin" and arch in ("arm64", "aarch64")
    total_gb, avail_gb, swap_gb = _detect_memory_gb()
    cuda_vram = _detect_cuda_vram_gb() if plat != "darwin" else None
    cpu_brand, gpu_cores = ("", None)
    if plat == "darwin":
        cpu_brand, gpu_cores = _detect_mac_chip()
    return Host(
        platform=plat,
        arch=arch,
        total_ram_gb=total_gb,
        available_ram_gb=avail_gb,
        apple_silicon=apple_silicon,
        cpu_brand=cpu_brand,
        gpu_cores=gpu_cores,
        cuda_vram_gb=cuda_vram,
        has_metal=apple_silicon,
        swap_used_gb=swap_gb,
    )


def _detect_memory_gb() -> tuple[float, float, float]:
    total_gb = 16.0
    avail_gb = 12.0
    swap_gb = 0.0

    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        total_gb = vm.total / 1024**3
        avail_gb = vm.available / 1024**3
        swap_gb = sw.used / 1024**3
        if sys.platform == "darwin":
            mac_total, mac_avail = _mac_precise_memory()
            if mac_total:
                total_gb = mac_total
            if mac_avail is not None:
                avail_gb = mac_avail
        return total_gb, avail_gb, swap_gb
    except Exception:
        pass

    if sys.platform == "darwin":
        try:
            mac_total, mac_avail = _mac_precise_memory()
            if mac_total:
                total_gb = mac_total
            if mac_avail is not None:
                avail_gb = mac_avail
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                data = {ln.split(":")[0]: int(ln.split()[1]) for ln in f if ":" in ln}
            total_gb = data["MemTotal"] / 1024**2
            avail_gb = data.get("MemAvailable", data["MemTotal"]) / 1024**2
            swap_gb = max(0, data.get("SwapTotal", 0) - data.get("SwapFree", 0)) / 1024**2
        except Exception:
            pass
    return total_gb, avail_gb, swap_gb


def _mac_precise_memory() -> tuple[float | None, float | None]:
    total = None
    avail = None
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        total = int(out.strip()) / 1024**3
    except Exception:
        pass
    try:
        vm = subprocess.check_output(["vm_stat"], text=True)
        pages = {}
        page_size = 16384
        for ln in vm.splitlines():
            if "page size of" in ln:
                try:
                    page_size = int(ln.split("page size of")[1].split()[0])
                except Exception:
                    pass
                continue
            if ":" in ln:
                k, v = ln.split(":", 1)
                v = v.strip().rstrip(".").replace(",", "")
                if v.isdigit():
                    pages[k.strip()] = int(v)
        free = pages.get("Pages free", 0)
        inactive = pages.get("Pages inactive", 0)
        speculative = pages.get("Pages speculative", 0)
        purgeable = pages.get("Pages purgeable", 0)
        file_backed = pages.get("File-backed pages", 0)
        avail_bytes = (free + inactive + speculative + purgeable + file_backed) * page_size
        avail = avail_bytes / 1024**3
    except Exception:
        pass
    return total, avail


def _detect_mac_chip() -> tuple[str, int | None]:
    brand, gpu_cores = "", None
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=5,
        )
        for ln in out.splitlines():
            s = ln.strip()
            if s.lower().startswith("total number of cores:"):
                try:
                    gpu_cores = int(s.split(":")[1].strip())
                    break
                except Exception:
                    pass
    except Exception:
        pass
    return brand, gpu_cores


def _detect_cuda_vram_gb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        totals = [int(x.strip()) / 1024 for x in out.strip().splitlines() if x.strip()]
        return max(totals) if totals else None
    except Exception:
        return None


def _kv_budget_gb(ctx_tokens: int) -> float:
    return (ctx_tokens / 1000) * KV_CACHE_GB_PER_K_TOKENS * 8 / 2


def _budget_gb(host: Host, target_ctx: int) -> tuple[float, float, float]:
    """Return (working_total, usable_for_weights, breakdown_reserve)."""
    if host.cuda_vram_gb:
        working = host.cuda_vram_gb
        reserve = 2.0
    else:
        working = host.total_ram_gb
        reserve = _system_reserve_gb(host.total_ram_gb, host.platform)

    kv_gb = _kv_budget_gb(target_ctx)
    runtime_overhead_gb = 1.5
    usable = working - reserve - kv_gb - runtime_overhead_gb
    return working, usable, reserve + kv_gb + runtime_overhead_gb


def recommend(host: Host, target_ctx: int = 32_768) -> list[tuple[str, str]]:
    cfg = load()
    _, usable, _ = _budget_gb(host, target_ctx)
    overhead = 1.12

    gguf_candidates = [q for q in cfg.gguf if q.expected_size_gb * overhead <= usable]
    gguf_candidates.sort(key=lambda q: -q.expected_size_gb)
    gguf_best = gguf_candidates[0] if gguf_candidates else min(cfg.gguf, key=lambda q: q.expected_size_gb)

    picks: list[tuple[str, str]] = []
    if host.apple_silicon:
        mlx_candidates = [q for q in cfg.mlx if q.expected_size_gb * overhead <= usable]
        mlx_candidates.sort(key=lambda q: -q.expected_size_gb)
        if mlx_candidates:
            picks.append(("mlx", mlx_candidates[0].id))
    picks.append(("gguf", gguf_best.id))
    return picks


def print_plan(host: Host, target_ctx: int) -> list[tuple[str, str]]:
    cfg = load()
    working, usable, overhead_total = _budget_gb(host, target_ctx)
    reserve = _system_reserve_gb(host.total_ram_gb, host.platform) if not host.cuda_vram_gb else 2.0
    kv_gb = _kv_budget_gb(target_ctx)

    console.print(f"[bold]Host:[/] {host.describe()}")

    if host.platform == "darwin" and host.available_ram_gb < host.total_ram_gb * 0.6:
        console.print(f"[yellow]Warnung:[/] Nur {host.available_ram_gb:.1f} von {host.total_ram_gb:.1f} GB gerade frei. "
                      "Schließe Chrome / Docker / andere Apps vor dem Modell-Load.")

    console.print(f"[bold]Target context:[/] {target_ctx:,} tokens  (KV ~ {kv_gb:.1f} GB)")
    console.print(f"[bold]System reserve:[/] {reserve:.1f} GB")
    console.print(f"[bold]Runtime/LM-Studio-Overhead:[/] 1.5 GB")
    console.print(f"[bold]Usable for weights:[/] ~ {usable:.1f} GB "
                  f"(aus {working:.1f} GB - {overhead_total:.1f} GB Overhead)\n")

    t = Table(title="Candidates (fits=OK, tight=~, too-big=NO)")
    t.add_column("Kind"); t.add_column("Variant"); t.add_column("Size GB")
    t.add_column("Verdict"); t.add_column("Headroom GB")
    all_q: list[tuple[str, GGUFQuant | MLXQuant]] = [("gguf", q) for q in cfg.gguf] + [("mlx", q) for q in cfg.mlx]
    for kind, q in sorted(all_q, key=lambda p: -p[1].expected_size_gb):
        need = q.expected_size_gb * 1.12
        if kind == "mlx" and not host.apple_silicon:
            v, hr = "n/a (not Apple Silicon)", "-"
        elif need <= usable:
            v, hr = "OK fits", f"+{usable - need:.1f}"
        elif need <= usable + 2:
            v, hr = "~ tight", f"{usable - need:.1f}"
        else:
            v, hr = "NO too big", f"{usable - need:.1f}"
        t.add_row(kind.upper(), q.id, f"{q.expected_size_gb:.1f}", v, hr)
    console.print(t)

    picks = recommend(host, target_ctx)
    console.print(f"\n[bold green]Recommended:[/] " + ", ".join(f"{k}:{v}" for k, v in picks))
    return picks
