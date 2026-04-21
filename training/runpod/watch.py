"""Live-Monitor für SFT-Training auf RunPod.

Zeigt in einem lokalen Terminal parallel:
  - Tail des /workspace/sft.log (Trainer-Output, geparste Steps & Loss)
  - nvidia-smi live (Util, VRAM, Power, Temp)
  - Ableitungen: aktueller Step, ETA, durchschnittliche s/it

Usage:
  pip install rich
  python training/runpod/watch.py

Stoppen mit Ctrl+C — Training auf dem Pod läuft weiter (nohup).
"""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("rich fehlt — installiere mit: pip install rich")

# ---------- Pod-Verbindung (aus ENV überschreibbar) ----------
SSH_HOST = os.environ.get("POD_HOST", "38.143.35.131")
SSH_PORT = os.environ.get("POD_PORT", "15960")
SSH_USER = os.environ.get("POD_USER", "root")
SSH_KEY = os.environ.get("POD_KEY", os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go"))
LOG_PATH = os.environ.get("POD_LOG", "/workspace/sft.log")
GPU_POLL_SEC = float(os.environ.get("GPU_POLL_SEC", "2"))

SSH_BASE = [
    "ssh", "-i", SSH_KEY,
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-p", SSH_PORT,
    f"{SSH_USER}@{SSH_HOST}",
]

# ---------- State ----------
@dataclass
class TrainState:
    current_step: int = 0
    total_steps: int = 0
    last_loss: Optional[float] = None
    last_lr: Optional[float] = None
    last_grad_norm: Optional[float] = None
    step_times: deque = field(default_factory=lambda: deque(maxlen=20))
    started_at: Optional[datetime] = None
    training_started_at: Optional[datetime] = None  # Zeitpunkt Step 1 begann
    last_line: str = ""
    log_tail: deque = field(default_factory=lambda: deque(maxlen=18))
    status: str = "verbinde…"
    phase: str = "Init"

@dataclass
class GpuState:
    name: str = "—"
    util_gpu: int = 0
    util_mem: int = 0
    mem_used: int = 0
    mem_total: int = 0
    temp: int = 0
    power: float = 0.0
    power_max: float = 0.0
    last_poll: Optional[datetime] = None
    alive: bool = False
    last_error: str = ""

TRAIN = TrainState()
GPU = GpuState()
STOP = threading.Event()

# ---------- Parser ----------
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[([^\]]+)\]")
LOSS_RE = re.compile(r"'loss'\s*:\s*([0-9.eE+-]+)")
LR_RE = re.compile(r"'learning_rate'\s*:\s*([0-9.eE+-]+)")
GRAD_RE = re.compile(r"'grad_norm'\s*:\s*([0-9.eE+-]+)")
TOTAL_STEPS_RE = re.compile(r"Total steps\s*=\s*([0-9,]+)")
STEP_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)(s|min)/it")

PHASE_HINTS = [
    ("Loading weights", "Weights laden (Shards)"),
    ("Fast downloading", "Model-Download"),
    ("tokenizing",       "Tokenisierung"),
    ("Map",              "Dataset-Preprocessing"),
    ("packing",          "Packing"),
    ("Num examples",     "Trainer init"),
]

def detect_phase(line: str) -> Optional[str]:
    low = line.lower()
    for needle, label in PHASE_HINTS:
        if needle.lower() in low:
            return label
    return None

def parse_log_line(line: str) -> None:
    line = line.rstrip("\n")
    if not line:
        return
    TRAIN.last_line = line
    TRAIN.log_tail.append(line)

    if m := TOTAL_STEPS_RE.search(line):
        TRAIN.total_steps = int(m.group(1).replace(",", ""))
        if TRAIN.started_at is None:
            TRAIN.started_at = datetime.now()

    if phase := detect_phase(line):
        TRAIN.phase = phase

    # Step nur akzeptieren wenn total mit bekannten total_steps übereinstimmt —
    # sonst matchen auch "Loading weights: 531/531" und ds.map-Progress-Bars.
    is_train_step_line = False
    if (m := STEP_RE.search(line)) and TRAIN.total_steps > 0:
        step = int(m.group(1))
        total = int(m.group(2))
        if total == TRAIN.total_steps:
            is_train_step_line = True
            TRAIN.phase = "Training"
            if step > TRAIN.current_step:
                TRAIN.current_step = step
                if TRAIN.training_started_at is None and step >= 1:
                    TRAIN.training_started_at = datetime.now()
                if TRAIN.started_at is None:
                    TRAIN.started_at = datetime.now()

    # Step-times nur aus Trainings-Progress-Bars übernehmen, nicht aus Weight-Loading
    if is_train_step_line and (m := STEP_TIME_RE.search(line)):
        val = float(m.group(1))
        unit = m.group(2)
        secs = val * 60 if unit == "min" else val
        TRAIN.step_times.append(secs)

    if m := LOSS_RE.search(line):
        TRAIN.last_loss = float(m.group(1))
    if m := LR_RE.search(line):
        TRAIN.last_lr = float(m.group(1))
    if m := GRAD_RE.search(line):
        TRAIN.last_grad_norm = float(m.group(1))

    lower = line.lower()
    if "traceback" in lower or "cuda out of memory" in lower:
        TRAIN.status = "[bold red]FEHLER im Log[/bold red]"
    elif "sft done" in lower:
        TRAIN.status = "[bold green]FERTIG[/bold green]"
    elif TRAIN.current_step > 0 and TRAIN.total_steps > 0:
        TRAIN.status = f"[green]läuft · Step {TRAIN.current_step} von {TRAIN.total_steps}[/green]"
    elif TRAIN.total_steps > 0:
        TRAIN.status = f"[yellow]{TRAIN.phase} · wartet auf ersten Step…[/yellow]"
    else:
        TRAIN.status = f"[yellow]{TRAIN.phase}…[/yellow]"

# ---------- Worker ----------
def log_tail_worker() -> None:
    cmd = SSH_BASE + [f"tail -F -n 200 {shlex.quote(LOG_PATH)}"]
    while not STOP.is_set():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if STOP.is_set():
                    proc.terminate()
                    break
                parse_log_line(line)
            rc = proc.wait()
            if rc != 0 and not STOP.is_set():
                TRAIN.log_tail.append(f"[ssh tail exit rc={rc} — reconnecting in 3s]")
        except Exception as e:
            TRAIN.log_tail.append(f"[ssh tail error: {e}]")
        if not STOP.is_set():
            time.sleep(3)

NVIDIA_QUERY = ",".join([
    "name", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "temperature.gpu",
    "power.draw", "power.limit",
])

def gpu_poll_worker() -> None:
    remote = f"nvidia-smi --query-gpu={NVIDIA_QUERY} --format=csv,noheader,nounits"
    cmd = SSH_BASE + [remote]
    while not STOP.is_set():
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=20, text=True)
            parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
            if len(parts) >= 8:
                GPU.name = parts[0]
                GPU.util_gpu = int(float(parts[1]))
                GPU.util_mem = int(float(parts[2]))
                GPU.mem_used = int(float(parts[3]))
                GPU.mem_total = int(float(parts[4]))
                GPU.temp = int(float(parts[5]))
                GPU.power = float(parts[6])
                GPU.power_max = float(parts[7])
                GPU.last_poll = datetime.now()
                GPU.alive = True
                GPU.last_error = ""
        except subprocess.CalledProcessError as e:
            GPU.alive = False
            GPU.last_error = (e.output or "").strip().splitlines()[-1] if e.output else f"rc={e.returncode}"
        except subprocess.TimeoutExpired:
            GPU.alive = False
            GPU.last_error = "SSH timeout (>20s)"
        except Exception as e:
            GPU.alive = False
            GPU.last_error = f"{type(e).__name__}: {e}"
        STOP.wait(GPU_POLL_SEC)

# ---------- Rendering ----------
def fmt_td(sec: float) -> str:
    if sec < 0 or sec != sec:
        return "—"
    return str(timedelta(seconds=int(sec)))

def header_panel() -> Panel:
    avg = (sum(TRAIN.step_times) / len(TRAIN.step_times)) if TRAIN.step_times else 0
    done = max(0, min(TRAIN.current_step, TRAIN.total_steps)) if TRAIN.total_steps else 0
    total = TRAIN.total_steps or 0
    remaining = max(total - done, 0)
    eta = remaining * avg if avg and total else 0
    pct = (done / total * 100) if total else 0

    bar_w = 30
    filled = int(bar_w * pct / 100) if total else 0
    bar = "█" * filled + "░" * (bar_w - filled)

    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan"); t.add_column()
    t.add_row("Phase", TRAIN.phase)
    t.add_row("Status", TRAIN.status)

    if total > 0:
        progress_text = f"[cyan]{bar}[/cyan]  Step {done} von {total}  ({pct:5.1f}%)"
    else:
        progress_text = "[dim]noch kein Trainings-Start (Setup läuft)[/dim]"
    t.add_row("Progress", progress_text)

    t.add_row("Step-time (ø20)", f"{avg:.1f}s/it" if avg else "[dim]—[/dim]")
    t.add_row("ETA", fmt_td(eta) if eta else "[dim]—[/dim]")
    if TRAIN.training_started_at:
        t.add_row("Training seit", fmt_td((datetime.now() - TRAIN.training_started_at).total_seconds()))
    elif TRAIN.started_at:
        t.add_row("Setup seit", fmt_td((datetime.now() - TRAIN.started_at).total_seconds()))
    return Panel(t, title="[bold]SFT Training[/bold]", border_style="cyan")

def metrics_panel() -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold magenta"); t.add_column()
    t.add_row("Loss", f"{TRAIN.last_loss:.4f}" if TRAIN.last_loss is not None else "—")
    t.add_row("LR", f"{TRAIN.last_lr:.2e}" if TRAIN.last_lr is not None else "—")
    t.add_row("Grad-norm", f"{TRAIN.last_grad_norm:.3f}" if TRAIN.last_grad_norm is not None else "—")
    return Panel(t, title="[bold]Trainer-Metrics[/bold]", border_style="magenta")

def gpu_panel() -> Panel:
    if not GPU.alive:
        msg = f"[red]keine GPU-Daten[/red]\n[dim]{GPU.last_error or 'verbinde…'}[/dim]"
        return Panel(msg, title="[bold]GPU[/bold]", border_style="red")
    mem_pct = (GPU.mem_used / GPU.mem_total * 100) if GPU.mem_total else 0
    pwr_pct = (GPU.power / GPU.power_max * 100) if GPU.power_max else 0
    util_color = "green" if GPU.util_gpu > 70 else ("yellow" if GPU.util_gpu > 30 else "red")
    mem_color = "red" if mem_pct > 95 else ("yellow" if mem_pct > 85 else "green")

    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold yellow"); t.add_column()
    t.add_row("Device", GPU.name)
    t.add_row("GPU-Util", f"[{util_color}]{GPU.util_gpu:3d}%[/{util_color}]  (MemBus {GPU.util_mem}%)")
    t.add_row("VRAM", f"[{mem_color}]{GPU.mem_used:>6d} / {GPU.mem_total} MiB ({mem_pct:4.1f}%)[/{mem_color}]")
    t.add_row("Power", f"{GPU.power:5.1f} / {GPU.power_max:.0f} W ({pwr_pct:4.1f}%)")
    t.add_row("Temp", f"{GPU.temp}°C")
    if GPU.last_poll:
        t.add_row("Letzter Poll", GPU.last_poll.strftime("%H:%M:%S"))
    return Panel(t, title="[bold]GPU[/bold]", border_style="yellow")

def log_panel() -> Panel:
    if not TRAIN.log_tail:
        body = Text("warte auf Log-Zeilen…", style="dim")
    else:
        lines = list(TRAIN.log_tail)[-16:]
        body = Text()
        for ln in lines:
            style = "white"
            low = ln.lower()
            if "error" in low or "traceback" in low or "oom" in low:
                style = "bold red"
            elif "warn" in low:
                style = "yellow"
            elif "loss" in low and "grad_norm" in low:
                style = "bold green"
            elif "/" in ln and "it/s" in ln or "s/it" in ln:
                style = "cyan"
            body.append(ln[-240:] + "\n", style=style)
    return Panel(body, title=f"[bold]{LOG_PATH} (letzte Zeilen)[/bold]", border_style="white")

def build_view() -> Group:
    top = Layout()
    top.split_row(
        Layout(header_panel(), name="h", ratio=2),
        Layout(metrics_panel(), name="m", ratio=1),
        Layout(gpu_panel(), name="g", ratio=2),
    )
    return Group(top, log_panel())

# ---------- Main ----------
def preflight_ssh(console: Console) -> bool:
    """Teste SSH-Verbindung bevor TUI startet — zeigt klaren Fehler falls Key/Host falsch."""
    console.print(f"[dim]Preflight: teste SSH zu {SSH_USER}@{SSH_HOST}:{SSH_PORT}…[/dim]")
    cmd = SSH_BASE + ["echo ok"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15, text=True)
        if "ok" in out:
            console.print("[green]✓ SSH ok[/green]")
            return True
        console.print(f"[red]SSH unerwartete Ausgabe:[/red] {out!r}")
        return False
    except subprocess.CalledProcessError as e:
        console.print("[red]✗ SSH fehlgeschlagen:[/red]")
        console.print(f"[red]{e.output.strip() if e.output else f'rc={e.returncode}'}[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[red]✗ SSH-Timeout nach 15s[/red]")
        return False

def main() -> None:
    # Fallback-Konsole (force_terminal), damit Rich auch in VSCode-Terminal live rendert
    console = Console(force_terminal=True)

    if not preflight_ssh(console):
        console.print("\n[bold yellow]Hinweis:[/bold yellow] prüfe POD_HOST/POD_PORT/POD_KEY-ENV oder teste manuell:")
        console.print(f"[dim]  ssh -i {SSH_KEY} -p {SSH_PORT} {SSH_USER}@{SSH_HOST} 'echo ok'[/dim]")
        sys.exit(1)

    console.print(f"[dim]starte Live-Monitor — log={LOG_PATH}  gpu-poll={GPU_POLL_SEC}s[/dim]")
    console.print("[dim]Ctrl+C zum Beenden · Training im Pod läuft unabhängig weiter[/dim]\n")
    time.sleep(0.6)

    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())

    t_log = threading.Thread(target=log_tail_worker, daemon=True)
    t_gpu = threading.Thread(target=gpu_poll_worker, daemon=True)
    t_log.start()
    t_gpu.start()

    try:
        # screen=True → alt-screen buffer, sauberes Clearing in jedem Terminal
        # auto_refresh=True → Live refresht selbst, kein manuelles update() nötig
        with Live(build_view(), console=console, refresh_per_second=4,
                  screen=True, auto_refresh=True) as live:
            while not STOP.is_set():
                live.update(build_view())
                time.sleep(0.25)
    except KeyboardInterrupt:
        STOP.set()
    finally:
        STOP.set()
        console.print("\n[dim]Monitor beendet — Training im Pod läuft weiter.[/dim]")

if __name__ == "__main__":
    main()
