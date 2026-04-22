"""Live monitor for SFT training on RunPod.

Shows in a local terminal in parallel:
  - tail of /workspace/sft.log (trainer output, parsed steps & loss)
  - nvidia-smi live (util, VRAM, power, temp)
  - disk usage (/workspace, checkpoints, HF cache)
  - cost tracking ($/h × elapsed + projection)
  - derived metrics: current step, ETA, avg s/it, tokens/sec,
    loss & grad-norm sparklines, eval-loss trend, WandB run URL.

Usage:
  pip install rich
  # Default: direct-TCP (auto-detected via runpodctl)
  python training/runpod/watch.py

  # RunPod proxy fallback (no public-TCP mapping, PTY required):
  POD_PROXY_USER=alpqgkmttz9dhl-64411961 python training/runpod/watch.py

Quit with Ctrl+C — training on the pod keeps running (nohup).
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
    sys.exit("rich missing — install with: pip install rich")

# ---------- Pod connection ----------
# Two connection modes, picked automatically:
#   1. Direct TCP (default): host+port come from env (POD_HOST/POD_PORT) or are
#      auto-detected via `runpodctl get pod <id>`. Fast, no PTY.
#   2. RunPod SSH proxy: set POD_PROXY_USER="<pod-id>-<account-id>" — then we
#      connect as POD_PROXY_USER@ssh.runpod.io. Needed when the pod has no
#      public-TCP SSH mapping (the proxy requires a PTY — we pass -tt).
# POD_ID is the stable identifier for direct-TCP auto-detection.
POD_ID = os.environ.get("POD_ID", "3xc1b6nzhkgmqq")
POD_PROXY_USER = os.environ.get("POD_PROXY_USER")  # e.g. "alpqgkmttz9dhl-64411961"
SSH_USER = os.environ.get("POD_USER", "root")
SSH_KEY = os.environ.get("POD_KEY", os.path.expanduser("~/.ssh/id_ed25519"))
LOG_PATH = os.environ.get("POD_LOG", "/workspace/pipeline.log")
CKPT_DIR = os.environ.get("POD_CKPT", "/workspace/checkpoints/sft-phase-a")
WORKSPACE_PATH = os.environ.get("POD_WORKSPACE", "/workspace")
GPU_POLL_SEC = float(os.environ.get("GPU_POLL_SEC", "2"))
DISK_POLL_SEC = float(os.environ.get("DISK_POLL_SEC", "30"))
# Pod hourly rate for the cost tracker (fallback $2.59 for H100 NVL reserved).
HOURLY_RATE = float(os.environ.get("POD_HOURLY_RATE", "2.59"))
SAVE_STEPS = int(os.environ.get("POD_SAVE_STEPS", "20"))  # used to predict next save
# Tokens/s calculation: effective batch * max_seq_length ≈ tokens seen per step.
# SFT default: grad_accum 64 * bsz 1 * max_seq_length 6144 = 393,216.
EFFECTIVE_BATCH_TOKENS = int(os.environ.get("POD_TOKENS_PER_STEP", str(64 * 6144)))

_PORT_RE = re.compile(r"(\d+\.\d+\.\d+\.\d+):(\d+)->22\s*\(pub,tcp\)")

def detect_ssh_endpoint() -> tuple[str, str]:
    """Return (host, port) for SSH — env overrides win, otherwise ask runpodctl."""
    host = os.environ.get("POD_HOST")
    port = os.environ.get("POD_PORT")
    if host and port:
        return host, port
    try:
        out = subprocess.check_output(
            ["runpodctl", "get", "pod", POD_ID, "-a"],
            stderr=subprocess.STDOUT, timeout=15, text=True,
        )
        m = _PORT_RE.search(out)
        if m:
            return m.group(1), m.group(2)
    except FileNotFoundError:
        sys.exit("runpodctl not installed — install it or set POD_HOST/POD_PORT env vars")
    except subprocess.CalledProcessError as e:
        sys.exit(f"runpodctl error: {(e.output or '').strip()[:200]}")
    except subprocess.TimeoutExpired:
        sys.exit("runpodctl get pod → timeout (>15s)")
    sys.exit(f"could not find pub-tcp SSH port for {POD_ID} in runpodctl output")

if POD_PROXY_USER:
    # RunPod proxy: <pod-id>-<account-id>@ssh.runpod.io, no -p, PTY required.
    # BatchMode is intentionally omitted — the proxy needs the key-agent flow.
    SSH_HOST = "ssh.runpod.io"
    SSH_PORT = "22"
    SSH_BASE = [
        "ssh", "-tt", "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=30",
        "-o", "ConnectTimeout=15",
        f"{POD_PROXY_USER}@{SSH_HOST}",
    ]
else:
    SSH_HOST, SSH_PORT = detect_ssh_endpoint()
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
    training_started_at: Optional[datetime] = None  # timestamp when step 1 began
    last_line: str = ""
    log_tail: deque = field(default_factory=lambda: deque(maxlen=18))
    status: str = "connecting…"
    phase: str = "Init"
    # Loss history for sparkline (step, loss)
    loss_history: deque = field(default_factory=lambda: deque(maxlen=200))
    # Grad-norm history — shows training stability (spikes = LR issues)
    grad_history: deque = field(default_factory=lambda: deque(maxlen=200))
    # WandB URL parsed from log (only set if WANDB_API_KEY was active)
    wandb_url: Optional[str] = None
    # Eval metrics (from eval_steps)
    last_eval_loss: Optional[float] = None
    last_eval_step: int = 0
    last_eval_runtime: Optional[float] = None
    first_eval_loss: Optional[float] = None  # baseline for the delta arrow
    # Agent-eval metrics (from AgentEvalCallback — see _agent_eval.py)
    last_agent_parse_rate: Optional[float] = None
    last_agent_name_match: Optional[float] = None
    last_agent_avg_tokens: Optional[float] = None
    last_agent_step: int = 0
    first_agent_avg_tokens: Optional[float] = None  # baseline for Mac-speed trend
    # Checkpoint tracking
    saves: list = field(default_factory=list)  # sorted step numbers
    last_save_step: Optional[int] = None
    last_save_at: Optional[datetime] = None
    hub_push_last: Optional[datetime] = None

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
    sm_clock: int = 0  # MHz
    last_poll: Optional[datetime] = None
    alive: bool = False
    last_error: str = ""

@dataclass
class DiskState:
    workspace_used_gb: float = 0.0
    workspace_total_gb: float = 0.0
    ckpt_size_gb: float = 0.0
    hf_cache_gb: float = 0.0
    last_poll: Optional[datetime] = None
    alive: bool = False

TRAIN = TrainState()
GPU = GpuState()
DISK = DiskState()
STOP = threading.Event()

# ---------- Parser ----------
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[([^\]]+)\]")
# Trainer logs numbers as either bare or quoted depending on log config.
# Accept both: 'loss': 0.7 AND 'loss': '0.7'
LOSS_RE = re.compile(r"'loss'\s*:\s*'?([0-9.eE+-]+)'?")
LR_RE = re.compile(r"'learning_rate'\s*:\s*'?([0-9.eE+-]+)'?")
GRAD_RE = re.compile(r"'grad_norm'\s*:\s*'?([0-9.eE+-]+)'?")
EPOCH_RE = re.compile(r"'epoch'\s*:\s*'?([0-9.eE+-]+)'?")
EVAL_LOSS_RE = re.compile(r"'eval_loss'\s*:\s*'?([0-9.eE+-]+)'?")
EVAL_RUNTIME_RE = re.compile(r"'eval_runtime'\s*:\s*'?([0-9.eE+-]+)'?")
# AgentEvalCallback logs lines like:
#   [agent-eval] step=100 {'agent/parse_rate': 0.94, 'agent/name_match': 0.88, 'agent/avg_output_tokens': 72.5}
AGENT_STEP_RE   = re.compile(r"\[agent-eval\]\s+step=(\d+)")
AGENT_PARSE_RE  = re.compile(r"'agent/parse_rate'\s*:\s*([0-9.eE+-]+)")
AGENT_NAME_RE   = re.compile(r"'agent/name_match'\s*:\s*([0-9.eE+-]+)")
AGENT_TOKENS_RE = re.compile(r"'agent/avg_output_tokens'\s*:\s*([0-9.eE+-]+)")
TOTAL_STEPS_RE = re.compile(r"Total steps\s*=\s*([0-9,]+)")
STEP_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)(s|min)/it")
SAVE_STEP_RE = re.compile(r"checkpoint-(\d+)")
WANDB_URL_RE = re.compile(r"https://wandb\.ai/\S+?/runs/\S+")

PHASE_HINTS = [
    ("PHASE 01",         "Dataset-Build"),
    ("PHASE 02",         "SFT Training"),
    ("PHASE 03",         "DPO Training"),
    ("PHASE 04",         "LongCtx Training"),
    ("PHASE 05",         "Export / Merge"),
    ("Loading weights",  "Loading weights (shards)"),
    ("Fast downloading", "Model download"),
    ("tokenizing",       "Tokenization"),
    ("Map",              "Dataset preprocessing"),
    ("packing",          "Packing"),
    ("Num examples",     "Trainer init"),
    ("loading SFT",      "Dataset-Build: SFT sources"),
    ("loading DPO",      "Dataset-Build: DPO sources"),
    ("SFT total",        "Dataset-Build: SFT done"),
    ("DPO total",        "Dataset-Build: DPO done"),
    ("LongCtx total",    "Dataset-Build: LongCtx done"),
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

    # Only accept a step match if total == known total_steps — otherwise
    # "Loading weights: 531/531" and ds.map progress bars would all match.
    # Fallback: if we're already in Training phase (loss seen) but total_steps
    # was never parsed (e.g. the banner scrolled out of the tail buffer),
    # accept any plausible X/Y with Y > 10 as the training total.
    is_train_step_line = False
    if m := STEP_RE.search(line):
        step = int(m.group(1))
        total = int(m.group(2))
        if TRAIN.total_steps == 0 and TRAIN.phase == "Training" and total > 10 and step <= total:
            TRAIN.total_steps = total
        if TRAIN.total_steps > 0 and total == TRAIN.total_steps:
            is_train_step_line = True
            TRAIN.phase = "Training"
            if step > TRAIN.current_step:
                TRAIN.current_step = step
                if TRAIN.training_started_at is None and step >= 1:
                    TRAIN.training_started_at = datetime.now()
                if TRAIN.started_at is None:
                    TRAIN.started_at = datetime.now()

    # Step-time samples only from training progress bars, not weight-loading.
    if is_train_step_line and (m := STEP_TIME_RE.search(line)):
        val = float(m.group(1))
        unit = m.group(2)
        secs = val * 60 if unit == "min" else val
        TRAIN.step_times.append(secs)

    if m := LOSS_RE.search(line):
        loss_val = float(m.group(1))
        TRAIN.last_loss = loss_val
        # A 'loss' line proves we're in training — fix up phase/timers even if
        # the "Total steps = N" banner was already out of the log tail buffer.
        TRAIN.phase = "Training"
        if TRAIN.training_started_at is None:
            TRAIN.training_started_at = datetime.now()
        if TRAIN.started_at is None:
            TRAIN.started_at = datetime.now()
        if TRAIN.current_step > 0:
            # Only train loss — eval_loss has its own regex above.
            if not TRAIN.loss_history or TRAIN.loss_history[-1][0] != TRAIN.current_step:
                TRAIN.loss_history.append((TRAIN.current_step, loss_val))
    if m := LR_RE.search(line):
        TRAIN.last_lr = float(m.group(1))
    if m := GRAD_RE.search(line):
        gn = float(m.group(1))
        TRAIN.last_grad_norm = gn
        if TRAIN.current_step > 0:
            if not TRAIN.grad_history or TRAIN.grad_history[-1][0] != TRAIN.current_step:
                TRAIN.grad_history.append((TRAIN.current_step, gn))
    if m := EVAL_LOSS_RE.search(line):
        val = float(m.group(1))
        TRAIN.last_eval_loss = val
        TRAIN.last_eval_step = TRAIN.current_step
        if TRAIN.first_eval_loss is None:
            TRAIN.first_eval_loss = val
    if m := EVAL_RUNTIME_RE.search(line):
        TRAIN.last_eval_runtime = float(m.group(1))

    # Agent-eval metrics — emitted by AgentEvalCallback after each evaluate().
    if "[agent-eval]" in line:
        if m := AGENT_STEP_RE.search(line):
            TRAIN.last_agent_step = int(m.group(1))
        if m := AGENT_PARSE_RE.search(line):
            TRAIN.last_agent_parse_rate = float(m.group(1))
        if m := AGENT_NAME_RE.search(line):
            TRAIN.last_agent_name_match = float(m.group(1))
        if m := AGENT_TOKENS_RE.search(line):
            val = float(m.group(1))
            TRAIN.last_agent_avg_tokens = val
            if TRAIN.first_agent_avg_tokens is None:
                TRAIN.first_agent_avg_tokens = val

    # Checkpoint save detection — trainer logs "Saving model checkpoint to .../checkpoint-N".
    if "Saving" in line and "checkpoint" in line.lower():
        if m := SAVE_STEP_RE.search(line):
            s = int(m.group(1))
            if s not in TRAIN.saves:
                TRAIN.saves.append(s)
                TRAIN.saves.sort()
                TRAIN.last_save_step = s
                TRAIN.last_save_at = datetime.now()
    # Hub push detection
    if ("Upload" in line or "Pushing" in line) and "Hub" in line:
        TRAIN.hub_push_last = datetime.now()
    # Capture the WandB URL once (stays stable for the whole run).
    if TRAIN.wandb_url is None:
        if m := WANDB_URL_RE.search(line):
            TRAIN.wandb_url = m.group(0)

    lower = line.lower()
    if "traceback" in lower or "cuda out of memory" in lower:
        TRAIN.status = "[bold red]ERROR in log[/bold red]"
    elif "sft done" in lower:
        TRAIN.status = "[bold green]DONE[/bold green]"
    elif TRAIN.current_step > 0 and TRAIN.total_steps > 0:
        TRAIN.status = f"[green]running · step {TRAIN.current_step} of {TRAIN.total_steps}[/green]"
    elif TRAIN.total_steps > 0:
        TRAIN.status = f"[yellow]{TRAIN.phase} · waiting for first step…[/yellow]"
    else:
        TRAIN.status = f"[yellow]{TRAIN.phase}…[/yellow]"

# ---------- Worker ----------
def log_tail_worker() -> None:
    # Use a large backfill window so early lines like "Total steps = 166" are
    # still visible when we attach to a long-running training run. 5000 lines
    # covers several hours of trainer output.
    cmd = SSH_BASE + [f"tail -F -n 5000 {shlex.quote(LOG_PATH)}"]
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
    "power.draw", "power.limit", "clocks.sm",
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
                if len(parts) >= 9:
                    try:
                        GPU.sm_clock = int(float(parts[8]))
                    except ValueError:
                        GPU.sm_clock = 0
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

def disk_poll_worker() -> None:
    """Polls disk info (checkpoints, HF cache, /workspace fill level) at a slow cadence."""
    ws = shlex.quote(WORKSPACE_PATH)
    ckpt = shlex.quote(CKPT_DIR)
    hf = shlex.quote(f"{WORKSPACE_PATH}/huggingface")
    # One combined SSH query: df + 2× du. `du -BG` reports GB (rounded).
    remote = (
        f"df -BG {ws} | tail -1 | awk '{{print $2\" \"$3}}' | tr -d G; "
        f"echo '---'; du -sBG {ckpt} 2>/dev/null | awk '{{print $1}}' | tr -d G || echo 0; "
        f"echo '---'; du -sBG {hf} 2>/dev/null | awk '{{print $1}}' | tr -d G || echo 0"
    )
    cmd = SSH_BASE + [remote]
    while not STOP.is_set():
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30, text=True)
            blocks = out.strip().split("---")
            if len(blocks) >= 1:
                df_parts = blocks[0].strip().split()
                if len(df_parts) >= 2:
                    DISK.workspace_total_gb = float(df_parts[0])
                    DISK.workspace_used_gb = float(df_parts[1])
            if len(blocks) >= 2:
                try:
                    DISK.ckpt_size_gb = float(blocks[1].strip() or "0")
                except ValueError:
                    DISK.ckpt_size_gb = 0
            if len(blocks) >= 3:
                try:
                    DISK.hf_cache_gb = float(blocks[2].strip() or "0")
                except ValueError:
                    DISK.hf_cache_gb = 0
            DISK.last_poll = datetime.now()
            DISK.alive = True
        except Exception:
            DISK.alive = False
        STOP.wait(DISK_POLL_SEC)

# ---------- Rendering ----------
def fmt_td(sec: float) -> str:
    if sec < 0 or sec != sec:
        return "—"
    return str(timedelta(seconds=int(sec)))

def sparkline(values: list, width: int = 60) -> str:
    """Compact Unicode sparkline from a list of floats. Higher values = taller blocks."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    vmin, vmax = min(values), max(values)
    rng = (vmax - vmin) or 1e-9
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(7, int((v - vmin) / rng * 7))] for v in sampled)

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
        progress_text = f"[cyan]{bar}[/cyan]  Step {done} of {total}  ({pct:5.1f}%)"
    else:
        progress_text = "[dim]training not started yet (setup running)[/dim]"
    t.add_row("Progress", progress_text)

    t.add_row("Step time (ø20)", f"{avg:.1f}s/it" if avg else "[dim]—[/dim]")
    # Throughput: tokens/sec derived from effective batch × seq-len per step
    if avg:
        tok_per_sec = EFFECTIVE_BATCH_TOKENS / avg
        t.add_row("Throughput", f"{tok_per_sec:,.0f} tok/s  ({EFFECTIVE_BATCH_TOKENS/1000:.0f}k tok/step)")
    t.add_row("ETA", fmt_td(eta) if eta else "[dim]—[/dim]")
    if TRAIN.training_started_at:
        t.add_row("Training for", fmt_td((datetime.now() - TRAIN.training_started_at).total_seconds()))
    elif TRAIN.started_at:
        t.add_row("Setup for", fmt_td((datetime.now() - TRAIN.started_at).total_seconds()))
    return Panel(t, title="[bold]Fine-Tuning Progress[/bold]", border_style="cyan")

def metrics_panel() -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold magenta"); t.add_column()
    # If we're in a non-training phase, show phase info instead of empty metrics
    is_training_phase = TRAIN.phase in ("Training", "SFT Training", "DPO Training", "LongCtx Training")
    if not is_training_phase and TRAIN.last_loss is None:
        t.add_row("Phase", TRAIN.phase or "[dim]starting…[/dim]")
        t.add_row("Metrics", "[dim]will appear once training begins[/dim]")
    else:
        t.add_row("Loss", f"{TRAIN.last_loss:.4f}" if TRAIN.last_loss is not None else "[dim]—[/dim]")
        t.add_row("LR", f"{TRAIN.last_lr:.2e}" if TRAIN.last_lr is not None else "[dim]—[/dim]")
        t.add_row("Grad-norm", f"{TRAIN.last_grad_norm:.3f}" if TRAIN.last_grad_norm is not None else "[dim]—[/dim]")
    t.add_row("", "")  # Separator
    if TRAIN.last_eval_loss is not None:
        trend = ""
        if TRAIN.first_eval_loss and TRAIN.last_eval_loss != TRAIN.first_eval_loss:
            delta = TRAIN.last_eval_loss - TRAIN.first_eval_loss
            arrow = "↓" if delta < 0 else "↑"
            color = "green" if delta < 0 else "red"
            trend = f" [{color}]{arrow}{abs(delta):.3f}[/{color}]"
        t.add_row("Eval loss", f"{TRAIN.last_eval_loss:.4f} (step {TRAIN.last_eval_step}){trend}")
        if TRAIN.last_eval_runtime:
            t.add_row("Eval time", f"{TRAIN.last_eval_runtime:.1f}s")
    else:
        t.add_row("Eval", "[dim]starts at step 20[/dim]" if is_training_phase else "[dim]—[/dim]")

    # ── Agent-Eval (Tool-Call-Qualität + Mac-Speed-Proxy) ─────────────────
    has_agent = (
        TRAIN.last_agent_parse_rate is not None
        or TRAIN.last_agent_name_match is not None
        or TRAIN.last_agent_avg_tokens is not None
    )
    if has_agent:
        t.add_row("", "")  # Separator vor Agent-Block
        step_suffix = f" (step {TRAIN.last_agent_step})" if TRAIN.last_agent_step else ""
        if TRAIN.last_agent_parse_rate is not None:
            pct = TRAIN.last_agent_parse_rate * 100
            color = "green" if pct >= 90 else "yellow" if pct >= 70 else "red"
            t.add_row("Agent parse", f"[{color}]{pct:5.1f}%[/{color}]{step_suffix}")
        if TRAIN.last_agent_name_match is not None:
            pct = TRAIN.last_agent_name_match * 100
            color = "green" if pct >= 85 else "yellow" if pct >= 60 else "red"
            t.add_row("Tool-name match", f"[{color}]{pct:5.1f}%[/{color}]")
        if TRAIN.last_agent_avg_tokens is not None:
            trend = ""
            if (TRAIN.first_agent_avg_tokens is not None
                    and TRAIN.first_agent_avg_tokens != TRAIN.last_agent_avg_tokens):
                delta = TRAIN.last_agent_avg_tokens - TRAIN.first_agent_avg_tokens
                # Weniger Tokens = schneller auf Mac → ↓ ist grün.
                arrow = "↓" if delta < 0 else "↑"
                color = "green" if delta < 0 else "red"
                trend = f" [{color}]{arrow}{abs(delta):.1f}[/{color}]"
            t.add_row("Out-tokens/call", f"{TRAIN.last_agent_avg_tokens:.1f}{trend}")

    if TRAIN.wandb_url:
        short = TRAIN.wandb_url.split("/runs/", 1)[-1][:24]
        t.add_row("WandB", f"[link={TRAIN.wandb_url}]…/runs/{short}[/link]")
    return Panel(t, title="[bold]Metrics[/bold]", border_style="magenta")

def gpu_panel() -> Panel:
    if not GPU.alive:
        msg = f"[red]no GPU data[/red]\n[dim]{GPU.last_error or 'connecting…'}[/dim]"
        return Panel(msg, title="[bold]GPU[/bold]", border_style="red")
    mem_pct = (GPU.mem_used / GPU.mem_total * 100) if GPU.mem_total else 0
    pwr_pct = (GPU.power / GPU.power_max * 100) if GPU.power_max else 0
    util_color = "green" if GPU.util_gpu > 70 else ("yellow" if GPU.util_gpu > 30 else "red")
    mem_color = "red" if mem_pct > 95 else ("yellow" if mem_pct > 85 else "green")

    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold yellow"); t.add_column()
    t.add_row("Device", GPU.name)
    t.add_row("GPU util", f"[{util_color}]{GPU.util_gpu:3d}%[/{util_color}]  (MemBus {GPU.util_mem}%)")
    t.add_row("VRAM", f"[{mem_color}]{GPU.mem_used:>6d} / {GPU.mem_total} MiB ({mem_pct:4.1f}%)[/{mem_color}]")
    t.add_row("Power", f"{GPU.power:5.1f} / {GPU.power_max:.0f} W ({pwr_pct:4.1f}%)")
    if GPU.sm_clock:
        t.add_row("SM clock", f"{GPU.sm_clock} MHz")
    t.add_row("Temp", f"{GPU.temp}°C")
    if GPU.last_poll:
        t.add_row("Last poll", GPU.last_poll.strftime("%H:%M:%S"))
    return Panel(t, title="[bold]GPU[/bold]", border_style="yellow")

def _sparkline_row(history, color, label, width=60):
    """Render one sparkline line with stats summary."""
    if not history:
        return None, f"[dim]{label}: no data yet[/dim]"
    values = [v for _, v in history]
    steps_seen = [s for s, _ in history]
    spark = sparkline(values, width=width)
    first, last = values[0], values[-1]
    vmin, vmax = min(values), max(values)
    delta = last - first
    arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "·")
    arrow_color = "green" if delta < 0 else ("red" if delta > 0 else "white")
    summary = (
        f"[dim]{label}: {first:.3f} → {last:.3f}  "
        f"[{arrow_color}]{arrow}{abs(delta):.3f}[/{arrow_color}]  ·  "
        f"min {vmin:.3f}  max {vmax:.3f}  ·  step {steps_seen[0]}–{steps_seen[-1]}  (n={len(values)})[/dim]"
    )
    return f"[{color}]{spark}[/{color}]", summary

def loss_panel() -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column()
    has_any = False
    if TRAIN.loss_history:
        has_any = True
        spark, summary = _sparkline_row(TRAIN.loss_history, "green", "loss")
        t.add_row(spark)
        t.add_row(summary)
    if TRAIN.grad_history:
        has_any = True
        if TRAIN.loss_history:
            t.add_row("")  # spacer
        spark, summary = _sparkline_row(TRAIN.grad_history, "magenta", "grad-norm")
        t.add_row(spark)
        t.add_row(summary)
    if not has_any:
        phase_hint = f" — current phase: {TRAIN.phase}" if TRAIN.phase else ""
        t.add_row(f"[dim]no training metrics yet{phase_hint}. Loss & grad-norm appear once SFT/DPO/LongCtx training starts.[/dim]")
    return Panel(t, title="[bold]Loss & Grad-Norm[/bold]", border_style="green")

def cost_panel() -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold blue"); t.add_column()
    t.add_row("Rate", f"${HOURLY_RATE:.2f}/h")

    anchor = TRAIN.started_at
    if anchor:
        elapsed = (datetime.now() - anchor).total_seconds()
        current_cost = HOURLY_RATE * elapsed / 3600
        t.add_row("So far", f"[bold]${current_cost:.2f}[/bold]")

        avg = (sum(TRAIN.step_times) / len(TRAIN.step_times)) if TRAIN.step_times else 0
        remaining = max(TRAIN.total_steps - TRAIN.current_step, 0) if TRAIN.total_steps else 0
        eta = remaining * avg if avg else 0
        if eta:
            projected_total = HOURLY_RATE * (elapsed + eta) / 3600
            t.add_row("Projection", f"${projected_total:.2f}")
            t.add_row("Remaining", f"${HOURLY_RATE * eta / 3600:.2f}")
    else:
        t.add_row("So far", "[dim]—[/dim]")
    return Panel(t, title="[bold]Cost[/bold]", border_style="blue")

def checkpoint_panel() -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan"); t.add_column()

    if TRAIN.saves:
        t.add_row("Saves", ", ".join(str(s) for s in TRAIN.saves[-4:]))
        ago = ""
        if TRAIN.last_save_at:
            secs = (datetime.now() - TRAIN.last_save_at).total_seconds()
            ago = f"  ({fmt_td(secs)} ago)"
        t.add_row("Last", f"step {TRAIN.last_save_step}{ago}")
    else:
        t.add_row("Saves", "[dim]none yet[/dim]")

    if TRAIN.current_step > 0 and TRAIN.total_steps > 0:
        next_save = ((TRAIN.current_step // SAVE_STEPS) + 1) * SAVE_STEPS
        if next_save <= TRAIN.total_steps:
            steps_to_go = next_save - TRAIN.current_step
            avg = (sum(TRAIN.step_times) / len(TRAIN.step_times)) if TRAIN.step_times else 0
            eta_save = fmt_td(steps_to_go * avg) if avg else "—"
            t.add_row("Next", f"step {next_save}  (in ~{eta_save})")

    if TRAIN.hub_push_last:
        secs = (datetime.now() - TRAIN.hub_push_last).total_seconds()
        t.add_row("Hub push", f"{fmt_td(secs)} ago")

    if DISK.alive:
        if DISK.ckpt_size_gb > 0:
            t.add_row("Ckpt size", f"{DISK.ckpt_size_gb:.1f} GB")
        if DISK.workspace_total_gb:
            used_pct = DISK.workspace_used_gb / DISK.workspace_total_gb * 100
            color = "red" if used_pct > 90 else "yellow" if used_pct > 75 else "green"
            t.add_row(
                "/workspace",
                f"[{color}]{DISK.workspace_used_gb:.0f} / {DISK.workspace_total_gb:.0f} GB "
                f"({used_pct:.0f}%)[/{color}]",
            )
        if DISK.hf_cache_gb:
            t.add_row("HF cache", f"{DISK.hf_cache_gb:.1f} GB")
    return Panel(t, title="[bold]Checkpoints & disk[/bold]", border_style="cyan")

def build_view() -> Layout:
    # Log-Tail wurde entfernt — das Layout hat nur noch top (Metriken) + mid
    # (Sparklines + Cost + Checkpoints). Beide Zeilen teilen sich den Platz.
    root = Layout()
    root.split_column(
        Layout(name="top", size=14),   # header + metrics (incl. agent-eval) + GPU
        Layout(name="mid"),            # sparklines + cost + checkpoints — Rest
    )
    root["top"].split_row(
        Layout(header_panel(), name="h", ratio=2),
        Layout(metrics_panel(), name="m", ratio=2),
        Layout(gpu_panel(), name="g", ratio=2),
    )
    root["mid"].split_row(
        Layout(loss_panel(), name="loss", ratio=3),
        Layout(cost_panel(), name="cost", ratio=1),
        Layout(checkpoint_panel(), name="ckpt", ratio=2),
    )
    return root

# ---------- Main ----------
def preflight_ssh(console: Console) -> bool:
    """Test the SSH connection before launching the TUI — clear error on bad key/host."""
    who = POD_PROXY_USER if POD_PROXY_USER else f"{SSH_USER}@{SSH_HOST}:{SSH_PORT}"
    mode = "proxy" if POD_PROXY_USER else "direct"
    console.print(f"[dim]Preflight: testing SSH ({mode}) to {who}…[/dim]")
    cmd = SSH_BASE + ["echo ok"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15, text=True)
        if "ok" in out:
            console.print("[green]✓ SSH ok[/green]")
            return True
        console.print(f"[red]unexpected SSH output:[/red] {out!r}")
        return False
    except subprocess.CalledProcessError as e:
        console.print("[red]✗ SSH failed:[/red]")
        console.print(f"[red]{e.output.strip() if e.output else f'rc={e.returncode}'}[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[red]✗ SSH timeout after 15s[/red]")
        return False

def main() -> None:
    # force_terminal so Rich's Live still renders inside the VSCode integrated terminal.
    console = Console(force_terminal=True)

    if not preflight_ssh(console):
        console.print("\n[bold yellow]Hint:[/bold yellow] check POD_HOST / POD_PORT / POD_KEY env vars, or try manually:")
        if POD_PROXY_USER:
            console.print(f"[dim]  ssh -tt -i {SSH_KEY} {POD_PROXY_USER}@ssh.runpod.io 'echo ok'[/dim]")
        else:
            console.print(f"[dim]  ssh -i {SSH_KEY} -p {SSH_PORT} {SSH_USER}@{SSH_HOST} 'echo ok'[/dim]")
            console.print("[dim]Proxy fallback: export POD_PROXY_USER=<pod-id>-<account-id>  (see RunPod 'Connect → SSH over exposed TCP / Proxy')[/dim]")
        sys.exit(1)

    console.print(f"[dim]starting live monitor — log={LOG_PATH}  gpu-poll={GPU_POLL_SEC}s  disk-poll={DISK_POLL_SEC}s[/dim]")
    console.print("[dim]Ctrl+C to quit · training keeps running on the pod[/dim]\n")
    time.sleep(0.6)

    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())

    t_log = threading.Thread(target=log_tail_worker, daemon=True)
    t_gpu = threading.Thread(target=gpu_poll_worker, daemon=True)
    t_disk = threading.Thread(target=disk_poll_worker, daemon=True)
    t_log.start()
    t_gpu.start()
    t_disk.start()

    try:
        # screen=True → alt-screen buffer, clean redraws in any terminal
        # auto_refresh=True → Live refreshes itself, no manual update() needed
        with Live(build_view(), console=console, refresh_per_second=4,
                  screen=True, auto_refresh=True) as live:
            while not STOP.is_set():
                live.update(build_view())
                time.sleep(0.25)
    except KeyboardInterrupt:
        STOP.set()
    finally:
        STOP.set()
        console.print("\n[dim]monitor stopped — training keeps running on the pod.[/dim]")

if __name__ == "__main__":
    main()
