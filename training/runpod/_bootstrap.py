"""
RunPod-Bootstrap — kein Colab, keine Drive-Mounts.

- Erwartet: HF_TOKEN als ENV-Var (vom Pod gesetzt oder aus /workspace/.env)
- Arbeitet in /workspace (Container-Disk oder Network Volume)
- Installiert Deps nur wenn Marker-File fehlt (idempotent über Spot-Restarts)
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("CODERLLM_WORKSPACE", "/workspace"))
REPO_DIR = WORKSPACE / "CoderLLM"
CHECKPOINT_ROOT = WORKSPACE / "checkpoints"
CACHE_ROOT = WORKSPACE / "hf_cache"
DEPS_MARKER = WORKSPACE / ".deps_installed"

PIP_PACKAGES = [
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
    "unsloth_zoo",
    "trl>=0.12", "transformers>=4.46", "datasets>=3.0",
    "peft>=0.13", "accelerate>=1.0", "bitsandbytes", "wandb", "pyyaml",
    "huggingface_hub>=0.25", "tqdm",
]

# Pod-Images kommen teils mit torchao vorinstalliert, das `torch.int1`
# erwartet (torch >=2.6). Wir pinnen zurück auf eine torch-2.4/2.5-kompatible
# Version, damit der Transformers-Import nicht bricht.
PIP_PINS_PRE = [
    "torchao==0.7.0",
]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def bootstrap(*, install: bool = True) -> Path:
    _load_env_file(WORKSPACE / ".env")

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN nicht gesetzt. Setz ENV-Var oder lege /workspace/.env an "
            "mit Zeile: HF_TOKEN=hf_xxx"
        )
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if install and not DEPS_MARKER.exists():
        print("pinning torchao (torch.int1 conflict fix) …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "--force-reinstall", "--no-deps", *PIP_PINS_PRE],
            check=True,
        )
        print("installing pip deps (once per pod) …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES],
            check=True,
        )
        DEPS_MARKER.touch()
        print("deps installed.")
    else:
        print("deps already installed — skipping.")

    training = REPO_DIR / "training"
    if not training.exists():
        raise RuntimeError(
            f"{training} fehlt. Pipeline muss aus geklontem Repo laufen — "
            f"clone zuerst nach {REPO_DIR}."
        )
    os.chdir(training)
    print(f"cwd = {training}")
    return training


def apply_gpu_optims() -> None:
    """Hopper-spezifische Optimierungen — vor Model-Load aufrufen."""
    import torch
    # Force-load torch._inductor.config — unsloth_zoo >=2026 greift als
    # Attribut drauf zu, torch 2.4 lädt es aber nicht eager.
    try:
        import torch._inductor.config  # noqa: F401
    except Exception:
        pass
    # torch 2.4 fehlt nn.Module.set_submodule (ab 2.5), transformers 5.x braucht es.
    import torch.nn as _nn
    if not hasattr(_nn.Module, "set_submodule"):
        def _set_submodule(self, target: str, module: _nn.Module) -> None:
            atoms = target.split(".")
            if len(atoms) == 1:
                setattr(self, atoms[0], module)
                return
            parent = self.get_submodule(".".join(atoms[:-1]))
            setattr(parent, atoms[-1], module)
        _nn.Module.set_submodule = _set_submodule
        print("torch compat: nn.Module.set_submodule backported")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    print(f"GPU: {props.name}  ({total_gb:.1f} GB, SM {props.major}.{props.minor})")
    print(f"TF32 matmul: on, cuDNN benchmark: on")


def checkpoint_dir(phase: str) -> Path:
    d = CHECKPOINT_ROOT / phase
    d.mkdir(parents=True, exist_ok=True)
    return d


def patch_unsloth_telemetry() -> None:
    """No-op Unsloth's HF stats-endpoint — hängt in Colab/RunPod ~120s."""
    try:
        import torch._inductor.config  # noqa: F401
    except Exception:
        pass
    import unsloth.models._utils as _u
    _u._get_statistics = lambda *a, **kw: None
    _u.time_limited_stats_check = lambda *a, **kw: None
