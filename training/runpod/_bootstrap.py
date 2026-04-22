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

# PyTorch muss zuerst mit dem korrekten CUDA-Index installiert werden,
# damit unsloth_zoo >=2026.4.8 funktioniert (braucht torch.int1 etc.).
TORCH_PACKAGES = [
    "torch==2.6.0+cu124",
    "torchvision==0.21.0+cu124",
    "torchaudio==2.6.0+cu124",
]

PIP_PACKAGES = [
    "torchao==0.13.0",
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
    "unsloth_zoo",
    "trl>=0.12", "transformers>=4.46", "datasets>=3.0",
    "peft>=0.13", "accelerate>=1.0", "bitsandbytes", "wandb", "pyyaml",
    "huggingface_hub>=0.25", "tqdm",
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


def _backport_torch_compat() -> None:
    """Back-port torch 2.5 APIs that transformers 5.x uses but torch 2.4 lacks.

    RunPod H100 base images ship torch 2.4.1, but transformers >=5.0 calls
    `model.set_submodule(...)` during the bitsandbytes 4-bit integration
    (transformers/integrations/bitsandbytes.py). That method landed in
    torch 2.5 — without it, `FastLanguageModel.from_pretrained` dies with:

        AttributeError: 'Qwen3MoeForCausalLM' object has no attribute 'set_submodule'

    Idempotent: bails out if torch already provides the attribute, so it's
    safe to call from both `bootstrap()` and `apply_gpu_optims()`.
    """
    try:
        import torch.nn as nn
    except ImportError:
        return  # torch not installed yet — bootstrap() will retry post-install

    if hasattr(nn.Module, "set_submodule"):
        return

    def set_submodule(
        self,
        target: str,
        module: "nn.Module",
        strict: bool = False,
    ) -> None:
        # Signature mirrors pytorch 2.5's nn.Module.set_submodule.
        if not isinstance(target, str):
            raise TypeError(
                f"`target` must be a string, got {type(target).__name__}"
            )
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        atoms = target.split(".")
        name = atoms[-1]
        parent_path = ".".join(atoms[:-1])
        parent = self.get_submodule(parent_path) if parent_path else self

        if strict and not hasattr(parent, name):
            raise AttributeError(
                f"{parent._get_name()} has no attribute `{name}`"
            )
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"`module` must be an nn.Module, got {type(module).__name__}"
            )
        setattr(parent, name, module)

    nn.Module.set_submodule = set_submodule
    print("torch compat: backported nn.Module.set_submodule (torch <2.5)")


def _configure_moe_backend() -> None:
    """Pick an Unsloth MoE backend compatible with the installed triton.

    Unsloth's default `unsloth_triton` grouped-GEMM path uses
    `tl.make_tensor_descriptor` (triton >=3.2, ships with torch >=2.6).
    RunPod torch 2.4.1 images ship triton 3.0 → kernel-compile fails with:

        AttributeError: module 'triton.language' has no attribute 'make_tensor_descriptor'

    `select_moe_backend()` in unsloth_zoo is `@lru_cache`d, so the env var
    must be set *before* the first MoE forward. We set it here in bootstrap,
    long before unsloth is imported.

    Respects user override — set UNSLOTH_MOE_BACKEND yourself to force
    `grouped_mm` (torch >=2.8) or `unsloth_triton` (triton >=3.2).
    """
    if os.environ.get("UNSLOTH_MOE_BACKEND"):
        print(f"UNSLOTH_MOE_BACKEND (user-set): {os.environ['UNSLOTH_MOE_BACKEND']}")
        return
    try:
        import triton
        import triton.language as tl
        tver = getattr(triton, "__version__", "?")
    except ImportError:
        print("UNSLOTH_MOE_BACKEND: triton not installed — skipping autoconfig")
        return
    # Unsloth's grouped-GEMM kernel literally references `tl.make_tensor_descriptor`
    # (the non-experimental name, triton >=3.2). Triton 3.0 only has
    # `_experimental_make_tensor_descriptor`, which won't satisfy the AST lookup
    # — so *only* checking the exact attribute the kernel uses is correct here.
    has_tma = hasattr(tl, "make_tensor_descriptor")
    if has_tma:
        print(f"UNSLOTH_MOE_BACKEND=auto (triton {tver} has tl.make_tensor_descriptor)")
        return
    os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
    print(
        f"UNSLOTH_MOE_BACKEND=native_torch "
        f"(triton {tver} lacks tl.make_tensor_descriptor → native loop fallback)"
    )


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
        print("installing pip deps (once per pod) …")
        # 1. PyTorch zuerst — RunPod-Image hat 2.4.1, unsloth_zoo braucht >=2.6
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "--index-url", "https://download.pytorch.org/whl/cu124",
                *TORCH_PACKAGES,
            ],
            check=True,
        )
        # 2. Restliche Packages
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
    # Patch early: greift bevor transformers/unsloth importiert werden.
    _backport_torch_compat()
    _configure_moe_backend()
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
    _backport_torch_compat()
    _configure_moe_backend()
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
