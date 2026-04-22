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

# PyTorch für CUDA 12.8 + H100. torch 2.8 zieht triton 3.3+ automatisch mit,
# das aktiviert den schnellen unsloth_triton / grouped_mm MoE-Backend.
# (Alter cu124-Pfad mit torch 2.6 + triton 3.2 war ~20× langsamer.)
TORCH_PACKAGES = [
    "torch==2.8.0+cu128",
    "torchvision==0.23.0+cu128",
    "torchaudio==2.8.0+cu128",
]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# Flash-Attention-2 Wheel passend zu torch 2.8 + cu12 + cxx11abi=TRUE
# (torch 2.8+cu128 default bei CUDA-Index-Build). Wheel-Namenskonvention bei
# Dao-AILab: "cu12" (nicht "cu128"), "torch2.8", "cxx11abiTRUE"/"FALSE"
# muss zu torch._C._GLIBCXX_USE_CXX11_ABI passen. v2.8.3 hat torch 2.8 wheels;
# v2.8.1 nicht (nur torch 2.10) — daher v2.8.3.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-"
    "cp311-cp311-linux_x86_64.whl"
)

PIP_PACKAGES = [
    # Triton NICHT pinnen — torch 2.8 bringt die passende Version (≥3.3, mit
    # tl.make_tensor_descriptor-API) als strikte Dependency. Ein Pin würde
    # entweder ignoriert oder zu Resolver-Konflikten führen.
    "torchao==0.13.0",
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
    "unsloth_zoo",
    "trl>=0.12", "transformers>=4.46", "datasets>=3.0",
    "peft>=0.13", "accelerate>=1.0", "bitsandbytes", "wandb", "pyyaml",
    "huggingface_hub>=0.25", "tqdm",
    # Flash Attention 2 direkt von GitHub-Release (kein build-isolation nötig).
    FLASH_ATTN_WHEEL,
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


def _verify_stack() -> None:
    """Printet eine klare Stack-Summary am Start jeder Phase — macht sofort
    sichtbar ob die kritische Kombination (torch 2.8 + triton 3.3+ TMA + FA2)
    tatsächlich geladen wurde. Rote Flaggen zeigen wir mit Warn-Prefix,
    damit sie im Log nicht übersehen werden.
    """
    import importlib
    rows = []

    # torch
    try:
        import torch
        t_ver = torch.__version__
        t_cuda = getattr(torch.version, "cuda", None)
        ok = t_ver >= "2.7"
        rows.append((("✓" if ok else "⚠"), "torch", f"{t_ver}  (CUDA rt: {t_cuda})",
                     "≥2.7 für unsloth_triton MoE-Kernel"))
    except ImportError:
        rows.append(("✗", "torch", "NOT INSTALLED", "Pipeline wird crashen"))

    # triton
    try:
        import triton
        from triton import language as tl
        has_tma = hasattr(tl, "make_tensor_descriptor")
        rows.append((("✓" if has_tma else "⚠"), "triton",
                     f"{triton.__version__}  TMA={'yes' if has_tma else 'NO'}",
                     "TMA nötig für schnellen MoE-Kernel"))
    except ImportError:
        rows.append(("✗", "triton", "NOT INSTALLED", ""))

    # flash-attn
    try:
        import flash_attn
        rows.append(("✓", "flash_attn", flash_attn.__version__,
                     "Attention-Speedup vs. xformers"))
    except ImportError:
        rows.append(("⚠", "flash_attn", "NOT INSTALLED", "fällt auf xformers zurück (langsamer)"))

    # unsloth + unsloth_zoo
    for pkg in ("unsloth", "unsloth_zoo"):
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            rows.append(("✓", pkg, ver, ""))
        except ImportError:
            rows.append(("✗", pkg, "NOT INSTALLED", "Pipeline wird crashen"))

    # transformers + trl + peft (häufige Version-Konflikte)
    for pkg in ("transformers", "trl", "peft", "accelerate", "bitsandbytes"):
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            rows.append(("✓", pkg, ver, ""))
        except ImportError:
            rows.append(("✗", pkg, "NOT INSTALLED", ""))

    w_pkg = max(len(r[1]) for r in rows)
    w_ver = max(len(r[2]) for r in rows)
    print("┌── STACK SUMMARY ───────────────────────────────────────────────")
    for flag, pkg, ver, note in rows:
        line = f"│ {flag}  {pkg:<{w_pkg}}  {ver:<{w_ver}}"
        if note:
            line += f"   {note}"
        print(line)
    print("└───────────────────────────────────────────────────────────────")


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
        # 1. PyTorch zuerst via CUDA-Index — TORCH_INDEX_URL ist auf cu128 gesetzt,
        #    damit torch 2.8 + triton 3.3+ (mit TMA) kommt. Auf älteren Images
        #    mit cu124 würde das fehlschlagen — dann muss der Pod neu mit cu128
        #    oder höher aufgebaut werden.
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "--index-url", TORCH_INDEX_URL,
                *TORCH_PACKAGES,
            ],
            check=True,
        )
        # 2. Restliche Packages (inkl. flash-attn wheel). triton wird NICHT
        #    gepinnt — torch 2.8 zieht die passende Version automatisch mit.
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
    # Stack-Summary printet nach dem MoE-Backend-Pick — zeigt sofort ob
    # alles stimmt (torch 2.8 + triton TMA + FA2 usw).
    _verify_stack()
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
