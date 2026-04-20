"""
Shared Colab setup — kopiere diese Funktion an den Anfang jedes Notebook-Runs
(oder die Zellen oben in jedem 0X_*.py enthalten den gleichen Block).

Erkennt Colab, installiert Dependencies, cloned das Repo falls Configs fehlen,
und zieht HF_TOKEN aus den Colab-Secrets.
"""
from __future__ import annotations
import os, subprocess, sys, shutil
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def colab_setup(
    repo_url: str = "https://github.com/zurd46/CoderLLM.git",
    repo_dir: str = "/content/CoderLLM",
    install: bool = True,
) -> Path:
    """Clone repo (if needed), install deps, load HF_TOKEN from Colab secrets.

    Returns the Path to training/ inside the clone.
    """
    if in_colab():
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
            if tok:
                os.environ["HF_TOKEN"] = tok
                os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
                print("✔ HF_TOKEN loaded from Colab secrets")
        except Exception:
            if not os.environ.get("HF_TOKEN"):
                print("✘ HF_TOKEN missing — add it via Colab → Secrets")

    root = Path(repo_dir)
    if not root.exists():
        print(f"cloning {repo_url} → {root}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(root)], check=True)
    elif in_colab():
        try:
            subprocess.run(["git", "-C", str(root), "pull", "--ff-only"], check=False)
        except Exception:
            pass

    training = root / "training"
    os.chdir(training)
    print(f"cwd = {training}")

    if install:
        req = [
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
            "trl>=0.12", "transformers>=4.46", "datasets>=3.0",
            "peft>=0.13", "accelerate>=1.0", "bitsandbytes", "wandb", "pyyaml",
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *req],
            check=False,
        )

    return training
