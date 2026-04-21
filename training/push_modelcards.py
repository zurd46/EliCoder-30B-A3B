"""
Push model cards (README.md) directly to HF Hub — no pipeline run required.

Usage (from repo root on Mac):
    python training/push_modelcards.py               # all four repos
    python training/push_modelcards.py --only merged # only the BF16 base
    python training/push_modelcards.py --only sft
    python training/push_modelcards.py --only dpo
    python training/push_modelcards.py --only longctx

HF_TOKEN is read from env or from .env / build_pipeline/.env in the repo root.
Repos are created as private if they don't exist yet.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training" / "runpod"))

from _modelcard import MERGED_CARD, adapter_card  # noqa: E402

OWNER = os.environ.get("CODERLLM_HF_OWNER", "zurd46")

REPOS = {
    "sft":     (f"{OWNER}/EliCoder-30B-A3B-LoRA-SFT",     lambda: adapter_card("sft")),
    "dpo":     (f"{OWNER}/EliCoder-30B-A3B-LoRA-DPO",     lambda: adapter_card("dpo")),
    "longctx": (f"{OWNER}/EliCoder-30B-A3B-LoRA-LongCtx", lambda: adapter_card("longctx")),
    "merged":  (f"{OWNER}/EliCoder-30B-A3B",    lambda: MERGED_CARD),
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def push(repo_id: str, content: str, token: str) -> None:
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id)
    except RepositoryNotFoundError:
        print(f"  creating {repo_id} (private) …")
        create_repo(repo_id, private=True, exist_ok=True, token=token)

    import io
    buf = io.BytesIO(content.encode("utf-8"))
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="docs: professional model card",
    )
    print(f"  ✓ README pushed → https://huggingface.co/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(REPOS), help="push only one card")
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    load_env(REPO_ROOT / "build_pipeline" / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("FEHLER: HF_TOKEN nicht gesetzt (weder ENV noch .env)", file=sys.stderr)
        return 1

    keys = [args.only] if args.only else list(REPOS)
    for k in keys:
        repo_id, card_fn = REPOS[k]
        print(f"[{k}] → {repo_id}")
        push(repo_id, card_fn(), token)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
