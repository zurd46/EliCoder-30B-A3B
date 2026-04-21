"""
Rename HF Hub repos from coder-16b-dyn-* → EliCoder-*.

Datasets are renamed (content preserved via HF move_repo API).
Model repos are deleted (they only held placeholder READMEs / partial checkpoints
that will be re-pushed by the fresh training run).

Usage:
    python training/rename_repos.py            # dry run
    python training/rename_repos.py --apply    # really do it
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_RENAMES = [
    ("zurd46/coder-16b-dyn-sft",     "zurd46/EliCoder-Dataset-SFT"),
    ("zurd46/coder-16b-dyn-dpo",     "zurd46/EliCoder-Dataset-DPO"),
    ("zurd46/coder-16b-dyn-longctx", "zurd46/EliCoder-Dataset-LongCtx"),
]

MODEL_DELETIONS = [
    "zurd46/coder-16b-dyn-lora-sft",
    "zurd46/coder-16b-dyn-lora-dpo",
    "zurd46/coder-16b-dyn-lora-longctx",
    "zurd46/coder-16b-dyn-base-fp16",
]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="really perform the changes")
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    load_env(REPO_ROOT / "build_pipeline" / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("FEHLER: HF_TOKEN nicht gesetzt", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi(token=token)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] rename/delete plan:\n")

    # 1. Datasets — rename (preserves content)
    for old, new in DATASET_RENAMES:
        try:
            api.repo_info(old, repo_type="dataset")
            exists = True
        except RepositoryNotFoundError:
            exists = False
        if not exists:
            print(f"  [dataset] {old} — not found, skip")
            continue
        print(f"  [dataset] rename {old} → {new}")
        if args.apply:
            try:
                api.move_repo(from_id=old, to_id=new, repo_type="dataset", token=token)
                print(f"            ✓ renamed")
            except Exception as e:
                print(f"            ✗ {e}")

    print()

    # 2. Model repos — delete (only held placeholder/partial data)
    for repo in MODEL_DELETIONS:
        try:
            api.repo_info(repo, repo_type="model")
            exists = True
        except RepositoryNotFoundError:
            exists = False
        if not exists:
            print(f"  [model] {repo} — not found, skip")
            continue
        print(f"  [model] delete {repo}")
        if args.apply:
            try:
                api.delete_repo(repo_id=repo, repo_type="model", token=token)
                print(f"          ✓ deleted")
            except Exception as e:
                print(f"          ✗ {e}")

    if not args.apply:
        print("\nthis was a dry-run. re-run with --apply to execute.")
    else:
        print("\ndone. next: re-run training/push_modelcards.py to recreate READMEs on new names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
