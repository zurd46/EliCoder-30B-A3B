"""Convert every 0X_*.py in this directory to a .ipynb (Colab-friendly)."""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def split_cells(text: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    current_kind = "code"
    current_src: list[str] = []

    def flush():
        if current_src:
            src = "\n".join(current_src).strip("\n")
            if src:
                cells.append((current_kind, src))

    for line in text.splitlines():
        m = re.match(r"^# %% \[(\w+)\]\s*$", line)
        if m:
            flush()
            current_kind = "markdown" if m.group(1) == "markdown" else "code"
            current_src = []
            continue
        if line.strip() == "# %%":
            flush()
            current_kind = "code"
            current_src = []
            continue
        if current_kind == "markdown":
            if line.startswith("# "):
                current_src.append(line[2:])
            elif line.startswith("#"):
                current_src.append(line[1:].lstrip())
            else:
                current_src.append(line)
        else:
            current_src.append(line)
    flush()
    return cells


def to_ipynb(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, src in cells:
        if kind == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": src.splitlines(keepends=True),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": src.splitlines(keepends=True),
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for py in sorted(HERE.glob("0[1-9]_*.py")):
        ipynb = py.with_suffix(".ipynb")
        cells = split_cells(py.read_text())
        ipynb.write_text(json.dumps(to_ipynb(cells), indent=1))
        print(f"  {py.name} → {ipynb.name}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
