import json, sys
from pathlib import Path
from _common import RESULTS_DIR, MODEL_ID

BASELINE_PATH = RESULTS_DIR / "baseline.json"
REGRESSION_THRESHOLD = 0.02


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def main():
    metrics = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "baseline.json":
            continue
        data = json.loads(f.read_text())
        metrics[f.stem] = flatten(data)

    summary = RESULTS_DIR / "summary.md"
    lines = [f"# Eval summary — {MODEL_ID}\n"]
    for bench, m in metrics.items():
        lines.append(f"## {bench}")
        for k, v in sorted(m.items()):
            lines.append(f"- `{k}` = **{v:.4f}**" if isinstance(v, float) else f"- `{k}` = {v}")
        lines.append("")
    summary.write_text("\n".join(lines))
    print(f"wrote {summary}")

    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())
        regressions = []
        for bench, m in metrics.items():
            for k, v in m.items():
                b = baseline.get(bench, {}).get(k)
                if b is None or not isinstance(b, (int, float)):
                    continue
                if v + REGRESSION_THRESHOLD < b:
                    regressions.append(f"{bench}.{k}: {v:.4f} < baseline {b:.4f}")
        if regressions:
            print("REGRESSIONS:")
            for r in regressions:
                print(" -", r)
            sys.exit(1)
    print("no regressions.")


if __name__ == "__main__":
    main()
