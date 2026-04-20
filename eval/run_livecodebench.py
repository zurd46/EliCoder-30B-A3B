import json, subprocess, tempfile, signal
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from _common import chat, save, extract_code, MODEL_ID

TIMEOUT_S = 10


def run_in_subprocess(code: str, stdin: str, timeout: int = TIMEOUT_S) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sol.py"
        p.write_text(code)
        try:
            out = subprocess.run(
                ["python", str(p)],
                input=stdin, capture_output=True, text=True, timeout=timeout,
            )
            return out.stdout.strip(), out.returncode == 0
        except subprocess.TimeoutExpired:
            return "", False
        except Exception:
            return "", False


def main():
    ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
    # Filtering: only tasks from 2026-01 onwards to avoid train contamination
    passed = 0; attempted = 0
    for row in tqdm(ds, desc="LiveCodeBench"):
        if row.get("release_date", "0") < "2026-01-01":
            continue
        attempted += 1
        prompt = f"Write a complete, standalone Python program that reads from stdin and writes to stdout.\n\nProblem:\n{row['question_content']}\n\nReturn only the code in a ```python block."
        resp = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048)
        code = extract_code(resp.choices[0].message.content)

        ok = True
        for test in row.get("public_test_cases", [])[:3]:
            stdout, exit_ok = run_in_subprocess(code, test.get("input", ""))
            if not exit_ok or stdout.strip() != test.get("output", "").strip():
                ok = False
                break
        if ok:
            passed += 1

    acc = passed / max(attempted, 1)
    save("livecodebench", {"model": MODEL_ID, "attempted": attempted, "passed": passed, "pass_rate": acc})
    print(f"LiveCodeBench pass@1 (2026-01+): {acc:.2%}  ({passed}/{attempted})")


if __name__ == "__main__":
    main()
