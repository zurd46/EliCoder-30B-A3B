import json, subprocess, tempfile
from pathlib import Path
from tqdm import tqdm
from evalplus.data import get_human_eval_plus
from _common import chat, save, extract_code, MODEL_ID

PROMPT_TMPL = """Complete the following Python function. Output ONLY the complete function definition in a ```python code block. Do not include tests or imports beyond what's needed.

{prompt}"""


def main():
    problems = get_human_eval_plus()
    samples = []
    for task_id, task in tqdm(problems.items(), desc="HumanEval+"):
        msg = [{"role": "user", "content": PROMPT_TMPL.format(prompt=task["prompt"])}]
        resp = chat(msg, temperature=0.0, max_tokens=1024)
        completion = extract_code(resp.choices[0].message.content)
        samples.append({"task_id": task_id, "solution": completion})

    jsonl = Path(tempfile.mkdtemp()) / "samples.jsonl"
    jsonl.write_text("\n".join(json.dumps(s) for s in samples))

    out = subprocess.run(
        ["evalplus.evaluate", "--dataset", "humaneval", "--samples", str(jsonl)],
        capture_output=True, text=True,
    )
    print(out.stdout)
    passk = _parse_passk(out.stdout)
    save("humaneval_plus", {"model": MODEL_ID, "results": passk, "n_samples": len(samples)})


def _parse_passk(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        for metric in ("pass@1", "pass@5", "pass@10"):
            if metric in line and ":" in line:
                try:
                    out[metric] = float(line.split(":")[-1].strip().rstrip("%")) / (100 if "%" in line else 1)
                except ValueError:
                    pass
    return out


if __name__ == "__main__":
    main()
