import json, subprocess, tempfile
from pathlib import Path
from tqdm import tqdm
from evalplus.data import get_mbpp_plus
from _common import chat, save, extract_code, MODEL_ID

PROMPT_TMPL = """You are an expert Python programmer. Write a function that solves the problem below.
Output ONLY the function definition in a ```python code block, no tests.

Problem:
{prompt}"""


def main():
    problems = get_mbpp_plus()
    samples = []
    for task_id, task in tqdm(problems.items(), desc="MBPP+"):
        prompt = task.get("prompt", task.get("text", ""))
        resp = chat([{"role": "user", "content": PROMPT_TMPL.format(prompt=prompt)}], temperature=0.0, max_tokens=1024)
        samples.append({"task_id": task_id, "solution": extract_code(resp.choices[0].message.content)})

    jsonl = Path(tempfile.mkdtemp()) / "samples.jsonl"
    jsonl.write_text("\n".join(json.dumps(s) for s in samples))

    out = subprocess.run(
        ["evalplus.evaluate", "--dataset", "mbpp", "--samples", str(jsonl)],
        capture_output=True, text=True,
    )
    print(out.stdout)
    save("mbpp_plus", {"model": MODEL_ID, "raw": out.stdout, "n_samples": len(samples)})


if __name__ == "__main__":
    main()
