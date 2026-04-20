import os, json, time
from pathlib import Path
from typing import Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "lm-studio")
MODEL_ID = os.environ.get("MODEL_ID", "coder-16b-dyn-UD-Q4_K_XL")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def chat(messages, tools=None, temperature=0.0, max_tokens=2048, stop=None) -> Any:
    return client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )


def save(name: str, obj: dict) -> Path:
    obj["_meta"] = {"model": MODEL_ID, "base_url": BASE_URL, "timestamp": time.time()}
    p = RESULTS_DIR / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2))
    return p


def extract_code(text: str) -> str:
    if "```" not in text:
        return text
    block = text.split("```", 2)[1]
    if block.startswith(("python", "py")):
        block = block.split("\n", 1)[1]
    return block.split("```")[0]
