import json
from datasets import load_dataset
from jsonschema import validate, ValidationError
from tqdm import tqdm
from _common import chat, save, MODEL_ID


def tool_call_valid_json(resp) -> bool:
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        return False
    for tc in msg.tool_calls:
        try:
            json.loads(tc.function.arguments)
        except Exception:
            return False
    return True


def tool_call_matches_expected(resp, expected_name: str, expected_args_schema: dict | None) -> bool:
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        return False
    got = msg.tool_calls[0]
    if got.function.name != expected_name:
        return False
    try:
        args = json.loads(got.function.arguments)
    except Exception:
        return False
    if expected_args_schema:
        try:
            validate(args, expected_args_schema)
        except ValidationError:
            return False
    return True


def main():
    ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="test")

    stats = {"n": 0, "valid_json": 0, "correct_name": 0, "schema_ok": 0}
    per_category: dict[str, dict] = {}

    for row in tqdm(ds, desc="BFCL v3"):
        cat = row.get("category", "default")
        pc = per_category.setdefault(cat, {"n": 0, "correct": 0})

        messages = [
            {"role": "system", "content": "You are a helpful assistant that calls functions to answer the user's question. Respond with a single tool call."},
            {"role": "user", "content": row["question"]},
        ]
        tools = row["functions"]
        expected_name = row.get("expected_function")
        schema = row.get("expected_args_schema")

        try:
            resp = chat(messages, tools=tools, temperature=0.0, max_tokens=512)
        except Exception:
            stats["n"] += 1
            pc["n"] += 1
            continue

        stats["n"] += 1
        pc["n"] += 1
        if tool_call_valid_json(resp):
            stats["valid_json"] += 1
        if expected_name and tool_call_matches_expected(resp, expected_name, schema):
            stats["correct_name"] += 1
            stats["schema_ok"] += 1
            pc["correct"] += 1

    n = max(stats["n"], 1)
    summary = {
        "model": MODEL_ID,
        "n": n,
        "valid_json_rate": stats["valid_json"] / n,
        "correct_name_rate": stats["correct_name"] / n,
        "schema_ok_rate": stats["schema_ok"] / n,
        "per_category": {k: {"acc": v["correct"] / max(v["n"], 1), "n": v["n"]} for k, v in per_category.items()},
    }
    save("bfcl", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
