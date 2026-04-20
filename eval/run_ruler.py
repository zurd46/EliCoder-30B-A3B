import random
from tqdm import tqdm
from _common import chat, save, MODEL_ID

random.seed(42)

LENGTHS = [8_000, 16_000, 32_000, 64_000, 128_000, 200_000]
N_PER_LEN = 20
CORPUS = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ".split()


def synthesize(target_tokens: int) -> tuple[str, str]:
    magic = f"{random.randint(10**9, 10**10)}"
    body_tokens = target_tokens - 60
    words = [random.choice(CORPUS) for _ in range(body_tokens)]
    insert = random.randint(body_tokens // 4, 3 * body_tokens // 4)
    words.insert(insert, f"PASSPHRASE={magic}.")
    return " ".join(words), magic


def main():
    results = {}
    for L in LENGTHS:
        hits = 0
        for _ in tqdm(range(N_PER_LEN), desc=f"RULER-{L}"):
            doc, truth = synthesize(L)
            resp = chat([
                {"role": "system", "content": "Answer with the digits only, nothing else."},
                {"role": "user", "content": f"Document:\n{doc}\n\nQuestion: What is the value of PASSPHRASE? Answer with the number only."},
            ], temperature=0.0, max_tokens=32)
            ans = (resp.choices[0].message.content or "").strip()
            if truth in ans:
                hits += 1
        acc = hits / N_PER_LEN
        results[str(L)] = acc
        print(f"  len={L:>7}  acc={acc:.2f}")
    save("ruler", {"model": MODEL_ID, "accuracies": results})


if __name__ == "__main__":
    main()
