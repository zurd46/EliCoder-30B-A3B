# Evaluation Harness

Alle Benchmarks laufen gegen eine **OpenAI-kompatible URL** (default `http://localhost:1234/v1` = LM Studio). Das erlaubt die gleiche Harness für Base-Modell, quantisiertes GGUF, MLX, oder sogar gegen OpenAI/Anthropic-APIs (zum Vergleichen).

## Setup

```bash
cd eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ausführen

```bash
# LM Studio muss laufen mit dem Modell geladen
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=lm-studio
export MODEL_ID=coder-16b-dyn-UD-Q4_K_XL

python run_humaneval.py
python run_mbpp.py
python run_bfcl.py           # Tool-Use BFCL v3
python run_ruler.py          # Long-Context
python run_livecodebench.py  # Kontaminationsfreier Code-Bench
python run_swebench.py       # teuer (6-10h), Docker nötig

python aggregate.py          # schreibt results/summary.md
```

## Regression-Guard

`aggregate.py` vergleicht mit `results/baseline.json` und exit-codet != 0 wenn
irgendeine Metrik > 2 % unter dem Baseline fällt. Für CI-Integration.
