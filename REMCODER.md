# RemCoder Project Memo

## Overview
Ein Open-Source Coder-Modell + Agent-Runtime für Consumer-Hardware (Mac M-Series oder Windows/Linux mit CUDA/Vulkan). Das Modell läuft in **LM Studio**, der mitgelieferte Agent [`coder`](coder/) spricht OpenAI-kompatibel gegen `localhost:1234/v1` und bringt eine vollständige Tool-Suite (Filesystem, Git, GitHub, Shell, Tests, Projekt-Scaffolding, LSP, De

## Tech Stack
- Python
- MLX
- GGUF
- Hugging Face
- LM Studio
- Git
- GitHub
- Playwright
- Jupyter Notebooks

## Structure
- `build_pipeline`: End-to-end Pipeline für Modell-Quantisierung und Upload
- `coder`: Agent Runtime für OpenAI-kompatiblen API-Call und Tool-Suite
- `docs`: Dokumentation für Architektur und Fine-Tuning
- `eval`: Bewertungstools für verschiedene Benchmarks
- `training`: Trainingspipeline mit verschiedenen Phasen (Data Build, SFT, DPO, Long Context, Export)

## Build / Test / Lint
- `coderllm-build all` für Build-Pipeline
- `coder-agent run` für Agent Runtime
- `run_tests`, `run_lint`, `run_typecheck` für Tests und Linting

## Known TODOs
- none found

## Conventions
- Python-Code mit `coder-agent`-CLI
- Jupyter Notebooks für Trainingspipeline
- Git und GitHub für Versionierung und Collaboration