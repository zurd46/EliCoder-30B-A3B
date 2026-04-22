# RunPod-Pipeline — Abhängigkeiten & Versionen

> Stand: 2026-04-22
> Zielsystem: RunPod H100-NVL Pod mit `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

Der Pipeline-Stack ist **strikt versionsgebunden** — jede Zeile ist aus schmerzhaftem Trial-and-Error. Bitte keine Versionen lockern ohne zu prüfen.

---

## 1. Lock-File

Alle Pflicht-Versionen. Wird von [_bootstrap.py](_bootstrap.py) exakt so installiert, idempotent über `/workspace/.deps_installed`-Marker.

### CUDA-Toolkit (vom Pod-Image)
| Paket | Version | Quelle |
|---|---|---|
| CUDA Runtime | **12.4** | RunPod-Image (`pytorch:2.4.0-…-cuda12.4.1-…`) |
| Driver | ≥ 550 | Pod-Host |

### PyTorch (gepinnt auf CUDA-12.4-Wheels)
| Paket | Version | Warum diese Version |
|---|---|---|
| `torch` | **2.6.0+cu124** | Maximum für `cu124`. `unsloth_zoo ≥ 2026.4.8` braucht `torch.int1`-API (torch ≥ 2.6). |
| `torchvision` | **0.21.0+cu124** | Passt exakt zu `torch 2.6.0`. |
| `torchaudio` | **2.6.0+cu124** | Passt exakt zu `torch 2.6.0`. |

Installiert via:
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
```

### Kernel- & Compiler-Stack
| Paket | Version | Warum diese Version |
|---|---|---|
| `triton` | **3.2.0** (explicit pin) | Strict dep von `torch 2.6`. Höhere Versionen brechen `torch._inductor` (in 3.4 wurde `AttrsDescriptor` entfernt). Nachteil: kein `tl.make_tensor_descriptor` → Unsloth MoE-Kernel fällt auf `native_torch` zurück. |
| `torchao` | **0.13.0** | Passender 4-bit-Quantization-Stack für torch 2.6. |

### Attention
| Paket | Version | Warum diese Version |
|---|---|---|
| `flash-attn` | **2.7.4.post1** | Ohne FA2 fällt Unsloth auf xformers zurück (~1.5–2× langsamer auf H100). FA3 ist für diesen Pfad (BF16 + 30B-MoE + Qwen3) noch nicht stabil. |

Prebuilt-Wheel (verhindert 20-min-Kompilieren):
```bash
pip install --no-build-isolation \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Matrix-Schlüssel beim Wheel: `cu12 · torch 2.6 · cp311 · cxx11abiFALSE · x86_64`.

### Training-Framework
| Paket | Version | Warum diese Version |
|---|---|---|
| `unsloth` | `git+https://github.com/unslothai/unsloth.git` (latest main) | Qwen3-MoE-Patches sind nur in main — kein pinned-Release hat die Änderung. |
| `unsloth_zoo` | `latest` | Muss zu `unsloth` main passen. |
| `trl` | **≥ 0.12** | `SFTConfig` · `DPOConfig` mit den in unseren Scripts genutzten Keys. |
| `transformers` | **≥ 4.46** | Nötig für `save_strategy` · `eval_strategy` getrennt + Qwen3-MoE-Support. |
| `peft` | **≥ 0.13** | LoRA-Adapter-Merge ohne RuntimeError bei MoE-Expert-Projektionen. |
| `accelerate` | **≥ 1.0** | Kompatibel mit transformers 4.46+. |
| `bitsandbytes` | `latest` | 4-bit QLoRA. |
| `datasets` | **≥ 3.0** | `map(…, num_proc=8)` stabil. |

### Helpers
| Paket | Zweck |
|---|---|
| `wandb` | Optional — Metriken-Sink. Ohne `WANDB_API_KEY` wird `report_to="none"` und Agent-Eval-Metriken landen nur in stdout (reicht für [watch.py](watch.py)). |
| `huggingface_hub` ≥ 0.25 | Dataset-Push + `get_last_checkpoint` via Hub. |
| `pyyaml` | Configs in [../configs/](../configs/). |
| `tqdm` | Progress-Bars. |

---

## 2. Bekannte Version-Konflikte (nicht touchen!)

### `triton > 3.2.0` + `torch 2.6` = 💥
```
ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'
```
Ab Triton 3.4 entfernt, aber `torch._inductor.runtime.hints` importiert es beim Modul-Start. Heilt erst ab `torch 2.7` — und das gibt's nur für `cu126`/`cu128`, nicht für unseren cu124-Pod.

**Fix:** `triton==3.2.0` explizit pinnen.

### `triton < 3.2.0` = schlafender Kernel
```
UNSLOTH_MOE_BACKEND=native_torch (triton X.Y.Z lacks tl.make_tensor_descriptor → native loop fallback)
```
Unsloth-Triton grouped-GEMM ist dann tot. Läuft, aber ~10–30× langsamer als der schnelle Kernel. Mit FA2 bleibt's trotzdem erträglich.

**Workaround bis Pod-Image mit cu126 verfügbar:** Mit native_torch-Fallback leben. Kompensation: FA2 + `bsz=1, grad_accum=64, max_seq=4096`.

### `flash-attn` aus PyPI ohne Wheel = 20-min-Build
Das PyPI-Paket triggert einen full source build (nvcc + ninja). Beim RunPod-Pod ist das jedes Mal Zeitverschwendung.

**Fix:** Das prebuilt GitHub-Release-Wheel (Link oben).

---

## 3. Ideal-Stack (sobald cu126/cu128 Pod-Image verfügbar ist)

Wenn wir einen Pod auf `cuda12.6` oder `cuda12.8` hochziehen können:

| Paket | Ideal-Version | Effekt |
|---|---|---|
| `torch` | `2.8.0+cu128` | `torch._inductor` kompatibel mit `triton ≥ 3.4`. `grouped_mm` MoE-Backend wird verfügbar. |
| `triton` | `3.4.0` (unpinned, von torch gezogen) | `tl.make_tensor_descriptor` vorhanden → Unsloth-Triton-MoE aktiv (~20 s/step statt ~500 s/step auf 30B-MoE + max_seq=4096). |
| `flash-attn` | `2.7.4.post1` (torch 2.8 wheel) | Bleibt FA2. |

Dann würde `UNSLOTH_MOE_BACKEND` auf `unsloth_triton` oder `grouped_mm` springen und SFT ginge in ~2 h statt 5–6 h durch.

---

## 4. Schnelle Diagnose-Commands

```bash
# Versionen prüfen
python3 -c "import torch, triton, flash_attn; \
  print(f'torch={torch.__version__}, triton={triton.__version__}, flash_attn={flash_attn.__version__}')"

# TMA-API (= schneller MoE-Kernel verfügbar?)
python3 -c "from triton import language as tl; print('TMA:', hasattr(tl, 'make_tensor_descriptor'))"

# inductor-Kompatibilität (= torch 2.6 + triton 3.4 kaputt?)
python3 -c "from triton.compiler.compiler import AttrsDescriptor; print('AttrsDescriptor: OK')"

# Welches MoE-Backend ist gerade gesetzt?
python3 -c "import os; print('UNSLOTH_MOE_BACKEND=', os.environ.get('UNSLOTH_MOE_BACKEND','auto'))"
```

---

## 5. Komplette Install-Reihenfolge (bei vollem Reset)

```bash
# 1. DEPS_MARKER löschen (falls vorhanden)
rm -f /workspace/.deps_installed

# 2. Alle alten triton/flash-attn-Spuren entfernen (Sicherheit)
pip uninstall -y triton flash-attn 2>/dev/null

# 3. PyTorch zuerst (wegen cu124-Index)
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

# 4. Alles weitere (triton wird wegen Pin NICHT geupgradet)
pip install \
  triton==3.2.0 \
  torchao==0.13.0 \
  "unsloth @ git+https://github.com/unslothai/unsloth.git" \
  unsloth_zoo \
  "trl>=0.12" "transformers>=4.46" "datasets>=3.0" \
  "peft>=0.13" "accelerate>=1.0" bitsandbytes wandb pyyaml \
  "huggingface_hub>=0.25" tqdm \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 5. Marker setzen (bootstrap skipt beim nächsten Lauf)
touch /workspace/.deps_installed
```

Oder einfach: `python runpod/02_sft.py` — [_bootstrap.py](_bootstrap.py) macht Schritte 3–5 automatisch beim ersten Aufruf.
