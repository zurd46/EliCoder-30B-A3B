# Training auf Google Colab (H100)

## Vorbereitung (einmalig, 3 Min)

**1. Colab Pro + H100**
- https://colab.research.google.com → oben **Runtime → Change runtime type** → **A100 / H100**
- Für H100 brauchst du Colab Pro oder Colab Pro+ Compute Units

**2. HuggingFace Token als Colab-Secret**
- https://huggingface.co/settings/tokens → **New token** mit Scope `write`
- In Colab: Schlüssel-Symbol links („Secrets") → **+ Add new secret**
  - Name: `HF_TOKEN`
  - Value: dein Token
  - Toggle **Notebook access: ON**
- Dieser Secret wird von allen Notebooks automatisch gelesen.

**3. Repo muss auf GitHub sein**
- Die Notebooks clonen beim Start automatisch `https://github.com/zurd46/CoderLLM.git`
- Falls du einen anderen Namespace benutzt: in jedem `0X_*.ipynb` in der `_bootstrap()`-Funktion die URL anpassen

---

## Ausführen (strikte Reihenfolge)

**Link-Muster für Colab:**
```
https://colab.research.google.com/github/zurd46/CoderLLM/blob/main/training/<NAME>.ipynb
```

Alternativ: in Colab **File → Open notebook → GitHub-Tab → `zurd46/CoderLLM`**.

| # | Notebook | Dauer H100 | Output |
|---|---|---|---|
| 1 | `01_data_build.ipynb` | 20 min | 3 HF-Datasets: `coder-16b-dyn-{sft,dpo,longctx}` |
| 2 | `02_sft_unsloth.ipynb` | ~12 h | LoRA-v1: `coder-16b-dyn-lora-sft` |
| 3 | `03_dpo_unsloth.ipynb` | ~4 h | LoRA-v2: `coder-16b-dyn-lora-dpo` |
| 4 | `04_longctx.ipynb` | ~2 h | LoRA-v3: `coder-16b-dyn-lora-longctx` |
| 5 | `05_export.ipynb` | ~40 min | Gemergter BF16-Checkpoint auf HF: `coder-16b-dyn-base-fp16` |

**Wichtig:** Jede Phase liest den LoRA der vorherigen Phase. Also nicht parallel ausführen.

Nach Phase 5: lokal auf Mac:

```bash
cd build_pipeline
# configs/quants.yaml temporär auf base: zurd46/coder-16b-dyn-base-fp16 ändern
python -m build.cli auto --yes --build-from-source
```

→ produziert MLX + GGUF Quants deiner fine-getunten Version für LM Studio.

---

## Was die Notebooks beim Start tun (automatisch)

Jedes Notebook hat als **erste Zelle** einen `_bootstrap()`-Aufruf. Der macht:

1. Erkennt, ob in Colab
2. Liest `HF_TOKEN` aus Colab-Secrets (`userdata.get("HF_TOKEN")`)
3. Cloned das Repo nach `/content/CoderLLM` (falls nicht da)
4. `cd /content/CoderLLM/training` (damit Configs gefunden werden)
5. `pip install` aller nötigen Pakete (Unsloth, TRL, etc.)

Du musst nichts davon manuell machen.

---

## Source-of-truth vs. Notebooks

Die `.py`-Dateien sind die Quelle. Die `.ipynb`-Dateien werden generiert:

```bash
cd training
python make_notebooks.py   # regeneriert alle *.ipynb aus *.py
```

Wenn du die Trainings-Logik ändern willst: **bearbeite die `.py`**, dann
`python make_notebooks.py`, dann committe beide. Colab öffnet immer die `.ipynb`.

---

## Troubleshooting

**OOM auf H100 trotz 80 GB?**
- In `configs/sft.yaml`: `max_seq_length` von 32768 → 16384 reduzieren, oder `per_device_train_batch_size` → 1.

**Colab-Timeout nach 12 h:**
- Alle Notebooks haben `save_steps: 500` → checkpoint-resume funktioniert. Einfach Notebook neu starten, es nimmt letzten Checkpoint auf (TRL findet ihn in `output_dir`).

**Dataset-Push schlägt fehl (403):**
- HF_TOKEN hat keinen write-Scope. Neu erstellen mit **Write**-Rolle.

**Unsloth install failt:**
- Runtime-Typ prüfen: muss GPU sein (A100 oder H100). Auf CPU-Runtime failed bitsandbytes.

---

## Fine-Tune Dauer / Kosten (Schätzung)

| Phase | H100-Stunden | Compute-Units |
|---|---|---|
| 01 Data-Build (CPU-only) | - | 0 |
| 02 SFT 3 epochs × 940M tokens | 12 h | ~14 CU |
| 03 DPO 1 epoch | 4 h | ~5 CU |
| 04 LongCtx Layer 30-47 | 2 h | ~3 CU |
| 05 Export / merge | 0.5 h | ~1 CU |
| **Total** | **~18 h** | **~23 CU** |

Bei Colab Pro: ~12 USD. Bei Colab Pro+: im Monats-Quota enthalten.
