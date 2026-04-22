# Fine-Tuning Speedup-Plan — Schneller ohne Qualitätsverlust

> Stand: 2026-04-22  
> Scope: `training/` (Colab + RunPod)  
> Ziel: Trainingszeit reduzieren, BLEU/Pass@1/HumanEval-Score beibehalten oder verbessern.

---

## TL;DR — Top-Maßnahmen nach Impact

| # | Maßnahme | Geschätzter Speedup | Aufwand |
|---|---|---|---|
| 1 | MoE-Backend-Autokonfig + TF32 auch in Colab | **20–40 %** | Klein |
| 2 | Batch-Size 2 statt 1 auf H100 (beide Pipelines) | **15–25 %** | Klein |
| 3 | Eval-Frequenz von `steps:20` → `steps:100` | **5–10 %** | Minimal |
| 4 | Dataset-Map mit `cache_file_name` persistieren | **10–15 %** bei Restart | Klein |
| 5 | Logging `steps:5` → `steps:20` + WandB-Batch | **1–3 %** | Minimal |
| 6 | Hub-Upload auf `strategy="end"` oder `every_save` | **3–5 %** | Minimal |
| 7 | `torch.compile(model, mode="reduce-overhead")` testen | **10–20 %** (exp.) | Mittel |
| 8 | Gradient-Checkpointing nur bei Bedarf | **20–30 %** wenn aus | Klein |

> **Gesamtpotenzial:** Wenn alle Maßnahmen greifen, ist eine **Reduktion von ~30–50 % Trainingszeit** realistisch (z.B. SFT von 12 h → 7–8 h).

---

## 1. MoE-Backend + Hopper-Optimierungen in Colab aktivieren

**Problem:**  
Die RunPod-Pipeline hat `_configure_moe_backend()` und `apply_gpu_optims()` (TF32, cuDNN benchmark, torch-2.4-Backport). Die Colab-Versionen haben das alles **nicht**.  
Colab H100 läuft auf Hopper — genau wie RunPod. Ohne diese Patches:
- Unsloth fällt auf `native_torch` MoE-Loop zurück (langsamer Faktor 2–3×).
- TF32-Matmul bleibt deaktiviert (verpasster Speedup ~10–20 %).

**Lösung:**  
Die Optimierungen aus `runpod/_bootstrap.py` in die Colab-Bootstrap-Zelle übernehmen:
```python
# Vor jedem import torch / unsloth
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
```
Und die MoE-Backend-Erkennung (`tl.make_tensor_descriptor` Check + `UNSLOTH_MOE_BACKEND=native_torch` Fallback).

**Risiko:** Keines — reine Performance-Verbesserung, identische Ergebnisse.

---

## 2. Batch-Size 2 statt 1 auf H100 / H100-NVL

**Problem:**  
SFT läuft mit `per_device_train_batch_size=1, gradient_accumulation_steps=64`.  
MoE-Modelle (Qwen3-Coder-30B-A3B) haben **grouped-GEMM-Kernel**, die bei mehr Tokens pro Batch deutlich effizienter werden (bessere SM-Auslastung, weniger Kernel-Launch-Overhead).

**Lösung:**  
Wie RunPod bereits macht — dynamischer Override:
```python
if torch.cuda.get_device_properties(0).total_memory >= 80e9:  # 80 GB+
    batch_size = 2
    grad_accum = max(1, grad_accum // 2)
```
Dadurch bleibt der effektive Batch gleich (64), aber die GPU arbeitet pro Step effizienter.

**Risiko:** Minimal — bei OOM automatisch auf 1 zurückfallen. Mit Unsloth + 4-bit ist H100 80GB für bsz=2, seq=6144, MoE sicher genug.

---

## 3. Evaluierungsfrequenz reduzieren

**Problem:**  
- SFT: `eval_steps: 20` → bei ~2.300 Gesamtsteps wird **115× evaluiert**.
- DPO: `eval_steps: 200` → hier ist es OK.
- Jede Eval-Runde lädt den kompletten Eval-Split, führt Forward durch und berechnet Loss.

**Lösung:**  
- SFT: `eval_steps: 100` oder `200` (statt 20).
- LongCtx: `eval_strategy: epoch` ist bereits gut.

**Qualitäts-Impact:** Keiner — man verpasst keinen kritischen Moment. Mit `save_strategy: steps` bleibt das beste Modell trotzdem gesichert.

---

## 4. Dataset-Preprocessing cachen

**Problem:**  
Bei jedem Restart (Colab Timeout, Spot-Interrupt) wird das komplette Dataset neu tokenisiert:
```python
ds = ds.map(fmt, remove_columns=ds.column_names)
```
Bei 170k Samples × Chat-Template = ~3–5 Minuten Verlust pro Restart.

**Lösung:**  
`datasets`-Cache nutzen:
```python
ds = ds.map(
    fmt,
    remove_columns=ds.column_names,
    cache_file_name=f"/workspace/cache/{dataset_name}_tokenized.arrow",
    load_from_cache_file=True,
)
```
In Colab: `/content/cache/` verwenden (oder Google Drive mounten für Persistenz).

**Bonus:** Auch `01_data_build.py` könnte die finalen Datasets lokal cachen, bevor sie gepusht werden.

---

## 5. Logging & WandB Overhead reduzieren

**Problem:**  
- `logging_steps: 5` → alle ~30 Sekunden ein Log-Eintrag + WandB-Upload.
- WandB sync im Hintergrund blockiert manchmal den Training-Loop.

**Lösung:**  
- `logging_steps: 20` (SFT), `logging_steps: 20` (DPO).
- WandB in Offline-Mode starten, dann am Ende syncen:
  ```python
  os.environ["WANDB_MODE"] = "offline"
  # ... training ...
  # am Ende:
  subprocess.run(["wandb", "sync", wandb_dir])
  ```

**Risiko:** Keiner — Metriken sind identisch, nur zeitversetzt.

---

## 6. Hub-Upload Strategie anpassen

**Problem:**  
RunPod nutzt `hub_strategy="checkpoint"` → **jedes Mal** wenn ein Checkpoint gespeichert wird, läuft ein Upload zu HuggingFace im Hintergrund. Das blockiert teilweise den Prozess oder verbraucht Bandbreite.

**Lösung:**  
- `hub_strategy="every_save"` → Upload nur bei Save (identisch, aber explizit).
- **Besser:** `hub_strategy="end"` → nur am Ende hochladen.  
  Wenn `save_strategy: steps` + `save_total_limit` lokal funktioniert, reicht der End-Upload. Bei Spot-Instances bleibt das Risiko, aber RunPod hat ja lokale Checkpoints in `/workspace/checkpoints/`.

**Trade-off:** Bei Spot-Interrupt ohne HF-Upload geht der letzte Checkpoint verloren. Ausgleich: `save_total_limit` lokal hochsetzen und nur alle N Saves uploaden.

---

## 7. `torch.compile` evaluieren (experimentell)

**Problem:**  
PyTorch 2.x kann Modelle mit `torch.compile()` beschleunigen. Bei Unsloth-Modellen ist das lange problematisch, aber mit neueren Unsloth-Versionen sollte `mode="reduce-overhead"` funktionieren.

**Lösung:**  
Nach dem `FastLanguageModel.from_pretrained(...)` testweise:
```python
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("torch.compile active")
except Exception as e:
    print(f"torch.compile skipped: {e}")
```

**Risiko:** Kann bei ersten Steps länger brauchen (Kompilierung), danach aber 10–20 % schneller. Wenn es crasht, einfach im `except` überspringen.

---

## 8. Gradient-Checkpointing strategisch ein-/ausschalten

**Problem:**  
- `sft.yaml`: `gradient_checkpointing: false` → schneller, braucht mehr VRAM.
- RunPod erzwingt `gradient_checkpointing=True` bei `native_torch` MoE-Backend.
- Gradient-Checkpointing kostet **~20–30 % Speed** (zusätzlicher Forward-Pass).

**Lösung:**  
Wenn die MoE-Optimierungen aus Punkt 1 greifen (grouped-GEMM statt native_torch), bleibt mehr VRAM frei. Dann kann man **Gradient-Checkpointing abschalten** und gewinnt 20–30 % Speed.

Logik (wie RunPod, aber invertiert):
```python
if moe_backend_has_fast_kernel:
    gradient_checkpointing = False  # mehr Speed
else:
    gradient_checkpointing = True   # sonst OOM
```

**Risiko:** OOM wenn falsch eingeschätzt. Mit H100 80GB + Unsloth 4-bit + bsz=2 + seq=6144 ist das aber sicher.

---

## 9. `dataloader_num_workers` für DPO/LongCtx

**Problem:**  
Standard ist `dataloader_num_workers=0` (alles im Main-Process). Bei nicht-gepackten Datasets (DPO, LongCtx) kann das Data-Loading eine kleine Lücke zwischen den Steps erzeugen.

**Lösung:**  
```python
dataloader_num_workers=2,  # oder 4
```
Nur dort wo `packing=False` ist (DPO, LongCtx). Bei `packing=True` (SFT) ist das Dataset bereits vorbereitet, also kein Impact.

---

## 10. `pin_memory=True` bei DataLoader

**Problem:**  
`dataloader_pin_memory=False` in `sft.yaml`. Das bedeutet, CPU→GPU-Transfer ist synchron.

**Lösung:**  
```python
dataloader_pin_memory=True,
```
Besonders bei `batch_size=2` und H100 bringt das einen messbaren, wenn auch kleinen, Speedup.

---

## 11. Unsloth-spezifisch: `dtype` explizit setzen

**Problem:**  
`dtype=None` lässt Unsloth entscheiden. Manchmal wählt es automatisch FP16 statt BF16.

**Lösung:**  
Explizit `dtype=torch.bfloat16` in `from_pretrained(...)` überall setzen. BF16 ist auf H100 identisch schnell wie FP16, aber stabiler (kein Overflow).

---

## 12. Datenaufbereitung beschleunigen (`01_data_build.py`)

**Problem:**  
- Datasets werden sequentiell geladen.
- `concatenate_datasets(parts).shuffle(seed=42)` shufflet das komplette Array im RAM.

**Lösung:**  
- **Streaming** für große Quellen (Nemotron 40k, OpenCodeReasoning 40k) nutzen, wenn nur ein Subset gebraucht wird.
- `.shuffle(seed=42)` auf jeder Quelle **vor** dem `concatenate` ausführen, nicht danach. Das spart einen Full-RAM-Shuffle auf dem riesigen kombinierten Dataset.
- `synth_long_ctx` parallelisieren mit `multiprocessing.Pool` (die 8.000 Samples sind unabhängig voneinander).

---

## Zusammenfassung: Konsolidierter Plan (Fehler + Speedup)

Um beides auf einmal umzusetzen, empfehle ich diese **Reihenfolge**:

### Phase A — Kritische Fixes (Blocksiert Training)
1. Dataset-Namen angleichen (`01_data_build.py` Colab)
2. `PYTORCH_CUDA_ALLOC_CONF` Typo fixen
3. `eval_fraction` in Colab-Skripten implementieren
4. Export `max_seq_length` auf Long-Context-Wert setzen

### Phase B — Speed-Optimierungen (30–50 % schneller)
5. MoE-Backend + TF32 + cuDNN in Colab aktivieren
6. Batch-Size 2 Override für H100
7. Eval-Frequenz reduzieren (SFT: 20→100)
8. Dataset-Map Caching einbauen
9. Gradient-Checkpointing bedingt abschalten (wenn schneller MoE-Kernel aktiv)

### Phase C — Polish (weitere 5–10 %)
10. Hub-Upload `strategy="end"`
11. Logging-Frequenz anpassen
12. `torch.compile` experimentell testen

---

*Plan erstellt durch Audit der aktuellen Pipeline. Keine Code-Änderungen durchgeführt.*
