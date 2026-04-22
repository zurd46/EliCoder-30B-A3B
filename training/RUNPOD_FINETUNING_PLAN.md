# RunPod Fine-Tuning Master-Plan

> Stand: 2026-04-22  
> **Quellsystem:** `training/runpod/` (H100, primäre Pipeline)  
> **Sekundär:** `training/*.py` / `*.ipynb` (Colab, Fallback/Doku)  
> Ziel: Fehler beheben + 30–50 % schneller auf RunPod, Qualität erhalten.

---

## Architektur-Entscheidung

RunPod ist die **einzige produktive Pipeline**. Die Colab-Dateien (`training/0X_*.py`, `*.ipynb`) dienen nur noch als Fallback/Referenz.  
**Konsequenz:** Alle kritischen Fixes und Speed-Optimierungen landen zuerst in `training/runpod/`. Die Colab-Dateien werden danach synchronisiert (oder als veraltet markiert).

---

## Teil 1 — Kritische Bugs (müssen vor dem nächsten Training gefixt werden)

### 1.1 Export zerstört Long-Context (`runpod/05_export.py`)
**Datei:** `training/runpod/05_export.py`  
**Zeile 32:** `max_seq_length=4096`

Das gemergte BF16-Modell wird mit einem Tokenizer/Config limitiert auf **4k Kontext**. Das überschreibt die in Phase C (LongCtx) trainierte YaRN-Rescaling-Erweiterung auf 131k–262k. Das exportierte Modell vergisst effektiv alles über 4k.

**Fix:**
```python
max_seq_length=131072  # oder 262144, je nach longctx.yaml Ziel
```

Alternativ: Den Wert aus `configs/longctx.yaml` (`target_max_position_embeddings: 262144`) auslesen und setzen.

**Impact wenn nicht gefixt:** Phase C (2h Training, ~3 CU) ist komplett verschwendet.

---

### 1.2 `original_max_position_embeddings` in `configs/longctx.yaml` prüfen
**Datei:** `training/configs/longctx.yaml`  
**Zeile 7:** `original_max_position_embeddings: 65536`

Qwen3-Coder-30B-A3B-Instruct hat in seiner `config.json` typischerweise `max_position_embeddings: 32768` (nicht 65536).  
Wenn YaRN mit `original=65536` und `factor=4.0` rechnet, skaliert es auf `262144`. Wenn das Modell aber nur auf `32768` trainiert wurde, ist die Frequenzberechnung falsch — das Modell sieht Positionen, die es nie gelernt hat, mit falschen Winkeln.

**Fix:**
1. `config.json` von `Qwen/Qwen3-Coder-30B-A3B-Instruct` prüfen.
2. `original_max_position_embeddings` auf den echten Wert setzen (wahrscheinlich `32768`).
3. `factor` anpassen, damit `32768 × factor ≈ 262144` → `factor = 8.0`.

**Impact wenn nicht gefixt:** Long-Context-Retention bricht bei >65k ein (Needle-in-Haystack Test wird failen).

---

### 1.3 Dataset-Namen in `configs/*.yaml` vs. `runpod/01_data_build.py`
**Status:** ✅ **Bereits konsistent** in RunPod.  
- `runpod/01_data_build.py` pushed `EliCoder-Dataset-SFT/DPO/LongCtx`
- `configs/*.yaml` laden `EliCoder-Dataset-SFT/DPO/LongCtx`

**Aber:** `training/01_data_build.py` (Colab) ist inkonsistent → Colab als veraltet markieren oder synchronisieren.

---

### 1.4 `save_total_limit` fehlt in `configs/dpo.yaml`
**Datei:** `training/configs/dpo.yaml`

Im RunPod-Code gibt es einen Fallback (`T.get("save_total_limit", 2)`), aber die Config sollte es explizit definieren. Ohne expliziten Wert riskiert man, dass eine neue TRL-Version den Fallback ändert und die Disk voll läuft.

**Fix:** `save_total_limit: 2` in `configs/dpo.yaml` ergänzen.

---

## Teil 2 — Speed-Optimierungen für RunPod (priorisiert)

### 2.1 Dataset-Map Caching (hoher Impact bei Restart)
**Dateien:** `runpod/02_sft.py`, `runpod/03_dpo.py`, `runpod/04_longctx.py`

Bei jedem Pod-Restart (Spot-Preemption) wird das Dataset neu tokenisiert:
```python
ds = ds.map(fmt, remove_columns=ds.column_names)
```
Das kostet 3–5 Minuten pro Phase.

**Fix:**
```python
import hashlib
cache_key = hashlib.md5(
    f"{TRAIN['dataset']}_{TRAIN['max_seq_length']}_{tokenizer.name_or_path}".encode()
).hexdigest()
cache_file = f"/workspace/cache/{cache_key}.arrow"

ds = ds.map(
    fmt,
    remove_columns=ds.column_names,
    cache_file_name=cache_file,
    load_from_cache_file=True,
)
```
Das `/workspace/cache`-Verzeichnis persistiert auf Network Volume.

**Speedup:** 3–5 Minuten pro Restart. Bei mehreren Spot-Interrupts pro Tag summiert sich das.

---

### 2.2 Hub-Upload Strategie optimieren
**Dateien:** `runpod/02_sft.py`, `runpod/03_dpo.py`, `runpod/04_longctx.py`

Aktuell: `hub_strategy="checkpoint"` → jedes `save_steps` wird sofort zu HF hochgeladen. Das blockiert den Training-Loop oder verbraucht Upload-Bandbreite.

**Fix:**
```python
hub_strategy="end",  # oder "every_save" wenn du Zwischenstände willst
```
Mit `save_strategy="steps"` + `save_total_limit` bleiben die Checkpoints lokal in `/workspace/checkpoints/` erhalten. Am Ende wird das finale Modell hochgeladen.

**Trade-off:** Wenn der Pod mitten im Training stirbt (und nicht resumed), gehen die letzten Steps verloren. Aber RunPod hat ja bereits `get_last_checkpoint()` + lokale Resume. Der HF-Upload ist nur ein zusätzlicher Mirror.

**Speedup:** 3–5 % weniger Overhead während des Trainings.

---

### 2.3 Eval-Frequenz reduzieren
**Datei:** `configs/sft.yaml`

Aktuell: `eval_steps: 20` bei ~2.300 Gesamtsteps → **115 Eval-Runden**.  
Jede Runde lädt den Eval-Split und führt Forward durch.

**Fix:**
```yaml
eval_steps: 100  # oder 200
```

LongCtx hat bereits `eval_strategy: epoch` → gut so.
DPO hat `eval_steps: 200` → akzeptabel.

**Speedup:** ~5–8 % bei SFT.

---

### 2.4 `torch.compile` experimentell testen (potenziell 10–20 %)
**Dateien:** `runpod/02_sft.py`, `runpod/03_dpo.py`, `runpod/04_longctx.py`

Nach dem Model-Load:
```python
try:
    # "reduce-overhead" ist am besten für kleine Batches
    model = torch.compile(model, mode="reduce-overhead")
    print("✓ torch.compile active")
except Exception as e:
    print(f"✗ torch.compile skipped: {e}")
```

**Risiko:** Erste Steps brauchen länger (Kompilierung). Wenn Unsloth interne Triton-Kernels damit kollidieren, crasht es. Deshalb `try/except`.

**Empfohlene Vorgehensweise:** Einmal in Phase 02 (SFT) testen. Wenn es nach 50 Steps stabil läuft und der Throughput (`samples/s`) steigt, für alle Phasen aktivieren.

---

### 2.5 `dataloader_num_workers` + `pin_memory`
**Dateien:** `runpod/03_dpo.py`, `runpod/04_longctx.py`

Bei `packing=False` (DPO, LongCtx) kann das Data-Loading eine kleine Lücke erzeugen:
```python
args = SFTConfig(
    # ... bestehende Args ...
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)
```

Bei SFT ist `packing=True` → das Dataset ist bereits vorbereitet, also kein Impact.

**Speedup:** 2–5 % bei DPO/LongCtx.

---

### 2.6 Logging-Frequenz anpassen
**Datei:** `configs/sft.yaml`

`logging_steps: 5` ist sehr frequent. Alle 20 Steps reichen:
```yaml
logging_steps: 20
```

Minimaler Speedup (~1 %), aber weniger Noise in den Logs.

---

### 2.7 Gradient-Checkpointing bedingt abschalten (wenn schneller MoE-Kernel aktiv)
**Datei:** `runpod/02_sft.py`

Die RunPod-Version erzwingt `gradient_checkpointing=True` bei `native_torch` MoE-Backend. Wenn der schnelle grouped-GEMM-Kernel aktiv ist (`unsloth_triton`), bleibt mehr VRAM frei.

Aktuell wird GC bei `native_torch` erzwungen, aber bei `unsloth_triton` **nicht explizit ausgeschaltet**. Die Config hat `gradient_checkpointing: false`, aber wenn Unsloth intern was ändert, könnte es trotzdem aktiv sein.

**Fix:** Explizit sicherstellen, dass GC nur bei Bedarf aktiv ist:
```python
if os.environ.get("UNSLOTH_MOE_BACKEND") == "native_torch":
    TRAIN["gradient_checkpointing"] = True
    print("native_torch detected → gradient_checkpointing=True (OOM safety)")
else:
    TRAIN["gradient_checkpointing"] = False
    print("fast MoE kernel detected → gradient_checkpointing=False (speed)")
```

**Speedup:** 20–30 % wenn GC ausgeschaltet werden kann.

---

### 2.8 Datenaufbereitung parallelisieren (`runpod/01_data_build.py`)
**Datei:** `training/runpod/01_data_build.py`

Die Dataset-Loader laufen sequentiell. Die ersten 3–4 Quellen könnten parallel geladen werden (CPU-bound Netzwerk-Requests).

**Fix:** `concurrent.futures.ThreadPoolExecutor` für die unabhängigen Loader:
```python
from concurrent.futures import ThreadPoolExecutor

def load_with_label(fn_key):
    fn, key = fn_key
    return fn(MAX_PER_SOURCE[key]), key

with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(load_with_label, loaders))

parts = []
for d, key in results:
    print(f"  {key}: {len(d)}")
    parts.append(d)
```

**Speedup:** 2–3× schnellerer Dataset-Build (von ~25 min → ~10 min).

---

### 2.9 `torch.compile` für den Export (`runpod/05_export.py`)
**Datei:** `training/runpod/05_export.py`

Beim Merge-and-Unload könnte `torch.compile` helfen, aber der eigentliche Bottleneck ist das Speichern der Safetensors. Nicht kritisch.

**Niedrigere Priorität.**

---

### 2.10 `bf16` explizit setzen (Stabilität + Performance)
**Dateien:** Alle RunPod-Training-Skripte

Aktuell: `dtype=None` in `FastLanguageModel.from_pretrained()`.  
Unsloth wählt automatisch — meistens korrekt, aber nicht garantiert.

**Fix:**
```python
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"],
    max_seq_length=...,
    dtype=torch.bfloat16,  # explizit
    load_in_4bit=True,
    # ...
)
```

Sicherstellt, dass H100 Tensor Cores mit BF16 arbeiten (optimal).

---

## Teil 3 — Colab-Dateien (Sekundär, niedrige Prio)

Da RunPod primär ist, müssen die Colab-Dateien nicht sofort gefixt werden. Empfohlene Vorgehensweise:

1. **Option A:** Colab-Dateien löschen und durch einen Hinweis in `README.md` ersetzen: "Colab: Importiere die `.ipynb` aus dem letzten Release-Tag, wenn nötig."
2. **Option B:** Colab-Skripte automatisch aus `runpod/` generieren (z.B. via `make_notebooks.py` erweitern, dass es auch `runpod/` verarbeitet).
3. **Option C:** Colab-Dateien manuell synchronisieren (aufwändig, fehleranfällig).

**Empfehlung:** Option A oder B. Die RunPod-Skripte sind reiner Python-Code ohne Notebook-Magie — sie lassen sich 1:1 in Colab-Zellen kopieren.

---

## Teil 4 — Testplan nach den Änderungen

Vor dem nächsten vollen Training sollte ein **Dry-Run** durchgeführt werden:

### 4.1 Phase 01 (Dataset-Build)
- [ ] Läuft in <15 Minuten durch?
- [ ] Datasets sind unter `zurd46/EliCoder-Dataset-*` erreichbar?
- [ ] `analyze_seqlen.py` zeigt plausible Token-Längen?

### 4.2 Phase 02 (SFT) — nur 50 Steps testen
- [ ] Kein OOM bei bsz=2, seq=6144?
- [ ] `torch.compile` läuft stabil (falls aktiviert)?
- [ ] Throughput > 2.0 samples/s?
- [ ] Eval-Loss wird berechnet (wenn `eval_fraction > 0`)?

### 4.3 Phase 03 (DPO) — nur 50 Steps testen
- [ ] `ref_model=None` DPO lädt korrekt?
- [ ] Kein OOM?

### 4.4 Phase 04 (LongCtx) — nur 20 Steps testen
- [ ] YaRN-Config wird korrekt geladen (`rope_scaling` im Modell-Config sichtbar)?
- [ ] `original_max_position_embeddings` stimmt mit Qwen-Config überein?

### 4.5 Phase 05 (Export)
- [ ] Gemergtes Modell hat `max_position_embeddings >= 131072` in `config.json`?
- [ ] Needle-in-Haystack Test mit 32k/64k/128k funktioniert?

---

## Zusammenfassung: Empfohlene Umsetzungsreihenfolge

| # | Aufgabe | Datei(en) | Geschätzte Zeit |
|---|---|---|---|
| 1 | Export `max_seq_length` fixen | `runpod/05_export.py` | 2 min |
| 2 | `original_max_position_embeddings` prüfen/fixen | `configs/longctx.yaml` | 5 min |
| 3 | `save_total_limit` in DPO-Config | `configs/dpo.yaml` | 1 min |
| 4 | Dataset-Map Caching | `runpod/02-04_*.py` | 15 min |
| 5 | Hub-Upload `strategy="end"` | `runpod/02-04_*.py` | 10 min |
| 6 | Eval-Frequenz + Logging anpassen | `configs/sft.yaml`, `configs/dpo.yaml` | 5 min |
| 7 | Gradient-Checkpointing bedingt | `runpod/02_sft.py` | 10 min |
| 8 | `dataloader_num_workers` + `pin_memory` | `runpod/03-04_*.py` | 10 min |
| 9 | `torch.compile` experimentell | `runpod/02_sft.py` | 20 min |
| 10 | Datenaufbereitung parallel | `runpod/01_data_build.py` | 20 min |
| 11 | `bf16` explizit setzen | Alle `runpod/0*.py` | 10 min |
| 12 | Colab-Dateien veralten/markieren | `training/README.md` | 5 min |

**Gesamtaufwand:** ~2 Stunden Implementierung + 1 Stunde Dry-Run-Test.

---

*Plan erstellt für RunPod-H100 als primäre Pipeline.*
