# RunPod Fine-Tuning — Audit & Speedup Plan

> **Scope:** `training/runpod/` ONLY  
> **Hardware:** RunPod H100 (Hopper, 80GB VRAM)  
> **Basis-Modell:** `Qwen/Qwen3-Coder-30B-A3B-Instruct` (MoE)  
> **Ziel:** Bugs fixen + 30–50% schneller, Qualität erhalten

---

## Teil 1 — KRITISCHE BUGS (Training würde fehlschlagen oder Ergebnisse zerstören)

### BUG-01: Export zerstört Long-Context (`05_export.py`)
**Datei:** `training/runpod/05_export.py` · Zeile 32

```python
max_seq_length=4096  # ❌ ZERSTÖRT die in Phase C trainierten 200k Context-Fähigkeiten
```

Das gemergte BF16-Modell wird mit einem Context-Limit von 4k exportiert. Der Tokenizer und die Model-Config werden auf 4096 beschränkt. Alle Needle-in-Haystack-Trainings aus Phase C sind damit wirkungslos.

**Fix:**
```python
max_seq_length=131072  # oder 262144 — muss mit longctx.yaml übereinstimmen
```

**Impact wenn nicht gefixt:** Phase C (~2h, ~3 CU) ist komplett verschwendet. Das finale Modell kann keine >4k Contexte verarbeiten.

---

### BUG-02: `original_max_position_embeddings` stimmt nicht mit Qwen3-Modell überein
**Datei:** `training/configs/longctx.yaml` · Zeile 7

Aktuell: `original_max_position_embeddings: 65536`

Qwen3-Coder-30B-A3B-Instruct hat in seiner `config.json` den Wert `32768` (nicht 65536). YaRN berechnet die Frequenz-Skalierung aus `original × factor`. Wenn der Original-Wert falsch ist, sind alle >32k Positionen mit falschen RoPE-Winkeln kodiert.

**Fix:**
1. `config.json` von `Qwen/Qwen3-Coder-30B-A3B-Instruct` prüfen.
2. `original_max_position_embeddings` auf echten Wert setzen (wahrscheinlich `32768`).
3. `factor` anpassen: `32768 × 8.0 = 262144`.

**Impact wenn nicht gefixt:** Needle-in-Haystack bricht bei >65k ein. Model "vergisst" Passphrasen in langen Dokumenten.

---

### BUG-03: `save_total_limit` fehlt in `configs/dpo.yaml`
**Datei:** `training/configs/dpo.yaml`

Im Code gibt es einen Fallback (`T.get("save_total_limit", 2)`), aber die Config selbst definiert es nicht. Bei einer neuen TRL-Version oder unerwartetem Verhalten kann das Disk-Full führen.

**Fix:** `save_total_limit: 2` explizit in `configs/dpo.yaml` ergänzen.

---

### BUG-04: Dataset-Map wird bei jedem Restart neu berechnet
**Dateien:** `runpod/02_sft.py`, `runpod/03_dpo.py`, `runpod/04_longctx.py`

Bei Spot-Preemption oder manuellem Restart wird `ds.map(fmt, ...)` jedes Mal neu ausgeführt. Das kostet 3–5 Minuten pro Phase.

**Fix:** `cache_file_name` setzen:
```python
cache_file = "/workspace/cache/sft_tokenized.arrow"
ds = ds.map(fmt, remove_columns=ds.column_names, cache_file_name=cache_file)
```

---

## Teil 2 — SPEED-OPTIMIERUNGEN (ohne Qualitätsverlust)

### SPEED-01: Gradient-Checkpointing nur bei native_torch MoE-Backend
**Datei:** `runpod/02_sft.py`

Aktuell wird GC bei `native_torch` erzwungen. Wenn der schnelle `unsloth_triton` Kernel aktiv ist, könnte GC **ausbleiben** und 20–30% Speed bringen.

Die Logik sollte explizit sein:
```python
if os.environ.get("UNSLOTH_MOE_BACKEND") == "native_torch":
    TRAIN["gradient_checkpointing"] = True
else:
    TRAIN["gradient_checkpointing"] = False  # Speed!
```

**Speedup:** 20–30% bei SFT, wenn schneller Kernel verfügbar.

---

### SPEED-02: Hub-Upload auf `strategy="end"`
**Dateien:** `runpod/02_sft.py`, `runpod/03_dpo.py`, `runpod/04_longctx.py`

Aktuell: `hub_strategy="checkpoint"` → Upload bei jedem Save-Step. Das blockiert den Loop.

**Fix:** `hub_strategy="end"` → nur am Ende hochladen. Lokale Checkpoints in `/workspace/checkpoints/` bleiben erhalten.

**Speedup:** 3–5% weniger Overhead.

---

### SPEED-03: Eval-Frequenz reduzieren
**Datei:** `configs/sft.yaml`

Aktuell: `eval_steps: 20`. Bei ~2.300 Steps = 115 Eval-Runden.

**Fix:** `eval_steps: 100` oder `200`.

**Speedup:** 5–8% bei SFT.

---

### SPEED-04: `torch.compile` experimentell testen
**Dateien:** `runpod/02_sft.py`, `runpod/03_dpo.py`, `runpod/04_longctx.py`

```python
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("torch.compile active")
except Exception as e:
    print(f"torch.compile skipped: {e}")
```

**Speedup:** Potenziell 10–20%. Risiko: Erste Steps langsamer (Kompilierung).

---

### SPEED-05: Datenaufbereitung parallelisieren
**Datei:** `runpod/01_data_build.py`

Dataset-Loader laufen sequentiell. Mit `ThreadPoolExecutor` parallelisieren:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda t: (t[0](MAX_PER_SOURCE[t[1]]), t[1]), loaders))
```

**Speedup:** 2–3× bei Phase 01 (von ~25 min → ~10 min).

---

### SPEED-06: `dataloader_num_workers` + `pin_memory`
**Dateien:** `runpod/03_dpo.py`, `runpod/04_longctx.py`

Bei `packing=False` (DPO, LongCtx) kann Data-Loading eine Lücke erzeugen:
```python
args = SFTConfig(
    # ...
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)
```

**Speedup:** 2–5% bei DPO/LongCtx.

---

### SPEED-07: `bf16` explizit setzen
**Dateien:** Alle `runpod/0*.py`

Aktuell: `dtype=None` (Unsloth entscheidet). Explizit `torch.bfloat16` sicherstellen:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    dtype=torch.bfloat16,  # statt None
    # ...
)
```

**Impact:** Sicherstellung, dass H100 Tensor Cores mit BF16 arbeiten.

---

## Teil 3 — IMPLEMENTIERUNGSREIHENFOLGE

| Reihenfolge | Bug/Speedup | Datei | Aufwand |
|---|---|---|---|
| 1 | **BUG-01** Export max_seq_length fixen | `05_export.py` | 2 min |
| 2 | **BUG-02** original_max_position_embeddings prüfen | `configs/longctx.yaml` | 5 min |
| 3 | **BUG-03** save_total_limit in DPO | `configs/dpo.yaml` | 1 min |
| 4 | **BUG-04** Dataset-Map Caching | `02-04_*.py` | 15 min |
| 5 | **SPEED-01** Gradient-Checkpointing bedingt | `02_sft.py` | 10 min |
| 6 | **SPEED-02** Hub-Upload "end" | `02-04_*.py` | 10 min |
| 7 | **SPEED-03** Eval-Frequenz | `configs/sft.yaml` | 2 min |
| 8 | **SPEED-04** torch.compile testen | `02_sft.py` | 20 min |
| 9 | **SPEED-05** Daten parallel | `01_data_build.py` | 20 min |
| 10 | **SPEED-06** dataloader workers | `03-04_*.py` | 10 min |
| 11 | **SPEED-07** bf16 explizit | Alle `0*.py` | 10 min |

---

## Teil 4 — TESTPLAN (Dry-Run vor vollem Training)

### Phase 01 (Dataset-Build)
- [ ] Läuft durch in <15 Minuten
- [ ] Datasets unter `zurd46/EliCoder-Dataset-*` erreichbar

### Phase 02 (SFT) — nur 50 Steps
- [ ] Kein OOM bei bsz=2, seq=6144
- [ ] Throughput messen (Vorher vs. Nachher)
- [ ] torch.compile stabil (falls aktiviert)

### Phase 03 (DPO) — nur 50 Steps
- [ ] Adapter-Load von SFT funktioniert
- [ ] Kein OOM

### Phase 04 (LongCtx) — nur 20 Steps
- [ ] YaRN-Config korrekt im Modell
- [ ] `original_max_position_embeddings` stimmt

### Phase 05 (Export)
- [ ] `config.json` zeigt `max_position_embeddings >= 131072`
- [ ] Needle-in-Haystack 32k/64k/128k funktioniert

---

*Plan erstellt ausschließlich für `training/runpod/` · Colab/Root-Dateien werden ignoriert.*
