# Fine-Tuning Audit — Gefundene Probleme & Behebungsplan

> Stand: 2026-04-22  
> Scope: `training/` (Colab-Notebooks + RunPod-Pipeline)  
> Aktion: **NUR Plan** — keine Code-Änderungen durchgeführt.

---

## Legende

| Prio | Bedeutung |
|---|---|
| 🔴 Kritisch | Führt mit hoher Wahrscheinlichkeit zu Crash, OOM oder inkonsistenten Ergebnissen |
| 🟡 Mittel | Suboptimale Ergebnisse, versteckte Bugs oder Wartungsprobleme |
| 🟢 Niedrig | Kosmetisch / Architektur-Verbesserung |

---

## 🔴 Kritisch

### 1. Dataset-Namens-Inkonsistenz (Colab `01_data_build.py` ↔ Configs)
**Beschreibung:**  
Die Colab-Version von `01_data_build.py` pushed Datasets nach:
- `zurd46/coder-16b-dyn-sft`
- `zurd46/coder-16b-dyn-dpo`
- `zurd46/coder-16b-dyn-longctx`

Alle `configs/*.yaml` (und damit `02_sft_unsloth.py`, `03_dpo_unsloth.py`, `04_longctx.ipynb`) laden aber von:
- `zurd46/EliCoder-Dataset-SFT`
- `zurd46/EliCoder-Dataset-DPO`
- `zurd46/EliCoder-Dataset-LongCtx`

Die RunPod-Version (`runpod/01_data_build.py`) nutzt korrekterweise die `EliCoder-*` Namen.

**Impact:** Colab-Training schlägt fehl (DatasetNotFoundError) oder lädt veraltete/alte Datasets.

**Behebung:**
1. `training/01_data_build.py`: `OWNER/coder-16b-dyn-*` → `OWNER/EliCoder-Dataset-*` angleichen.
2. Alternativ: `configs/*.yaml` auf `coder-16b-dyn-*` ändern (nicht empfohlen, da `rename_repos.py` + `push_modelcards.py` auf EliCoder-* setzen).

---

### 2. Export-Skript `05_export.py` (Colab) — Hardcodes & falsche `max_seq_length`
**Beschreibung:**
- `ADAPTER = f"{OWNER}/coder-16b-dyn-lora-longctx"` stimmt nicht mit `longctx.yaml` Output (`EliCoder-30B-A3B-LoRA-LongCtx`) überein.
- `MERGED_REPO = f"{OWNER}/coder-16b-dyn-base-fp16"` stimmt nicht mit `push_modelcards.py` (`EliCoder-30B-A3B`) überein.
- `max_seq_length=4096` beim Export **überschreibt** den Tokenizer/Model-Context auf 4k. Damit werden die in Phase C teuer trainierten Long-Context-Fähigkeiten (bis 200k) im exportierten Modell deaktiviert.

**Impact:** Exportiertes Modell kann keine Long-Context-Aufgaben lösen, obwohl dafür trainiert wurde. Adapter-Repo wird nicht gefunden.

**Behebung:**
1. `ADAPTER` und `MERGED_REPO` aus `longctx.yaml` / Umgebungsvariablen lesen.
2. `max_seq_length` auf den Wert aus `longctx.yaml` (`40960` oder höher, z.B. `131072`) setzen.

---

### 3. `PYTORCH_ALLOC_CONF` Typo in `02_sft_unsloth.py` (Colab)
**Beschreibung:**  
Zeile 112 setzt `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"`.  
Korrekt wäre: `PYTORCH_CUDA_ALLOC_CONF`.

**Impact:** `expandable_segments` wird nie aktiviert. Erhöhtes OOM-Risiko auf H100 bei langen Sequenzen + Packing.

**Behebung:** Variable in Zeile 12 und 112 konsistent auf `PYTORCH_CUDA_ALLOC_CONF` ändern.

---

### 4. `eval_fraction` in Configs definiert, aber in Colab-Trainings nicht verwendet
**Beschreibung:**  
`sft.yaml`, `dpo.yaml`, `longctx.yaml` definieren `eval_fraction` (0.01, 0.005, 0.02) und `eval_strategy: steps/epoch`.  
Die Colab-Skripte (`02_sft_unsloth.py`, `03_dpo_unsloth.py`, `04_longctx.ipynb/.py`) erstellen aber **kein** `eval_dataset` und übergeben es nicht an den Trainer.

**Impact:** Wenn `eval_strategy != "no"`, crasht der Trainer mit `ValueError: evaluation requires an eval_dataset` oder ignoriert die Evaluierung stillschweigend (je nach TRL-Version).

**Behebung:** Colab-Skripte um `ds.train_test_split(test_size=eval_fraction)` erweitern und `eval_dataset` an Trainer übergeben (wie in RunPod-Version bereits implementiert).

---

### 5. `save_total_limit` fehlt in `dpo.yaml` (und teilweise in Colab-Code)
**Beschreibung:**  
`dpo.yaml` hat kein `save_total_limit`. Der Colab-Code von `03_dpo_unsloth.py` liest es direkt aus der Config (`CFG["training"]["save_total_limit"]`) ohne Fallback.

**Impact:** Alle Checkpoints werden aufbewahrt → volle Disk bei längerem Training oder vielen `save_steps`.

**Behebung:** `save_total_limit: 2` in `dpo.yaml` ergänzen. Im Colab-Code optionalen Fallback hinzufügen.

---

## 🟡 Mittel

### 6. `original_max_position_embeddings` in `longctx.yaml` stimmt nicht mit Basis-Modell überein
**Beschreibung:**  
`longctx.yaml` setzt `original_max_position_embeddings: 65536`.  
Qwen3-Coder-30B-A3B-Instruct hat laut HuggingFace-Config jedoch typischerweise `32768` (oder `4096`).

**Impact:** Wenn `original_max_position_embeddings > tatsächliche_embeddings`, kann Unsloth/HF beim RoPE-Scaling eine falsche Frequenz berechnen oder einen Index-Error werfen.

**Behebung:** Wert aus dem tatsächlichen Basis-Modell (`config.json` von `Qwen/Qwen3-Coder-30B-A3B-Instruct`) auslesen und in `longctx.yaml` korrigieren.

---

### 7. Long-Context synthetische Daten (Colab) — schlechte Qualität & Plattform-Abhängigkeit
**Beschreibung:**  
- `01_data_build.py` (Colab) nutzt `/usr/share/dict/words` (nur auf Linux vorhanden). Fallback ist `["the"] * 5000`.
- Token-Längen-Schätzung mit `L // 6` (Zeichen pro Token) ist extrem ungenau.
- RunPod-Version (`runpod/01_data_build.py`) ist hier deutlich besser: Python-Code als Haystack.

**Impact:** Colab-LongCtx-Dataset ist realitätsfremd oder zu kurz. Training generalisiert schlecht auf echte Codebases.

**Behebung:** Colab-Version an RunPod-Version angleichen (`synth_long_ctx_code` übernehmen).

---

### 8. `gradient_checkpointing: false` in `sft.yaml` bei MoE-Modell
**Beschreibung:**  
`sft.yaml` hat `gradient_checkpointing: false`. Die RunPod-Version erkennt im `native_torch` MoE-Fall das OOM-Risiko und erzwingt `gradient_checkpointing=True`.

**Impact:** Bei native_torch MoE-Backend (oder größerem effektivem Batch durch Packing) kann es zu OOM kommen, obwohl H100 80GB genug haben sollte.

**Behebung:** Entweder `gradient_checkpointing: true` in `sft.yaml` setzen, oder den RunPod-Override-Mechanismus (`native_torch`-Erkennung) auch in die Colab-Version übernehmen.

---

### 9. Export `max_seq_length=4096` auch in RunPod `05_export.py`
**Beschreibung:**  
Auch `runpod/05_export.py` lädt das Modell mit `max_seq_length=4096`.

**Impact:** Wie bei Punkt 2 — Long-Context geht verloren, auch in der "produktionsreifen" RunPod-Pipeline.

**Behebung:** `max_seq_length` auf `131072` (oder den Wert aus `longctx.yaml`) setzen.

---

### 10. Kein expliziter `dtype` in `04_longctx.ipynb/.py` (Colab)
**Beschreibung:**  
`FastLanguageModel.from_pretrained(..., dtype=None)` lässt Unsloth entscheiden. Bei Long-Context sollte explizit `torch.bfloat16` gesetzt werden.

**Impact:** Mögliche automatische FP16-Wahl bei älteren GPUs, was zu Gradient-Overflow führen kann.

**Behebung:** `dtype=torch.bfloat16` explizit setzen.

---

### 11. `wandb.login()` wird nie aufgerufen
**Beschreibung:**  
Alle Skripte setzen `WANDB_PROJECT` und `WANDB_NAME`, aber keines ruft `wandb.login()` oder prüft, ob ein Login besteht.

**Impact:** In frischen Umgebungen (neuer Colab-Runtime, neuer RunPod) hängt der Training-Start, weil WandB nach einem Login promptet oder stille Datenverluste erzeugt.

**Behebung:** Optional `wandb.login()` aufrufen oder `report_to="none"` falls kein `WANDB_API_KEY` gesetzt ist (RunPod macht das teilweise schon).

---

### 12. Code-Duplikation: `_bootstrap` in allen Colab-Dateien
**Beschreibung:**  
Die komplette `_bootstrap()` Funktion ist identisch in `01_data_build.py` bis `05_export.py` (Colab) kopiert.

**Impact:** Wartungsaufwand. Änderungen (z.B. neues Paket, neuer Secret-Name) müssen an 5 Stellen gepflegt werden. Risiko von Divergenz.

**Behebung:**
- `_bootstrap.py` (bereits vorhanden im Root) importieren, oder
- `from _bootstrap import bootstrap` nutzen (wie RunPod-Version).

---

## 🟢 Niedrig

### 13. `make_notebooks.py` — fehlende Validierung
**Beschreibung:**  
Wenn eine `.py`-Datei keine `# %%` Marker hat, wird sie als eine einzige riesige Code-Zelle exportiert. Es gibt keine Prüfung, ob der Output syntaktisch gültig ist.

**Impact:** Gering — aktuell funktioniert es.

**Behebung:** Optionale JSON-Validierung oder Cell-Count-Warnung hinzufügen.

---

### 14. `device_map={"": 0}` überall hardcoded
**Beschreibung:**  
Keine Multi-GPU Unterstützung. Für Colab/RunPod Single-GPU OK, aber nicht flexibel.

**Impact:** Gering — aktuelles Setup ist Single-GPU.

**Behebung:** Optional aus Config lesen oder `auto` erlauben.

---

### 15. `rename_repos.py` löscht Modell-Repos ohne Backup-Warnung
**Beschreibung:**  
Das Skript löscht `coder-16b-dyn-lora-*` und `coder-16b-dyn-base-fp16` Repos, wenn sie existieren.

**Impact:** Wenn Checkpoints oder wichtige READMEs in den alten Repos sind, sind sie unwiderruflich weg.

**Behebung:** `--dry-run` als Default, `--apply` erst nach Bestätigung. Oder Backup-Hinweis im Skript.

---

## Zusammenfassung: Empfohlene Reihenfolge der Behebung

| Reihenfolge | Problem | Dateien |
|---|---|---|
| 1 | Dataset-Namens-Inkonsistenz | `01_data_build.py`, ggf. `configs/*.yaml` |
| 2 | `PYTORCH_ALLOC_CONF` Typo | `02_sft_unsloth.py` |
| 3 | `eval_fraction` nicht verwendet (Colab) | `02_sft_unsloth.py`, `03_dpo_unsloth.py`, `04_longctx.py` |
| 4 | Export `max_seq_length` & Hardcodes | `05_export.py`, `runpod/05_export.py` |
| 5 | `save_total_limit` in `dpo.yaml` | `configs/dpo.yaml` |
| 6 | `original_max_position_embeddings` prüfen | `configs/longctx.yaml` |
| 7 | Long-Context Datensynthese verbessern | `01_data_build.py` |
| 8 | `_bootstrap` Deduplizierung | Alle `0X_*.py` (Colab) |
| 9 | `wandb.login()` / Reporting-Sicherheit | Alle Trainings-Skripte |
| 10 | `gradient_checkpointing` konsistent | `configs/sft.yaml` oder Colab-Override |

---

*Plan erstellt durch Audit des `training/`-Verzeichnisses (Colab + RunPod).*  
*Keine Code-Änderungen wurden vorgenommen.*
