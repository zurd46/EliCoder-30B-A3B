# Training auf RunPod (End-to-End automatisiert)

Alternative zu Colab — ein Pod, alle 5 Phasen automatisch, ~19 h auf H100, **Budget ~$25–50**.

## Warum RunPod statt Colab

| | Colab Pro+ | RunPod |
|---|---|---|
| Timeout | 24 h hart | kein Timeout |
| GPU-Garantie | "best effort" | GPU buchbar + dediziert |
| Spot-Interrupts | — | ja, aber Auto-Resume ist drin |
| Kosten für full run | in CU enthalten | ~$25–50 je nach GPU |
| Setup | 0 | 15 min einmalig |

---

## 1. Vorbereitung (einmalig, ~10 min)

### 1.1 runpodctl installieren
```bash
curl -L https://github.com/runpod/runpodctl/releases/download/v1.14.4/runpodctl-darwin-arm64 \
  -o runpodctl && chmod +x runpodctl && sudo mv runpodctl /usr/local/bin/

mkdir -p ~/.runpod && touch ~/.runpod/config.yaml
runpodctl config --apiKey <DEIN_RUNPOD_KEY>
```

### 1.2 HF-Token + optionaler WandB-Key
Den **HuggingFace-Token** (Scope: write) brauchst du — er landet später im Pod als ENV-Var.

Optional: `WANDB_API_KEY` für Training-Metriken auf wandb.ai.

### 1.3 GitHub-Zugang
Die Pipeline klont `https://github.com/zurd46/CoderLLM.git`. Repo muss **public** sein — oder setze `CODERLLM_REPO_URL` mit Token: `https://<TOKEN>@github.com/<user>/CoderLLM.git`.

---

## 2. GPU-Wahl

| GPU | VRAM | $/h (On-Demand) | $/h (Spot) | Wall-Clock | Erw. Kosten | Empfehlung |
|---|---|---|---|---|---|---|
| **H100 NVL** | **94 GB** | **$2.59** | $1.40 | **~16 h** | **$41 (OD) / $22 (Spot)** | 🏆 beste GPU, On-Demand passt in $50 |
| H100 80GB HBM3 | 80 GB | $2.69 | $1.50 | ~16 h | $43 / $24 | Alternative, gleich schnell |
| A100-SXM4-80GB | 80 GB | $1.39 | $0.79 | ~26 h | $36 / $21 | sparsamer, langsamer |
| A100 80GB PCIe | 80 GB | $1.19 | $0.60 | ~30 h | $36 / $18 | günstigst on-demand |

**Gewählt: H100 NVL On-Demand** — Hopper-Architektur, 94 GB (entspannt für Long-Context 131k), keine Spot-Interrupts, läuft in einem Tag durch.

Zeitschätzung für H100 NVL:
- 01 Data-Build (CPU): ~20 min
- 02 SFT: ~10 h (durch GPU-Opts schneller als Colab-12h)
- 03 DPO: ~3.5 h
- 04 LongCtx: ~2 h
- 05 Export: ~40 min
- **Total ~16 h × $2.59 = ~$41**

---

## 3. Pod starten

### 3.1 Pod erstellen (On-Demand H100 NVL)

```bash
runpodctl create pod \
  --name coderllm-sft \
  --gpuType "NVIDIA H100 NVL" \
  --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 200 \
  --volumeSize 0 \
  --mem 64 \
  --vcpu 8 \
  --ports "22/tcp,8888/http" \
  --env "HF_TOKEN=hf_DEIN_TOKEN_HIER" \
  --env "CODERLLM_REPO_URL=https://github.com/zurd46/CoderLLM.git"
```

Kontrolliere Pod-ID + Status:
```bash
runpodctl get pod
```

### 3.2 Pipeline starten (SSH rein)

```bash
# Pod-ID aus runpodctl get pod
runpodctl exec python --pod <POD_ID> -c "print('alive')"

# Oder direkt SSH (ssh-key ist durch runpodctl config bereits uploaded)
ssh root@<POD_IP> -p <POD_SSH_PORT>
```

Im Pod:
```bash
# RunPod-API-Key + Pod-ID für Auto-Shutdown setzen
export RUNPOD_API_KEY="rpa_..."
export RUNPOD_POD_ID="<POD_ID>"

# Pipeline klonen + starten
git clone --depth 1 https://github.com/zurd46/CoderLLM.git /workspace/CoderLLM
cd /workspace/CoderLLM/training
bash runpod/pipeline.sh 2>&1 | tee -a /workspace/pipeline.log &
disown
```

Mit `&` + `disown` läuft es weiter auch wenn SSH getrennt wird.

---

## 4. Progress beobachten

```bash
# Log streamen
ssh root@<POD_IP> -p <PORT> "tail -f /workspace/pipeline.log"

# Phasen-Status
ssh root@<POD_IP> -p <PORT> "ls /workspace/.phase*_done 2>/dev/null"

# GPU-Auslastung
ssh root@<POD_IP> -p <PORT> "nvidia-smi"

# Disk-Usage
ssh root@<POD_IP> -p <PORT> "df -h /workspace"
```

Oder via WandB Dashboard wenn `WANDB_API_KEY` gesetzt war.

---

## 5. Auto-Resume (Spot-Interrupts, OOM, Pod-Restart)

Jede Trainings-Phase nutzt **drei Schutzmechanismen**:

1. **Checkpoints lokal** — `/workspace/checkpoints/<phase>/checkpoint-<step>` (persistiert solange Container-Disk lebt)
2. **Checkpoints auf HF Hub** — via `hub_strategy="checkpoint"` nach jedem `save_steps` (Backup falls Pod komplett stirbt)
3. **Auto-Resume** — `get_last_checkpoint()` beim Trainer-Start findet letzten Checkpoint automatisch

Falls der Pod komplett neu gestartet wird, einfach `bash runpod/pipeline.sh` nochmal laufen lassen — `/workspace/.phaseNN_done` Marker sorgen dafür, dass abgeschlossene Phasen übersprungen werden.

---

## 6. Auto-Shutdown

Wenn `RUNPOD_API_KEY` + `RUNPOD_POD_ID` als ENV gesetzt sind, stoppt `pipeline.sh` den Pod nach Phase 5 automatisch via GraphQL. Spart das Kosten-Delta falls du nicht sofort da bist.

Manuell stoppen/löschen:
```bash
runpodctl stop pod <POD_ID>
runpodctl remove pod <POD_ID>   # endgültig, Disk weg
```

---

## 7. Dateien-Übersicht

```
training/
├── runpod/                     # RunPod-spezifisch (diese Files)
│   ├── _bootstrap.py           # ENV, Deps, TF32, Hopper-Opts
│   ├── 01_data_build.py        # CPU · Datasets bauen + HF-Push
│   ├── 02_sft.py               # SFT · Qwen3-30B 4bit + LoRA, H100-NVL-optimiert
│   ├── 03_dpo.py               # DPO · baut auf SFT-LoRA auf
│   ├── 04_longctx.py           # YaRN 262k · Layer 30-47 fine-tune
│   ├── 05_export.py            # Merge LoRA → BF16 base → HF Push
│   └── pipeline.sh             # Orchestrator, idempotent
├── 02_sft_unsloth.py           # Colab-Version (unverändert)
├── 03_dpo_unsloth.py           # Colab-Version (unverändert)
├── 04_longctx.py               # Colab-Version (unverändert)
├── 05_export.py                # Colab-Version (unverändert)
└── configs/                    # Shared — Colab UND RunPod nutzen die gleichen configs
    ├── sft.yaml
    ├── dpo.yaml
    └── longctx.yaml
```

---

## 8. GPU-Optimierungen gegenüber Colab

Aktiviert in [runpod/_bootstrap.py](runpod/_bootstrap.py) + [runpod/02_sft.py](runpod/02_sft.py):

- **TF32 matmul** — `torch.backends.cuda.matmul.allow_tf32 = True`
- **cuDNN benchmark** — auto-tunes conv-kernels
- **expandable_segments** — reduziert Fragmentation
- **Batch-Auto-Scaling** — auf H100 NVL (94 GB) wird `per_device_train_batch_size` von 1 → 2 erhöht, `gradient_accumulation_steps` halbiert → gleicher effektiver Batch, ~15 % weniger Wall-Clock
- **packing=True** im SFT — mehrere kurze Samples pro Sequenz, nutzt Context besser
- **Unsloth FlashAttention-2** — automatisch bei `FastLanguageModel.from_pretrained`

---

## 9. Troubleshooting

**`runpodctl: command not found` nach Reboot**
→ `/usr/local/bin` nicht in PATH. `export PATH=/usr/local/bin:$PATH` in `~/.zshrc`.

**Pod startet nicht, "out of capacity"**
→ H100 NVL ist belegt. Auf H100 80GB HBM3 wechseln (`--gpuType "NVIDIA H100 80GB HBM3"`) oder A100-SXM4.

**`HF_TOKEN not set` im Pod**
→ Beim `create pod` nicht als `--env` mitgegeben. Nachträglich: `ssh` rein, `echo 'HF_TOKEN=hf_...' >> /workspace/.env`.

**`CUDA out of memory` in Phase 4 (LongCtx)**
→ In `configs/longctx.yaml` temporär `max_seq_length: 131072` → `65536` reduzieren. Auch H100 NVL kann bei 131k eng werden.

**Phase hängt ohne Progress**
→ Unsloth Telemetry-Patch sollte das abfangen. Falls doch: `pkill -f python` im Pod, dann `bash runpod/pipeline.sh` neu — Auto-Resume übernimmt.

**Pipeline läuft, Pod-Bill explodiert**
→ `runpodctl stop pod <ID>` sofort. Log checken warum Auto-Shutdown nicht ausgelöst hat (RUNPOD_API_KEY fehlt wahrscheinlich).

---

## 10. Nach erfolgreichem Run

Phase 5 hat `zurd46/coder-16b-dyn-base-fp16` auf HF gepusht. Lokal auf deinem Mac:

```bash
cd build_pipeline
# configs/quants.yaml temporär: base.hf_repo = zurd46/coder-16b-dyn-base-fp16
python -m build.cli auto --yes --build-from-source
```

→ MLX + GGUF Quants für LM Studio.

Danach Pod entfernen (spart Storage-Kosten):
```bash
runpodctl remove pod <POD_ID>
```
