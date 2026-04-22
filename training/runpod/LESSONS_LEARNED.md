# RunPod Training Pipeline — Erkenntnisse & Workflow

> **Zielgruppe:** Jeder der diese Pipeline weiter entwickelt oder einen neuen Pod hochzieht.
> **Stand:** 2026-04-22

Dieses Dokument bündelt Dinge, die nicht offensichtlich aus dem Code hervorgehen, aber Stunden + Kosten gekostet haben bis sie klar waren.

---

## 1. Pod-Image ist der wichtigste Hebel für Training-Speed

### Entscheidung: CUDA-Version beim Pod-Create

Auf H100-NVL mit Qwen3-Coder-30B-A3B (MoE) ist der MoE-Kernel der einzige relevante Speed-Bottleneck. Drei Backends, gesteuert über `UNSLOTH_MOE_BACKEND`:

| Backend | Benötigt | Step-Time (30B-A3B, max_seq=4096, bsz=1, ga=64) |
|---|---|---|
| `native_torch` | Immer verfügbar (Python-Loop über Experts) | **~200–500 s/step** 🔴 |
| `unsloth_triton` | Triton ≥ 3.3 mit `tl.make_tensor_descriptor` | **~15–30 s/step** 🟢 |
| `grouped_mm` | torch ≥ 2.8 | **~8–20 s/step** 🟢🟢 |

### Welches Pod-Image bringt welchen Backend?

| RunPod-Image | CUDA | torch | triton | MoE-Backend möglich |
|---|---|---|---|---|
| `pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | 12.4 | ≤ 2.6 | 3.2 | **nur native_torch** 🔴 |
| `pytorch:2.7.0-py3.11-cuda12.6.3-cudnn-devel-ubuntu22.04` | 12.6 | 2.7 | 3.3 | `unsloth_triton` ✅ |
| `pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` | 12.8 | 2.8 | 3.4 | `grouped_mm` ✅✅ |

**Immer cu128 wählen, wenn verfügbar.** Der kleine Zeitverlust beim Pull wird in <1 h Training wieder reingespielt.

### Kostenrechnung (konkret aus unserer Session)

- **Wrong image (cu124/torch 2.6/triton 3.2):** 465 s/step × 370 Steps × $3.07/h = **~$144** nur für SFT
- **Right image (cu128/torch 2.8/triton 3.4):** 15 s/step × 370 Steps × $3.07/h = **~$5** für SFT
- **Delta:** $139 Ersparnis, 43 Stunden kürzere Laufzeit

---

## 2. Triton-Version ist strikt von torch abhängig

### Inkompatibilitäts-Matrix

Der pip-Resolver lässt sich bei `torch 2.6` NICHT überreden, eine neuere Triton zu akzeptieren — und wenn man's erzwingt, bricht torch:

```
ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'
```

Ursache: `torch._inductor.runtime.hints` importiert `AttrsDescriptor` am Modul-Start. `AttrsDescriptor` wurde in Triton 3.4 entfernt. Heilt erst in torch 2.7.

### Umgekehrt: torch 2.6 + triton 3.2 = kein TMA

```
UNSLOTH_MOE_BACKEND=native_torch (triton 3.2.0 lacks tl.make_tensor_descriptor → native loop fallback)
```

Der stable `tl.make_tensor_descriptor`-API kam erst in Triton 3.4. Vorher nur `_experimental_make_tensor_descriptor`, und Unsloth's Kernel-AST-Lookup akzeptiert nur den non-experimental Namen.

### Faustregel

**Niemals triton einzeln pinnen — es ist torch's dependency.** Version wird automatisch gezogen vom korrekten pod-image. Nur wenn explizit gepinnt werden muss (z.B. um Up-Resolver-Drift zu verhindern), dann exakt auf die Torch-kompatible Version.

---

## 3. Flash Attention 2 ist nicht optional

Ohne FA2 fällt Unsloth auf xformers zurück. Auf H100 mit BF16 bedeutet das **1.5–2× Slowdown pro Attention-Layer**. Beim 30B-Modell mit 48 Layern summiert sich das enorm.

### Installation

**NIEMALS** aus PyPI (`pip install flash-attn`) — das triggert einen 20-min-Source-Build mit nvcc.

**IMMER** prebuilt wheel vom Dao-AILab Release.

### Wheel-URL-Format (wichtige Fallen)

`flash_attn-{VERSION}+{CUDA}torch{TORCH}cxx11abi{ABI}-cp3{PY}-cp3{PY}-linux_x86_64.whl`

| Feld | Wert bei torch 2.8 + cu128 + Python 3.11 | Anmerkung |
|---|---|---|
| VERSION | **2.8.3** (NICHT 2.8.1) | v2.8.1 hat nur torch 2.10 wheels — eine Release-Nummer sagt NICHTS über die torch-Matrix aus. |
| CUDA | **cu12** (NICHT cu128) | Dao nutzt nur cu11/cu12 als Major, nicht minor. |
| TORCH | **torch2.8** | Exakte Major.Minor — cu128 Torch braucht torch 2.8, nicht torch 2.7 wheel. |
| ABI | **TRUE** oder **FALSE** | Prüfen mit: `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` — PyPI cu128-Wheels sind `TRUE`, manylinux-Wheels `FALSE`. |
| PY | **311** | `cp310`/`cp311`/`cp312` etc. |

### Richtige URL dynamisch finden

```bash
curl -s "https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3" \
  | python3 -c "
import json, sys
for a in json.load(sys.stdin)['assets']:
    if 'torch2.8' in a['name'] and 'cp311' in a['name'] and 'abiTRUE' in a['name']:
        print(a['browser_download_url'])"
```

Die Wheel-Liste ist **nicht chronologisch** — neue Torch-Versionen landen oft erst in späteren `v2.8.x`-Releases.

### Aktueller Pin (torch 2.8 + cu128 + cp311)

```
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

### Verifikation im Unsloth-Log

Richtig:
```
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
```

Falsch:
```
Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead.
```

---

## 4. Pod-Deployment Workflow

### Per `runpodctl create pod` (CLI)

```bash
runpodctl create pod \
  --name coderllm-sft \
  --gpuType "NVIDIA H100 NVL" \
  --imageName "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04" \
  --gpuCount 1 \
  --containerDiskSize 200 \
  --volumeSize 150 \
  --volumePath /workspace \
  --mem 150 \
  --vcpu 19 \
  --ports "22/tcp" \
  --secureCloud
```

**Nachteil:** bekommt on-demand-Preis ($3.07/h H100-NVL), kein Reserved-Rate ($2.59/h) — für Reserved muss man über die UI.

### SSH-Daemon läuft NICHT automatisch

RunPod-pytorch-Images haben openssh-server installiert, aber der Daemon wird **nicht gestartet** und Host-Keys fehlen. Direct-TCP (Port 22 durch Pod's public-TCP-mapping) liefert `Connection refused` bis du folgendes im Pod-Web-Terminal machst:

```bash
ssh-keygen -A && service ssh start
```

Dann läuft direct-TCP auf dem `--ports 22/tcp`-Public-Port (steht in `runpodctl get pod -a`).

### SSH-Proxy (Fallback) ist sofort verfügbar

`ssh <pod-id>-<account-id>@ssh.runpod.io -i ~/.ssh/id_ed25519` funktioniert sofort nach dem Deploy, auch ohne lokalen sshd — RunPod's Proxy hat seinen eigenen Auth-Layer auf Basis der Account-SSH-Keys.

**Problem:** Proxy braucht PTY (`-tt`). Das tut sich automatisiert schlecht — für Scripts lieber direct-TCP nach einmaligem Setup.

### Public-Key für direct-TCP einmalig hinterlegen

Nach `ssh-keygen -A && service ssh start` im Pod-Terminal noch:

```bash
mkdir -p /root/.ssh && \
echo "ssh-ed25519 AAAA…" >> /root/.ssh/authorized_keys && \
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
```

(Der Account-Key vom RunPod-Dashboard wird **nicht automatisch** in die Container-`authorized_keys` kopiert — nur der Proxy kennt ihn.)

---

## 5. `.env` für Pipeline-Credentials

[pipeline.sh](pipeline.sh) sourced `/workspace/.env` wenn `HF_TOKEN` in der Shell nicht gesetzt ist. Minimum:

```bash
HF_TOKEN=hf_xxx
WANDB_PROJECT=CoderLLM        # optional — ohne key nur stdout
WANDB_API_KEY=xxx             # optional
RUNPOD_API_KEY=rpa_xxx        # für Auto-Stop am Ende von Phase 06
RUNPOD_POD_ID=<current-pod-id>  # für Auto-Stop
CODERLLM_REPO_URL=https://github.com/zurd46/CoderLLM.git
```

`chmod 600 /workspace/.env` — sonst sieht jeder der auf der Pod-Maschine sitzt den Token.

---

## 6. Gated HF-Datasets brauchen Account-Freigabe

Salesforce/xlam-function-calling-60k ist **gated** — nur mit Token aufrufbar wenn der HF-Account (der zum Token gehört) Zugriff hat. Das geht nur manuell:

1. https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
2. "Agree and access repository"
3. Freigabe (meistens binnen Minuten, manchmal Stunden)

Ohne Freigabe crasht Phase 01 beim DPO- und LongCtx-Build, weil beide xLAM als Tool-Schema-Quelle nutzen.

**Im Code:** [01_data_build.py:load_xlam_tool_calls()](01_data_build.py) fängt `PermissionError` ab und skipped — aber `build_agent_dpo()` und `synth_long_ctx_agentic()` crashen dann.

**Fallback-Plan:** Glaive-Function-Calling-v2 als Tool-Schema-Pool nehmen. Noch nicht implementiert.

---

## 7. Agentik-Fine-Tuning — Was wirklich zählt

### Das Core-Problem

Qwen3-Coder-30B-A3B ist bereits auf Milliarden Code-Tokens trainiert. **Noch mehr Coding-Daten bringen nichts.** Was fehlt, ist Tool-Use-Verhalten:

1. Direkte Tool-Calls ohne Preamble
2. Korrektes Schema-Parsing
3. Multi-turn Tool-Loops

### Dataset-Komposition (reiner Agentik-Fokus)

| Dataset | Samples | Zweck |
|---|---|---|
| xLAM-function-calling-60k | 60k | BFCL-Style single-turn (gold standard) |
| Glaive-function-calling-v2 | 25k | Diverse Tool-Schemas, Dialog-Kontext |
| ToolACE | 15k | Multi-turn Agent-Loops |
| Hermes-function-calling-v1 | 15k | OpenAI-kompatible Calls |
| SWE-bench_Verified | 500 | Real repo patches (Struktur) |

**Absichtlich weggelassen:** Coding-General-Datasets (Magicoder, CodeFeedback, Self-OSS-Instruct, OpenCodeReasoning, Nemotron-OpenCode). Jedes Coding-Sample zieht das Modell Richtung geschwätziger Prosa — genau das Gegenteil vom Ziel.

### Die drei entscheidenden Config-Knobs

In [../configs/sft.yaml](../configs/sft.yaml):

1. **`max_seq_length: 4096`** — Unter 4k passen Multi-Turn Tool-Traces nicht rein. Wichtiger als du denkst.
2. **`target_modules`: inkl. `gate_proj`, `up_proj`, `down_proj`** — MoE-Router für Tool-Routing adaptieren. Ohne das lernt das Modell keine neuen Tool-Auswahl-Entscheidungen.
3. **`packing: true`** — Packt 130k Samples auf ~11k pack-sequences → 370 Steps. Ohne Packing: ~2000 Steps, 5× längeres Training.

### Conciseness-DPO — der Mac-Speed-Hebel

[build_agent_dpo()](01_data_build.py) generiert synthetische DPO-Paare:

- `chosen` = direkter `<tool_call>`-Block ohne Prosa
- `rejected` = Preamble ("Of course! Let me help…") + exakt derselbe Tool-Call

DPO trainiert also *nur* den Unterschied "kurz vs. geschwätzig" — Korrektheit bleibt konstant. Output-Tokens pro Call sinken messbar von ~180 auf ~60. Das ist **3× schnellere Decode auf M-Series Macs**.

---

## 8. Watch.py + Agent-Eval Integration

### [_agent_eval.py](_agent_eval.py) — Callback für Trainer

Feuert nach jedem `trainer.evaluate()` (also an `eval_steps`). Zieht 50 Held-out xLAM-Samples (Shuffle-Seed 999 statt 42), generated Tool-Calls, berechnet:

- `agent/parse_rate` — % valide `<tool_call>`-JSON-Blöcke
- `agent/name_match` — % korrekter Tool-Name
- `agent/avg_output_tokens` — **Mac-Speed-Proxy**. Sinkt er über Training, wirkt Conciseness.

Metriken gehen in stdout (garantiert) + W&B (wenn konfiguriert).

### [watch.py](watch.py) — Live-Monitor

Parsed `[agent-eval] step=N {…}`-Zeilen aus `/workspace/pipeline.log` via tail-F. Neue Felder im Metrics-Panel:

- `Agent parse` — farbcodiert (grün ≥90%, gelb ≥70%, rot darunter)
- `Tool-name match` — farbcodiert
- `Out-tokens/call` — Wert + Trend-Pfeil ggü. erstem Messpunkt (↓ grün = Mac schneller)

Default-Pod-ID ist im Code gepinnt (`POD_ID = "3xc1b6nzhkgmqq"` oder der jeweils aktuelle). Für einen neuen Pod: entweder Default editieren + commit, oder mit `POD_ID=xxx python watch.py` überschreiben.

---

## 9. Pipeline-Idempotenz + Resume

[pipeline.sh](pipeline.sh) schreibt `/workspace/.phaseNN_done`-Marker pro erfolgreicher Phase. Beim Restart werden Phasen mit existierendem Marker übersprungen.

**Wann Marker manuell löschen:**
- Nach Code-Änderung in `01_data_build.py` (Dataset wird neu gebaut)
- Nach Config-Änderung in `sft.yaml`/`dpo.yaml`/`longctx.yaml` (alte Checkpoints inkompatibel)
- Nach Pod-Rebuild (Volume neu → Marker eh weg, keine Aktion nötig)

```bash
rm -f /workspace/.phase0*_done /workspace/pipeline.log
rm -rf /workspace/checkpoints/
```

**Checkpoint-Resume innerhalb einer Phase** läuft automatisch via Transformers' `get_last_checkpoint()` — solange die Config-Shape identisch bleibt. Bei Config-Änderung (z.B. max_seq 2048→4096) crasht Resume.

---

## 10. Schnell-Diagnose wenn SFT plötzlich langsam wird

Reihenfolge zur Fehlersuche:

```bash
# 1. Welcher MoE-Backend ist aktiv?
grep "UNSLOTH_MOE_BACKEND" /workspace/pipeline.log
# Wenn "native_torch" → cu124-Pod, du hast das falsche Image.

# 2. Flash Attention 2 aktiv?
grep "FA2 = " /workspace/pipeline.log
# "FA2 = True" muss stehen — sonst fehlt flash-attn.

# 3. Triton/torch versions?
python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"
# torch 2.6 + triton 3.2 = 🔴
# torch 2.7 + triton 3.3 = 🟢
# torch 2.8 + triton 3.4 = 🟢🟢

# 4. TMA-API verfügbar (= unsloth_triton möglich)?
python3 -c "from triton import language as tl; print(hasattr(tl,'make_tensor_descriptor'))"

# 5. GPU-Util während Training (sollte 60–95% sein bei gesundem Setup)
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader
# Power << 250W + Util < 50% → CPU/Kernel-Bottleneck, nicht GPU-Compute
```

---

## 11. Häufige Fallstricke — Quick Reference

| Symptom | Ursache | Fix |
|---|---|---|
| `ImportError: AttrsDescriptor` | triton 3.4 + torch 2.6 | `pip install triton==3.2.0 --force-reinstall --no-deps` + neues pod-image |
| `465s/step` auf H100 | native_torch MoE | Pod-Image auf cu128 |
| `Connection refused` auf port 34xxx | sshd läuft nicht | `ssh-keygen -A && service ssh start` im web-term |
| `Permission denied (publickey)` | authorized_keys fehlt | Public-Key manuell einfügen |
| `Dataset is gated` (xLAM) | HF-Account ohne Freigabe | HF-Webseite → "Agree and access" |
| `CUDA out of memory` beim ersten Forward | max_seq zu groß für bsz=2 | `bsz=1 + gradient_checkpointing=True` |
| `torch.compile incompatible with 4-bit + PEFT` | bekannt, nicht fixbar | nicht nutzen — `compile=False` |
| `FA2 = False` trotz Install | falsche wheel-Matrix | Matrix prüfen (cu12/torch2.6/cp311/cxx11abiFALSE) |
| `size of tensor a (2048) must match tensor b (768)` beim 1. Step | `grouped_mm` × LoRA-auf-MoE-Experts | `UNSLOTH_MOE_BACKEND=unsloth_triton` in `.env` — siehe Sektion 11b |
| `RuntimeError` beim ersten Training-Step, GPU util = 0 | Backend/Kernel-Mismatch | Stack-Trace in pipeline.log prüfen — oft MoE × LoRA oder FA2-Kollision |

---

## 11b. GROSSE FALLE: grouped_mm × LoRA-auf-MoE-Experts = Shape-Crash 🔴

**Symptom** (im ersten Training-Step):

```
File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 823, in forward
    second_gemm_output = second_gemm_output + lora_delta
RuntimeError: The size of tensor a (2048) must match the size of tensor b (768) at non-singleton dimension 1
```

**Ursache:** Wenn `configs/sft.yaml` `target_modules` **`gate_proj`, `up_proj`, `down_proj` enthält** (also LoRA auf MoE-Expert-Projektionen), kollidiert das mit Unsloth's schnellstem MoE-Backend `grouped_mm`:

- `grouped_mm` verarbeitet **alle Experts zusammen** in einem GEMM → Output-Shape `[total_tokens_across_experts × hidden]` (z. B. 2048)
- LoRA-Adapter produzieren ein Delta **pro Expert** mit Shape `[single_expert_tokens × hidden]` (z. B. 768)
- Addition scheitert an non-singleton dim 1

Das tritt **nur** beim ersten Training-Step auf — Loading, Tokenize, Packing, Trainer-Init sind alle ok. Man verliert dadurch ~15 min Setup-Zeit pro Crash.

**Welches Backend wählt Unsloth automatisch?** Bei `UNSLOTH_MOE_BACKEND=auto` mit torch 2.8 + triton 3.4 (TMA verfügbar) pickt Unsloth `grouped_mm` als fastest. Das crasht mit unseren MoE-LoRA-Targets.

### Fix: `finetune_all_experts=True` an Unsloth durchreichen

**Das ist der EIGENTLICHE Fix** — `UNSLOTH_MOE_BACKEND=unsloth_triton` alleine **behebt das Problem NICHT**. Der Crash tritt auch mit `unsloth_triton` auf, weil das Problem nicht im Kernel liegt, sondern in der **LoRA-Adapter-Verkabelung auf MoE-Experts**.

In [02_sft.py](02_sft.py) an `FastLanguageModel.get_peft_model()`:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=…,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    # Unsloth-spezifisch — wrappt LoRA so dass deltas korrekt pro-Expert
    # addiert werden. Ohne True: Shape-Mismatch beim ersten Forward.
    finetune_all_experts=True,
    …
)
```

Und in [../configs/sft.yaml](../configs/sft.yaml):

```yaml
lora:
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  finetune_all_experts: true   # PFLICHT wenn MoE-Experts in target_modules
```

### Optional zusätzlich: `unsloth_triton` statt `grouped_mm`

`grouped_mm` ist ~2× schneller als `unsloth_triton`, funktioniert aber nicht mit finetuned Experts auf allen Versionen. Sicherer Pfad:

```bash
# /workspace/.env
UNSLOTH_MOE_BACKEND=unsloth_triton
```

Beides zusammen (`finetune_all_experts=True` + `unsloth_triton`) ist die getestete-stabile Kombo.

### Kompatibilitäts-Matrix MoE-Backend × LoRA-Target

| Backend | LoRA auf q/k/v/o | LoRA auf gate/up/down (MoE-Experts) | Speed |
|---|---|---|---|
| `native_torch` | ✓ | ✓ | 🔴 sehr langsam |
| `unsloth_triton` | ✓ | ✓ | 🟢 schnell (TMA) |
| `grouped_mm` | ✓ | **✗** — Shape-Crash | 🟢🟢 am schnellsten |

**Faustregel:** Wenn `target_modules` in sft.yaml MoE-Experts enthält → immer `UNSLOTH_MOE_BACKEND=unsloth_triton` explizit setzen. Nicht auf `auto` vertrauen.

### Alternative: LoRA auf MoE-Experts weglassen

```yaml
# sft.yaml
lora:
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    # gate_proj / up_proj / down_proj NICHT drin
```

Dann kann `grouped_mm` aktiv bleiben (~2× schneller als `unsloth_triton`). **Nachteil:** MoE-Router-Adaption fällt weg — für reine Agentik-Runs ist das aber oft die bessere Wahl, weil die Tool-Routing-Entscheidung eher aus Attention als aus Expert-Routing kommt.

---

## 12. Besonders perfide Fallstricke (2026-04-22 Session)

### Bootstrap überschreibt Image-Torch silently
RunPod-Image `pytorch:2.8.0-py3.11-cuda12.8.1` kommt **mit torch 2.8.0.dev+cu128 vorinstalliert**. Wenn `_bootstrap.py` aber `TORCH_PACKAGES=["torch==2.6.0+cu124"]` hat, wird dieser Image-Zustand **bei jedem Pipeline-Start heruntergestuft auf torch 2.6+cu124**. Ohne Error-Message. Das Resultat: cu128-Pod läuft mit cu124-Stack und verliert den schnellen MoE-Kernel.

**Fix:** `TORCH_PACKAGES` und `TORCH_INDEX_URL` müssen zum Image-CUDA-Level passen. Vor jedem Push prüfen:
```bash
# Auf Image: welche CUDA-Runtime hat torch?
python3 -c "import torch; print(torch.version.cuda)"
# → 12.8 → TORCH_INDEX_URL muss cu128 sein
```

### DEPS_MARKER vortäuscht sauberen Stack
`/workspace/.deps_installed` ist nur ein `touch`-File. Es sagt nur "bootstrap lief mindestens einmal durch" — NICHTS über welche Versionen oder ob der Install erfolgreich war. Bei Wechsel auf neues Pod-Image: **Marker explicit löschen**, sonst wird der falsche Stack weiterverwendet.

```bash
rm -f /workspace/.deps_installed   # force re-bootstrap
```

### Paralleler Pipeline-Run auf einem Pod
Ein `nohup bash pipeline.sh &` während noch ein alter `02_sft.py` läuft → **beide Prozesse teilen sich VRAM** → OOM oder Thrashing. `pgrep/pkill` vor jedem Restart ist kritisch. Immer so:
```bash
pkill -9 -f "python.*0[0-9]_\|pipeline.sh" ; sleep 2
# erst DANACH neuer Start
```

### Version 2.8.1 ≠ Version 2.8.3 bei flash-attn
Flash-Attention-Releases sind **fragmentiert nach torch-Kompatibilität**, nicht chronologisch. `v2.8.1` hat nur Wheels für torch 2.10, `v2.8.3` hat die für torch 2.8. Ein neuerer Release-Tag ≠ neuerer torch-Support. **Immer via GitHub-API-Query die passende Version finden**, nicht aus der README.

### `pip install -U` geht ins user-site statt system-site
Bei `root`-User im RunPod-Container: `pip install -U triton` kann ins `~/.local/lib/python3.11/site-packages/` installieren, NICHT ins System `/usr/local/lib/python3.11/dist-packages/`. Der Python-Import nimmt dann die alte System-Version, nicht die User-Version. Fix: `pip install --force-reinstall --no-deps` oder explizit `--target`.

---

## 13. Was beim nächsten Mal anders zu machen wäre

1. **Immer mit dem neuesten CUDA-Image starten** — nicht mit dem Default-pytorch:2.4-Image. Das spart mehrere $100 pro Training.
2. **xLAM-Freigabe VOR dem ersten Pod-Start** einholen. Gated-Datasets blockieren die Pipeline sonst in Phase 01.
3. **Einmal `runpodctl create pod` automatisieren** mit SSH-daemon-Setup-Inline als `--args` oder via Template. Spart 5 min pro Rebuild.
4. **Ein kleines "Smoke-Test"-Script** vor dem vollen Training: nur 10 SFT-Steps + 10 DPO-Steps, dauert 5 min, verrät ob der Stack läuft bevor 6 h durchlaufen.
5. **Bessere Config-Validierung im bootstrap:** Print eine Zeile `EXPECTED: torch 2.8, triton 3.4, FA2=True` vs `ACTUAL: ...` — direkte Visual Red-Flag wenn's nicht passt.
