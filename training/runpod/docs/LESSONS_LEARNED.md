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

---

## 14. Session-Timeline 2026-04-22 — alle Fehler + Fixes im Überblick

Wir haben heute dieselbe Pipeline mehrfach neu gestartet, weil jeder Fix einen neuen Fehler aufdeckte. Diese Liste ist **vollständig** und zeigt die Reihenfolge — so dass bei einem ähnlichen Debug-Pfad später nichts übersehen wird.

### #1 — Native MoE-Kernel → 465 s/step (statt ~15 s)
**Symptom:** step_time = 464.9 s/it auf H100 NVL. GPU util = 43 %, Power 148 W (idle).
**Log:** `UNSLOTH_MOE_BACKEND=native_torch (triton 3.2.0 lacks tl.make_tensor_descriptor → native loop fallback)`
**Ursache:** Pod-Image `cuda12.4.1` pinnt torch 2.6 → strict dep triton 3.2.0 → kein TMA.
**Versuchter Fix (gescheitert):** `pip install -U triton` → `ImportError: AttrsDescriptor` weil torch 2.6._inductor triton 3.4 nicht supported.
**Wirklicher Fix:** Pod mit cu128-Image neu aufbauen.

### #2 — Flash Attention 2 nicht aktiv → xformers-Fallback
**Symptom:** Unsloth-Log: `Your Flash Attention 2 installation seems to be broken. Using Xformers instead.`
**Ursache:** `flash-attn` nicht im Bootstrap-PIP_PACKAGES.
**Fix:** Prebuilt wheel aus Dao-AILab-GitHub-Release einbinden. **Nicht** `pip install flash-attn` aus PyPI (20 min source build).

### #3 — Dataset `Salesforce/xlam-function-calling-60k` gated
**Symptom:** `PermissionError` in Phase 01. `xlam_tool_calls: SKIPPED (gated)`.
**Ursache:** HF-Dataset braucht manuelle "Agree and access"-Freigabe.
**Fix:** Auf HF-Web-UI Freigabe anfordern (5 min – 24 h Bearbeitung).
**Konsequenz:** ohne Fix crasht auch `build_agent_dpo()` und `synth_long_ctx_agentic()`, weil beide xLAM als Tool-Schema-Pool nutzen.

### #4 — Pod-SSH-Daemon läuft nicht automatisch
**Symptom:** `ssh … Connection refused` auf Public-TCP-Port nach Pod-Deploy.
**Ursache:** RunPod-pytorch-Images haben openssh-server installiert, aber der Daemon startet nicht, und Host-Keys fehlen.
**Fix:** Im Web-Terminal einmalig:
```bash
ssh-keygen -A && service ssh start && \
mkdir -p /root/.ssh && \
echo "<public-key>" >> /root/.ssh/authorized_keys && \
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
```
**Not-Fallback:** `ssh <pod-id>-<account-id>@ssh.runpod.io -i ~/.ssh/id_ed25519` (Proxy, braucht PTY).

### #5 — torch 2.7 nicht für cu124 verfügbar
**Symptom:** `pip install torch==2.7.0+cu124` → `ERROR: No matching distribution found`.
**Ursache:** PyTorch-Wheel-Index bietet cu124 nur bis torch 2.6.0.
**Fix:** Neuer Pod mit cu126/cu128-Image. (Die `runpod/pytorch:2.8.0-…-cuda12.8.1` Image ist die richtige.)

### #6 — Bootstrap überschreibt Image-Torch silently
**Symptom:** cu128-Image hat torch 2.8.0.dev+cu128 vorinstalliert. Nach `bootstrap()` zeigt `python -c "import torch; print(torch.__version__)"` → `2.6.0+cu124`.
**Ursache:** `_bootstrap.py` hatte hardcoded `TORCH_PACKAGES=["torch==2.6.0+cu124"]` mit `--index-url cu124`. Ohne Error-Message downgradet es den Image-Stack.
**Fix:** `TORCH_PACKAGES` + `TORCH_INDEX_URL` an das verwendete CUDA-Level anpassen. Jetzt cu128.

### #7 — `pip install -U` landet im user-site bei root
**Symptom:** `pip install -U triton` meldet Erfolg, `triton.__version__` zeigt aber noch alte Version.
**Ursache:** Mit root-User installiert pip manchmal ins user-site statt system-site. `sys.path` liest dann die alte System-Version zuerst.
**Fix:** `pip uninstall -y <pkg>` zuerst, dann `pip install --force-reinstall --no-deps <pkg>==<version>`.

### #8 — Flash-attn v2.8.1 hat KEINE torch-2.8-Wheels (nur 2.10)
**Symptom:** URL `flash_attn-2.8.1+cu128torch2.8…` → 404 Not Found.
**Ursache:** Flash-Attention-Releases sind **nicht linear** nach Torch-Kompat geordnet. v2.8.1 wurde nur für torch 2.10 gebaut; für torch 2.8 ist **v2.8.3** die erste verfügbare Version.
**Fix:** GitHub-API nach der richtigen Version fragen:
```bash
curl -s "https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3" \
  | jq -r '.assets[] | select(.name | contains("torch2.8") and contains("cp311")) | .browser_download_url'
```

### #9 — Flash-attn-Wheel-Name-Falle: `cu12` ≠ `cu128`, `abiTRUE` vs `FALSE`
**Symptom:** Auch bei richtiger Version schlägt der wheel-download fehl.
**Ursache:** Dao nutzt `cu12` als Major (nicht `cu124`/`cu128`), und `cxx11abi` muss zu torch passen.
**Fix:** Abi mit `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` prüfen (cu128-Wheels: **TRUE**, manylinux-Wheels: **FALSE**).

### #10 — `DEPS_MARKER` vortäuscht sauberen Stack
**Symptom:** Pipeline startet schnell "deps already installed — skipping" obwohl torch version falsch ist.
**Ursache:** `/workspace/.deps_installed` ist ein `touch`-File — sagt nur dass bootstrap mal durchlief, nicht dass die Versionen stimmen.
**Fix:** Bei Stack-Wechsel immer `rm -f /workspace/.deps_installed` erzwingen.

### #11 — `UNSLOTH_MOE_BACKEND=auto` pickt `grouped_mm` bei torch 2.8 + TMA
**Symptom:** `RuntimeError: size of tensor a (2048) must match tensor b (768) at non-singleton dimension 1` beim ersten Training-Step.
**Ursache:** `grouped_mm` verarbeitet alle MoE-Experts in einem GEMM, Shape stimmt nicht mit per-Expert-LoRA-Delta zusammen.
**Versuchter Fix (gescheitert):** `UNSLOTH_MOE_BACKEND=unsloth_triton` — gleicher Crash. Das Problem ist nicht der Kernel.

### #12 — **Echter Fix:** `finetune_all_experts=True` an `get_peft_model()`
**Symptom:** Identisch zu #11 — der Shape-Crash bleibt bei allen MoE-Backends solange LoRA-Adapter nicht pro-Expert gewrappt sind.
**Ursache:** Unsloth's `FastLanguageModel.get_peft_model()` nimmt einen `finetune_all_experts`-Flag. Ohne True landet ein LoRA-Delta pro Layer (768-wide), das MoE-Forward produziert 2048-wide Tensor.
**Fix:** In `02_sft.py` den Parameter durchreichen (der Config-Key in `sft.yaml` existierte schon, wurde nicht gelesen):
```python
model = FastLanguageModel.get_peft_model(
    model,
    target_modules=[..., "gate_proj", "up_proj", "down_proj"],
    finetune_all_experts=True,   # PFLICHT
    ...
)
```

### #13 — `pkill -f pipeline.sh` killt die SSH-Session
**Symptom:** `ssh … "pkill -f pipeline.sh; …"` liefert Exit 255 mitten in der Ausführung.
**Ursache:** Wenn der SSH-Prozess als Child unter pipeline.sh läuft, killt pkill ihn mit. Passiert auch via ssh.
**Fix:** PIDs explizit killen (`kill -9 <pid>`), oder `pkill` von außen mit separater Session. Immer danach mit `ps -ef | grep …` verifizieren.

### #14 — Parallele Pipeline-Runs teilen VRAM
**Symptom:** Nach `nohup bash pipeline.sh &` ist VRAM knapp, alter Run lief noch → OOM.
**Ursache:** `pgrep` findet alte Runs nicht immer (z.B. wenn sie `python runpod/02_sft.py` sind, nicht `pipeline.sh`).
**Fix:** Regex-breiter killen: `pkill -9 -f "python.*0[0-9]_\|pipeline.sh"`, dann `ps | grep` verifizieren, dann erst neu starten.

---

## 15. Pre-Flight Checklist — vor jedem Full SFT-Run

Diese Liste strikt vor `nohup bash runpod/pipeline.sh …` durchgehen. Verhindert die meisten Fallen aus Sektion 14:

### Pod-Setup
- [ ] Pod-Image ist `pytorch:2.8+` mit `cuda12.6+` (idealer 12.8) — **nicht** das Default cu124.
- [ ] `ssh-keygen -A && service ssh start` im Pod-Web-Terminal ausgeführt.
- [ ] Eigener public-key in `/root/.ssh/authorized_keys` hinterlegt + chmod 700/600.
- [ ] SSH direct-TCP funktioniert: `ssh -p <port> root@<ip> 'echo ok'` liefert `ok`.

### Dataset-Zugänge
- [ ] HF-Account hat Zugriff auf `Salesforce/xlam-function-calling-60k` (gated).
- [ ] `Team-ACE/ToolACE`, `NousResearch/hermes-function-calling-v1`, `glaiveai/glaive-function-calling-v2`, `princeton-nlp/SWE-bench_Verified` — alle offen, kein Freigabe nötig.

### /workspace/.env
```
HF_TOKEN=hf_…
WANDB_PROJECT=CoderLLM
WANDB_API_KEY=…                  # optional
RUNPOD_API_KEY=rpa_…
RUNPOD_POD_ID=<aktueller-pod>
UNSLOTH_MOE_BACKEND=unsloth_triton   # falls MoE-Experts in target_modules
```

### Config-Konsistenz-Check (`configs/sft.yaml`)
- [ ] `max_seq_length: 4096` (oder größer für Tool-Traces).
- [ ] Wenn `target_modules` enthält `gate_proj` / `up_proj` / `down_proj`:
  - [ ] `finetune_all_experts: true` MUSS gesetzt sein.
  - [ ] In [02_sft.py](02_sft.py): `finetune_all_experts=CFG["lora"]["finetune_all_experts"]` wird durchgereicht.
- [ ] `packing: true` (sonst 2000 statt 370 Steps).
- [ ] `gradient_checkpointing: true` (sonst OOM bei max_seq 4096).

### Bootstrap-Sanity
- [ ] `rm -f /workspace/.deps_installed` wenn Pod-Image oder Dep-Versions gewechselt.
- [ ] Beim ersten Start auf den Stack-Summary-Block im Log achten:
  ```
  ┌── STACK SUMMARY ───────────────────────
  │ ✓  torch       2.8.0+cu128
  │ ✓  triton      3.4.0  TMA=yes
  │ ✓  flash_attn  2.8.3
  │ ✓  unsloth     2026.4+
  │ ✓  unsloth_zoo 2026.4+
  │ …
  └──────────────────────────────────────
  ```
- [ ] **Alle Zeilen müssen `✓` zeigen.** Jedes `⚠` oder `✗` abbrechen und fixen.

### State-Cleanup vor Restart (wenn schon ein alter Run war)
```bash
pkill -9 -f "python.*0[0-9]_\|pipeline.sh"; sleep 2
ps -ef | grep -E "pipeline|0[0-9]_" | grep -v grep   # muss leer sein
rm -rf /workspace/checkpoints                         # wenn Config gewechselt
rm -f /workspace/.phase0[2-5]_done                    # Phase 01 OK lassen wenn Datasets schon da
rm -f /workspace/pipeline.log
```

### Smoke-Test empfohlen
- [ ] Ersten Full-Run limitieren: `TRAIN["num_train_epochs"] = 0.02` in `02_sft.py` → ~7 Steps, 2-3 min.
- [ ] Wenn dieser durchläuft ohne Crash → Limit wieder auf 2 Epochen setzen, ernst trainieren.

---

## 16. AgentEvalCallback OOM bei Step 100 (2026-04-22 Fortsetzung)

Nachdem Training bei Step 100 erfolgreich eine eval-Runde durchführte (`eval_loss=0.6048`) und der `AgentEvalCallback` getriggert wurde, **crashte das gesamte Training** beim ersten `model.generate()`-Aufruf:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 93.10 GiB
of which 86.88 MiB is free. Process 845747 has 93.00 GiB memory in use.
  File "moe_utils.py", line 1252, in forward_triton_grouped_gemm
    second_gemm_output.view(num_tokens, top_k, hidden_dim)
```

### Was passierte

Während Training (Step 100) sind ~92 GB VRAM belegt:
- LoRA-gewrapptes 30B-MoE Modell in 4-bit (~22 GB)
- Optimizer-State (paged_adamw_8bit für Attention-LoRA) + Gradients
- Gradient-Checkpointing-Buffer + activation recomputation
- Packed 4k-Sequenz Activation-Caches

Der `AgentEvalCallback.on_evaluate()` ruft dann `model.generate()` für 50 held-out Samples mit `max_new_tokens=128` und `max_prompt_tokens=3072` auf. Jeder generate-Call alloziert:
- KV-Cache pro Layer
- Neue Forward-Activations (weil `model.eval()` Gradient-Checkpointing deaktiviert → Activations bleiben im VRAM statt rekomputed)
- Zusätzliche Decode-Buffer

**Resultat: OOM im allerersten Sample.** Training-Prozess dead.

### Ergebnis bei Step 100 (vor Crash)

```
[agent-eval] step=100 {'agent/parse_rate': 0.0, 'agent/name_match': 0.0, 'agent/avg_output_tokens': 128.0}
```

Modell hat bei Step 100 noch KEIN valides Tool-Call-Format gelernt — parse_rate=0.0. Das ist erwartbar bei:
- 0.17 % trainable params (nur Attention-LoRA, MoE-Experts raus wegen bug)
- Erst 100/370 Steps = 27 % des SFT-Durchlaufs
- `max_new_tokens=128` — generate bricht oft mitten im Schema ab, valides JSON nicht erreicht

### Fix in `_agent_eval.py`

```python
def __init__(self, tokenizer,
             n_samples: int = 20,           # war 50
             max_new_tokens: int = 96,      # war 128
             max_prompt_tokens: int = 2048, # war 3072
             seed: int = 999):
    ...

def on_evaluate(self, args, state, control, **kwargs):
    ...
    model.eval()
    torch.cuda.empty_cache()  # NEU: vor Eval-Loop freigeben
    with torch.no_grad():
        for s in samples:
            ...
            out = model.generate(...)
            # ... parse & score ...
            del out, gen, inputs       # NEU: explizit freigeben
            torch.cuda.empty_cache()   # NEU: per-sample cleanup
```

### VRAM-Footprint vorher vs. nachher

| Config | generate-Peak | Fits with 92 GB training state? |
|---|---|---|
| Alt (50 × 128 × 3072) | ~1.5 GB transient | ❌ OOM |
| Neu (20 × 96 × 2048) | ~300 MB transient | ✅ passt mit 2 GB Luft |

### Checkpoint-Resume rettet den Progress

Weil `save_steps=20` im Trainer-Config, lagen `checkpoint-80` und `checkpoint-100` bereits auf Disk, als der OOM kam. `get_last_checkpoint()` in 02_sft.py findet sie automatisch beim Neustart → kein Training-Progress verloren.

### Faustregel

**Jeder Callback der `model.generate()` aufruft, muss aktiv VRAM aufräumen.** Bei 30B-Modellen mit MoE + Gradient-Checkpointing ist die Grenze ~2 GB transient. Verletzen = OOM, egal bei welchem Step.

Alternativen falls die Eval-Footprint nicht reicht:
1. **Gradient-Checkpointing während Eval kurz deaktivieren** — gibt Forward-Activations frei
2. **Eval nur auf Checkpoints** — separater Script nach jeder Phase statt In-Training-Callback
3. **Eval-Sample-Count weiter runter** (20 → 10)
