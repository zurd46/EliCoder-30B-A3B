runpodctl create pod \
  --name coderllm-sft \
  --gpuType "NVIDIA H100 NVL" \
  --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 200 \
  --mem 64 \
  --vcpu 8 \
  --ports "22/tcp" \
  --env "HF_TOKEN=<DEIN_HF_TOKEN>" \
  --env "CODERLLM_REPO_URL=https://github.com/zurd46/CoderLLM.git"


export RUNPOD_API_KEY="<DEIN_RUNPOD_API_KEY>"
export RUNPOD_POD_ID="<POD_ID>"
