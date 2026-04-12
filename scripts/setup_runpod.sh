#!/bin/bash
set -e

echo "=== Pre-flight checks ==="
DISK_FREE_GB=$(df -BG / | awk 'NR==2 {print int($4)}')
echo "Free disk: ${DISK_FREE_GB}GB"
if [ "$DISK_FREE_GB" -lt 30 ]; then
    echo "ERROR: Need at least 30GB free disk. Resize your RunPod volume."
    exit 1
fi

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU VRAM: ${VRAM_MB}MB"
if [ "$VRAM_MB" -lt 20000 ]; then
    echo "ERROR: Need at least 20GB VRAM for MoG inference. Select a larger GPU."
    exit 1
elif [ "$VRAM_MB" -lt 48000 ]; then
    echo "WARNING: 48GB VRAM recommended for full pipeline (depth + interpolation). Current: ${VRAM_MB}MB"
fi

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq rsync ffmpeg > /dev/null 2>&1

echo "=== Pinning torch to container version ==="
# CRITICAL: pin torch so no dep can upgrade it
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "Container torch: $TORCH_VER"

echo "=== Installing MoG deps (torch-safe) ==="
# Install deps that won't touch torch
pip install --no-cache-dir \
    omegaconf==2.3.0 \
    pytorch_lightning==2.1.4 \
    'open_clip_torch==2.22.0' \
    kornia \
    einops \
    decord \
    moviepy \
    easydict \
    rotary_embedding_torch \
    compel \
    fairscale \
    huggingface_hub \
    gdown \
    safetensors \
    diffusers \
    'transformers<4.40' \
    accelerate \
    peft \
    'numpy<2' \
    2>&1 | grep -E "Successfully|ERROR"

# xformers matching torch 2.4+cu124
pip install --no-cache-dir xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | grep -E "Successfully|ERROR"

# cupy for CUDA 12
pip install --no-cache-dir cupy-cuda12x 2>&1 | grep -E "Successfully|ERROR"

echo "=== Verifying ==="
python3 << 'PYEOF'
import torch
import xformers
import open_clip
import kornia
import omegaconf
import pytorch_lightning
import einops

print("torch=" + torch.__version__)
print("xformers=" + xformers.__version__)
print("cuda=" + str(torch.cuda.is_available()))
print("gpu=" + torch.cuda.get_device_name(0))

# cuDNN test
t = torch.randn(1, 3, 320, 512).cuda()
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
out = conv(t)
print("cuDNN OK: " + str(out.shape))
PYEOF

echo "=== Setup complete ==="
