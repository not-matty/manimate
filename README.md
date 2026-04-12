# Manimate

AI-assisted manga-to-animation tool. Draw every Nth frame, let AI fill in the rest — in the manga's own art style.

## What It Does

**Mode A — Keyframe Interpolation**: Animators draw keyframes at variable spacing. AI generates the in-between frames using diffusion-based video interpolation (MoG-VFI), producing smooth 24fps animation from sparse hand-drawn art.

**Mode B — AI Rotoscope** (planned): Feed in reference video. AI style-transfers each extracted keyframe into the manga's art style via ControlNet + LoRA, then interpolates between them using the same pipeline as Mode A.

Both modes share a LoRA style backbone trained on the source manga's panels, ensuring every generated frame matches the original art style.

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (20GB+ VRAM recommended, 8GB works with fp16)
- ffmpeg

### Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/not-matty/animation.git
cd animation

# Create environment
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Download model weights (~10GB)
python scripts/download_models.py
```

### Run Interpolation

```bash
# Basic usage — provide keyframes directory and output path
python scripts/interpolate.py keyframes=data/test_keyframes/ output=output/result.mp4

# Override inference settings
python scripts/interpolate.py keyframes=data/test_keyframes/ output=output/result.mp4 ddim_steps=25 half_precision=false

# Save individual frames alongside video
python scripts/interpolate.py keyframes=data/test_keyframes/ output=output/result.mp4 save_frames=true

# Output as frame sequence instead of video
python scripts/interpolate.py keyframes=data/test_keyframes/ output=output/frames/
```

Place keyframe images (PNG/JPG) in a directory, named in the order you want them interpolated (sorted lexicographically). The pipeline generates 14 intermediate frames between each adjacent pair at 320x512 resolution.

## Architecture

```
manimate/
├── interpolation/     # Frame interpolation engine
│   ├── base.py        # BaseInterpolator interface
│   ├── mog.py         # MoG-VFI wrapper (diffusion-based, anime checkpoint)
│   └── pipeline.py    # Multi-keyframe chaining
├── video/             # Video I/O (frame loading, ffmpeg assembly)
├── style_transfer/    # ControlNet + LoRA style transfer (planned)
├── lora/              # LoRA training pipeline (planned)
├── pose/              # Pose estimation for Mode B (planned)
├── cleanup/           # Correction tools (planned)
└── ui/                # Desktop UI (planned)
```

### Interpolation Pipeline

```
Keyframes:  K1 ──────────── K2 ──────────── K3
                    │                │
                MoG-VFI          MoG-VFI
                    │                │
Output:     K1 i i i i ... i K2 i i i i ... i K3
            (14 frames/pair → 24fps animation)
```

MoG-VFI uses optical flow guidance + latent video diffusion to generate temporally coherent frames. The anime checkpoint is tuned for illustrated/animated content.

## Configuration

Config is managed via [Hydra](https://hydra.cc). Base config at `configs/interpolation/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keyframes` | required | Path to keyframe image directory |
| `output` | required | Output path (.mp4 for video, directory for frames) |
| `ddim_steps` | 50 | Diffusion sampling steps (lower = faster, less quality) |
| `half_precision` | true | fp16 inference (fits 8GB GPUs) |
| `target_fps` | 24 | Output video frame rate |
| `guidance_scale` | 7.5 | Classifier-free guidance scale |
| `seed` | 42 | Random seed for reproducibility |
| `prompt` | "" | Optional text description of the scene |

Override any parameter via CLI: `python scripts/interpolate.py ddim_steps=25 seed=123`

## Cloud GPU (RunPod)

For larger runs or full-precision inference:

```bash
# On RunPod pod (A6000/A100 recommended)
bash scripts/setup_runpod.sh
python scripts/download_models.py
python scripts/interpolate.py keyframes=... output=... half_precision=false
```

## Development

```bash
# Lint + format
ruff check . && ruff format --check .

# Type check
pyright

# Tests
python -m pytest tests/ -x
```

## License

MIT
