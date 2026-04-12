# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Manimate is an AI-assisted animation tool for adapting manga into animation. Two core modes:

1. **Mode A — Keyframe Interpolation**: Animators draw every Nth frame. AI interpolates in-between frames, guided by a LoRA trained on the source manga's art style.
2. **Mode B — AI Rotoscope**: Reference video → extract keyframes → AI style-transfers each to manga style → human corrects → interpolate the rest.

Both modes share the same LoRA style backbone and interpolation pipeline. See `RESEARCH-BRIEF.md` for full design.

## Research Stack

| Technology | Purpose |
|------------|---------|
| PyTorch 2.x | Model inference and LoRA training |
| Stable Diffusion (SDXL/Flux) + ControlNet | Style transfer, keyframe cleanup, diffusion-based interpolation |
| PEFT / kohya-ss | LoRA training on manga panels |
| RIFE / FILM / AMT | Frame interpolation (primary in-betweening engine) |
| DWPose / OpenPose | Pose estimation for Mode B (reference video → pose conditioning) |
| Python + Qt or Dear ImGui | Standalone desktop UI |
| ffmpeg | Video I/O |
| RTX 3090/4090 | Local GPU inference |

## Commands

```bash
# LoRA training
python train_lora.py --config configs/lora/default.yaml --data_dir data/manga_panels/

# Mode A — Keyframe interpolation
python interpolate.py --keyframes path/to/keyframes/ --output output/ --lora models/loras/series.safetensors

# Mode B — Rotoscope from video
python rotoscope.py --video path/to/reference.mp4 --lora models/loras/series.safetensors --output output/

# Run UI
python -m manimate.ui

# Evaluate interpolation quality
python evaluate.py --generated output/ --reference ground_truth/

# Lint
ruff check . && ruff format --check .

# Type check
pyright
```

## Project Structure

```
manimate/
├── manimate/              # Main package
│   ├── interpolation/     # Frame interpolation (RIFE/FILM/AMT wrappers)
│   ├── style_transfer/    # ControlNet + LoRA style transfer pipeline
│   ├── lora/              # LoRA training and loading
│   ├── video/             # Video I/O, keyframe extraction, ffmpeg utils
│   ├── pose/              # Pose estimation (DWPose/OpenPose)
│   ├── cleanup/           # Correction tools (local repaint, batch fix, flicker detect)
│   ├── ui/                # Desktop UI (Qt or Dear ImGui)
│   └── utils/             # Shared utilities
├── configs/               # YAML configs
│   ├── lora/              # LoRA training configs (per manga series)
│   ├── interpolation/     # Interpolation model configs
│   └── style_transfer/    # Style transfer configs
├── models/                # Model weights (gitignored)
│   ├── loras/             # Trained LoRA weights per series
│   ├── interpolation/     # RIFE/FILM/AMT weights
│   └── diffusion/         # Base diffusion model + ControlNet
├── data/                  # Input data (gitignored)
│   ├── manga_panels/      # Source manga panels for LoRA training
│   └── reference_video/   # Reference footage for Mode B
├── results/               # Experiment outputs
│   ├── figures/           # Evaluation plots
│   └── comparisons/       # Side-by-side quality comparisons
├── scripts/               # One-off and utility scripts
├── tests/                 # Unit and integration tests
├── RESEARCH-BRIEF.md      # Project design and scope
├── EXPERIMENT-LOG.md      # Lab notebook (created by /init-project)
└── CLAUDE.md              # This file
```

## Code Patterns

### Config Management
- Standalone YAML files in `configs/`, one per experiment or model configuration
- Load with `yaml.safe_load`, no Hydra
- Every experiment gets a unique config — never overwrite old configs

### Model Wrappers
- Each external model (RIFE, FILM, ControlNet, etc.) gets a thin wrapper in its module
- Wrappers expose a consistent interface: `__init__(config)`, `__call__(inputs)` or descriptive method
- No business logic in wrappers — just translate our data format to the model's expected format

### Pipeline Pattern
- Mode A and Mode B converge on the same interpolation step
- Once you have manga-style keyframes (whether hand-drawn or style-transferred), the downstream pipeline is identical
- Each pipeline stage is a standalone function that reads files and writes files — no hidden state

### Naming Conventions
- Python: `snake_case` for files, functions, variables
- Classes: `PascalCase`
- Configs: `kebab-case.yaml`
- Output frames: `frame_{:06d}.png` (zero-padded 6 digits)

### Error Handling

**Fail loudly. Never silently accept unexpected state.**

- If a function receives unexpected input, crash with a clear error — don't return defaults or log warnings
- Only handle expected cases. No catch-all `except Exception` blocks
- No silent type coercion or value clamping

### Assertions

Use assertions liberally — they're free documentation and free bug detection.

```python
# Tensor shapes after every reshape/concatenation
assert x.shape == (batch, channels, height, width), f"Expected {(batch, channels, height, width)}, got {x.shape}"

# Frame counts
assert len(keyframes) >= 2, f"Need at least 2 keyframes, got {len(keyframes)}"

# Image value ranges
assert img.min() >= 0 and img.max() <= 1, f"Image values out of [0,1]: [{img.min()}, {img.max()}]"

# Config values
assert config.fps > 0, f"FPS must be positive, got {config.fps}"
assert config.keyframe_interval >= 1, f"Keyframe interval must be >= 1, got {config.keyframe_interval}"
```

## Key Files

| File | Purpose |
|------|---------|
| `RESEARCH-BRIEF.md` | Full project design, scope, experiments, architecture |
| `EXPERIMENT-LOG.md` | Lab notebook tracking all experiments and results |
| `configs/lora/default.yaml` | Default LoRA training config |
| `manimate/interpolation/` | Frame interpolation engine (core of both modes) |
| `manimate/style_transfer/` | ControlNet + LoRA style transfer (core of Mode B) |
| `manimate/lora/` | LoRA training pipeline |

## Experiment Tracking

- **Tool**: Weights & Biases
- **Project**: Set via `WANDB_PROJECT` in `.env`
- **Naming**: `exp-NNN-short-description` (e.g., `exp-004-rife-vs-film`)

## Reproducibility

- **Seeds**: Set in config YAML, applied to `torch`, `numpy`, `random` at startup
- **Environment**: `requirements.txt` with pinned versions
- **Configs**: One YAML per experiment, committed to git, never overwritten
- **LoRA weights**: Versioned per manga series in `models/loras/`

## Validation

```bash
ruff check .
ruff format --check .
pyright
python -m pytest tests/ -x
```

## On-Demand Context

| Topic | File |
|-------|------|
| MCP server suggestions | `.agents/reference/mcp-suggestions.md` |
| Project design & scope | `RESEARCH-BRIEF.md` |
