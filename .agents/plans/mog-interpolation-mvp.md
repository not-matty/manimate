# Task: MoG Interpolation MVP

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files etc.

## Task Description

Build the core Mode A pipeline: take a sequence of keyframe images and produce an interpolated animation video using MoG (Motion-Aware Generative Frame Interpolation) with its anime checkpoint. This is the foundational piece of Manimate — once this works, everything else (style transfer, cleanup, LoRA) plugs into it.

The implementation wraps MoG-VFI as a git submodule, provides a thin Python API, and exposes a CLI for end-to-end keyframe-to-video interpolation.

## Task Metadata

**Task Type**: Infrastructure + Code Change
**Estimated Complexity**: Medium-High
**Primary Systems Affected**: `manimate/interpolation/`, CLI entry point, model management
**Dependencies**: MoG-VFI repo, ani.ckpt + ours_t.ckpt model weights, PyTorch, CUDA GPU
**Supports Claim/Hypothesis**: H4 (human keyframes + AI in-betweening) — this is the in-betweening engine

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `CLAUDE.md` — Project conventions, patterns, naming
- `RESEARCH-BRIEF.md` (Section 3: Mode A, Section 7: Architecture) — Pipeline design
- `configs/interpolation/default.yaml` — Our interpolation config format
- `manimate/interpolation/__init__.py` — Target module (currently empty)
- `manimate/__init__.py` — Package root

### External Codebase: MoG-VFI (will be cloned as submodule)

**Key files to understand before implementing the wrapper:**

- `scripts/evaluation/inference.py` — The main inference script. Contains `run_inference()`, `image_guided_synthesis()`, and `load_model_checkpoint()`. This is what we wrap.
- `scripts/run_ani.sh` — The reference run command:
  ```bash
  python3 scripts/evaluation/inference.py \
    --seed 123 --ckpt_path checkpoints/ani.ckpt \
    --config configs/ani.yaml --savedir results/ani_test123 \
    --n_samples 1 --bs 1 --heigh 320 --width 512 \
    --unconditional_guidance_scale 7.5 --ddim_steps 50 \
    --ddim_eta 1.0 --prompt_dir prompts/ani_test \
    --text_input --video_length 16 --frame_stride 24 \
    --timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 \
    --perframe_ae --interp
  ```
- `configs/ani.yaml` — Model architecture config. Key details:
  - UNet: 320 base channels, attention at resolutions [4,2,1], temporal_length=16
  - First stage: AutoencoderKL_Dualref (VAE with dual reference)
  - Conditioning: CLIP text + CLIP image embeddings
  - Image size in latent space: [40, 64] (320x512 in pixel space)
- `emavfi/vfi_utils.py` — Optical flow computation used for motion guidance. `cal_flow()` takes the input video tensor and computes flow between first and last frames. `get_vfi_model()` loads the EMA-VFI flow model (needs `ours_t.ckpt`).
- `lvdm/` — The core latent video diffusion model (based on ToonCrafter)

**MoG inference flow:**
1. Load model from config + checkpoint
2. Load EMA-VFI flow model (for motion guidance)
3. For each frame pair: load images → resize to 320x512 → normalize to [-1,1]
4. Input format: `[B, C, T, H, W]` where T=16 frames (first and last are the input keyframes, 14 middle frames are generated)
5. `cal_flow()` computes optical flow from frame 0 to frame 15
6. Flow is injected as `motion_guidance` conditioning
7. DDIM sampling generates 16 latent frames
8. VAE decodes latents to pixel space
9. Output: 16 frames at 320x512

**Critical details:**
- MoG generates exactly 16 frames per pair (including the 2 input frames = 14 new frames)
- Resolution is fixed at 320x512 for the ani checkpoint
- Batch size must be 1
- Requires a text prompt per frame pair (can be empty string)
- Uses fp32 by default; fp16 can be enabled by uncommenting lines 333-334 in inference.py
- The EMA-VFI model (`ours_t.ckpt`) must be at `emavfi/ckpt/ours_t.ckpt` relative to the MoG repo root
- MoG requires Python 3.8+ and specific versions of some libs (xformers, cupy, etc.)

### New Files to Create

- `manimate/interpolation/mog.py` — MoG wrapper class
- `manimate/interpolation/base.py` — Base interpolation interface
- `manimate/interpolation/pipeline.py` — Multi-keyframe interpolation pipeline
- `manimate/video/io.py` — Frame I/O and video assembly utilities
- `scripts/interpolate.py` — CLI entry point for Mode A
- `scripts/download_models.py` — Script to download MoG checkpoints

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- MoG paper: https://arxiv.org/abs/2501.03699
  - Section 3: Method — dual-level guidance injection
  - Why: Understanding how motion guidance works for potential tuning
- MoG GitHub: https://github.com/MCG-NJU/MoG-VFI
  - README: Setup and usage instructions
  - Why: Installation and checkpoint paths
- MoG HuggingFace: https://huggingface.co/MCG-NJU/MoG
  - Why: Model weight downloads (ani.ckpt)
- ToonCrafter (base architecture): https://github.com/Doubiiu/ToonCrafter
  - Why: MoG's animation model is built on ToonCrafter. Understanding the base helps debug.

### Patterns to Follow

**Config Pattern (from CLAUDE.md):**
```python
import yaml

def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
```

**Model Wrapper Pattern (from CLAUDE.md):**
```python
class SomeModel:
    """Thin wrapper — translate our format to model's format."""
    def __init__(self, config: dict):
        # Load model, set device
        pass

    def __call__(self, inputs) -> outputs:
        # Run inference
        pass
```

**Error Handling (from CLAUDE.md):**
```python
# Fail loudly, liberal assertions
assert len(keyframes) >= 2, f"Need at least 2 keyframes, got {len(keyframes)}"
assert img.shape[-2:] == (320, 512), f"MoG requires 320x512, got {img.shape[-2:]}"
```

---

## IMPLEMENTATION PLAN

### Phase 1: Setup — Clone MoG, Download Weights, Verify

Get MoG running as-is before wrapping it.

**Tasks:**
1. Clone MoG-VFI as a git submodule under `vendor/MoG-VFI/`
2. Install MoG's dependencies into our venv
3. Download model weights (ani.ckpt + ours_t.ckpt)
4. Run MoG's own inference script on a test frame pair to verify it works
5. Benchmark: how long does one pair take? How much VRAM?

### Phase 2: Core Implementation — Wrapper + Pipeline

Build the Python API.

**Tasks:**
1. Create base interpolation interface (`base.py`)
2. Create MoG wrapper (`mog.py`) that loads the model once and exposes a `interpolate(frame_a, frame_b) -> list[frames]` method
3. Create multi-keyframe pipeline (`pipeline.py`) that takes N keyframes and runs pairwise interpolation, stitching results
4. Create video I/O utilities (`io.py`) for loading frames, saving frames, assembling to video via ffmpeg
5. Create CLI entry point (`scripts/interpolate.py`)

### Phase 3: Validation

**Tasks:**
1. Test with 2 keyframes → 16 frames output
2. Test with 3+ keyframes → chained interpolation
3. Test with ATD-12K triplets (use frame 1 and 3 as input, compare generated middle frame to ground truth frame 2)
4. Measure VRAM usage, inference time
5. Visual inspection of output quality

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### 1. ADD MoG-VFI as git submodule

```bash
cd /home/matthew/code/animation
git submodule add https://github.com/MCG-NJU/MoG-VFI.git vendor/MoG-VFI
```

Update `.gitignore` to ignore MoG checkpoint files but track the submodule:
```
# In .gitignore, add:
vendor/MoG-VFI/checkpoints/
vendor/MoG-VFI/emavfi/ckpt/
vendor/MoG-VFI/results/
```

- **VALIDATE**: `ls vendor/MoG-VFI/scripts/evaluation/inference.py` should exist

### 2. CREATE `scripts/download_models.py`

Script to download MoG checkpoints from HuggingFace.

- **IMPLEMENT**: Use `huggingface_hub` to download:
  - `MCG-NJU/MoG` → `ani.ckpt` → save to `vendor/MoG-VFI/checkpoints/ani.ckpt`
  - EMA-VFI checkpoint (`ours_t.ckpt`) → save to `vendor/MoG-VFI/emavfi/ckpt/ours_t.ckpt`
  - Note: The ours_t.ckpt is on Google Drive per the MoG README. Check the README for the exact link. May need `gdown` or manual download instructions.
- **IMPORTS**: `huggingface_hub`, `pathlib`, `argparse`
- **GOTCHA**: The EMA-VFI checkpoint may only be available on Google Drive, not HuggingFace. Check the MoG README for links. If Google Drive only, print instructions for manual download or use `gdown`.
- **VALIDATE**: `python scripts/download_models.py --help` runs without error

### 3. INSTALL MoG dependencies

MoG requires some specific packages. Install what's needed beyond what we already have:

```bash
source .venv/bin/activate
# Key additional deps for MoG:
pip install omegaconf pytorch_lightning open_clip_torch kornia xformers cupy-cuda12x einops decord moviepy fairscale rotary_embedding_torch easydict compel
```

- **GOTCHA**: MoG wants Python 3.8 and torch 2.1+cu121. We have Python 3.12 and torch 2.11+cu126. Version conflicts are likely, especially with xformers and cupy. May need to install compatible versions or patch.
- **GOTCHA**: `cupy-cuda12x` requires CUDA toolkit. If install fails, try `cupy-cuda12x==13.0.0` or install via conda.
- **GOTCHA**: xformers version must match PyTorch version. Check https://github.com/facebookresearch/xformers#installing-xformers for compatibility.
- **VALIDATE**: `python -c "from omegaconf import OmegaConf; from einops import rearrange; print('OK')"` succeeds

### 4. VERIFY MoG runs standalone

Before wrapping, confirm MoG works as-is.

- **IMPLEMENT**: Create a test prompt directory with two anime frames:
  ```bash
  mkdir -p vendor/MoG-VFI/prompts/test_pair
  # Copy or create two test images as 1_0.png and 1_1.png
  # Create prompt.txt with one line of text description
  ```
- **IMPLEMENT**: Run MoG inference:
  ```bash
  cd vendor/MoG-VFI
  python scripts/evaluation/inference.py \
    --seed 42 --ckpt_path checkpoints/ani.ckpt \
    --config configs/ani.yaml --savedir results/test \
    --n_samples 1 --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 7.5 --ddim_steps 50 \
    --ddim_eta 1.0 --prompt_dir prompts/test_pair \
    --text_input --video_length 16 --frame_stride 24 \
    --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
    --perframe_ae --interp
  ```
- **GOTCHA**: This will likely need ~10-12GB VRAM. If running on the RTX 3070 (8GB), enable fp16 by uncommenting lines 333-334 in `scripts/evaluation/inference.py`. If still OOM, this step must run on RunPod.
- **VALIDATE**: Output video exists at `vendor/MoG-VFI/results/test/samples_separate/`
- **NOTE**: Record VRAM usage and inference time for the plan

### 5. CREATE `manimate/interpolation/base.py`

Define the base interface for interpolation models.

- **IMPLEMENT**:
  ```python
  from abc import ABC, abstractmethod
  from pathlib import Path
  import torch
  from PIL import Image


  class BaseInterpolator(ABC):
      """Base class for frame interpolation models."""

      @abstractmethod
      def load(self) -> None:
          """Load model weights. Called once."""
          ...

      @abstractmethod
      def interpolate(
          self,
          frame_a: Image.Image,
          frame_b: Image.Image,
          num_frames: int = 14,
          prompt: str = "",
      ) -> list[Image.Image]:
          """Generate intermediate frames between frame_a and frame_b.
          
          Returns list of interpolated frames (NOT including the input frames).
          """
          ...

      @abstractmethod
      def unload(self) -> None:
          """Free GPU memory."""
          ...
  ```
- **VALIDATE**: `python -c "from manimate.interpolation.base import BaseInterpolator; print('OK')"`

### 6. CREATE `manimate/interpolation/mog.py`

The MoG wrapper. This is the core implementation.

- **IMPLEMENT**: Class `MoGInterpolator(BaseInterpolator)` that:
  1. `__init__(config)`: Stores config (checkpoint paths, inference params)
  2. `load()`: Adds `vendor/MoG-VFI` to `sys.path`, loads model from config + checkpoint, loads EMA-VFI flow model. Model stays on GPU.
  3. `interpolate(frame_a, frame_b, num_frames=14, prompt="")`:
     - Resize inputs to 320x512
     - Normalize to [-1, 1]
     - Stack into `[1, 3, 16, 320, 512]` tensor (first frame repeated 8x + last frame repeated 8x, same as MoG's `load_data_prompts` with `interp=True`)
     - Call `image_guided_synthesis()` with model's default params
     - Extract the 14 middle frames from the 16-frame output
     - Convert back to PIL Images
     - Return list of PIL Images
  4. `unload()`: Delete model, clear CUDA cache

- **IMPORTS**: Reuse MoG's internal functions. Key imports from vendor:
  ```python
  sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor" / "MoG-VFI"))
  from lvdm.models.samplers.ddim import DDIMSampler
  from utils.utils import instantiate_from_config
  from emavfi.vfi_utils import cal_flow, warp_fea
  from scripts.evaluation.inference import load_model_checkpoint, image_guided_synthesis, get_latent_z_with_hidden_states
  ```
  
- **PATTERN**: Follow the model wrapper pattern from CLAUDE.md — thin wrapper, consistent interface, no business logic in the wrapper.

- **GOTCHA**: MoG's `inference.py` uses `sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))` — our wrapper needs to set up the path correctly so MoG's internal imports resolve.
- **GOTCHA**: MoG generates exactly 16 frames. The first and last are the input frames. The 14 middle frames are the interpolation. If `num_frames` != 14, we'll need to handle this (e.g., run multiple passes or subsample). For MVP, fix at 14.
- **GOTCHA**: The EMA-VFI model auto-loads from `emavfi/ckpt/ours_t.ckpt` relative to CWD. We need to either `os.chdir` temporarily or patch the path.
- **GOTCHA**: fp16 for 8GB VRAM. Add a `half_precision: bool` config option that converts model and inputs to fp16.

- **VALIDATE**: 
  ```python
  from manimate.interpolation.mog import MoGInterpolator
  config = {"ckpt_path": "vendor/MoG-VFI/checkpoints/ani.ckpt", ...}
  interp = MoGInterpolator(config)
  interp.load()
  # Should load without error
  ```

### 7. CREATE `manimate/video/io.py`

Frame I/O and video utilities.

- **IMPLEMENT**:
  ```python
  def load_frames(directory: Path, pattern: str = "*.png") -> list[Image.Image]:
      """Load all frames from a directory, sorted by filename."""
      
  def save_frames(frames: list[Image.Image], directory: Path, prefix: str = "frame") -> list[Path]:
      """Save frames as frame_000000.png, frame_000001.png, etc."""

  def frames_to_video(frames: list[Image.Image], output_path: Path, fps: int = 24) -> Path:
      """Assemble frames into video using ffmpeg."""

  def video_to_frames(video_path: Path, output_dir: Path) -> list[Path]:
      """Extract frames from video using ffmpeg."""

  def load_image(path: Path) -> Image.Image:
      """Load a single image."""
  ```
- **IMPORTS**: `PIL.Image`, `pathlib.Path`, `subprocess` (for ffmpeg)
- **GOTCHA**: Use ffmpeg via subprocess, not moviepy (lighter dependency). Verify ffmpeg is installed.
- **VALIDATE**: `python -c "from manimate.video.io import load_image; print('OK')"`

### 8. CREATE `manimate/interpolation/pipeline.py`

The multi-keyframe pipeline that chains pairwise interpolation.

- **IMPLEMENT**: 
  ```python
  class InterpolationPipeline:
      """Interpolate between a sequence of keyframes."""
      
      def __init__(self, interpolator: BaseInterpolator):
          self.interpolator = interpolator
      
      def run(
          self,
          keyframes: list[Image.Image],
          num_intermediate: int = 14,
          prompts: list[str] | None = None,
      ) -> list[Image.Image]:
          """Interpolate between all adjacent keyframe pairs.
          
          Given keyframes [K1, K2, K3], produces:
          [K1, i1, i2, ..., i14, K2, i1, i2, ..., i14, K3]
          
          Returns the full frame sequence including keyframes.
          """
          assert len(keyframes) >= 2, f"Need at least 2 keyframes, got {len(keyframes)}"
          
          all_frames = [keyframes[0]]
          for i in range(len(keyframes) - 1):
              prompt = prompts[i] if prompts else ""
              intermediate = self.interpolator.interpolate(
                  keyframes[i], keyframes[i + 1], num_intermediate, prompt
              )
              all_frames.extend(intermediate)
              all_frames.append(keyframes[i + 1])
          
          return all_frames
  ```
- **VALIDATE**: Unit test with a mock interpolator

### 9. CREATE `scripts/interpolate.py`

CLI entry point for Mode A.

- **IMPLEMENT**: Argparse CLI that:
  1. Takes `--keyframes <dir>` (directory of numbered keyframe images)
  2. Takes `--output <path>` (output video path or frame directory)
  3. Takes `--config <yaml>` (interpolation config)
  4. Takes `--prompt <text>` (optional text description)
  5. Takes `--fps <int>` (output frame rate, default 24)
  6. Loads keyframes from directory (sorted by filename)
  7. Creates MoGInterpolator, loads model
  8. Runs InterpolationPipeline
  9. Saves output as video and/or frame sequence

- **PATTERN**: Follow the CLI command pattern from CLAUDE.md
- **VALIDATE**: `python scripts/interpolate.py --help` prints usage

### 10. UPDATE `manimate/interpolation/__init__.py`

Export the public API.

- **IMPLEMENT**:
  ```python
  from manimate.interpolation.base import BaseInterpolator
  from manimate.interpolation.mog import MoGInterpolator
  from manimate.interpolation.pipeline import InterpolationPipeline
  ```
- **VALIDATE**: `python -c "from manimate.interpolation import MoGInterpolator, InterpolationPipeline; print('OK')"`

### 11. UPDATE `configs/interpolation/default.yaml`

Update with MoG-specific settings.

- **IMPLEMENT**: Replace current placeholder config with:
  ```yaml
  # MoG interpolation config
  model: mog
  
  # MoG model paths
  mog:
    ckpt_path: vendor/MoG-VFI/checkpoints/ani.ckpt
    config_path: vendor/MoG-VFI/configs/ani.yaml
    vfi_ckpt_dir: vendor/MoG-VFI/emavfi/ckpt/
  
  # Inference settings
  height: 320
  width: 512
  video_length: 16         # MoG generates 16 frames per pair (14 intermediate)
  ddim_steps: 50
  ddim_eta: 1.0
  guidance_scale: 7.5
  guidance_rescale: 0.7
  frame_stride: 24
  timestep_spacing: uniform_trailing
  half_precision: false     # Set true for 8GB GPUs
  
  # Output
  target_fps: 24
  output_format: png
  seed: 42
  ```

### 12. TEST end-to-end with sample frames

- **IMPLEMENT**: Download or create 2-3 anime/manga keyframes for testing
  - Option A: Use frames from ATD-12K dataset (download first and last frames of triplets)
  - Option B: Find 2-3 CC-licensed anime frames online
  - Option C: Use frames from Pepper&Carrot webcomic
- **IMPLEMENT**: Run the full pipeline:
  ```bash
  python scripts/interpolate.py \
    --keyframes data/test_keyframes/ \
    --output output/test_interpolation.mp4 \
    --config configs/interpolation/default.yaml \
    --fps 24
  ```
- **VALIDATE**: Output video plays and shows smooth interpolation between keyframes
- **VALIDATE**: Frame count is correct: (num_keyframes - 1) * 14 + num_keyframes

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
source .venv/bin/activate
ruff check manimate/ scripts/
ruff format --check manimate/ scripts/
```

### Level 2: Imports & Structure

```bash
python -c "from manimate.interpolation import MoGInterpolator, InterpolationPipeline; print('imports OK')"
python -c "from manimate.video.io import load_frames, frames_to_video; print('io OK')"
python scripts/interpolate.py --help
```

### Level 3: Model Loading (requires GPU + checkpoints)

```bash
python -c "
from manimate.interpolation.mog import MoGInterpolator
import yaml
config = yaml.safe_load(open('configs/interpolation/default.yaml'))
m = MoGInterpolator(config)
m.load()
print('model loaded OK')
m.unload()
"
```

### Level 4: End-to-End (requires GPU + checkpoints + test data)

```bash
python scripts/interpolate.py \
  --keyframes data/test_keyframes/ \
  --output output/test_interpolation.mp4 \
  --config configs/interpolation/default.yaml
```

---

## ACCEPTANCE CRITERIA

- [ ] MoG-VFI cloned as git submodule under `vendor/`
- [ ] Model weights downloadable via script
- [ ] `MoGInterpolator` loads model and generates 14 intermediate frames from 2 input frames
- [ ] `InterpolationPipeline` chains multiple keyframe pairs
- [ ] CLI `scripts/interpolate.py` runs end-to-end: keyframe dir → output video
- [ ] Video I/O works (load frames, save frames, assemble with ffmpeg)
- [ ] Output video plays at 24fps with visually smooth interpolation
- [ ] Code passes `ruff check` and `ruff format --check`
- [ ] All assertions present (frame shapes, count, value ranges)
- [ ] Works on RTX 3070 8GB with fp16 enabled, OR documented that RunPod is required

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed
- [ ] All validation commands executed successfully
- [ ] Visual inspection of output video confirms quality
- [ ] VRAM usage and inference time documented
- [ ] Ready for `/commit`

---

## NOTES

### VRAM Concerns
MoG at fp32 needs ~10-12GB VRAM. The RTX 3070 has 8GB. Options:
1. **fp16**: Enable half precision (MoG supports this). May lose some quality.
2. **RunPod**: Run inference on a cloud GPU (A100/A6000). Better for experimentation.
3. **Reduced DDIM steps**: 50 → 25 steps. Faster, less VRAM, some quality loss.

Recommend: Try fp16 locally first. If quality is unacceptable, benchmark on RunPod with fp32.

### MoG Limitations
- Fixed output: 16 frames per pair (14 intermediate). Can't generate arbitrary frame counts.
- Fixed resolution: 320x512 for the ani checkpoint.
- Stochastic: Output varies per run (DDIM sampling). Set seed for reproducibility.
- Text prompt: Required but can be empty. Better prompts may improve quality.

### Future Integration Points
- **LoRA**: MoG is built on a diffusion model (SD-like architecture). LoRA can be injected into the UNet for style enforcement. This is a follow-up task.
- **RIFE fast path**: For simple motion, RIFE is much faster (~10ms vs ~60s per pair). Add as a fast-path option later.
- **Variable frame count**: For arbitrary frame counts, investigate ArbInterp or recursive MoG (interpolate the interpolation).
- **Higher resolution**: 320x512 is low. Super-resolution post-processing could help.
