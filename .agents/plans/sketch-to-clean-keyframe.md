# Task: Sketch-to-Clean Keyframe Generation (SDXL + ControlNet + IP-Adapter)

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files etc.

## Task Description

Implement the **Sketch-to-Clean Keyframe** step from Mode A (RESEARCH-BRIEF §3). Animators draw rough keyframe sketches. This module takes a rough sketch + a reference manga panel and outputs a clean, on-style manga keyframe suitable for the interpolation pipeline.

The stack combines three conditioning mechanisms on different pathways:

| Component | Role | Pathway | Model |
|-----------|------|---------|-------|
| **ControlNet** (scribble) | Pose/composition from rough sketch | Spatial conditioning | `xinsir/controlnet-scribble-sdxl-1.0` |
| **IP-Adapter** | Art style from reference manga panel | Cross-attention injection | `h94/IP-Adapter` (SDXL variant) |
| **LoRA** (optional) | Series-specific style (faces, line weight, screentones) | Weight fine-tune | Per-series, trained via kohya-ss |

**Base model**: SDXL (`stabilityai/stable-diffusion-xl-base-1.0`). Flux has better quality but needs 24GB+ VRAM and its ControlNet/IP-Adapter ecosystem is less mature.

## Task Metadata

**Task Type**: Code Change
**Estimated Complexity**: Medium-High
**Primary Systems Affected**: `manimate/style_transfer/` (currently empty), configs, scripts
**Dependencies**: `diffusers>=0.25` (already installed), `ip-adapter` weights from HF
**Supports Claim/Hypothesis**: Mode A workflow — rough keyframes → clean keyframes → interpolation (RESEARCH-BRIEF §3)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ THESE BEFORE IMPLEMENTING

- `manimate/interpolation/mog.py` (lines 29-110) — **Model wrapper pattern to mirror**: config dict constructor, `load()`/`unload()` lifecycle, VRAM staging (fp16 on CPU → CUDA), `torch.cuda.empty_cache()` on unload.
- `manimate/interpolation/base.py` (lines 1-41) — **ABC pattern**: same structure for BaseStyleTransfer.
- `configs/style_transfer/default.yaml` — **Existing config skeleton** (has SDXL base model, ControlNet, LoRA fields). Update this rather than creating from scratch.
- `configs/lora/default.yaml` — **LoRA training config** — shows the LoRA architecture choices (rank 32, alpha 32, target modules `[to_q, to_k, to_v, to_out.0]`).
- `manimate/video/io.py` (lines 1-30) — **Image loading pattern**: `load_image()` returns PIL RGB.
- `scripts/interpolate.py` — **Hydra CLI entry point pattern**: `@hydra.main`, `OmegaConf.to_container(cfg, resolve=True)`.
- `manimate/style_transfer/__init__.py` — **Currently empty**. Will add exports here.

### New Files to Create

- `manimate/style_transfer/sketch_to_clean.py` — Main pipeline wrapper
- `configs/style_transfer/sketch-to-clean.yaml` — Dedicated config for this workflow
- `scripts/test_sketch_to_clean.py` — Test script with sample sketch + reference panel

### Files to Modify

- `manimate/style_transfer/__init__.py` — Add exports
- `configs/style_transfer/default.yaml` — Update with correct model IDs and IP-Adapter fields
- `scripts/setup_runpod.sh` — Ensure diffusers deps are sufficient (should already be)

### Relevant Documentation — READ BEFORE IMPLEMENTING

- [Diffusers ControlNet SDXL guide](https://huggingface.co/docs/diffusers/using-diffusers/controlnet#stable-diffusion-xl) — Pipeline class, loading, inference
- [Diffusers IP-Adapter guide](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter) — `load_ip_adapter()`, `set_ip_adapter_scale()`, `ip_adapter_image` kwarg
- [xinsir/controlnet-scribble-sdxl-1.0 model card](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0) — Input format, conditioning scale
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) — SDXL weights: `sdxl_models/ip-adapter_sdxl.bin` or `sdxl_models/ip-adapter-plus_sdxl_vit-h.bin`
- [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) — Required for fp16 inference (default SDXL VAE has NaN issues in fp16)

### Patterns to Follow

**Model Wrapper Pattern** (from `mog.py`):
```python
class SketchToClean:
    def __init__(self, config: dict) -> None:
        self.config = config
        # Extract all config values
        self.pipe = None

    def load(self) -> None:
        # Load all components on CPU, then move to CUDA
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(...)
        self.pipe.load_ip_adapter(...)
        if self.lora_path:
            self.pipe.load_lora_weights(...)
        self.pipe = self.pipe.to("cuda")

    def generate(self, sketch: Image, reference: Image, prompt: str = "") -> Image:
        # Run inference
        ...

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
```

**Hydra Config Pattern** (from `configs/interpolation/default.yaml`):
```yaml
# Required fields use ???
sketch: ???
reference: ???
output: ???

# Model paths with defaults
base_model: stabilityai/stable-diffusion-xl-base-1.0

hydra:
  output_subdir: null
  run:
    dir: .
```

---

## IMPLEMENTATION PLAN

### Phase 1: Setup

- Update style_transfer config with correct model IDs
- Create sketch-to-clean specific config

### Phase 2: Core Implementation

- Implement SketchToClean pipeline wrapper
- Handle ControlNet + IP-Adapter + optional LoRA loading
- Implement sketch preprocessing (auto-detect whether to apply scribble HED)

### Phase 3: Test Script

- Create test script that generates a clean keyframe from a reference panel
- Test with manga panels as both "sketch" and "reference" (since we don't have actual rough sketches yet)

### Phase 4: Validation

- Verify pipeline loads and generates on RunPod
- Check VRAM usage in fp16
- Verify unload properly frees memory

---

## STEP-BY-STEP TASKS

### UPDATE `configs/style_transfer/default.yaml`

- **REFACTOR**: Update the existing skeleton config with correct model IDs and add IP-Adapter fields
  ```yaml
  base_model: stabilityai/stable-diffusion-xl-base-1.0
  vae: madebyollin/sdxl-vae-fp16-fix
  controlnet_model: xinsir/controlnet-scribble-sdxl-1.0
  ip_adapter:
    repo: h94/IP-Adapter
    subfolder: sdxl_models
    weight_name: ip-adapter-plus_sdxl_vit-h.bin
    scale: 0.6
  lora_path: null

  controlnet_conditioning_scale: 0.8
  guidance_scale: 5.0
  num_inference_steps: 30
  strength: 0.85

  height: 1024
  width: 1024
  half_precision: true
  seed: 42

  hydra:
    output_subdir: null
    run:
      dir: .
  ```
- **GOTCHA**: Use `madebyollin/sdxl-vae-fp16-fix` VAE — the default SDXL VAE produces NaN in fp16
- **VALIDATE**: `python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/style_transfer/default.yaml')"`

### CREATE `configs/style_transfer/sketch-to-clean.yaml`

- **IMPLEMENT**: CLI-oriented config that extends default with required I/O fields
  ```yaml
  defaults:
    - default

  sketch: ???       # Path to rough sketch image
  reference: ???    # Path to reference manga panel (style source)
  output: ???       # Output path for clean keyframe
  prompt: ""        # Optional text prompt (e.g., "manga style, detailed linework")
  negative_prompt: "blurry, low quality, deformed, 3d render, photo"
  ```
- **PATTERN**: Hydra defaults composition — inherits from `default.yaml`
- **VALIDATE**: `python -c "from hydra import compose, initialize_config_dir; initialize_config_dir('configs/style_transfer'); compose('sketch-to-clean')"`

### CREATE `manimate/style_transfer/sketch_to_clean.py`

- **IMPLEMENT**: Main pipeline wrapper class
- **IMPORTS**:
  ```python
  import torch
  from pathlib import Path
  from PIL import Image
  from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
  ```
- **CLASS**: `SketchToClean`
  - `__init__(self, config: dict)` — Extract all config values. Store model IDs. Set `self.pipe = None`.
  - `load(self)` — Load ControlNet, VAE, pipeline, IP-Adapter, optional LoRA. All in fp16 if configured.
    ```python
    def load(self) -> None:
        dtype = torch.float16 if self.half_precision else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model, torch_dtype=dtype
        )
        vae = AutoencoderKL.from_pretrained(self.vae_model, torch_dtype=dtype)

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
        )

        # IP-Adapter
        ip_cfg = self.config.get("ip_adapter", {})
        self.pipe.load_ip_adapter(
            ip_cfg["repo"],
            subfolder=ip_cfg["subfolder"],
            weight_name=ip_cfg["weight_name"],
        )
        self.pipe.set_ip_adapter_scale(ip_cfg.get("scale", 0.6))

        # Optional LoRA
        if self.lora_path and Path(self.lora_path).exists():
            self.pipe.load_lora_weights(self.lora_path)

        self.pipe = self.pipe.to("cuda")
    ```
  - `generate(self, sketch: Image.Image, reference: Image.Image, prompt: str = "", negative_prompt: str = "") -> Image.Image` — Run inference.
    ```python
    def generate(self, sketch: Image.Image, reference: Image.Image,
                 prompt: str = "", negative_prompt: str = "") -> Image.Image:
        assert self.pipe is not None, "Pipeline not loaded. Call load() first."

        from pytorch_lightning import seed_everything
        seed_everything(self.seed)

        # Resize inputs to configured dimensions
        sketch_resized = sketch.convert("RGB").resize((self.width, self.height))
        reference_resized = reference.convert("RGB").resize((self.width, self.height))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch_resized,                    # ControlNet input
            ip_adapter_image=reference_resized,      # IP-Adapter style reference
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
        ).images[0]

        return result
    ```
  - `unload(self)` — Delete pipeline, empty CUDA cache.
- **GOTCHA**: `diffusers` downloads models to HF cache on first use (~6.5GB for SDXL, ~2.5GB for ControlNet, ~1GB for IP-Adapter). On RunPod, ensure sufficient disk.
- **GOTCHA**: `load_ip_adapter()` must be called AFTER `from_pretrained()` but BEFORE `.to("cuda")`.
- **GOTCHA**: `seed_everything` from pytorch_lightning — same pattern as `mog.py`.
- **VALIDATE**: `python -c "from manimate.style_transfer.sketch_to_clean import SketchToClean; print('OK')"`

### UPDATE `manimate/style_transfer/__init__.py`

- **IMPLEMENT**: Add exports
  ```python
  from manimate.style_transfer.sketch_to_clean import SketchToClean

  __all__ = ["SketchToClean"]
  ```
- **VALIDATE**: `python -c "from manimate.style_transfer import SketchToClean"`

### CREATE `scripts/test_sketch_to_clean.py`

- **IMPLEMENT**: Test script that generates a clean keyframe
- **APPROACH**: Since we don't have rough sketches, use one manga panel as the "sketch" (structural input) and the other as the "reference" (style input). This tests the pipeline end-to-end even if the result is a style-blended image rather than a true sketch cleanup.
- **STRUCTURE**:
  ```python
  """Test sketch-to-clean keyframe generation.

  Uses manga panels as stand-in for rough sketches:
    - Panel A as sketch (pose/composition source)
    - Panel B as reference (style source)
    - Output: cleaned keyframe in Panel B's style with Panel A's composition

  RunPod:
      python scripts/test_sketch_to_clean.py

  Outputs to output/sketch_to_clean_test/:
    input_sketch.png    — the sketch input (Panel A)
    input_reference.png — the style reference (Panel B)
    output_clean.png    — generated clean keyframe
    output_grid.png     — side-by-side comparison
  """
  ```
- **CLI ARGS**:
  - `--sketch` — Path to sketch image (default: first panel)
  - `--reference` — Path to reference panel (default: second panel)
  - `--prompt` — Optional text prompt
  - `--half-precision` — fp16 (default: true)
  - `--steps` — Inference steps (default: 30)
  - `--controlnet-scale` — ControlNet conditioning scale (default: 0.8)
  - `--ip-adapter-scale` — IP-Adapter scale (default: 0.6)
- **OUTPUT**: Save inputs, output, and a side-by-side grid for comparison
- **VRAM**: Print peak VRAM usage after generation
- **VALIDATE**: `python scripts/test_sketch_to_clean.py --help`

### UPDATE `scripts/setup_runpod.sh`

- **ADD**: After existing verification block, add SDXL deps check. Most should already be installed (`diffusers`, `transformers`, `accelerate`, `peft`, `safetensors` are in the MoG deps). Verify:
  ```bash
  echo "=== Verifying style transfer deps ==="
  python3 -c "from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL; print('diffusers SDXL OK')"
  ```
- **GOTCHA**: `diffusers` is already installed for MoG. The IP-Adapter support is built into diffusers since v0.25 — no separate package needed.

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
```bash
ruff check manimate/style_transfer/ scripts/test_sketch_to_clean.py
ruff format --check manimate/style_transfer/ scripts/test_sketch_to_clean.py
```

### Level 2: Import Check
```bash
python -c "from manimate.style_transfer import SketchToClean; print('OK')"
python -c "from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel; print('diffusers OK')"
```

### Level 3: Config Validation
```bash
python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/style_transfer/default.yaml'); print(c.base_model)"
```

### Level 4: Dry Run (on RunPod)
```bash
python scripts/test_sketch_to_clean.py --steps 5  # Quick test with few steps
```

### Level 5: Full Run (on RunPod)
```bash
python scripts/test_sketch_to_clean.py
```

---

## ACCEPTANCE CRITERIA

- [ ] `SketchToClean` class loads SDXL + ControlNet + IP-Adapter pipeline
- [ ] `generate()` takes sketch + reference PIL Images, returns clean keyframe PIL Image
- [ ] Optional LoRA loading works (when path provided and file exists)
- [ ] fp16 inference works on RunPod A6000 (48GB)
- [ ] VRAM properly freed after `unload()`
- [ ] Generated keyframe preserves sketch composition (pose, layout)
- [ ] Generated keyframe reflects reference panel style (line weight, shading)
- [ ] Config follows Hydra pattern with `???` for required fields
- [ ] All lint/format checks pass
- [ ] Test script produces comparison grid

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed
- [ ] All validation commands executed
- [ ] Generated output visually inspected
- [ ] Ready for `/commit`

---

## NOTES

- **SDXL VAE fp16 fix is mandatory** — the default SDXL VAE (`stabilityai/stable-diffusion-xl-base-1.0` built-in VAE) produces NaN values in fp16. Always use `madebyollin/sdxl-vae-fp16-fix`.
- **IP-Adapter scale tuning** — Start at 0.6. Too high (>0.8) = output copies the reference image instead of just the style. Too low (<0.3) = style influence is negligible.
- **ControlNet scale tuning** — Start at 0.8. Too high (1.0) = output rigidly follows sketch lines (good for clean sketches, bad for rough ones). Lower (0.5-0.7) gives more creative freedom for rough sketches.
- **No actual rough sketches yet** — The test uses manga panels as stand-ins. Real evaluation needs actual rough animator sketches at varying roughness levels. This is exp-008 in RESEARCH-BRIEF.
- **LoRA integration is optional for MVP** — IP-Adapter alone provides zero-shot style transfer. LoRA adds series-specific consistency (character faces, screentone patterns) but requires training first. The pipeline should work with LoRA=None.
- **Resolution**: SDXL native is 1024x1024. The sketch-to-clean output will then be passed to MoG at 320x512. The high-res generation → downscale → interpolate flow preserves more detail than generating directly at 320x512.
- **VRAM budget**: SDXL + ControlNet + IP-Adapter = ~12GB fp16. On 48GB A6000, this leaves room for other models. On 8GB RTX 3070, need CPU offloading (`pipe.enable_model_cpu_offload()`).
- **Model download sizes**: ~10GB total (SDXL base ~6.5GB, ControlNet ~2.5GB, IP-Adapter ~1GB). First run will download to HF cache. On RunPod, this uses container disk — ensure ≥50GB.
