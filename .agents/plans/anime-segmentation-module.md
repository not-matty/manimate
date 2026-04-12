# Task: Anime Segmentation Module for Layer Decomposition

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files etc.

## Task Description

Replace the failed `rembg`/`isnet-anime` segmentation with a reliable anime-specific segmentation module. The current approach (tested in `scripts/test_layer_interpolation.py`) fails on B&W manga panels — `isnet-anime` detected only 13.5% foreground on panel A (character layer was entirely gray) vs 45.4% on panel B. The model was trained on color anime, not raw manga.

This module enables **layer decomposition**: separating character foreground from background so each layer can be interpolated independently and composited back together. This is listed under "Future Work" in RESEARCH-BRIEF.md.

**Two backends, one interface:**
1. **`SkyTNT/anime-segmentation`** (primary) — trained specifically on anime/illustration, lightweight (~200M params), single foreground mask
2. **`GroundingDINO + SAM 2`** (advanced) — text-prompted detection + precise segmentation, instance-level masks per character, heavier (~8GB combined)

## Task Metadata

**Task Type**: Code Change
**Estimated Complexity**: Medium
**Primary Systems Affected**: New `manimate/segmentation/` module, `scripts/test_layer_interpolation.py`, RunPod setup
**Dependencies**: `anime-segmentation` (git clone), `grounded-sam-2` (git clone), `supervision` (pip)
**Supports Claim/Hypothesis**: Layer decomposition for parallax/camera movement (RESEARCH-BRIEF §10)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ THESE BEFORE IMPLEMENTING

- `manimate/interpolation/base.py` (lines 1-41) — **BaseInterpolator ABC pattern to mirror** for BaseSegmenter. Same design: `__init__(config)`, `load()`, `unload()`, core method.
- `manimate/interpolation/mog.py` (lines 29-110) — **Model wrapper pattern**: config dict in constructor, vendor path management, VRAM staging (fp16 on CPU → move to CUDA), `torch.cuda.empty_cache()` on unload.
- `scripts/test_layer_interpolation.py` (lines 58-80) — **Current rembg approach that failed**: `new_session("isnet-anime")`, morphological cleanup, coverage metrics. This is what we're replacing.
- `scripts/test_layer_interpolation.py` (lines 83-97) — **Layer separation logic**: character on gray (128), background via `cv2.inpaint`. Keep this downstream code unchanged.
- `configs/interpolation/default.yaml` — **Hydra config pattern**: `???` for required fields, nested model config, `hydra.output_subdir: null`.
- `scripts/setup_runpod.sh` — **Dependency installation pattern**: pin versions, don't touch torch, verify imports.
- `pyproject.toml` (lines 12-35) — **Current dependencies** to not conflict with.

### New Files to Create

- `manimate/segmentation/__init__.py` — Module exports
- `manimate/segmentation/base.py` — BaseSegmenter ABC
- `manimate/segmentation/anime_seg.py` — SkyTNT/anime-segmentation wrapper
- `manimate/segmentation/grounded_sam.py` — GroundingDINO + SAM 2 wrapper
- `configs/segmentation/default.yaml` — Hydra config
- `scripts/test_segmentation.py` — Comparison test on manga panels

### Files to Modify

- `scripts/test_layer_interpolation.py` — Use new segmentation module instead of rembg
- `scripts/setup_runpod.sh` — Add segmentation deps
- `pyproject.toml` — Add `supervision` dependency

### Relevant Documentation — READ BEFORE IMPLEMENTING

- [SkyTNT/anime-segmentation README](https://github.com/SkyTNT/anime-segmentation) — Model loading, inference API
- [skytnt/anime-seg on HuggingFace](https://huggingface.co/skytnt/anime-seg) — Model weights (`isnet_is.ckpt`)
- [Grounded-SAM-2 repo](https://github.com/IDEA-Research/Grounded-SAM-2) — Install, demo scripts
- [Grounded-SAM-2 INSTALL.md](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/INSTALL.md) — Build from source
- [SAM 2 repo](https://github.com/facebookresearch/sam2) — Core SAM 2 API

### Patterns to Follow

**Model Wrapper Pattern** (from `mog.py`):
```python
class SomeWrapper:
    def __init__(self, config: dict) -> None:
        self.config = config
        # Extract config values in constructor
        self.model = None

    def load(self) -> None:
        # Load weights, move to GPU
        self.model = ...
        self.model.eval()
        self.model = self.model.cuda()

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
```

**Vendor Submodule Pattern** (from `mog.py`):
```python
VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "SomeVendor"

def _setup_imports() -> None:
    vendor_str = str(VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
```

---

## IMPLEMENTATION PLAN

### Phase 1: Setup

- Create `manimate/segmentation/` module structure
- Create Hydra config for segmentation
- Add vendor submodules or download scripts for model weights

### Phase 2: Core Implementation

- Implement BaseSegmenter ABC
- Implement anime-segmentation wrapper
- Implement GroundingDINO + SAM 2 wrapper
- Create comparison test script

### Phase 3: Integration

- Update test_layer_interpolation.py to use new module
- Update RunPod setup script
- Update pyproject.toml

### Phase 4: Validation

- Run segmentation on both manga panels
- Compare coverage metrics against rembg baseline (13.5% / 45.4%)
- Verify VRAM cleanup between segmentation and interpolation

---

## STEP-BY-STEP TASKS

### CREATE `manimate/segmentation/__init__.py`

- **IMPLEMENT**: Module exports
  ```python
  from manimate.segmentation.base import BaseSegmenter
  from manimate.segmentation.anime_seg import AnimeSegmenter

  __all__ = ["BaseSegmenter", "AnimeSegmenter"]
  ```
- **GOTCHA**: Don't import GroundedSAMSegmenter at module level — it has heavy deps (grounded-sam-2) that may not be installed. Import it lazily or leave it out of `__all__`.
- **VALIDATE**: `python -c "from manimate.segmentation import BaseSegmenter"`

### CREATE `manimate/segmentation/base.py`

- **IMPLEMENT**: Abstract base class mirroring `BaseInterpolator` pattern
  ```python
  from abc import ABC, abstractmethod
  from PIL import Image
  import numpy as np

  class BaseSegmenter(ABC):
      @abstractmethod
      def load(self) -> None: ...

      @abstractmethod
      def segment(self, image: Image.Image) -> np.ndarray:
          """Return foreground mask as uint8 numpy array [H, W], values 0-255."""
          ...

      def unload(self) -> None: ...
  ```
- **PATTERN**: Mirror `manimate/interpolation/base.py` — same `load()`/`unload()` lifecycle
- **GOTCHA**: Return mask as `np.ndarray` (uint8, 0-255) not PIL Image — downstream code in `test_layer_interpolation.py` expects numpy arrays for morphological ops and compositing

### CREATE `manimate/segmentation/anime_seg.py`

- **IMPLEMENT**: Wrapper around SkyTNT/anime-segmentation
- **MODEL**: `isnet_is` from HuggingFace `skytnt/anime-seg` — best quality at 1024x1024 input
- **APPROACH**: Since anime-segmentation has no pip package, download just the model weights and reimplement the minimal inference code. The core is: load ISNet model → preprocess (resize to 1024x1024, normalize) → forward pass → sigmoid → threshold → resize back to original dims
- **KEY DETAILS**:
  - Model architecture: ISNet (from `isnet.py` in anime-segmentation repo). It's a U²-Net variant. Rather than vendoring the whole repo, use `huggingface_hub` to download the ONNX or PyTorch weights and run inference directly.
  - Alternative: use the ONNX model via `onnxruntime` — simpler, no ISNet code needed. The HF repo has `.onnx` exports.
  - Input: PIL Image → resize to 1024x1024 → normalize → tensor [1, 3, 1024, 1024]
  - Output: mask tensor [1, 1, 1024, 1024] → sigmoid → threshold at 0.5 → resize to original → uint8 [H, W]
- **CONFIG**: 
  ```python
  self.model_path = config.get("model_path", "skytnt/anime-seg")  # HF repo or local path
  self.img_size = config.get("img_size", 1024)
  self.threshold = config.get("threshold", 0.5)
  ```
- **VRAM**: ~200MB — lightweight, can coexist with MoG on 48GB
- **GOTCHA**: Convert grayscale manga to RGB before inference (model expects 3 channels). Use `img.convert("RGB")`.
- **VALIDATE**: `python -c "from manimate.segmentation.anime_seg import AnimeSegmenter"`

### CREATE `manimate/segmentation/grounded_sam.py`

- **IMPLEMENT**: GroundingDINO + SAM 2 wrapper for text-prompted instance segmentation
- **APPROACH**: Use the `grounded-sam-2` combined repo. Install as git submodule at `vendor/Grounded-SAM-2/`.
- **KEY DETAILS**:
  - GroundingDINO: text prompt → bounding boxes. Use prompt `"character . person . figure"` (GDINO uses `.` as separator for multiple classes)
  - SAM 2: bounding boxes → precise masks
  - Combined output: union of all instance masks → single foreground mask [H, W]
  - Models: `groundingdino_swinb_cogcoor.pth` (~900MB), `sam2.1_hiera_large.pt` (~900MB)
- **CONFIG**:
  ```python
  self.text_prompt = config.get("text_prompt", "character . person . figure")
  self.box_threshold = config.get("box_threshold", 0.3)
  self.text_threshold = config.get("text_threshold", 0.25)
  self.sam_model = config.get("sam_model", "sam2.1_hiera_large")
  ```
- **VRAM**: ~8GB combined — must be unloaded before MoG on smaller GPUs
- **GOTCHA**: GroundingDINO and SAM 2 both have complex build requirements (custom CUDA kernels). This is the heavier option — document that `anime_seg` is recommended for most cases.
- **VALIDATE**: `python -c "from manimate.segmentation.grounded_sam import GroundedSAMSegmenter"`

### CREATE `configs/segmentation/default.yaml`

- **IMPLEMENT**: Hydra config for segmentation
  ```yaml
  backend: anime_seg  # Options: anime_seg, grounded_sam

  anime_seg:
    model_path: skytnt/anime-seg
    img_size: 1024
    threshold: 0.5

  grounded_sam:
    text_prompt: "character . person . figure"
    box_threshold: 0.3
    text_threshold: 0.25
    sam_model: sam2.1_hiera_large
    gdino_model: groundingdino_swinb_cogcoor

  # Post-processing
  morphology:
    close_kernel: 5
    close_iterations: 3
    open_kernel: 5
    open_iterations: 2

  hydra:
    output_subdir: null
    run:
      dir: .
  ```
- **PATTERN**: Mirror `configs/interpolation/default.yaml` structure

### CREATE `scripts/test_segmentation.py`

- **IMPLEMENT**: Comparison test that runs both segmentation backends on the manga panels and outputs masks + coverage metrics side by side
- **INPUTS**: `data/manga_panels/Screenshot*.png` (same panels used in layer test)
- **OUTPUTS**: `output/segmentation_test/`
  - `rembg_mask_a.png`, `rembg_mask_b.png` — baseline (rembg, for comparison)
  - `animeseg_mask_a.png`, `animeseg_mask_b.png` — anime-segmentation
  - `gsam_mask_a.png`, `gsam_mask_b.png` — GroundingDINO + SAM 2 (if available)
  - `comparison.txt` — coverage metrics table
- **VRAM STAGING**: Load segmenter → run both panels → unload → next segmenter
- **VALIDATE**: `python scripts/test_segmentation.py --backend anime_seg`

### UPDATE `scripts/test_layer_interpolation.py`

- **REFACTOR**: Replace `generate_masks()` function (lines 58-80) to use new segmentation module instead of rembg
- **CHANGE**: Import `AnimeSegmenter` from `manimate.segmentation`, use its `segment()` method
- **KEEP**: Morphological cleanup, layer separation, compositing — all unchanged
- **REMOVE**: `from rembg import new_session, remove` import
- **ADD**: CLI arg `--backend anime_seg` to select segmentation backend
- **VALIDATE**: `python scripts/test_layer_interpolation.py --backend anime_seg` (dry run — will need GPU)

### UPDATE `pyproject.toml`

- **ADD**: `onnxruntime-gpu` to optional deps (for anime-segmentation ONNX inference)
- **ADD**: `huggingface_hub` if not already present (for model downloads)
- **GOTCHA**: Don't add grounded-sam-2 to pyproject.toml — it's a vendor submodule with its own complex build

### UPDATE `scripts/setup_runpod.sh`

- **ADD**: After existing MoG deps block, add segmentation deps:
  ```bash
  echo "=== Installing segmentation deps ==="
  pip install --no-cache-dir onnxruntime-gpu 2>&1 | grep -E "Successfully|ERROR"
  ```
- **GOTCHA**: `onnxruntime-gpu` must match CUDA version on RunPod container

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
```bash
ruff check manimate/segmentation/ scripts/test_segmentation.py
ruff format --check manimate/segmentation/ scripts/test_segmentation.py
```

### Level 2: Import Check
```bash
python -c "from manimate.segmentation import BaseSegmenter, AnimeSegmenter; print('OK')"
```

### Level 3: Dry Run (requires GPU + model weights)
```bash
python scripts/test_segmentation.py --backend anime_seg
```

### Level 4: Integration (on RunPod)
```bash
python scripts/test_layer_interpolation.py --backend anime_seg
```

---

## ACCEPTANCE CRITERIA

- [ ] `BaseSegmenter` ABC with `load()`, `segment()`, `unload()` interface
- [ ] `AnimeSegmenter` loads model, produces uint8 mask [H, W] for any PIL Image
- [ ] `GroundedSAMSegmenter` loads models, produces mask from text prompt
- [ ] Coverage on manga panel A significantly better than 13.5% (rembg baseline)
- [ ] Coverage on manga panel B comparable to or better than 45.4% (rembg baseline)
- [ ] VRAM properly freed after segmenter.unload()
- [ ] `test_layer_interpolation.py` uses new module, no rembg dependency
- [ ] All lint/format checks pass
- [ ] Configs follow Hydra pattern

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed
- [ ] All validation commands executed
- [ ] Segmentation quality verified on both panels
- [ ] Ready for `/commit`

---

## NOTES

- **Start with anime-segmentation** — it's lighter, faster, and purpose-built for anime content. Only add GroundedSAM if anime-seg also fails on B&W manga.
- **ONNX inference preferred** over PyTorch for anime-segmentation — avoids vendoring ISNet model code, simpler dependency chain.
- **The real test is panel A** — panel B was already reasonable with rembg. If anime-segmentation also fails on panel A, the issue may be fundamental to this particular panel's composition (character fills most of the frame with similar line weight to background).
- **Don't modify compositing logic** — the downstream code (layer separation, interpolation, compositing) in `test_layer_interpolation.py` is fine. Only the mask generation needs improvement.
