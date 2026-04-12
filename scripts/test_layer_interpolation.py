#!/usr/bin/env python3
"""Test layer-separated interpolation on manga panels.

Pipeline:
  1. Segment foreground (character) from background using rembg (isnet-anime)
  2. Separate each panel into character layer + inpainted background layer
  3. Interpolate each layer pair independently with MoG-VFI (14 frames)
  4. Temporal resample to 24 frames for 1-second output at 24fps
  5. Composite character over background using blended masks
  6. Compare against baseline flat interpolation

Outputs to output/layer_test/:
  inputs/        — masks and separated layers for inspection
  baseline.mp4   — flat interpolation (no layer separation)
  char_layer.mp4 — character layer only
  bg_layer.mp4   — background layer only
  composited.mp4 — layers recombined

RunPod:
    pip install rembg
    python scripts/test_layer_interpolation.py
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from manimate.interpolation.mog import MoGInterpolator
from manimate.video.io import frames_to_video, save_frames

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "manga_panels"
OUTPUT_DIR = REPO_ROOT / "output" / "layer_test"

TARGET_FRAMES = 24
TARGET_FPS = 24


# --- Stage 1: Segmentation ---


def load_panels() -> tuple[Image.Image, Image.Image]:
    """Load the two manga panel screenshots."""
    panels = sorted(DATA_DIR.glob("Screenshot*.png"))
    assert len(panels) == 2, f"Expected 2 screenshots in {DATA_DIR}, found {len(panels)}"
    a = Image.open(panels[0]).convert("RGB")
    b = Image.open(panels[1]).convert("RGB")
    print(f"Panel A: {panels[0].name} {a.size}")
    print(f"Panel B: {panels[1].name} {b.size}")
    return a, b


def generate_masks(a: Image.Image, b: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Generate foreground masks using anime-specific background removal."""
    from rembg import new_session, remove

    print("Loading isnet-anime segmentation model...")
    session = new_session("isnet-anime")

    mask_a = np.array(remove(a, session=session, only_mask=True))
    mask_b = np.array(remove(b, session=session, only_mask=True))

    # Morphological cleanup — close small gaps, remove noise
    kernel = np.ones((5, 5), np.uint8)
    for mask in [mask_a, mask_b]:
        mask[:] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask[:] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    del session
    torch.cuda.empty_cache()

    coverage_a = mask_a.mean() / 255.0
    coverage_b = mask_b.mean() / 255.0
    print(f"Foreground coverage: A={coverage_a:.1%}, B={coverage_b:.1%}")
    return mask_a, mask_b


def separate_layers(img: Image.Image, mask: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Split image into character (on gray) and background (inpainted)."""
    img_np = np.array(img)
    fg_mask = mask > 127
    mask_3ch = np.stack([fg_mask] * 3, axis=-1)

    # Character on neutral gray — provides contrast with both black lines and white areas
    gray_bg = np.full_like(img_np, 128)
    char_layer = np.where(mask_3ch, img_np, gray_bg).astype(np.uint8)

    # Background — inpaint where the character was
    inpaint_mask = fg_mask.astype(np.uint8) * 255
    bg_layer = cv2.inpaint(img_np, inpaint_mask, 10, cv2.INPAINT_TELEA)

    return Image.fromarray(char_layer), Image.fromarray(bg_layer)


# --- Stage 2: Interpolation ---


def interpolate_pair(
    interpolator: MoGInterpolator,
    frame_a: Image.Image,
    frame_b: Image.Image,
    label: str,
) -> list[Image.Image]:
    """Interpolate a single pair, returning keyframes + intermediates."""
    print(f"  {label}...")
    intermediate = interpolator.interpolate(frame_a, frame_b)

    kf_a = interpolator.prepare_keyframe(frame_a)
    kf_b = interpolator.prepare_keyframe(frame_b)
    result = [kf_a, *intermediate, kf_b]
    print(f"    {len(result)} MoG frames ({len(intermediate)} intermediate)")
    return result


def resample_to_target(frames: list[Image.Image], target: int) -> list[Image.Image]:
    """Temporally resample frames to target count using linear blending.

    MoG gives us 16 frames; we need 24 for a 1-second clip at 24fps.
    First and last frames are preserved exactly.
    """
    n = len(frames)
    if n == target:
        return frames

    result = []
    for i in range(target):
        src_t = i * (n - 1) / (target - 1)
        idx_lo = int(src_t)
        idx_hi = min(idx_lo + 1, n - 1)
        alpha = src_t - idx_lo

        if alpha < 1e-3:
            result.append(frames[idx_lo])
        elif alpha > 1 - 1e-3:
            result.append(frames[idx_hi])
        else:
            a = np.array(frames[idx_lo]).astype(np.float32)
            b = np.array(frames[idx_hi]).astype(np.float32)
            blended = ((1 - alpha) * a + alpha * b).astype(np.uint8)
            result.append(Image.fromarray(blended))

    assert len(result) == target
    return result


# --- Stage 3: Composite ---


def composite_frames(
    char_frames: list[Image.Image],
    bg_frames: list[Image.Image],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> list[Image.Image]:
    """Composite character over background using linearly interpolated masks."""
    assert len(char_frames) == len(bg_frames)
    n = len(char_frames)
    w, h = char_frames[0].size

    # Resize source masks to match output frame dimensions
    ma = cv2.resize(mask_a, (w, h)).astype(np.float32) / 255.0
    mb = cv2.resize(mask_b, (w, h)).astype(np.float32) / 255.0

    composited = []
    for i in range(n):
        t = i / max(n - 1, 1)
        blended_mask = (1 - t) * ma + t * mb
        alpha = (blended_mask > 0.5).astype(np.float32)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        char_np = np.array(char_frames[i]).astype(np.float32)
        bg_np = np.array(bg_frames[i]).astype(np.float32)
        result = (char_np * alpha_3ch + bg_np * (1 - alpha_3ch)).astype(np.uint8)
        composited.append(Image.fromarray(result))

    return composited


# --- Output ---


def save_video(frames: list[Image.Image], name: str, fps: int) -> None:
    """Save frame list as video, cleaning up temp frames."""
    frame_dir = OUTPUT_DIR / f"{name}_frames"
    save_frames(frames, frame_dir)
    video_path = OUTPUT_DIR / f"{name}.mp4"
    frames_to_video(frame_dir, video_path, fps=fps)
    shutil.rmtree(frame_dir)
    print(f"  {name}.mp4 — {len(frames)} frames @ {fps}fps = {len(frames) / fps:.1f}s")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Test layer-separated interpolation")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--half-precision", action="store_true")
    args = parser.parse_args()

    config = {
        "mog": {
            "ckpt_path": str(REPO_ROOT / "vendor" / "MoG-VFI" / "checkpoints" / "ani.ckpt"),
            "config_path": str(REPO_ROOT / "vendor" / "MoG-VFI" / "configs" / "ani.yaml"),
        },
        "height": 320,
        "width": 512,
        "ddim_steps": args.ddim_steps,
        "ddim_eta": 1.0,
        "guidance_scale": 7.5,
        "guidance_rescale": 0.7,
        "frame_stride": 24,
        "timestep_spacing": "uniform_trailing",
        "half_precision": args.half_precision,
        "seed": 42,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs_dir = OUTPUT_DIR / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    # --- Stage 1: Segmentation (rembg model, then free VRAM) ---
    print("=== Stage 1: Segmentation ===")
    panel_a, panel_b = load_panels()
    mask_a, mask_b = generate_masks(panel_a, panel_b)

    char_a, bg_a = separate_layers(panel_a, mask_a)
    char_b, bg_b = separate_layers(panel_b, mask_b)

    # Save all inputs for inspection
    for name, img in [
        ("panel_a", panel_a),
        ("panel_b", panel_b),
        ("mask_a", Image.fromarray(mask_a)),
        ("mask_b", Image.fromarray(mask_b)),
        ("char_a", char_a),
        ("char_b", char_b),
        ("bg_a", bg_a),
        ("bg_b", bg_b),
    ]:
        img.save(inputs_dir / f"{name}.png")
    print(f"Inputs saved to {inputs_dir}")

    # --- Stage 2: Interpolation (single MoG load, three runs) ---
    print("\n=== Stage 2: MoG Interpolation ===")
    print("Loading MoG model...")
    interpolator = MoGInterpolator(config)
    interpolator.load()

    baseline_raw = interpolate_pair(interpolator, panel_a, panel_b, "baseline (flat)")
    char_raw = interpolate_pair(interpolator, char_a, char_b, "character layer")
    bg_raw = interpolate_pair(interpolator, bg_a, bg_b, "background layer")

    interpolator.unload()

    # Resample 16 MoG frames → 24 frames for 1s @ 24fps
    print(f"\nResampling {len(baseline_raw)} → {TARGET_FRAMES} frames...")
    baseline_frames = resample_to_target(baseline_raw, TARGET_FRAMES)
    char_frames = resample_to_target(char_raw, TARGET_FRAMES)
    bg_frames = resample_to_target(bg_raw, TARGET_FRAMES)

    # --- Stage 3: Composite ---
    print("\n=== Stage 3: Composite ===")
    composited = composite_frames(char_frames, bg_frames, mask_a, mask_b)

    # --- Save outputs ---
    print("\n=== Saving outputs ===")
    for name, frames in [
        ("baseline", baseline_frames),
        ("char_layer", char_frames),
        ("bg_layer", bg_frames),
        ("composited", composited),
    ]:
        save_video(frames, name, fps=TARGET_FPS)

    print(f"\nDone. All outputs in {OUTPUT_DIR}/")
    print("Compare baseline.mp4 vs composited.mp4 to evaluate layer separation.")


if __name__ == "__main__":
    main()
