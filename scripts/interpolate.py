"""CLI entry point for Mode A — Keyframe interpolation."""

import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from manimate.interpolation.mog import MoGInterpolator
from manimate.video.io import frames_to_video, load_frames


@hydra.main(config_path="../configs/interpolation", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    config: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    keyframes_dir = Path(cfg.keyframes)
    output_path = Path(cfg.output)
    fps = cfg.target_fps
    prompt = cfg.prompt

    # Load keyframes
    print(f"Loading keyframes from {keyframes_dir}")
    keyframes = load_frames(keyframes_dir)
    print(f"Loaded {len(keyframes)} keyframes")
    assert len(keyframes) >= 2, f"Need at least 2 keyframes, got {len(keyframes)}"

    # Initialize model
    print("Loading MoG model...")
    interpolator = MoGInterpolator(config)
    interpolator.load()

    # Determine output frame directory
    is_video = output_path.suffix in (".mp4", ".avi", ".mov", ".mkv")
    frame_dir = output_path.parent / f"{output_path.stem}_frames" if is_video else output_path
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Interpolate pair-by-pair with incremental saving — each pair's frames
    # are written to disk immediately so a crash doesn't lose prior GPU work.
    num_pairs = len(keyframes) - 1
    frame_idx = 0

    for i in range(num_pairs):
        # Save keyframe (prepared to match output resolution)
        prepared_kf = interpolator.prepare_keyframe(keyframes[i])
        prepared_kf.save(frame_dir / f"frame_{frame_idx:06d}.png")
        frame_idx += 1

        # Interpolate this pair
        print(f"Interpolating pair {i + 1}/{num_pairs}...")
        intermediate = interpolator.interpolate(keyframes[i], keyframes[i + 1], prompt=prompt)

        # Save interpolated frames immediately
        for frame in intermediate:
            frame.save(frame_dir / f"frame_{frame_idx:06d}.png")
            frame_idx += 1

        print(f"  Pair {i + 1} done — {len(intermediate)} frames saved (total: {frame_idx})")

    # Save last keyframe
    prepared_kf = interpolator.prepare_keyframe(keyframes[-1])
    prepared_kf.save(frame_dir / f"frame_{frame_idx:06d}.png")
    frame_idx += 1

    print(f"All {frame_idx} frames saved to {frame_dir}")

    # Assemble video if requested
    if is_video:
        print(f"Assembling video at {fps} fps -> {output_path}")
        frames_to_video(frame_dir, output_path, fps=fps)
        if not cfg.save_frames:
            shutil.rmtree(frame_dir)
        print(f"Output video: {output_path}")

    interpolator.unload()
    print("Done.")


if __name__ == "__main__":
    main()
