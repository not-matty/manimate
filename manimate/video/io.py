"""Video and frame I/O utilities."""

import subprocess
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_image(path: Path) -> Image.Image:
    """Load a single image."""
    path = Path(path)
    assert path.exists(), f"Image not found: {path}"
    return Image.open(path).convert("RGB")


def load_frames(directory: Path, pattern: str = "*") -> list[Image.Image]:
    """Load all image frames from a directory, sorted by filename.

    Accepts png, jpg, and jpeg files. Sorted lexicographically by stem.
    """
    directory = Path(directory)
    assert directory.is_dir(), f"Not a directory: {directory}"

    files = sorted(
        [f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda f: f.stem,
    )
    assert len(files) >= 1, f"No image files found in {directory}"

    return [Image.open(f).convert("RGB") for f in files]


def save_frames(
    frames: list[Image.Image],
    directory: Path,
    prefix: str = "frame",
) -> list[Path]:
    """Save frames as frame_000000.png, frame_000001.png, etc."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, frame in enumerate(frames):
        path = directory / f"{prefix}_{i:06d}.png"
        frame.save(path)
        paths.append(path)
    return paths


def frames_to_video(
    frame_dir: Path,
    output_path: Path,
    fps: int = 24,
    pattern: str = "frame_%06d.png",
) -> Path:
    """Assemble frames into video using ffmpeg.

    Expects frames named with the given pattern in frame_dir.
    """
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / pattern),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr}"
    assert output_path.exists(), f"Output video not created: {output_path}"
    return output_path


def video_to_frames(video_path: Path, output_dir: Path) -> list[Path]:
    """Extract frames from video using ffmpeg."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    assert video_path.exists(), f"Video not found: {video_path}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        str(output_dir / "frame_%06d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr}"

    frames = sorted(output_dir.glob("frame_*.png"))
    assert len(frames) > 0, f"No frames extracted from {video_path}"
    return frames
