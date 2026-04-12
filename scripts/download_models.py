"""Download MoG-VFI model checkpoints."""

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MOG_DIR = REPO_ROOT / "vendor" / "MoG-VFI"

HF_REPO = "MCG-NJU/MoG"
CHECKPOINTS = {
    "ani": {
        "hf_filename": "ani.ckpt",
        "dest": MOG_DIR / "checkpoints" / "ani.ckpt",
    },
    "real": {
        "hf_filename": "real.ckpt",
        "dest": MOG_DIR / "checkpoints" / "real.ckpt",
    },
}

# EMA-VFI checkpoint (Google Drive only)
EMAVFI_GDRIVE_FOLDER = "https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o"
EMAVFI_DEST = MOG_DIR / "emavfi" / "ckpt" / "ours_t.pkl"


def download_hf_checkpoints(models: list[str]) -> None:
    """Download checkpoints from HuggingFace using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call(["uv", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    for name in models:
        info = CHECKPOINTS[name]
        dest = info["dest"]
        if dest.exists():
            print(f"[skip] {name}: already exists at {dest}")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {name}: {HF_REPO}/{info['hf_filename']} -> {dest}")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=info["hf_filename"],
            local_dir=dest.parent,
            local_dir_use_symlinks=False,
        )
        print(f"[done] {name}: saved to {dest}")


def download_emavfi() -> None:
    """Download EMA-VFI checkpoint using gdown."""
    if EMAVFI_DEST.exists():
        print(f"[skip] ours_t.ckpt: already exists at {EMAVFI_DEST}")
        return

    EMAVFI_DEST.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore[import-not-found]
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call(["uv", "pip", "install", "gdown"])
        import gdown  # type: ignore[import-not-found]

    print(f"[download] ours_t.ckpt from Google Drive -> {EMAVFI_DEST.parent}")
    gdown.download_folder(
        url=EMAVFI_GDRIVE_FOLDER,
        output=str(EMAVFI_DEST.parent),
        quiet=False,
    )
    assert EMAVFI_DEST.exists(), (
        f"Expected {EMAVFI_DEST} after download. "
        f"If gdown failed, download manually from:\n  {EMAVFI_GDRIVE_FOLDER}\n"
        f"  Place ours_t.ckpt at: {EMAVFI_DEST}"
    )
    print(f"[done] ours_t.ckpt: saved to {EMAVFI_DEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MoG-VFI model checkpoints")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ani", "real", "all"],
        default=["ani"],
        help="Which HuggingFace checkpoints to download (default: ani)",
    )
    parser.add_argument(
        "--skip-emavfi",
        action="store_true",
        help="Skip downloading the EMA-VFI checkpoint (ours_t.ckpt)",
    )
    args = parser.parse_args()

    models = list(CHECKPOINTS.keys()) if "all" in args.models else args.models
    download_hf_checkpoints(models)

    if not args.skip_emavfi:
        download_emavfi()

    print("\nAll downloads complete.")
    print(f"  MoG checkpoints: {MOG_DIR / 'checkpoints'}")
    print(f"  EMA-VFI checkpoint: {EMAVFI_DEST}")


if __name__ == "__main__":
    main()
