"""MoG-VFI (Motion-Aware Generative Frame Interpolation) wrapper."""

import os
import sys
import threading
from pathlib import Path

import torch
import torchvision.transforms as transforms
from einops import repeat
from PIL import Image

from manimate.interpolation.base import BaseInterpolator

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "MoG-VFI"

# MoG generates exactly 16 frames per pair: 2 input + 14 interpolated.
MOG_TOTAL_FRAMES = 16
MOG_INTERMEDIATE_FRAMES = 14


def _setup_mog_imports() -> None:
    """Add MoG-VFI to sys.path so its internal imports resolve."""
    vendor_str = str(VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


class MoGInterpolator(BaseInterpolator):
    """Thin wrapper around MoG-VFI for manga-style frame interpolation.

    MoG generates 14 intermediate frames between two input frames at 320x512
    resolution using diffusion-based video generation with optical flow guidance.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        mog_cfg = config.get("mog", {})
        self.ckpt_path = Path(
            mog_cfg.get("ckpt_path", VENDOR_DIR / "checkpoints" / "ani.ckpt")
        ).resolve()
        self.config_path = Path(
            mog_cfg.get("config_path", VENDOR_DIR / "configs" / "ani.yaml")
        ).resolve()

        self.height = config.get("height", 320)
        self.width = config.get("width", 512)
        self.ddim_steps = config.get("ddim_steps", 50)
        self.ddim_eta = config.get("ddim_eta", 1.0)
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.guidance_rescale = config.get("guidance_rescale", 0.7)
        self.frame_stride = config.get("frame_stride", 24)
        self.timestep_spacing = config.get("timestep_spacing", "uniform_trailing")
        self.half_precision = config.get("half_precision", False)
        self.seed = config.get("seed", 42)

        self.model = None

    def load(self) -> None:
        """Load MoG model and EMA-VFI flow model."""
        assert threading.current_thread() is threading.main_thread(), (
            "MoG model loading requires main thread (uses os.chdir)"
        )
        _setup_mog_imports()

        from omegaconf import OmegaConf  # noqa: I001
        from utils.utils import instantiate_from_config  # type: ignore[import-not-found]

        assert self.ckpt_path.exists(), f"MoG checkpoint not found: {self.ckpt_path}"
        assert self.config_path.exists(), f"MoG config not found: {self.config_path}"

        # EMA-VFI loads relative to CWD, so we need to be in the vendor dir.
        original_cwd = os.getcwd()
        os.chdir(VENDOR_DIR)
        try:
            config = OmegaConf.load(str(self.config_path))
            model_config = config.pop("model", OmegaConf.create())  # type: ignore[call-arg]
            model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False

            # Patch EMA-VFI's Model.device() to no-op during init so VFI stays on CPU.
            # get_vfi_model() is called inside the model constructor and forces CUDA.
            import emavfi.Trainer as trainer_module  # type: ignore[import-not-found]  # noqa: I001

            _original_device = trainer_module.Model.device
            trainer_module.Model.device = lambda self: None  # type: ignore[assignment]
            try:
                self.model = instantiate_from_config(model_config)
            finally:
                trainer_module.Model.device = _original_device

            self.model.perframe_ae = True

            # Load checkpoint on CPU
            from scripts.evaluation.inference import load_model_checkpoint  # type: ignore[import-not-found]  # noqa: I001

            self.model = load_model_checkpoint(self.model, str(self.ckpt_path))
            self.model.eval()

            # Convert to fp16 on CPU before moving to GPU to fit in 8GB VRAM
            if self.half_precision:
                self.model = self.model.half()
                # Also convert VFI sub-model
                self.model.vfi.net = self.model.vfi.net.half()

            self.model = self.model.cuda()
            # Move VFI net to same device
            self.model.vfi.net = self.model.vfi.net.to("cuda")
        finally:
            os.chdir(original_cwd)

    def interpolate(
        self,
        frame_a: Image.Image,
        frame_b: Image.Image,
        num_frames: int = MOG_INTERMEDIATE_FRAMES,
        prompt: str = "",
    ) -> list[Image.Image]:
        """Generate intermediate frames between frame_a and frame_b.

        MoG always generates exactly 14 intermediate frames. If num_frames != 14,
        the output is uniformly subsampled to the requested count.

        Returns list of PIL Images (interpolated frames only, not including inputs).
        """
        assert self.model is not None, "Model not loaded. Call load() first."
        assert num_frames >= 1, f"num_frames must be >= 1, got {num_frames}"

        from pytorch_lightning import seed_everything

        seed_everything(self.seed)

        tensor_a = self._prepare_frame(frame_a)
        tensor_b = self._prepare_frame(frame_b)
        tensor_a = tensor_a.unsqueeze(1)  # [C, 1, H, W]
        tensor_b = tensor_b.unsqueeze(1)

        half_t = MOG_TOTAL_FRAMES // 2
        repeated_a = repeat(tensor_a, "c t h w -> c (repeat t) h w", repeat=half_t)
        repeated_b = repeat(tensor_b, "c t h w -> c (repeat t) h w", repeat=half_t)
        video_tensor = torch.cat([repeated_a, repeated_b], dim=1)  # [C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0).cuda()  # [1, C, T, H, W]

        if self.half_precision:
            video_tensor = video_tensor.half()

        # Compute noise shape in latent space
        h, w = self.height // 8, self.width // 8
        channels = self.model.model.diffusion_model.out_channels
        noise_shape = [1, channels, MOG_TOTAL_FRAMES, h, w]

        # Run inference
        _setup_mog_imports()
        from scripts.evaluation.inference import image_guided_synthesis  # type: ignore[import-not-found]  # noqa: I001

        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples = image_guided_synthesis(
                self.model,
                [prompt] if prompt else [""],
                video_tensor,
                noise_shape,
                n_samples=1,
                ddim_steps=self.ddim_steps,
                ddim_eta=self.ddim_eta,
                unconditional_guidance_scale=self.guidance_scale,
                cfg_img=None,
                fs=self.frame_stride,
                text_input=bool(prompt),
                multiple_cond_cfg=False,
                loop=False,
                interp=True,
                timestep_spacing=self.timestep_spacing,
                guidance_rescale=self.guidance_rescale,
            )

        # batch_samples shape: [1, n_samples, C, T, H, W]
        # Take first sample
        sample = batch_samples[0, 0]  # [C, T, H, W]
        assert sample.shape[1] == MOG_TOTAL_FRAMES, (
            f"Expected {MOG_TOTAL_FRAMES} frames, got {sample.shape[1]}"
        )

        # Extract the 14 intermediate frames (skip first and last which are input frames)
        intermediate = sample[:, 1:-1]  # [C, 14, H, W]

        # Convert to PIL Images
        all_frames = self._tensor_to_pil_list(intermediate)
        assert len(all_frames) == MOG_INTERMEDIATE_FRAMES

        # Subsample if fewer frames requested
        if num_frames < MOG_INTERMEDIATE_FRAMES:
            indices = self._uniform_subsample_indices(MOG_INTERMEDIATE_FRAMES, num_frames)
            all_frames = [all_frames[i] for i in indices]

        return all_frames

    def prepare_keyframe(self, frame: Image.Image) -> Image.Image:
        """Resize keyframe to match MoG output dimensions (aspect-preserving crop)."""
        return self._aspect_cover_crop(frame.convert("RGB"), self.height, self.width)

    def unload(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    @staticmethod
    def _aspect_cover_crop(img: Image.Image, height: int, width: int) -> Image.Image:
        """Resize to cover target area, then center-crop to exact dimensions.

        Unlike Resize+CenterCrop which zero-pads when the image is too narrow,
        this resizes so BOTH dimensions are >= target, then crops the excess.
        """
        w, h = img.size
        scale = max(height / h, width / w)
        new_h = round(h * scale)
        new_w = round(w * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return img.crop((left, top, left + width, top + height))

    def _prepare_frame(self, img: Image.Image) -> torch.Tensor:
        """Aspect-preserving resize + crop, then normalize to [-1, 1] tensor."""
        img = self._aspect_cover_crop(img.convert("RGB"), self.height, self.width)
        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(tensor)
        return tensor

    @staticmethod
    def _tensor_to_pil_list(tensor: torch.Tensor) -> list[Image.Image]:
        """Convert a [C, T, H, W] tensor in [-1, 1] range to list of PIL Images."""
        tensor = tensor.detach().cpu().float()
        tensor = torch.clamp(tensor, -1.0, 1.0)
        tensor = (tensor + 1.0) / 2.0  # [-1,1] -> [0,1]
        tensor = (tensor * 255).to(torch.uint8)

        frames = []
        for t in range(tensor.shape[1]):
            frame = tensor[:, t]  # [C, H, W]
            frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
            frames.append(Image.fromarray(frame))
        return frames

    @staticmethod
    def _uniform_subsample_indices(total: int, target: int) -> list[int]:
        """Pick `target` uniformly spaced indices from range(total)."""
        assert target <= total
        return [round(i * (total - 1) / (target - 1)) for i in range(target)]
