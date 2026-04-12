"""Base interface for frame interpolation models."""

from abc import ABC, abstractmethod

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

    def prepare_keyframe(self, frame: Image.Image) -> Image.Image:
        """Prepare a keyframe to match interpolated frame dimensions.

        Override for models with fixed output resolution. Default returns as-is.
        """
        return frame

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory."""
        ...
