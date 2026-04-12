"""Multi-keyframe interpolation pipeline."""

from PIL import Image

from manimate.interpolation.base import BaseInterpolator


class InterpolationPipeline:
    """Interpolate between a sequence of keyframes.

    Given keyframes [K1, K2, K3], produces:
    [K1, i1, i2, ..., iN, K2, i1, i2, ..., iN, K3]
    where N = num_intermediate frames per pair (may vary per pair).
    """

    def __init__(self, interpolator: BaseInterpolator) -> None:
        self.interpolator = interpolator

    def run(
        self,
        keyframes: list[Image.Image],
        num_intermediate: int | list[int] = 14,
        prompts: list[str] | None = None,
    ) -> list[Image.Image]:
        """Interpolate between all adjacent keyframe pairs.

        Args:
            keyframes: List of keyframe images (at least 2).
            num_intermediate: Frames per pair. Single int applies uniformly.
                List specifies per-pair counts (length must equal num pairs).
            prompts: Optional text prompts, one per keyframe pair.

        Returns:
            Full frame sequence including keyframes (all at interpolator's output resolution).
        """
        assert len(keyframes) >= 2, f"Need at least 2 keyframes, got {len(keyframes)}"

        num_pairs = len(keyframes) - 1

        if isinstance(num_intermediate, int):
            counts = [num_intermediate] * num_pairs
        else:
            assert len(num_intermediate) == num_pairs, (
                f"Need {num_pairs} frame counts for {len(keyframes)} keyframes, "
                f"got {len(num_intermediate)}"
            )
            counts = list(num_intermediate)

        if prompts is not None:
            assert len(prompts) == num_pairs, (
                f"Need {num_pairs} prompts for {len(keyframes)} keyframes, got {len(prompts)}"
            )

        expected_total = len(keyframes) + sum(counts)

        # Prepare keyframes to match interpolated frame dimensions
        prepared = [self.interpolator.prepare_keyframe(k) for k in keyframes]

        all_frames: list[Image.Image] = [prepared[0]]
        for i in range(num_pairs):
            prompt = prompts[i] if prompts else ""
            intermediate = self.interpolator.interpolate(
                keyframes[i], keyframes[i + 1], counts[i], prompt
            )
            assert len(intermediate) == counts[i], (
                f"Pair {i}: expected {counts[i]} frames, got {len(intermediate)}"
            )
            all_frames.extend(intermediate)
            all_frames.append(prepared[i + 1])

        assert len(all_frames) == expected_total, (
            f"Expected {expected_total} total frames, got {len(all_frames)}"
        )
        return all_frames
