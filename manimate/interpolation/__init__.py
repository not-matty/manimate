"""Frame interpolation engine."""

from manimate.interpolation.base import BaseInterpolator
from manimate.interpolation.mog import MoGInterpolator
from manimate.interpolation.pipeline import InterpolationPipeline

__all__ = ["BaseInterpolator", "MoGInterpolator", "InterpolationPipeline"]
