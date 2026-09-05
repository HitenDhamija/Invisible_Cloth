"""Pure background-aggregation functions.

These operate on lists of numpy arrays and have no camera or OpenCV
dependencies beyond numpy itself, making them easy to unit-test with
synthetic images.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AggregationError(Exception):
    """Raised when aggregation cannot be performed."""


def _validate_frames(frames: list[np.ndarray], min_frames: int = 1) -> None:
    """Check that *frames* is non-empty and all entries share the same shape.

    Raises:
        AggregationError: On empty input or shape mismatch.
    """
    if not frames:
        raise AggregationError("Cannot aggregate an empty list of frames")

    if len(frames) < min_frames:
        raise AggregationError(f"Need at least {min_frames} frame(s), got {len(frames)}")

    reference_shape = frames[0].shape
    for idx, frame in enumerate(frames[1:], start=2):
        if frame.shape != reference_shape:
            raise AggregationError(
                f"Frame 1 has shape {reference_shape}, but frame {idx} has shape {frame.shape}"
            )


def aggregate_mean(frames: list[np.ndarray]) -> np.ndarray:
    """Compute the element-wise mean across *frames*.

    Args:
        frames: List of BGR images with identical shapes.

    Returns:
        A single ``float64`` image averaged across all inputs.
    """
    _validate_frames(frames)
    stack = np.stack(frames, axis=0).astype(np.float64)
    result = np.mean(stack, axis=0)
    logger.debug("Mean aggregation: averaged %d frames", len(frames))
    return result


def aggregate_median(frames: list[np.ndarray]) -> np.ndarray:
    """Compute the element-wise median across *frames*.

    The median is robust to transient foreground objects: if a person
    walks through the scene in some but not all frames, the median at
    each pixel will be the background value (assuming the person
    occupies less than half the frames at that pixel location).

    Args:
        frames: List of BGR images with identical shapes.

    Returns:
        A single ``float64`` image representing the per-pixel median.
    """
    _validate_frames(frames)
    stack = np.stack(frames, axis=0).astype(np.float64)
    result = np.median(stack, axis=0)
    logger.debug("Median aggregation: median of %d frames", len(frames))
    return result
