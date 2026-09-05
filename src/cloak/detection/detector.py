"""Blue-cloth detection using HSV color-space thresholding.

Pipeline::

    BGR frame
    → optional Gaussian smoothing
    → BGR → HSV conversion
    → cv2.inRange() with configurable bounds
    → optional morphological cleanup
    → binary mask (uint8, 0 or 255)

Why HSV over BGR/RGB for color segmentation:

    In BGR/RGB, a single color (e.g. blue) is spread across three
    channels.  A blue pixel might be (200, 50, 30) and a shadow-blue
    pixel (100, 25, 15) — their Euclidean distance in BGR space is
    large even though both are perceptually "blue".  Lighting changes
    shift all three channels simultaneously, making fixed-threshold
    segmentation fragile.

    HSV decouples *chrominance* (Hue) from *intensity* (Value) and
    *purity* (Saturation).  A blue pixel always has Hue ≈ 100–130
    regardless of brightness, so a single Hue range covers bright
    blue, dark blue, and shadowed folds.  Saturation and Value bounds
    further filter out grey/white noise without affecting the Hue
    selectivity.
"""

from __future__ import annotations

import dataclasses
import logging

import cv2
import numpy as np

from cloak.config.schemas import DetectionConfig, ProcessingConfig

logger = logging.getLogger(__name__)

# Safety threshold: if more than this fraction of pixels are detected
# as "cloak", the thresholds are probably too broad.
_SAFETY_RATIO = 0.85


@dataclasses.dataclass(frozen=True)
class DetectionStats:
    """Per-frame detection statistics."""

    total_pixels: int
    cloak_pixels: int
    cloak_ratio: float
    warning: str | None


class BlueColorDetector:
    """Detect blue-cloth pixels via HSV thresholding.

    Example::

        detector = BlueColorDetector(detection_config, processing_config)
        mask = detector.detect(frame)
        stats = detector.last_stats
    """

    def __init__(
        self,
        detection: DetectionConfig,
        processing: ProcessingConfig,
    ) -> None:
        self._detection = detection
        self._processing = processing
        self._lower = np.array(detection.hsv_lower, dtype=np.uint8)
        self._upper = np.array(detection.hsv_upper, dtype=np.uint8)
        self._stats = DetectionStats(0, 0, 0.0, None)

    # -- public API -----------------------------------------------------------

    @property
    def last_stats(self) -> DetectionStats:
        return self._stats

    @property
    def lower_bound(self) -> np.ndarray:
        return self._lower.copy()

    @property
    def upper_bound(self) -> np.ndarray:
        return self._upper.copy()

    def set_bounds(self, lower: list[int], upper: list[int]) -> None:
        """Update HSV bounds at runtime (used by the calibrator)."""
        self._lower = np.array(lower, dtype=np.uint8)
        self._upper = np.array(upper, dtype=np.uint8)
        logger.debug("HSV bounds updated: %s — %s", lower, upper)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, DetectionStats]:
        """Run the full detection pipeline on a BGR frame.

        Args:
            frame: Input BGR image (uint8).

        Returns:
            A tuple of (binary_mask, stats) where *binary_mask* is a
            single-channel uint8 image with values 0 or 255.
        """
        smoothed = self._smooth(frame)
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = self._morphology(mask)
        self._stats = self._compute_stats(mask, frame.shape)
        return mask, self._stats

    def detect_blue_region(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return the original frame with non-blue areas blacked out."""
        return cv2.bitwise_and(frame, frame, mask=mask)

    # -- pipeline stages ------------------------------------------------------

    def _smooth(self, frame: np.ndarray) -> np.ndarray:
        k = self._processing.blur_kernel
        if k <= 1:
            return frame
        # Ensure kernel is odd
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    def _morphology(self, mask: np.ndarray) -> np.ndarray:
        k = self._processing.morphology_kernel
        if k <= 1:
            return mask
        k = k if k % 2 == 1 else k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # Close small holes inside the cloth
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Remove small noise outside the cloth
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # -- statistics -----------------------------------------------------------

    def _compute_stats(self, mask: np.ndarray, frame_shape: tuple[int, ...]) -> DetectionStats:
        total = mask.shape[0] * mask.shape[1]
        cloak = int(cv2.countNonZero(mask))
        ratio = cloak / total if total > 0 else 0.0

        warning: str | None = None
        if ratio > _SAFETY_RATIO:
            warning = f"Detection threshold may be too broad — {ratio:.0%} of frame is blue"
            logger.warning(warning)

        return DetectionStats(
            total_pixels=total,
            cloak_pixels=cloak,
            cloak_ratio=ratio,
            warning=warning,
        )
