"""Person-aware blue cloth detection.

Combines HSV color detection with person segmentation to
reduce false positives from blue objects not on a person.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from cloak.config.schemas import AIConfig, DetectionConfig, ProcessingConfig
from cloak.detection.detector import BlueColorDetector, DetectionStats
from cloak.detection.person import PersonDetector

logger = logging.getLogger(__name__)


class PersonAwareDetector:
    """Detect blue cloth only on detected person regions.

    Example::

        detector = PersonAwareDetector(detection_cfg, processing_cfg, ai_cfg)
        mask, stats = detector.detect(frame)
    """

    def __init__(
        self,
        detection_cfg: DetectionConfig,
        processing_cfg: ProcessingConfig,
        ai_cfg: AIConfig,
    ) -> None:
        self._hsv_detector = BlueColorDetector(detection_cfg, processing_cfg)
        self._person_detector = PersonDetector(ai_cfg)
        self._ai_cfg = ai_cfg
        self._stats = DetectionStats(
            total_pixels=0, cloak_pixels=0, cloak_ratio=0.0, warning=None,
        )

    @property
    def hsv_detector(self) -> BlueColorDetector:
        return self._hsv_detector

    @property
    def person_detector(self) -> PersonDetector:
        return self._person_detector

    @property
    def lower_bound(self) -> np.ndarray:
        return self._hsv_detector.lower_bound

    @property
    def upper_bound(self) -> np.ndarray:
        return self._hsv_detector.upper_bound

    def set_bounds(self, lower: list[int], upper: list[int]) -> None:
        self._hsv_detector.set_bounds(lower, upper)

    def detect(
        self, frame: np.ndarray, timestamp_ms: int = 0,
    ) -> tuple[np.ndarray, DetectionStats]:
        """Detect blue cloth constrained to person region.

        Args:
            frame: BGR frame (uint8).
            timestamp_ms: Frame timestamp for MediaPipe tracking.

        Returns:
            Tuple of (binary_mask, DetectionStats).
        """
        # Step 1: HSV blue detection (always runs)
        blue_mask, hsv_stats = self._hsv_detector.detect(frame)

        # Step 2: Person segmentation
        person_mask = self._person_detector.detect(frame, timestamp_ms)

        # Step 3: Intersect — blue AND person
        person_binary = (person_mask >= self._ai_cfg.person_threshold).astype(np.uint8) * 255

        # If no person detected and fallback enabled, use pure HSV mask
        person_pixels = np.count_nonzero(person_binary)
        if person_pixels == 0 and self._ai_cfg.fallback_to_hsv:
            logger.debug("No person detected, falling back to pure HSV mask")
            self._stats = hsv_stats
            return blue_mask, self._stats

        # Bitwise AND: keep only blue pixels that are on a person
        final_mask = cv2.bitwise_and(blue_mask, person_binary)

        # Recompute stats for the constrained mask
        total_pixels = frame.shape[0] * frame.shape[1]
        cloak_pixels = int(np.count_nonzero(final_mask))
        cloak_ratio = cloak_pixels / total_pixels if total_pixels > 0 else 0.0

        self._stats = DetectionStats(
            total_pixels=total_pixels,
            cloak_pixels=cloak_pixels,
            cloak_ratio=cloak_ratio,
            warning=None,
        )

        return final_mask, self._stats

    def detect_blue_region(
        self, frame: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        """Extract blue region using mask."""
        return self._hsv_detector.detect_blue_region(frame, mask)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._person_detector.close()
