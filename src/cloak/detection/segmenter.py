"""Segmenter protocol and AI-hybrid blue cloth detection.

Defines a common interface for all detectors and implements the
AI-hybrid pipeline: person segmentation (ONNX) + HSV color detection.

Architecture::

    SegmenterProtocol        -- common detect() interface
    │
    ├── BlueColorDetector    -- HSV-only (existing)
    ├── PersonAwareDetector  -- MediaPipe + HSV (existing)
    └── AIHybridDetector     -- ONNX person seg + HSV (new)

The AI hybrid detector does NOT assume the AI model knows anything
about "blue cloth". Instead:

    1. AI model segments the person region (class-agnostic or person class)
    2. HSV detector finds blue pixels within that region
    3. Intersection = clean cloak mask with zero background false positives

This hybrid approach is more reliable than either method alone because:
- HSV alone false-positives on blue backgrounds (posters, sky, furniture)
- AI alone cannot distinguish blue cloth from other clothing
- Combined: spatial prior (person) + color prior (blue) = robust
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from cloak.config.schemas import AIConfig, DetectionConfig, ProcessingConfig
from cloak.detection.detector import BlueColorDetector, DetectionStats
from cloak.detection.model_manager import ModelManager, ModelManagerError

logger = logging.getLogger(__name__)


class AIHybridDetector:
    """Detect blue cloth using AI person segmentation + HSV color detection.

    Hybrid pipeline per frame::

        frame
        ├── AI model  ->  person_mask (binary, full frame)
        ├── HSV det   ->  blue_mask (binary, full frame)
        └── AND       ->  final_mask (blue only on person)

    Falls back to pure HSV if the AI model is unavailable and
    ``fallback_to_hsv`` is enabled.

    Example::

        detector = AIHybridDetector(detection_cfg, processing_cfg, ai_cfg)
        mask, stats = detector.detect(frame)
        # stats includes ai_latency_ms for benchmarking
    """

    def __init__(
        self,
        detection_cfg: DetectionConfig,
        processing_cfg: ProcessingConfig,
        ai_cfg: AIConfig,
    ) -> None:
        self._hsv_detector = BlueColorDetector(detection_cfg, processing_cfg)
        self._model_manager = ModelManager(ai_cfg)
        self._ai_cfg = ai_cfg
        self._stats = DetectionStats(
            total_pixels=0, cloak_pixels=0, cloak_ratio=0.0, warning=None,
        )

        # Frame skipping for inference optimization
        self._frame_counter: int = 0
        self._cached_person_mask: np.ndarray | None = None
        self._ai_latency_ms: float = 0.0

    # -- public API -----------------------------------------------------------

    @property
    def hsv_detector(self) -> BlueColorDetector:
        return self._hsv_detector

    @property
    def model_manager(self) -> ModelManager:
        return self._model_manager

    @property
    def lower_bound(self) -> np.ndarray:
        return self._hsv_detector.lower_bound

    @property
    def upper_bound(self) -> np.ndarray:
        return self._hsv_detector.upper_bound

    @property
    def last_ai_latency_ms(self) -> float:
        """Latency of the last AI inference in milliseconds."""
        return self._ai_latency_ms

    @property
    def last_person_mask(self) -> np.ndarray | None:
        """Most recent AI person segmentation mask (for debugging)."""
        return self._cached_person_mask

    def set_bounds(self, lower: list[int], upper: list[int]) -> None:
        """Update HSV bounds at runtime."""
        self._hsv_detector.set_bounds(lower, upper)

    def detect(
        self, frame: np.ndarray, timestamp_ms: int = 0,
    ) -> tuple[np.ndarray, DetectionStats]:
        """Detect blue cloth constrained to AI-segmented person region.

        Args:
            frame: BGR frame (uint8, H x W x 3).
            timestamp_ms: Unused, kept for interface compatibility.

        Returns:
            Tuple of (binary_mask, DetectionStats).
        """
        h, w = frame.shape[:2]

        # Step 1: HSV blue detection (always runs)
        blue_mask, hsv_stats = self._hsv_detector.detect(frame)

        # Step 2: AI person segmentation (with frame skipping)
        person_mask = self._get_person_mask(frame, h, w)

        # Step 3: If no person detected, fall back to pure HSV
        if person_mask is None or np.count_nonzero(person_mask) == 0:
            if self._ai_cfg.fallback_to_hsv:
                logger.debug("No person detected by AI, falling back to pure HSV")
                self._stats = hsv_stats
                return blue_mask, self._stats
            else:
                # Return empty mask
                self._stats = DetectionStats(
                    total_pixels=h * w, cloak_pixels=0,
                    cloak_ratio=0.0, warning=None,
                )
                return np.zeros((h, w), dtype=np.uint8), self._stats

        # Step 4: Intersect — blue AND person
        final_mask = cv2.bitwise_and(blue_mask, person_mask)

        # Recompute stats
        total_pixels = h * w
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
        """Return original frame with non-blue areas blacked out."""
        return self._hsv_detector.detect_blue_region(frame, mask)

    def close(self) -> None:
        """Release model resources."""
        self._model_manager.close()

    # -- internals ------------------------------------------------------------

    def _get_person_mask(
        self, frame: np.ndarray, h: int, w: int,
    ) -> np.ndarray | None:
        """Get person mask with frame skipping and caching.

        Returns None if no person detected, or the cached mask from
        a previous inference frame.
        """
        self._frame_counter += 1
        skip = self._ai_cfg.inference_frame_skip

        # Run inference on first frame or every N-th frame
        if self._frame_counter % skip == 0 or self._cached_person_mask is None:
            try:
                self._model_manager.ensure_loaded()
                raw_mask = self._model_manager.predict(frame)
                self._ai_latency_ms = self._model_manager.last_latency_ms

                # Ensure mask matches frame dimensions
                if raw_mask.shape != (h, w):
                    raw_mask = cv2.resize(
                        raw_mask, (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                self._cached_person_mask = raw_mask
            except ModelManagerError as exc:
                logger.warning("AI person segmentation failed: %s", exc)
                self._ai_latency_ms = 0.0
                return None

        return self._cached_person_mask
