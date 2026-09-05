"""Optional illumination compensation before HSV detection.

Applies brightness normalization and/or CLAHE to the V channel
to compensate for changing lighting conditions (indoor lights,
window light, shadows).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from cloak.config.schemas import AdaptiveConfig

logger = logging.getLogger(__name__)


class AdaptivePreprocessor:
    """Preprocess frames to compensate for illumination changes.

    Example::

        preprocessor = AdaptivePreprocessor(config.adaptive)
        processed = preprocessor.preprocess(frame)
        raw_mask = detector.detect(processed)
    """

    def __init__(self, config: AdaptiveConfig) -> None:
        self._cfg = config
        self._clahe: cv2.CLAHE | None = None
        if config.enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip,
                tileGridSize=(config.clahe_grid, config.clahe_grid),
            )

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply illumination compensation to frame.

        Args:
            frame: Input BGR frame (uint8).

        Returns:
            Processed BGR frame (uint8). Returns original if disabled.
        """
        if not self._cfg.enabled:
            return frame

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Brightness normalization: scale V so mean stays near 128
        if self._cfg.brightness_normalize:
            v_mean = float(np.mean(v))
            if v_mean > 1.0:
                scale = 128.0 / v_mean
                v = np.clip(v.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        # CLAHE on V channel
        if self._clahe is not None:
            v = self._clahe.apply(v)

        # Reassemble and convert back to BGR
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
