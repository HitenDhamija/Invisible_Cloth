"""Temporal mask smoothing to reduce frame-to-frame flicker.

Uses Exponential Moving Average (EMA) on binary masks:

    accumulated = alpha * current + (1-alpha) * previous
    smoothed = (accumulated >= threshold) ? 255 : 0

Combined with per-pixel persistence counters to prevent rapid
disappearance of cloak pixels that fail detection for 1-2 frames.
"""

from __future__ import annotations

import logging

import numpy as np

from cloak.config.schemas import TemporalConfig

logger = logging.getLogger(__name__)


class TemporalMaskSmoother:
    """Smooth binary masks across frames using EMA and persistence.

    Example::

        smoother = TemporalMaskSmoother(config.temporal)
        for frame in video:
            raw_mask = detect(frame)
            smooth_mask = smoother.smooth(raw_mask)
            # use smooth_mask for compositing
    """

    def __init__(self, config: TemporalConfig) -> None:
        self._cfg = config
        self._accumulated: np.ndarray | None = None
        self._persistence: np.ndarray | None = None
        self._frame_count: int = 0

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def smooth(self, mask: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to a binary mask.

        Args:
            mask: Current binary mask (uint8, 0 or 255).

        Returns:
            Temporally smoothed binary mask (uint8, 0 or 255).
        """
        if not self._cfg.enabled:
            return mask

        h, w = mask.shape[:2]

        # Initialize state on first frame
        if self._accumulated is None or self._accumulated.shape != (h, w):
            self._accumulated = mask.astype(np.float32)
            self._persistence = np.zeros((h, w), dtype=np.uint8)
            self._frame_count = 0

            # Set persistence counters for active pixels on first frame
            persistence = self._cfg.persistence_frames
            if persistence > 0:
                active = mask == 255
                self._persistence[active] = np.uint8(persistence)

            return mask

        self._frame_count += 1
        alpha = self._cfg.ema_alpha

        # EMA accumulation
        current_float = mask.astype(np.float32)
        self._accumulated = alpha * current_float + (1.0 - alpha) * self._accumulated

        # Threshold: midpoint of the alpha-scaled range
        threshold = alpha * 127.0
        binary = np.where(
            self._accumulated >= threshold, np.uint8(255), np.uint8(0)
        )

        # Persistence: keep recently-active pixels ON
        persistence = self._cfg.persistence_frames
        if persistence > 0:
            # Pixels currently ON reset their counter
            active = mask == 255
            self._persistence[active] = np.uint8(persistence)

            # Pixels currently OFF but with counter > 0 stay ON
            warm = self._persistence > 0
            binary[warm] = np.uint8(255)

            # Decrement counters for OFF pixels
            self._persistence[~active & warm] -= 1

        return binary

    def reset(self) -> None:
        """Clear all accumulated state. Call on background recapture."""
        self._accumulated = None
        self._persistence = None
        self._frame_count = 0
        logger.debug("Temporal smoother reset")
