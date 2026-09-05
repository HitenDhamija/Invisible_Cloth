"""Compositing the invisibility-cloak effect.

Two rendering modes are available:

Hard mask (bitwise)::

    inverse_mask = 255 - mask
    cloak_region = background AND mask
    visible_region = frame AND inverse_mask
    output = cloak_region OR visible_region

This produces crisp, pixel-exact replacement. Every cloak pixel is
replaced 1:1 with the corresponding background pixel.

Soft blend (alpha)::

    alpha_mask = mask / 255.0          # float in [0, 1]
    output = alpha_mask * background
           + (1 - alpha_mask) * live_frame

This produces a smooth transition at mask boundaries. Cloth edges
appear slightly transparent rather than hard-cut, which hides
imperfect mask boundaries and avoids visible blue outlines.

The soft mask from the refiner (feathered edges) is used when
available; otherwise the binary mask is normalized to [0, 1].
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from cloak.config.schemas import RenderingConfig

logger = logging.getLogger(__name__)


class RenderError(Exception):
    """Raised when rendering cannot proceed due to invalid inputs."""


class InvisibilityRenderer:
    """Composite the background into the cloak region of each frame.

    Example::

        renderer = InvisibilityRenderer(config.rendering)
        output = renderer.render(frame, background, mask)
    """

    def __init__(self, config: RenderingConfig) -> None:
        self._cfg = config

    @property
    def use_soft_blend(self) -> bool:
        return self._cfg.use_soft_blend

    @use_soft_blend.setter
    def use_soft_blend(self, value: bool) -> None:
        self._cfg = self._cfg.model_copy(update={"use_soft_blend": value})

    def render(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        binary_mask: np.ndarray,
        soft_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Composite the invisibility effect onto *frame*.

        Args:
            frame: Current live BGR frame (uint8).
            background: Captured background BGR frame (uint8).
            binary_mask: Refined binary mask (uint8, 0 or 255).
            soft_mask: Optional feathered mask (float32, 0.0–1.0).
                Used when ``use_soft_blend`` is True. If *None* while
                soft blend is requested, the binary mask is used as fallback.

        Returns:
            The composited BGR frame (uint8).

        Raises:
            RenderError: If dimensions do not match or inputs are invalid.
        """
        self._validate(frame, background, binary_mask)

        if self._cfg.use_soft_blend:
            return self._soft_blend(frame, background, binary_mask, soft_mask)
        return self._hard_composite(frame, background, binary_mask)

    # -- hard composite -------------------------------------------------------

    @staticmethod
    def _hard_composite(
        frame: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Bitwise hard replacement.

        cloak_region = background AND mask
        visible_region = frame AND inverse_mask
        output = cloak_region OR visible_region
        """
        inverse_mask = cv2.bitwise_not(mask)

        cloak_region = cv2.bitwise_and(background, background, mask=mask)
        visible_region = cv2.bitwise_and(frame, frame, mask=inverse_mask)

        return cv2.bitwise_or(cloak_region, visible_region)

    # -- soft blend -----------------------------------------------------------

    @staticmethod
    def _soft_blend(
        frame: np.ndarray,
        background: np.ndarray,
        binary_mask: np.ndarray,
        soft_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Alpha-weighted blending.

        output = alpha * background + (1 - alpha) * frame

        Where alpha = soft_mask (or normalized binary_mask as fallback).
        """
        if soft_mask is not None and soft_mask.shape == frame.shape[:2]:
            alpha = soft_mask
        else:
            alpha = binary_mask.astype(np.float32) / 255.0

        # Ensure 3-channel for broadcasting
        alpha_3 = alpha[:, :, np.newaxis] if alpha.ndim == 2 else alpha

        bg_f = background.astype(np.float32)
        fg_f = frame.astype(np.float32)

        blended = alpha_3 * bg_f + (1.0 - alpha_3) * fg_f
        return np.clip(blended, 0, 255).astype(np.uint8)

    # -- validation -----------------------------------------------------------

    @staticmethod
    def _validate(
        frame: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if frame.shape != background.shape:
            raise RenderError(
                f"Frame shape {frame.shape} != background shape {background.shape}"
            )
        if mask.shape[:2] != frame.shape[:2]:
            raise RenderError(
                f"Mask shape {mask.shape} != frame shape {frame.shape}"
            )
