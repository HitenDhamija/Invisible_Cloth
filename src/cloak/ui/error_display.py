"""In-application error message display.

Shows user-correctable error messages on the frame instead of only
logging to console.

Usage::

    error_display = ErrorDisplay()
    error_display.show("No camera detected")
    error_display.render(frame)
"""

from __future__ import annotations

import time

import cv2
import numpy as np


class ErrorDisplay:
    """Display timed error messages on the frame.

    Example::

        err = ErrorDisplay()
        err.show("No camera detected — check connection")
        while True:
            err.render(frame)
    """

    def __init__(self, duration_seconds: float = 5.0) -> None:
        self._message: str | None = None
        self._start_time: float = 0.0
        self._duration = duration_seconds

    @property
    def active(self) -> bool:
        return self._message is not None

    @property
    def message(self) -> str | None:
        return self._message

    def show(self, message: str, duration: float | None = None) -> None:
        """Display an error message.

        Args:
            message: Error text to display.
            duration: How long to show (seconds). None = default.
        """
        self._message = message
        self._start_time = time.perf_counter()
        if duration is not None:
            self._duration = duration

    def clear(self) -> None:
        """Clear the error message."""
        self._message = None

    def render(self, frame: np.ndarray) -> None:
        """Draw error message on the frame (in-place).

        Auto-hides after the configured duration.
        """
        if self._message is None:
            return

        elapsed = time.perf_counter() - self._start_time
        if elapsed > self._duration:
            self._message = None
            return

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Fade out in last second
        remaining = self._duration - elapsed
        alpha = max(0.0, remaining) if remaining < 1.0 else 1.0

        # Error bar at top
        bar_h = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7 * alpha, frame, 1.0 - 0.7 * alpha, 0, frame)

        # Error icon and text
        text = f"! {self._message}"
        text_size = cv2.getTextSize(text, font, 0.55, 1)[0]
        tx = max(10, (w - text_size[0]) // 2)
        cv2.putText(
            frame, text, (tx, bar_h - 12),
            font, 0.55, (255, 255, 255), 1,
        )
