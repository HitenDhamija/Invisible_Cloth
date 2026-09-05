"""Background model: capture, aggregate, and store a clean background frame."""

from __future__ import annotations

import enum
import logging
import time

import cv2
import numpy as np

from cloak.capture.aggregator import (
    AggregationError,
    aggregate_mean,
    aggregate_median,
)
from cloak.config.schemas import BackgroundConfig

logger = logging.getLogger(__name__)

_AGGREGATORS = {
    "median": aggregate_median,
    "mean": aggregate_mean,
}


class CaptureState(enum.Enum):
    """High-level state of the background-capture subsystem."""

    IDLE = "idle"
    COUNTDOWN = "countdown"
    CAPTURING = "capturing"
    READY = "ready"


class BackgroundCaptureError(Exception):
    """Raised when background capture fails irrecoverably."""


class BackgroundModel:
    """Stores and manages a background image for the invisibility cloak.

    Lifecycle::

        model = BackgroundModel(config.background)
        model.start_capture()          # begin countdown + capture
        while model.state != CaptureState.READY:
            model.process_frame(frame)  # feed live frames
        bg = model.background           # clean background image
    """

    def __init__(self, config: BackgroundConfig) -> None:
        self._config = config
        self._state = CaptureState.IDLE
        self._background: np.ndarray | None = None
        self._frame_buffer: list[np.ndarray] = []
        self._frames_collected = 0
        self._countdown_start: float = 0.0
        self._capture_start: float = 0.0
        self._show_debug = False

    # -- public properties -----------------------------------------------------

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def background(self) -> np.ndarray | None:
        return self._background

    @property
    def has_background(self) -> bool:
        return self._background is not None

    @property
    def debug_enabled(self) -> bool:
        return self._show_debug

    @debug_enabled.setter
    def debug_enabled(self, value: bool) -> None:
        self._show_debug = value

    # -- capture flow ----------------------------------------------------------

    def start_capture(self) -> None:
        """Begin the countdown-then-capture sequence."""
        logger.info("Starting background capture sequence")
        self._state = CaptureState.COUNTDOWN
        self._countdown_start = time.perf_counter()
        self._frame_buffer.clear()
        self._frames_collected = 0

    def recapture(self) -> None:
        """Restart capture from countdown (called when user presses B)."""
        logger.info("Background recapture requested")
        self.start_capture()

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed a live camera frame into the capture state machine.

        Returns:
            The same frame with status overlays drawn on it.
            The caller should display the returned frame.
        """
        display = frame.copy()

        if self._state == CaptureState.IDLE:
            self._draw_idle_status(display)

        elif self._state == CaptureState.COUNTDOWN:
            self._process_countdown(display)

        elif self._state == CaptureState.CAPTURING:
            self._process_capture(display, frame)

        if self._show_debug and self._background is not None:
            self._draw_debug_panel(display)

        return display

    # -- internal state handlers -----------------------------------------------

    def _process_countdown(self, display: np.ndarray) -> None:
        elapsed = time.perf_counter() - self._countdown_start
        remaining = self._config.countdown_seconds - elapsed

        if remaining <= 0:
            logger.info("Countdown finished, beginning frame capture")
            self._state = CaptureState.CAPTURING
            self._capture_start = time.perf_counter()
            self._draw_capturing_status(display)
            return

        count = int(remaining) + 1
        self._draw_centered_text(display, str(count), scale=4.0, color=(0, 255, 255))
        self._draw_centered_text(
            display,
            "STEP OUT OF THE FRAME",
            y_offset=-80,
            scale=0.9,
            color=(255, 255, 255),
        )
        self._draw_centered_text(
            display,
            "Background will be captured automatically",
            y_offset=80,
            scale=0.6,
            color=(180, 180, 180),
        )

    def _process_capture(self, display: np.ndarray, frame: np.ndarray) -> None:
        required = self._config.capture_frames

        # Validate frame dimensions
        if self._frame_buffer and frame.shape != self._frame_buffer[0].shape:
            logger.warning(
                "Frame dimension mismatch: expected %s, got %s — skipping frame",
                self._frame_buffer[0].shape,
                frame.shape,
            )
            self._draw_capturing_status(display)
            return

        # Collect the frame
        self._frame_buffer.append(frame.copy())
        self._frames_collected += 1
        logger.debug("Captured frame %d/%d", self._frames_collected, required)

        self._draw_capturing_status(display)

        if self._frames_collected >= required:
            self._finalise_background()

    def _finalise_background(self) -> None:
        """Aggregate collected frames into the final background image."""
        method = self._config.aggregation_method
        aggregator = _AGGREGATORS.get(method)
        if aggregator is None:
            raise BackgroundCaptureError(f"Unknown aggregation method: {method!r}")

        try:
            bg_float = aggregator(self._frame_buffer)
        except AggregationError as exc:
            logger.error("Background aggregation failed: %s", exc)
            self._state = CaptureState.IDLE
            return

        self._background = bg_float.astype(np.uint8)
        self._frame_buffer.clear()
        self._state = CaptureState.READY
        logger.info(
            "Background ready (%s aggregation of %d frames)",
            method,
            self._frames_collected,
        )

    # -- drawing helpers -------------------------------------------------------

    def _draw_idle_status(self, frame: np.ndarray) -> None:
        self._draw_centered_text(
            frame,
            "Press B to capture background",
            scale=0.7,
            color=(200, 200, 200),
        )

    def _draw_capturing_status(self, frame: np.ndarray) -> None:
        required = self._config.capture_frames
        count = self._frames_collected
        pct = int(count / required * 100)

        bar_width = frame.shape[1] - 40
        filled = int(bar_width * count / required)
        cv2.rectangle(frame, (20, 50), (20 + filled, 70), (0, 200, 0), -1)
        cv2.rectangle(frame, (20, 50), (20 + bar_width, 70), (200, 200, 200), 2)

        self._draw_centered_text(
            frame,
            "KEEP STILL - Capturing background",
            y_offset=-80,
            scale=0.7,
            color=(0, 255, 0),
        )
        self._draw_centered_text(
            frame,
            f"{count}/{required} frames ({pct}%)",
            y_offset=100,
            scale=0.6,
            color=(255, 255, 255),
        )

    def _draw_debug_panel(self, frame: np.ndarray) -> None:
        """Show a small thumbnail of the stored background in the corner."""
        if self._background is None:
            return
        h, w = frame.shape[:2]
        thumb_h, thumb_w = 120, 160
        thumb = cv2.resize(self._background, (thumb_w, thumb_h))
        y0, y1 = h - thumb_h - 10, h - 10
        x0, x1 = w - thumb_w - 10, w - 10
        frame[y0:y1, x0:x1] = thumb
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(
            frame,
            "DEBUG: Background",
            (x0, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    @staticmethod
    def _draw_centered_text(
        frame: np.ndarray,
        text: str,
        *,
        y_offset: int = 0,
        scale: float = 1.0,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2 + y_offset
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)
