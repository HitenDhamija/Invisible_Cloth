"""Video recording using OpenCV VideoWriter.

Saves timestamped recordings under ``outputs/videos/``.

Usage::

    recorder = VideoRecorder(config)
    recorder.start(width, height, fps)
    recorder.write(frame)
    recorder.stop()
    # or use as context manager
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs"
_VIDEOS_DIR = _OUTPUTS_DIR / "videos"


class RecorderError(Exception):
    """Raised when recording fails."""


class VideoRecorder:
    """Record video frames to a file.

    Example::

        recorder = VideoRecorder()
        recorder.start(640, 480, 30.0)
        while capturing:
            recorder.write(frame)
        recorder.stop()
    """

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self._dir = Path(output_dir) if output_dir else _VIDEOS_DIR
        self._writer: cv2.VideoWriter | None = None
        self._recording = False
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._current_path: Path | None = None
        self._consecutive_failures: int = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def current_file(self) -> Path | None:
        return self._current_path

    def start(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> Path:
        """Start recording.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Frames per second.
            codec: FourCC codec string (e.g. 'mp4v', 'XVID').

        Returns:
            Path to the output file.

        Raises:
            RecorderError: If already recording or codec unavailable.
        """
        if self._recording:
            raise RecorderError("Already recording")

        self._dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"cloak_{timestamp}.mp4"
        path = self._dir / filename

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RecorderError(
                f"Could not open video writer for codec '{codec}'. "
                "Try 'XVID' or check if codec is available."
            )

        self._writer = writer
        self._recording = True
        self._start_time = time.perf_counter()
        self._frame_count = 0
        self._current_path = path

        logger.info("Recording started: %s (%dx%d @ %.0ffps)", path, width, height, fps)
        return path

    def write(self, frame: np.ndarray) -> bool:
        """Write a frame to the recording.

        Args:
            frame: BGR frame (uint8, H x W x 3).

        Returns:
            True if frame was written, False if not recording.
        """
        if not self._recording or self._writer is None:
            return False

        try:
            self._writer.write(frame)
            self._frame_count += 1
            self._consecutive_failures = 0
            return True
        except Exception as exc:
            self._consecutive_failures += 1
            logger.warning("Failed to write frame (%d consecutive): %s",
                          self._consecutive_failures, exc, exc_info=True)
            if self._consecutive_failures >= 10:
                logger.error("Too many consecutive write failures, stopping recording")
                self.stop()
            return False

    def stop(self) -> Path | None:
        """Stop recording and release resources.

        Returns:
            Path to the saved recording, or None if not recording.
        """
        if not self._recording:
            return None

        path = self._current_path
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._recording = False
        duration = time.perf_counter() - self._start_time

        if path:
            logger.info(
                "Recording stopped: %s (%d frames, %.1fs)",
                path, self._frame_count, duration,
            )

        return path

    def close(self) -> None:
        """Release resources (alias for stop)."""
        self.stop()
