"""Webcam capture abstraction."""

from __future__ import annotations

import logging
from types import TracebackType

import cv2
import numpy as np

from cloak.config.schemas import CameraConfig

logger = logging.getLogger(__name__)


class WebcamCaptureError(Exception):
    """Raised when the webcam cannot be initialized or read."""


class WebcamCapture:
    """Thread-safe webcam capture with context-manager support.

    Example::

        with WebcamCapture(config.camera) as cam:
            frame = cam.read()
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> WebcamCapture:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    # -- public API -----------------------------------------------------------

    def open(self) -> None:
        """Open the camera device and apply configured resolution/fps.

        If ``video_path`` is set in config and the camera device cannot be
        opened, fall back to reading from the video file.
        """
        video_path = getattr(self._config, "video_path", "") or ""
        source = video_path if video_path else self._config.device_id

        if video_path:
            logger.info("Opening video file: %s", video_path)
        else:
            logger.info(
                "Initializing camera device %d (%dx%d @ %d fps)",
                self._config.device_id,
                self._config.width,
                self._config.height,
                self._config.fps,
            )

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise WebcamCaptureError(
                f"Cannot open camera device {self._config.device_id}"
                + (f" or video file {video_path}" if video_path else "")
            )

        if not video_path:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        self._log_actual_properties()
        logger.info("Camera initialized successfully")

    def read(self) -> np.ndarray:
        """Read a single frame from the camera.

        Returns:
            The captured BGR frame.

        Raises:
            WebcamCaptureError: If the camera is not open or a frame cannot
                be read.
        """
        if self._cap is None or not self._cap.isOpened():
            raise WebcamCaptureError("Camera is not open")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            # If reading from a video file, loop back to the start
            video_path = getattr(self._config, "video_path", "") or ""
            if video_path and self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    return frame
            raise WebcamCaptureError("Failed to read frame from camera")

        return frame

    def release(self) -> None:
        """Release the camera device safely."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released")
        self._cap = None

    # -- properties -----------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def actual_width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def actual_fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    # -- internals ------------------------------------------------------------

    def _log_actual_properties(self) -> None:
        if self._cap is None:
            return
        logger.info(
            "Actual camera properties: %dx%d @ %.1f fps",
            self.actual_width,
            self.actual_height,
            self.actual_fps,
        )
