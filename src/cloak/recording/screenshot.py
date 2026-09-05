"""Screenshot capture to disk.

Saves timestamped screenshots under ``outputs/screenshots/``.

Usage::

    screencapper = ScreenCapturer()
    path = screencapper.capture(frame, "render")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs"
_SCREENSHOTS_DIR = _OUTPUTS_DIR / "screenshots"


class ScreenCapturer:
    """Capture screenshots to disk.

    Example::

        cap = ScreenCapturer()
        path = cap.capture(frame, label="render")
    """

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self._dir = Path(output_dir) if output_dir else _SCREENSHOTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._count: int = 0

    @property
    def count(self) -> int:
        return self._count

    def capture(
        self,
        frame: np.ndarray,
        label: str = "",
        quality: int = 95,
    ) -> Path:
        """Save a screenshot.

        Args:
            frame: BGR frame (uint8, H x W x 3).
            label: Optional label for the filename (e.g. 'render', 'debug').
            quality: JPEG compression quality (1-100).

        Returns:
            Path to the saved screenshot.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        filename = f"cloak_{timestamp}{suffix}.jpg"
        path = self._dir / filename

        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success = cv2.imwrite(str(path), frame, params)

        if success:
            self._count += 1
            logger.info("Screenshot saved: %s", path)
        else:
            logger.warning("Failed to save screenshot: %s", path)

        return path

    def capture_pair(
        self,
        render_frame: np.ndarray,
        debug_frame: np.ndarray | None = None,
    ) -> list[Path]:
        """Capture both render and optional debug screenshots.

        Args:
            render_frame: The final rendered output.
            debug_frame: Optional debug view frame.

        Returns:
            List of paths to saved screenshots.
        """
        paths = [self.capture(render_frame, "render")]
        if debug_frame is not None:
            paths.append(self.capture(debug_frame, "debug"))
        return paths
