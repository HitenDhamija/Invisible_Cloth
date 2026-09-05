"""Interactive HSV calibration mode using OpenCV trackbars.

Opens a window with six trackbars (H/S/V min and max) that update the
detector bounds in real time.  The current values are printed to the
console so they can be copied into ``configs/default.yaml``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from cloak.detection.detector import BlueColorDetector

logger = logging.getLogger(__name__)

_WINDOW = "HSV Calibrator"
_TRACKBAR_H_MIN = "H min"
_TRACKBAR_H_MAX = "H max"
_TRACKBAR_S_MIN = "S min"
_TRACKBAR_S_MAX = "S max"
_TRACKBAR_V_MIN = "V min"
_TRACKBAR_V_MAX = "V max"


class HSVCalibrator:
    """Interactive trackbar-based HSV range calibrator.

    Usage::

        cal = HSVCalibrator(detector)
        while True:
            frame = cam.read()
            key = cal.step(frame)
            if key == 27:
                break
    """

    def __init__(self, detector: BlueColorDetector) -> None:
        self._detector = detector
        self._ready = False

    def setup(self, initial_lower: list[int], initial_upper: list[int]) -> None:
        """Create the calibrator window and trackbars."""
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WINDOW, 600, 400)

        cv2.createTrackbar(
            _TRACKBAR_H_MIN, _WINDOW, initial_lower[0], 179, lambda _: None
        )
        cv2.createTrackbar(
            _TRACKBAR_H_MAX, _WINDOW, initial_upper[0], 179, lambda _: None
        )
        cv2.createTrackbar(
            _TRACKBAR_S_MIN, _WINDOW, initial_lower[1], 255, lambda _: None
        )
        cv2.createTrackbar(
            _TRACKBAR_S_MAX, _WINDOW, initial_upper[1], 255, lambda _: None
        )
        cv2.createTrackbar(
            _TRACKBAR_V_MIN, _WINDOW, initial_lower[2], 255, lambda _: None
        )
        cv2.createTrackbar(
            _TRACKBAR_V_MAX, _WINDOW, initial_upper[2], 255, lambda _: None
        )

        self._ready = True
        logger.info("HSV Calibrator window created")

    def step(self, frame: np.ndarray) -> int:
        """Read trackbars, update detector, display panels, return last key.

        Returns the key code from ``cv2.waitKey``.
        """
        if not self._ready:
            self.setup(
                self._detector.lower_bound.tolist(),
                self._detector.upper_bound.tolist(),
            )

        lower, upper = self._read_trackbars()
        self._detector.set_bounds(lower, upper)

        mask, stats = self._detector.detect(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_region = self._detector.detect_blue_region(frame, mask)

        self._display_panels(frame, hsv, mask, blue_region, lower, upper, stats)

        return cv2.waitKey(1) & 0xFF

    def print_values(self) -> None:
        """Print the current HSV bounds to the console."""
        lower = self._detector.lower_bound.tolist()
        upper = self._detector.upper_bound.tolist()
        print("\n--- Current HSV Calibration ---")
        print(f"hsv_lower: {lower}")
        print(f"hsv_upper: {upper}")
        print("Copy these values into configs/default.yaml\n")

    def destroy(self) -> None:
        cv2.destroyWindow(_WINDOW)

    # -- internals ------------------------------------------------------------

    def _read_trackbars(self) -> tuple[list[int], list[int]]:
        h_min = cv2.getTrackbarPos(_TRACKBAR_H_MIN, _WINDOW)
        h_max = cv2.getTrackbarPos(_TRACKBAR_H_MAX, _WINDOW)
        s_min = cv2.getTrackbarPos(_TRACKBAR_S_MIN, _WINDOW)
        s_max = cv2.getTrackbarPos(_TRACKBAR_S_MAX, _WINDOW)
        v_min = cv2.getTrackbarPos(_TRACKBAR_V_MIN, _WINDOW)
        v_max = cv2.getTrackbarPos(_TRACKBAR_V_MAX, _WINDOW)
        lower = [h_min, s_min, v_min]
        upper = [h_max, s_max, v_max]
        return lower, upper

    @staticmethod
    def _display_panels(
        frame: np.ndarray,
        hsv: np.ndarray,
        mask: np.ndarray,
        blue_region: np.ndarray,
        lower: list[int],
        upper: list[int],
        stats: object,
    ) -> None:
        h, w = frame.shape[:2]
        panel_h, panel_w = h // 2, w // 2

        orig_small = cv2.resize(frame, (panel_w, panel_h))
        hsv_small = cv2.resize(hsv, (panel_w, panel_h))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_bgr, (panel_w, panel_h))
        blue_small = cv2.resize(blue_region, (panel_w, panel_h))

        top = np.hstack([orig_small, hsv_small])
        bottom = np.hstack([mask_small, blue_small])
        combined = np.vstack([top, bottom])

        # Overlay labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [
            ("Original", 10, 25),
            ("HSV", panel_w + 10, 25),
            ("Mask", 10, panel_h + 25),
            ("Blue Region", panel_w + 10, panel_h + 25),
        ]
        for text, x, y in labels:
            cv2.putText(combined, text, (x, y), font, 0.6, (0, 255, 255), 2)

        # Bounds + stats
        info = f"Lower: {lower}  Upper: {upper}  Blue: {stats.cloak_ratio:.1%}"
        cv2.putText(
            combined, info, (10, combined.shape[0] - 10),
            font, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow(_WINDOW, combined)
